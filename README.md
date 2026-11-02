# TuringServe

An LLM inference server built from scratch for NVIDIA **Turing** GPUs: the free Kaggle/Colab **T4**,
the **GTX 16xx** and the **RTX 20xx**. OpenAI-compatible API, paged KV cache, continuous batching,
chunked prefill, Triton **tensor-core** kernels, CUDA graphs, prefix caching, INT8 weights,
speculative decoding (prompt lookup + draft model) and a self-tuning controller called **Autopilot**.

> Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (what and why) ·
> Procedure: [docs/BUILD_GUIDE.md](docs/BUILD_GUIDE.md) (order, commands, tests) ·
> Measurements and decisions: [docs/lab_notes.md](docs/lab_notes.md)

**Why Turing needs its own engine.** FlashAttention-2 needs sm80+. Triton >= 3.3 emits no tensor-core
(MMA) instructions for compute capability < 8.0, so the default attention in current vLLM on a T4 runs on
FMA units (vLLM issue #60821: 10.2 s for an 8,000-token prefill on Qwen2.5-1.5B). TuringServe pins
**torch 2.6.0 / Triton 3.2.0**, the last release that still emits `mma.sync` on sm_75, and writes its own
paged-attention kernels. *The headline claim is only made after it is measured* (`benchmarks/REPORT.md`).

## Status

Everything in the architecture is implemented. What has and has not been verified so far:

| Area | State |
|---|---|
| Model (Qwen2), loader, paged KV cache, scheduler, engine, sampler, detokenizer, stop strings, API, metrics, prefix cache, speculative decoding (n-gram + draft), INT8 reference path, Autopilot controller, tuner cache | Implemented; **covered by CPU tests that pass** (tiny Qwen2 vs Hugging Face: logits <= 1e-4, layer-wise, greedy tokens identical) |
| Triton kernels (rmsnorm, silu_mul, rope, kv_write, decode v1/v2/v3, prefill, w8a16), tuner measurement, CUDA graphs, memory profiler | Implemented to the spec; **GPU tests are written (`pytest -m gpu`) but have not been run yet** - they need a CUDA machine with torch 2.6 / Triton 3.2 |
| Benchmarks, quality evals, Kaggle notebooks, soak/chaos | Harness written (`chaos.py` passes against the tiny CPU server); **no T4 numbers exist yet** |

No performance number in this repository is claimed until it appears in `benchmarks/REPORT.md`,
which is generated only from committed result JSON files.

## Architecture

```
 Clients (curl, OpenAI SDK, RAG app, lm-eval, benchmark client)
      | HTTP + SSE
      v
 API SERVER (FastAPI, 1 uvicorn worker)   /v1/chat/completions /v1/completions /v1/models /health /metrics
      |  queue.Queue                                  ^ loop.call_soon_threadsafe
      v                                               | RequestOutput
 ENGINE THREAD: one loop, one step at a time
   SCHEDULER -> KV CACHE MANAGER (block allocator, block tables, PREFIX CACHE hash + LRU)
   INPUT BUILDER (numpy -> pinned CPU -> GPU)  ->  MODEL RUNNER (Qwen2 on a flat token batch; CUDA-graph replay
   for pure decode)  ->  SPECULATIVE DECODER  ->  SAMPLER (one GPU->CPU copy)  ->  OUTPUT PROCESSOR
   AUTOPILOT (startup tuner + runtime rule controller)   METRICS (Prometheus, JSON logs, profiler)
      v
 GPU: weights (fp16 or INT8) · KV cache [L, 2, num_blocks, KVH, 16, D] · every op has a reference twin
```

Principles: one step is one forward pass over a flat token batch (continuous batching and chunked prefill
are the same mechanism); write-then-attend; decodes first; a pure-PyTorch reference for every kernel
(`--kernels reference` runs the whole engine on the CPU); one GPU->CPU sync per step.

## How it works (one paragraph per component)

- **API** (`turingserve/api`): OpenAI-compatible chat/completions with SSE streaming, `stream_options.include_usage`,
  vLLM-style extensions (`top_k`, `repetition_penalty`, `min_tokens`, `ignore_eos`, `stop_token_ids`),
  strict 400/404/429/503 errors, disconnect -> abort. The engine runs in its own thread behind `AsyncEngine`.
- **Scheduler** (`scheduler/`): three passes - running decodes (preempting the newest sequence by *recompute* if
  blocks run out), running chunked prefills, then FCFS admission under a token budget and a free-block watermark.
- **KV cache** (`kv/`): 16-token blocks, null block 0, ref-counted allocator with an LRU of evictable hashed
  blocks; a blake2b hash-chain prefix cache shares only full immutable blocks (so no copy-on-write).
- **Model** (`model/`, `ops/`): Qwen2 with HF numerics copied exactly (RMSNorm cast order, fp16 residual adds,
  rotate-half RoPE with fp32 tables). Attention reads everything from the paged cache after the new K/V are written.
- **Kernels** (`ops/triton_*.py`): decode v1 (no `tl.dot`), v2 (GQA group padded to 16 rows, tensor cores),
  v3 (split-K "flash-decoding"); varlen paged prefill; fused RMSNorm+residual, SiLU*up, RoPE, KV-write; W8A16 GEMM.
  Turing-safe tile configs only (64 KiB shared memory, no `cp.async`).
- **Sampler** (`sampling/`): penalties, min_tokens, temperature, top-k, top-p and an *exponential race*
  (`argmax(p / Exp(1))`) so per-request seeds are reproducible regardless of batch composition.
- **Streaming text** (`engine/`): Rust `DecodeStream` detokenizer (UTF-8 safe) and stop-string hold-back.
- **Speculative decoding** (`spec/`): prompt-lookup n-gram or a 0.5B draft model sharing the target's block
  tables; one verification forward; exact greedy acceptance and distribution-preserving rejection sampling.
- **INT8** (`quant/`): per-row symmetric weights, fp16 activations; Triton kernel for small M, dequant + cuBLAS above `M_cross`.
- **Autopilot** (`autopilot/`): startup = kernel tuner (JSON cache) -> memory profile -> CUDA graphs -> chunk sweep;
  runtime = a pure `decide(state, signals)` rule function (R0-R7) with 3-window hysteresis and cooldowns.
  Rules, not reinforcement learning: explainable, testable, stable.
- **Observability** (`metrics/`, `deploy/observability`): every metric of ARCH 6.16 at `/metrics`, JSON-lines logs,
  `POST /debug/profile`, and Prometheus/Grafana as plain binaries.

## Correctness and testing

```
pytest -m "not gpu and not model and not slow and not kaggle" -q     # what CI runs (CPU, ~1-2 min)
pytest tests/kernels -m gpu -q                                      # every Triton kernel vs reference (needs CUDA)
pytest -q                                                           # everything available on this machine
```

Test pyramid: unit (allocator, prefix cache, scheduler with `FakeModelRunner`, sampler chi-square tests,
detokenizer, stop strings, controller, tuner cache) -> correctness (vs Hugging Face; paged == contiguous;
batch invariance; preemption/prefix-cache/speculation equivalence) -> e2e (real server subprocess, OpenAI SDK).
Tolerance policy and needle tests: [ARCHITECTURE.md section 9](docs/ARCHITECTURE.md). Never loosen a threshold
without recording the measured numbers in `docs/lab_notes.md`.

## How to run

Pinned stack (ARCH 2.5): Python 3.12, `torch==2.6.0+cu124`, Triton 3.2.0 (installed by torch),
`transformers>=4.45,<5`. Use WSL2/Linux (as the Build Guide does); never upgrade torch or Triton.

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install "transformers>=4.45,<5" safetensors numpy fastapi "uvicorn[standard]" pydantic httpx openai \
    prometheus-client psutil pytest pytest-asyncio pytest-timeout hypothesis ruff datasets pandas matplotlib
uv pip install -e . --no-deps
python scripts/check_versions.py && python scripts/probe_triton_mma.py   # tensor-core probe (needs a GPU)
```

Quick start (unload Ollama first - it holds GPU memory):

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir ~/models/Qwen2.5-0.5B-Instruct
turingserve serve --model ~/models/Qwen2.5-0.5B-Instruct --kernels reference     # debug path
turingserve serve --model ~/models/Qwen2.5-0.5B-Instruct                          # Triton + CUDA graphs
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Say hi in five words"}],"max_tokens":32}'
```

CPU-only (no GPU needed): `turingserve serve --model <dir> --device cpu --kernels reference --dtype float32`.
Useful flags: `--quantization int8`, `--spec-method ngram|draft`, `--autopilot off|startup|full`,
`--max-num-batched-tokens`, `--no-prefix-caching`, `--debug-endpoints`, `--profile-dir`. Full table: ARCH section 4.

Benchmarks:
```bash
python -m benchmarks.serve_bench --base-url http://localhost:8000 --model Qwen2.5-0.5B-Instruct \
  --tokenizer $TS_MODEL_05 --dataset sharegpt --num-prompts 200 --mode concurrency --concurrency 1,4,16 \
  --ignore-eos --label laptop-test --out benchmarks/results
python -m benchmarks.make_report
```

## Limitations and future work

Out of scope for v1.0: multi-GPU, LoRA, multimodal, tool calling, JSON-schema decoding, logprobs, `n > 1`,
embeddings, models other than Qwen2, KV offload. Stretch (ARCH 13): Llama 3.2 support, grammar-constrained decoding,
INT4 AWQ, EAGLE, overlap scheduling. The GTX 1650 has no tensor cores - laptop numbers are for correctness only.

License: Apache-2.0 (replace the notice in `LICENSE` with the full text before publishing).
