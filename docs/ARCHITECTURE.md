# TuringServe — Architecture (v3)

> An LLM inference server, built from scratch, for NVIDIA **Turing** GPUs: the free Kaggle/Colab
> **T4**, the **GTX 16xx** and the **RTX 20xx**. It has an OpenAI-compatible API, a paged KV cache,
> continuous batching, chunked prefill, Triton **tensor-core** kernels, CUDA graphs, prefix caching,
> INT8 weights, speculative decoding, and a self-tuning controller called **Autopilot**.

**How to use this file.** This file defines *what* to build and *why*, with exact formulas, data
structures and defaults. The companion file **"TuringServe - Complete Build Guide.md"** defines the
*order*, the commands and the tests. If they disagree: this file decides the design, the guide decides
the procedure, and you then fix the other file so they agree again.

**Labels used in this file**
- **[V]** verified from a source listed in §15
- **[M]** measured on your own machines during this review (27 Oct 2026)
- **[S]** senior-engineering judgement (well-known practice, not re-verified)

**Contents**
1. Goal and differentiation · 2. Facts and constraints · 3. System overview · 4. Configuration and CLI ·
5. Data model · 6. Components · 7. Request lifecycle · 8. Repository layout · 9. Testing strategy ·
10. Benchmark methodology · 11. Risks · 12. Methods that work / don't · 13. Out of scope and stretch ·
14. Review log · 15. Sources

---

## 1. Goal and differentiation

**Goal:** a serving engine you wrote yourself, that you can explain line by line, proven correct by
tests, and measured honestly against real engines on the same GPU.

| | nano-vllm (well-known mini engine) | vLLM (latest) on a T4 | **TuringServe** |
|---|---|---|---|
| Runs on Turing | No: depends on `flash-attn`, which needs sm80+ [V] | Yes. It auto-selects `TRITON_ATTN` or `FLEX_ATTENTION` [V] | **Yes. Built for it** |
| Tensor cores in attention on T4 | n/a | **No:** Triton ≥ 3.3 emits no MMA (tensor-core) instructions for compute capability < 8.0, so it falls back to FMA [V] | **Yes:** Triton is pinned to 3.2, the last release before that change [V] |
| 8,000-token prefill, Qwen2.5-1.5B, T4 | n/a | **10.2 s** with the default backend (1.1 s with FlashInfer, which vLLM disables on SM75 by default) [V] | **Target ≤ 2.0 s** |
| Tuning | Fixed | Many manual flags | **Autopilot:** profiles the GPU at startup, adapts while running |
| Speculative decoding | None | Several methods | Prompt lookup (n-gram) + draft model, switched on/off automatically |
| Proof | One throughput number | Production-grade | Test pyramid in CI, standard metrics, load and soak tests, used by your RAG app |

**The honest headline claim** (only after you measure it):
"On Turing GPUs, TuringServe's attention uses tensor cores because it pins Triton 3.2. The default
attention in current vLLM on a T4 does not, and is up to ~9× slower at long prefill."
vLLM may fix this (PR #55380 would enable FlashInfer on SM75 [V]). Re-measure right before you
publish and report whatever is true at that moment.

**Rule for reading other engines' code:** build each component yourself first. Then read how
nano-vllm or vLLM did it, and write the differences in `docs/lab_notes.md`. Never copy code. You must
be able to explain every line in an interview.

---

## 2. Facts and constraints

### 2.1 Your machines [M]

| Item | Value | Consequence |
|---|---|---|
| Laptop GPU | GTX 1650, 4 GB (4096 MiB), Turing sm_75, driver 572.70 (CUDA 12.8), WDDM | Development and correctness only. All published numbers come from the T4. |
| Display | Not on the NVIDIA GPU ("Disp.A Off") | Almost all 4 GB is usable once other apps release it |
| **Ollama** | Its `llama-server.exe` holds **2,341 MiB** of the GPU | **Unload Ollama models before every session** (Guide §0.1) |
| CPU / RAM | i7-9750H (6 cores) / 15.8 GB | WSL gets ~8 GB by default; set 10 GB (Guide §0.2) |
| Disk | C: **16.5 GB free**; D: 80.2 GB free | **Move WSL to D:** (Guide §0.2); C: is too small for the venvs, models and datasets |
| WSL | WSL 2.6.3, distro `Ubuntu` = **Ubuntu 26.04 LTS** | `wsl --manage --move` is available (needs ≥ 2.3.24) [V] |
| System Python in WSL | **3.14.4** | **PyTorch 2.6 has no Python 3.14 wheels.** Use a uv-managed Python 3.12 |
| CUDA driver library in WSL | `/usr/lib/wsl/lib/libcuda.so` present | Triton can find it; if not, set `TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib` |
| Docker | Docker Desktop installed | Not used. Observability tools run as plain binaries inside WSL (simpler networking) |

### 2.2 Benchmark GPU: Kaggle T4
- Tesla T4: Turing sm_75, **40 SMs**, 16 GB GDDR6 (15,360 MiB visible), peak memory bandwidth
  **320 GB/s**, FP16 tensor-core peak ~65 TFLOPS, INT8 ~130 TOPS, **70 W power cap** [S].
  The power cap causes clock throttling, so benchmarks must warm up, log clocks and repeat (§10).
- Kaggle free GPU quota is about **30 h/week**; your account settings show the real number [V].
  Choose **"GPU T4"** or **"GPU T4 ×2"**, never P100 (P100 is Pascal sm_60, not Turing).

### 2.3 Turing (sm_75) hardware limits
- **No bf16 and no fp8** [V]. Everything runs in **fp16 with fp32 accumulation**.
- **No `cp.async`** (asynchronous global → shared copies arrived with Ampere) [S]. Triton software
  pipelining (`num_stages`) helps little on Turing. Use `num_stages` 1–2 in kernel configs.
- **Shared memory per thread block: max 64 KiB** [S]. Tiles larger than this fail to compile with
  "out of resource: shared memory". This limits block sizes (§6.8).
- T4 has fp16/int8 **tensor cores**. The GTX 1650 (TU117) has **no tensor cores** [S], so tensor-core
  instructions may be slow there. Guide §0.7 probes this.

### 2.4 Software facts
- **Triton ≥ 3.3 stopped emitting MMA (tensor-core) instructions for compute capability < 8.0**
  (triton-lang PR #5066). An attempt to restore Turing support was declined (PR #9970). Triton 3.1.0 was
  confirmed to emit `mma.sync` on sm_75. 3.2 is the last release before the change [V].
- **PyTorch 2.6.0 ships with Triton 3.2.0** [V]. PyTorch 2.5.1 ships with Triton 3.1.0 [S].
- **FlashAttention-2 needs sm80+** [V]. In vLLM, FlashInfer is gated to sm80+ [V].
- **PyTorch 2.6 wheels exist for Python 3.9–3.13 only** [S].
- **transformers 4.x requires `huggingface_hub` < 1.0** [S]. Never upgrade `huggingface_hub` on its own.
- **Qwen2.5-Instruct `generation_config.json`:** `do_sample: true, temperature 0.7, top_p 0.8, top_k 20,
  repetition_penalty 1.1, eos_token_id [151645, 151643]` [V]. HF `generate()` applies these defaults
  unless you override them. **The repetition penalty applies even with `do_sample=False`.**
- **PyTorch SDPA with `is_causal=True` aligns the mask top-left** when query length ≠ key length [S].
  Chunked prefill needs a bottom-right mask, so the reference path builds masks explicitly.
- **Triton interpreter mode** (`TRITON_INTERPRET=1`) runs kernels on the CPU with numpy, which helps debugging [V].
- **lm-eval's `local-chat-completions` model supports generation tasks only** (no loglikelihood without
  logprobs) [V]. HellaSwag-style tasks are therefore out; GSM8K works.

### 2.5 Pinned software stack

| Component | Version | Why |
|---|---|---|
| Python | **3.12** (installed by `uv`) | PyTorch 2.6 wheels; includes `Python.h`, which Triton needs to compile its launcher |
| PyTorch | **2.6.0+cu124** | Pairs with Triton 3.2.0 |
| Triton | **3.2.0** (installed by PyTorch) | Last release with Turing MMA |
| transformers | `>=4.45,<5` (exact version frozen in `requirements.lock`) | Reference model; v5 changes APIs |
| tokenizers | as required by transformers (≥ 0.21) | `DecodeStream` incremental detokenizer |
| fastapi, uvicorn[standard], pydantic v2 | latest at install, then frozen | API server |
| prometheus-client, psutil | latest, frozen | Metrics, soak test |
| numpy, safetensors, huggingface_hub | installed with transformers, frozen | Loading |
| httpx, openai | latest, frozen | Benchmark client, E2E tests |
| pytest, pytest-asyncio, pytest-timeout, hypothesis, ruff | latest, frozen | Tests, lint |
| datasets, pandas, matplotlib | latest, frozen | Data, reports |
| lm-eval[api] | latest, **separate venv** | Quality evaluation through the API |

- **Fallback stack:** `torch==2.5.1+cu124` (Triton 3.1.0), used only if the MMA probe (Guide §0.7) fails.
- **Upgrade policy:** never upgrade torch or Triton during the project. Any change → re-run the MMA probe and all kernel tests.
- **Baselines (vLLM, llama.cpp) run in separate Kaggle notebooks**, because vLLM installs its own torch.

### 2.6 Models [V]

| Field | Qwen2.5-0.5B-Instruct (dev, laptop) | Qwen2.5-1.5B-Instruct (benchmark, T4) |
|---|---|---|
| hidden_size | 896 | 1536 |
| intermediate_size | 4864 | 8960 |
| num_hidden_layers | 24 | 28 |
| num_attention_heads (H) | 14 | 12 |
| num_key_value_heads (KVH) | 2 | 2 |
| head_dim (D = hidden / H) | 64 | 128 |
| GQA group size (G = H / KVH) | **7** (not a power of 2) | **6** (not a power of 2) |
| vocab_size | 151,936 | 151,936 |
| rope_theta | 1,000,000 | 1,000,000 |
| rms_norm_eps | 1e-6 | 1e-6 |
| max_position_embeddings | 32,768 | 32,768 |
| tie_word_embeddings | true | true |
| sliding window | disabled (`use_sliding_window: false`) | disabled |
| checkpoint dtype | bfloat16 (converted to fp16 at load) | bfloat16 |
| parameters (total / non-embedding) | 0.49 B / 0.36 B | 1.54 B / 1.31 B |
| license | Apache-2.0 | Apache-2.0 |

- Architecture details: RMSNorm; RoPE in the "rotate half" layout (non-interleaved); **bias on the
  Q/K/V projections only** (no bias on o_proj or the MLP); SwiGLU MLP; tied input/output embeddings.
- Special tokens: `<|endoftext|>` = 151643 (BOS/PAD), `<|im_end|>` = 151645. **Stop on both** (from `generation_config.json`).
- Chat template: use `tokenizer.apply_chat_template(messages, add_generation_prompt=True)`. When the
  messages contain no system message, Qwen's template inserts a default one.
- **Draft model for speculation:** the 0.5B, which has the same tokenizer and vocabulary as the 1.5B.
- **Fallback family** (only if the Day-1 fp16 gate fails): SmolLM2-360M-Instruct / SmolLM2-1.7B-Instruct
  (Llama architecture: no QKV bias, different heads and rope theta; read `config.json`).

### 2.7 Numbers you must know (and say in interviews)

**Weight bytes (fp16):** 0.5B ≈ 0.99 GB · 1.5B ≈ 3.09 GB.
**INT8 weight bytes** (embedding stays fp16 because it is tied): 0.5B ≈ 0.36 + 0.27 = **0.63 GB** ·
1.5B ≈ 1.31 + 0.47 = **1.78 GB**.

**KV cache bytes** = 2 (K and V) × layers × KVH × D × 2 bytes per token:
```
0.5B: 2 × 24 × 2 × 64  × 2 = 12,288 B = 12 KiB/token → block of 16 tokens = 192 KiB
1.5B: 2 × 28 × 2 × 128 × 2 = 28,672 B = 28 KiB/token → block of 16 tokens = 448 KiB
```

**Logits memory:** T × 151,936 × 4 B. For a 2,048-token step that is **1.24 GB in fp32**. So logits are
computed **only for the rows that sample a token** (§6.7). This is mandatory on a 4 GB GPU.

**Speed of light for decode at batch 1** (memory-bandwidth bound: all weights are read once per token):
```
tokens/s ≤ measured_bandwidth / weight_bytes
T4, 1.5B fp16: 320 GB/s / 3.09 GB ≈ 103 tok/s at peak bandwidth (use your measured bandwidth, Guide §0.8)
T4, 1.5B int8: 320 / 1.78 ≈ 180 tok/s
```
Report "% of speed of light" next to every batch-1 number.

**Prefill compute** for 8,000 tokens on the 1.5B ≈ 2 × 1.31e9 × 8000 (linear layers) + ~5.5e12 (causal
attention) ≈ **26.5 TFLOP**. At a realistic 25–40 TFLOPS on the T4 that is ~0.7–1.1 s, which matches
vLLM's FlashInfer result (1.1 s) [V]. Hence the target of ≤ 2.0 s.

**Memory budget at startup** (formula in §6.10.3):
- T4, 1.5B fp16, utilization 0.85: about 12.75 GiB budget − 2.9 GiB weights − ~0.3 GiB activations −
  0.25 GiB graph reserve − ~0.4 GiB other → **~8.9 GiB KV ≈ 20,000 blocks ≈ 330,000 tokens**.
- Laptop, 0.5B fp16, utilization 0.80, Ollama unloaded: **~1.5 GiB KV ≈ 8,000 blocks ≈ 128,000 tokens**.
- Laptop, 1.5B: fp16 is borderline (the 2.9 GiB of weights almost fills the budget). **Use INT8 for the 1.5B on the laptop.**

---

## 3. System overview

```
 Clients (curl, OpenAI SDK, RAG app, lm-eval, benchmark client)
      │ HTTP + SSE
      ▼
 ┌──────────────────────────── API SERVER (FastAPI, 1 uvicorn worker) ─────────────────────────────┐
 │ /v1/chat/completions  /v1/completions  /v1/models  /health  /metrics  (/debug/state optional)   │
 │ validate → chat template → tokenize → AsyncEngine.add()      SSE streaming · disconnect → abort │
 └───────────────┬───────────────────────────────────────────────────────────────▲─────────────────┘
       queue.Queue (thread-safe)                                   loop.call_soon_threadsafe(...)
                 ▼                                                                │ RequestOutput
 ┌──────────────────────────── ENGINE THREAD: one loop, one step at a time ──────────────────────────┐
 │  SCHEDULER ──► KV CACHE MANAGER (block allocator, block tables, PREFIX CACHE: hash + LRU)          │
 │     │ BatchPlan                                                                                   │
 │     ▼                                                                                             │
 │  INPUT BUILDER (numpy → pinned CPU buffers → GPU buffers)                                         │
 │     ▼                                                                                             │
 │  MODEL RUNNER: Qwen2 forward on a flat token batch · CUDA-graph replay for pure-decode steps      │
 │     │      └─ SPECULATIVE DECODER (n-gram / draft model) for small pure-decode batches            │
 │     ▼                                                                                             │
 │  SAMPLER (penalties, temperature, top-k, top-p, seeded exponential race) → ONE GPU→CPU copy       │
 │     ▼                                                                                             │
 │  OUTPUT PROCESSOR (DecodeStream detokenizer, stop strings with hold-back, finish reasons)         │
 │                                                                                                   │
 │  AUTOPILOT: startup (tuner, memory profile, graphs, chunk sweep) + runtime controller (rules)     │
 │  METRICS: Prometheus counters/histograms, step-time breakdown, JSON logs, profiler hooks          │
 └───────────────────────────────────────────────────────────────────────────────────────────────────┘
                 ▼
 ┌──────────────────────────── GPU ─────────────────────────────────────────────────────────────────┐
 │ Weights (fp16 or INT8+scales) · KV cache tensor [L, 2, num_blocks, KVH, 16, D] fp16 · graph pool  │
 │ ops: rmsnorm(+residual) · rope · silu_mul · kv_write · decode attention · prefill attention ·     │
 │      w8a16 GEMM · cuBLAS (torch.matmul) for fp16 linear layers                                     │
 │ Every op has a REFERENCE (pure PyTorch) and a TRITON implementation, chosen by --kernels           │
 └───────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Design principles**
1. **One step = one forward pass over a flat token batch.** Decode tokens and prefill chunks of many
   requests are concatenated into one `[T]` token array, with no padding. Continuous batching and chunked
   prefill are then the same mechanism (as in vLLM V1 and Sarathi-Serve).
2. **One uniform rule for prefill and decode:** each sequence has `token_ids` and `num_computed_tokens`.
   A step computes the next `n` uncomputed tokens. If the chunk reaches the end of `token_ids`, the
   sequence samples one new token. Decode is simply `n = 1`.
3. **Write-then-attend:** each layer first writes the new K/V into the paged cache, then attention reads
   *all* keys/values (prefix + new) from the cache. Decode and prefill attention share one memory path.
4. **Decodes first:** in every batch the decode rows come first, then the prefill chunks.
5. **Reference path for everything:** every kernel has a slow, obviously-correct PyTorch twin. Tests
   compare the two. `--kernels reference` runs the whole engine without Triton, even on the CPU.
6. **One GPU→CPU synchronization per step** (the sampled tokens). No `.item()` anywhere else in the loop.
7. **Measure before optimizing:** profiling decides what to optimize; ablations prove each gain.

---

## 4. Configuration and CLI

Commands: `turingserve serve --model PATH [flags]` (the server) and `turingserve tune --model PATH [--retune]`
(runs only the kernel tuner, Phase 7). All flags live in one `EngineConfig` dataclass (`turingserve/config.py`).

| Flag | Default | Meaning |
|---|---|---|
| `--model` | (required) | Directory with `config.json`, `generation_config.json`, `*.safetensors`, tokenizer files |
| `--served-model-name` | model directory name | Name clients must send in `"model"` |
| `--dtype` | `float16` | `float16` or `float32` (float32 is for tests) |
| `--device` | `cuda` | `cuda` or `cpu` (cpu requires `--kernels reference`) |
| `--kernels` | `triton` | `triton` or `reference` |
| `--max-model-len` | `4096` | Max prompt + output tokens per request (≤ 32768) |
| `--block-size` | `16` | Tokens per KV block (fixed; kernels assume 16) |
| `--num-blocks` | auto | Testing only: force the number of KV blocks and skip the memory profiler |
| `--gpu-memory-utilization` | `0.85` | Fraction of total GPU memory the engine may use (laptop: 0.80) |
| `--max-num-seqs` | `128` | Max running sequences (laptop: 32) |
| `--max-num-batched-tokens` | `2048` | Token budget per step (the prefill chunk limit) |
| `--max-waiting-requests` | `256` | Beyond this the API returns HTTP 429 |
| `--watermark` | `0.01` | Fraction of KV blocks kept free when admitting new requests |
| `--cuda-graphs` / `--no-cuda-graphs` | on (GPU only) | Graph replay for pure-decode steps |
| `--graph-batch-sizes` | `1,2,4,8,16,32,64,128` | Sizes to capture; values above `--max-num-seqs` are dropped |
| `--prefix-caching` / `--no-prefix-caching` | on | Reuse KV blocks of identical prefixes |
| `--quantization` | `none` | `none` or `int8` (W8A16) |
| `--int8-lm-head` | off | Extra INT8 copy of the tied embedding, for the logits matmul (ablation) |
| `--spec-method` | `none` | `none`, `ngram` or `draft` |
| `--spec-k` | `4` | Max proposed tokens per step (1–8) |
| `--spec-ngram-max` / `--spec-ngram-min` | `3` / `1` | n-gram sizes tried, from largest to smallest |
| `--spec-draft-model` | none | Path to the draft model (required for `draft`) |
| `--spec-max-batch` | `8` | Speculation only when the pure-decode batch is ≤ this size |
| `--autopilot` | `startup` | `off`, `startup` or `full` (§6.15) |
| `--ttft-target-ms` / `--tpot-target-ms` | `2000` / `50` | Latency targets (SLOs) used by Autopilot and benchmarks |
| `--tuning-dir` | `~/.turingserve/tuning` | Where tuning results are cached |
| `--retune` | off | Ignore the cached tuning and measure again |
| `--host` / `--port` | `0.0.0.0` / `8000` | Server address |
| `--log-level` | `info` | `debug`, `info`, `warning` |
| `--log-requests` | off | Log one JSON line per finished request |
| `--debug-endpoints` | off | Enables `GET /debug/state` (used by tests and the chaos/soak scripts) |
| `--profile-dir` | none | If set, `POST /debug/profile?steps=N` writes a torch profiler trace there |
| `--seed` | `0` | Seed for the global sampling generator |

**Validation at startup** (fail fast with a clear message): block size 16; `max_model_len ≤
max_position_embeddings`; `max_num_batched_tokens ≥ 16`; `--kernels triton` needs CUDA; `--spec-method
draft` needs `--spec-draft-model` with the same vocabulary; `--quantization int8` needs `--kernels
triton` (the reference path dequantizes instead, which is allowed for tests via `--kernels reference`).

---

## 5. Data model

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0        # 0 → greedy
    top_p: float = 1.0              # (0, 1]
    top_k: int = -1                 # -1 = off, else ≥ 1
    seed: int | None = None
    max_tokens: int = 16            # completions default (OpenAI); chat default = max_model_len - prompt_len
    min_tokens: int = 0             # EOS/stop tokens are masked until this many tokens exist
    stop: list[str] = []            # ≤ 16 strings, each non-empty
    stop_token_ids: list[int] = []
    ignore_eos: bool = False        # benchmarks: generate exactly max_tokens
    repetition_penalty: float = 1.0 # HF semantics, > 0
    presence_penalty: float = 0.0   # OpenAI semantics, [-2, 2]
    frequency_penalty: float = 0.0  # OpenAI semantics, [-2, 2]

class SeqStatus(Enum): WAITING, RUNNING, FINISHED

@dataclass
class Sequence:
    request_id: str
    token_ids: list[int]            # prompt + generated so far
    num_prompt_tokens: int
    num_computed_tokens: int = 0    # tokens whose K/V are in the cache
    block_table: list[int] = []     # physical block ids, in order
    num_hashed_blocks: int = 0      # prefix cache: blocks registered so far
    draft_computed: int = 0         # spec decoding (draft model): tokens in the draft KV cache
    status: SeqStatus = WAITING
    params: SamplingParams
    generator: torch.Generator | None   # created from params.seed
    detok: DecodeStream; text: str = ""; emitted_len: int = 0
    finish_reason: str | None = None    # "stop" | "length" | "abort" | "error"
    arrival_time, first_scheduled_time, first_token_time, last_token_time, finish_time: float
    num_cached_prompt_tokens: int = 0   # prefix-cache hits (for usage and metrics)
    # derived:
    #   num_uncomputed = len(token_ids) - num_computed_tokens
    #   num_output = len(token_ids) - num_prompt_tokens

@dataclass
class Scheduled:   seq: Sequence; num_tokens: int; samples: bool    # samples = chunk reaches the end
@dataclass
class BatchPlan:   items: list[Scheduled]; num_decode: int; num_tokens: int; preempted: list[Sequence]
                   # items: decodes first (num_tokens == 1), then prefill chunks

@dataclass
class AttentionMetadata:            # all GPU tensors, int32 unless noted
    query_start_loc: Tensor         # [B+1]  start of each sequence's query tokens in the flat batch
    seq_lens: Tensor                # [B]    context length AFTER this step (prefix + new tokens)
    block_tables: Tensor            # [B, max_blocks_per_seq]  padded with 0 (block 0 = null block)
    slot_mapping: Tensor            # [T] int64, cache slot per new token; -1 = padding (skip)
    num_decode: int                 # first num_decode rows have query_len 1 → decode kernel
    max_query_len: int; max_seq_len: int     # host ints (used for grids outside graphs)

@dataclass
class ModelInput:  input_ids: Tensor [T] int64; positions: Tensor [T] int64;
                   meta: AttentionMetadata; sample_indices: Tensor [S] int64   # rows of T that sample

@dataclass
class RequestOutput:               # engine → API, one per request per step that produced output
    request_id: str; delta_text: str; delta_token_ids: list[int]
    finished: bool; finish_reason: str | None
    prompt_tokens: int; completion_tokens: int; cached_prompt_tokens: int
```

---

## 6. Components

### 6.1 API server (`turingserve/api/`)

**Endpoints**

| Method + path | Behavior |
|---|---|
| `POST /v1/chat/completions` | Chat, streaming and non-streaming |
| `POST /v1/completions` | Raw prompt (string or list of token ids), streaming and non-streaming |
| `GET /v1/models` | `{"object":"list","data":[{"id":NAME,"object":"model","owned_by":"turingserve","max_model_len":N}]}` |
| `GET /health` | 200 `{"status":"ok"}` while the engine thread is alive and healthy; 503 otherwise |
| `GET /metrics` | Prometheus text format (§6.16) |
| `GET /debug/state` | Only with `--debug-endpoints`: `{num_running, num_waiting, num_free_blocks, num_cached_blocks, num_total_blocks, num_requests_tracked}` |
| `POST /debug/profile?steps=N` | Only with `--profile-dir`: trace the next N engine steps |

**Request fields accepted**
- Chat: `model`, `messages` (roles `system`/`user`/`assistant`; content is a string or a list of
  `{"type":"text","text":...}` parts), `max_tokens` or `max_completion_tokens`, `temperature`, `top_p`,
  `n` (must be 1), `stream`, `stream_options.include_usage`, `stop` (string or list), `seed`,
  `presence_penalty`, `frequency_penalty`, `user` (ignored), `response_format` (only `{"type":"text"}`).
- Completions: `model`, `prompt` (one string or one list of ints; lists of prompts are rejected), same sampling fields.
- **Extensions** (vLLM-compatible names): `top_k`, `repetition_penalty`, `min_tokens`, `ignore_eos`, `stop_token_ids`.
- **Rejected with 400:** `n > 1`, `logprobs`, `top_logprobs`, `echo`, `best_of`, `suffix`,
  `tools`/`tool_choice`/`functions`, non-empty `logit_bias`, `response_format` other than text, unknown roles,
  empty prompt (0 tokens), more than 16 stop strings.
- `model` must equal `--served-model-name`, otherwise **404** `model_not_found`.
- Defaults: `temperature` 1.0, `top_p` 1.0; completions `max_tokens` 16; chat `max_tokens` = `max_model_len − prompt_tokens`.
- `prompt_tokens ≥ max_model_len` or `prompt_tokens + max_tokens > max_model_len` → **400** `context_length_exceeded`.

**Non-streaming chat response**
```json
{"id":"chatcmpl-<uuid hex>","object":"chat.completion","created":1761580000,"model":"qwen2.5-1.5b",
 "choices":[{"index":0,"message":{"role":"assistant","content":"..."},"finish_reason":"stop"}],
 "usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17}}
```
(Completions: `"object":"text_completion"`, `choices[0].text`, id prefix `cmpl-`.)

**Streaming (SSE)**: headers `Content-Type: text/event-stream`, `Cache-Control: no-cache`. Each event is
`data: <json>\n\n`, ending with `data: [DONE]\n\n`. Chat order:
```
data: {"id":"chatcmpl-1a2b","object":"chat.completion.chunk","created":1761580000,"model":"qwen2.5-1.5b","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}
data: {... "choices":[{"index":0,"delta":{"content":"Hel"},"finish_reason":null}]}
data: {... "choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":null}]}
data: {... "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: {... "choices":[],"usage":{"prompt_tokens":12,"completion_tokens":5,"total_tokens":17}}   ← only if stream_options.include_usage
data: [DONE]
```
Empty deltas (for example, while the detokenizer waits for a complete UTF-8 character) are **not** sent.
`usage` also contains `prompt_tokens_details.cached_tokens` (prompt tokens served from the prefix cache;
this is OpenAI's field name).

**Errors:** `{"error":{"message":"...","type":"invalid_request_error","param":"<field or null>","code":"<code>"}}`
with status 400 (validation), 404 (model), 429 (`server_overloaded`, header `Retry-After: 1`),
500 (`internal_error`), 503 (`engine_unhealthy`).

**Cancellation:** the streaming generator wraps its loop in `try/finally`. If the client disconnects
(Starlette raises `CancelledError`/`GeneratorExit`) and the request has not finished, call
`async_engine.abort(request_id)`. Non-streaming handlers do the same when cancelled.

**Server process:** exactly **one uvicorn worker** (each worker would load its own engine).

### 6.2 Async bridge and threading (`turingserve/api/async_engine.py`)

```
class AsyncEngine:
    input_q: queue.Queue[Request]; abort_q: queue.Queue[str]
    sinks: dict[request_id, (asyncio loop, asyncio.Queue[RequestOutput])]
    start():  start the engine thread (daemon) running _loop()
    _loop():
        while not stopping:
            drain abort_q → engine.abort(id); drain input_q → engine.add_request(req)
            if not engine.has_unfinished(): block up to 50 ms on input_q.get(); add what arrives; continue
            try:    outputs = engine.step()
            except torch.cuda.OutOfMemoryError: engine.recover_from_oom(); continue    (§6.17)
            except Exception: log stack trace; fail every request with "error"; healthy = False; break
            for out in outputs:
                loop, q = sinks[out.request_id]; loop.call_soon_threadsafe(q.put_nowait, out)
                if out.finished: del sinks[out.request_id]
    async generate(request) -> AsyncIterator[RequestOutput]:  register sink, put in input_q, yield until finished
    abort(request_id):  put in abort_q (thread-safe, idempotent)
```
- Tokenization and the chat template run in the API (event-loop) thread. Detokenization runs in the engine thread.
- **Planned upgrade, only if profiling shows the GIL is a bottleneck:** move the engine to its own process
  (ZeroMQ, as vLLM V1 does). Measure "engine CPU time per step" first (§6.16).

### 6.3 Engine core (`turingserve/engine/engine.py`)

```
class LLMEngine:
    add_request(request_id, prompt_token_ids, params, arrival_time)
    abort(request_id)            # finish_reason "abort"; frees blocks at the next step
    has_unfinished() -> bool
    step() -> list[RequestOutput]
    generate(prompts, params) -> list[final outputs]     # offline helper used by tests and benchmarks

step():
    plan = scheduler.schedule()
    if plan is empty: return []
    if spec_decoder.should_run(plan): return spec_decoder.step(plan)            (§6.13)
    inp = input_builder.build(plan)
    hidden = runner.forward(inp)                      # CUDA-graph replay if pure decode and size fits
    logits = runner.compute_logits(hidden, inp.sample_indices)    # [S, V] fp32
    tokens, bad = sampler.sample(logits, sampling rows)           # ONE GPU→CPU copy (tokens + NaN flags)
    for item in plan.items:
        item.seq.num_computed_tokens += item.num_tokens
        prefix_cache.register_full_blocks(item.seq)    # only blocks full of COMPUTED tokens
    for each sampling seq (in order) with its token:
        if bad[row]: finish(seq, "error"); continue
        seq.token_ids.append(token)
        outputs += output_processor.process(seq, [token])         # stop checks, detok, finish
    for finished seqs: scheduler.finish(seq) → kv.free(seq)
    metrics.record_step(...); autopilot.observe(...)
    return outputs
```

### 6.4 Scheduler (`turingserve/scheduler/scheduler.py`)

Queues: `waiting` (deque, FCFS) and `running` (list, oldest first).

```
schedule():
    budget = max_num_batched_tokens; plan = []; preempted = []
    # Pass 1: running sequences that are decoding (num_uncomputed == 1), oldest first
    for seq in [s for s in running if s.num_uncomputed == 1]:
        if seq in preempted: continue
        while not kv.can_append(seq, 1):
            victim = newest seq in running that is not in plan and is not seq  (or seq itself if none)
            preempt(victim); preempted.append(victim)
            if victim is seq: break
        if seq in preempted: continue
        kv.allocate(seq, 1); plan.append(Scheduled(seq, 1, samples=True)); budget -= 1
    # Pass 2: running sequences in the middle of a chunked prefill (num_uncomputed > 1)
    for seq in [s for s in running if s.num_uncomputed > 1 and s not in preempted]:
        if budget == 0: break
        n = min(seq.num_uncomputed, budget)
        if not kv.can_append(seq, n): break               # never preempt for a prefill; retry next step
        kv.allocate(seq, n); plan.append(Scheduled(seq, n, samples = (n == seq.num_uncomputed))); budget -= n
    # Pass 3: admit waiting sequences, FCFS, only if nothing was preempted in this step
    while waiting and budget > 0 and not preempted and len(running) < max_num_seqs:
        seq = waiting[0]
        if prefix caching and seq.num_computed_tokens == 0: prefix_cache.attach_hits(seq)   (§6.6)
        n = min(seq.num_uncomputed, budget)
        if not kv.can_append(seq, n, keep_free = watermark_blocks):
            prefix_cache.detach_hits(seq); break          # undo, stay first in line
        waiting.popleft(); running.append(seq); kv.allocate(seq, n)
        plan.append(Scheduled(seq, n, samples = (n == seq.num_uncomputed))); budget -= n
    return BatchPlan(decodes first, then prefills; num_decode; num_tokens; preempted)

preempt(seq):   # recompute-style
    kv.free(seq)                      # decref blocks; hashed blocks become evictable (LRU), not lost
    seq.num_computed_tokens = 0; seq.block_table = []; seq.num_hashed_blocks = 0; seq.draft_computed = 0
    running.remove(seq); waiting.appendleft(seq); metrics.preemptions += 1
    # generated tokens stay in token_ids; on re-admission they are recomputed like prompt tokens
```
- `watermark_blocks = max(1, int(watermark × num_blocks))`.
- `max_model_len`: a sequence finishes with "length" when `len(token_ids) == max_model_len`.
- **Invariants** (property-tested): the step's token count ≤ budget; every running sequence has
  `len(block_table) == ceil(num_computed_or_scheduled / 16)`; no block id is both free and owned; every
  request finishes (no starvation), because running sequences always finish within `max_tokens`.

### 6.5 KV cache (`turingserve/kv/`)

**Layout:** one tensor allocated once: `kv_cache = empty([L, 2, num_blocks, KVH, 16, D], fp16)`.
Layer views: `k_cache[l] = kv_cache[l, 0]` and `v_cache[l] = kv_cache[l, 1]`, each `[num_blocks, KVH, 16, D]`.
Each (block, head) slice is 16 × D contiguous values, which suits per-head loads in the attention kernels.

**Block 0 is the null block.** It is never handed out. Padded block-table entries point to it, and
padded slots use −1 (the KV-write kernel skips negative slots).

**Block allocator** (`block_allocator.py`, pure Python, CPU-testable):
```
free: deque[int] (initially 1..num_blocks-1) · ref: list[int] · evictable: OrderedDict[int, None] (LRU, §6.6)
allocate() -> int            # pop from free; if free is empty, evict the oldest evictable block; else raise NoFreeBlocks
incref(b), decref(b)         # decref to 0 → into `evictable` if the block has a hash, else back to `free`
num_available() = len(free) + len(evictable)
```
**KV manager** (`kv_manager.py`):
```
blocks_needed(seq, n) = ceil((seq.num_computed_tokens + n) / 16) - len(seq.block_table)
can_append(seq, n, keep_free=0) = num_available() - blocks_needed(seq, n) >= keep_free
allocate(seq, n): append blocks_needed(...) new blocks to seq.block_table
free(seq): decref every block in seq.block_table; clear it
slot(seq, pos) = seq.block_table[pos // 16] * 16 + pos % 16
```

### 6.6 Prefix cache (`turingserve/kv/prefix_cache.py`)

- **Hash chain:** for full block i of a sequence, `h_i = blake2b(h_{i-1} + int32_bytes(tokens[16i : 16i+16]), digest_size=16)`, with `h_{-1} = b""`.
- **Maps:** `hash_to_block: dict[bytes, int]`, `block_hash: list[bytes | None]`.
- **Register** (after every step): for each block of the sequence that is now *full of computed tokens*
  (`index < num_computed_tokens // 16`) and not yet registered, compute its hash. If the hash is not yet
  in the map, register it. If another block already has that hash, leave this block unregistered (private).
- **attach_hits(seq)** at admission: walk the full blocks of `seq.token_ids` in order. For each, compute the
  chain hash; on a hit, `incref` the block (removing it from `evictable` if its ref was 0) and append it
  to the block table; stop at the first miss. Then `num_computed_tokens = hits × 16`. **If that equals
  `len(token_ids)`, drop the last hit block** (decref, and subtract 16): at least one token must be
  computed to produce logits.
- **detach_hits(seq):** undo attach (decref the attached blocks, reset counters).
- **Eviction:** in `allocate()`, when `free` is empty, take the oldest block in `evictable`, delete its
  hash mapping, and reuse it. Only ref-0 blocks are ever evictable, so a running sequence's blocks are never evicted.
- **Metrics:** prompt tokens queried, tokens hit, so hit rate = hit / queried.
- Only full, immutable blocks are shared, so no copy-on-write is needed (`n = 1` only).

### 6.7 Model (`turingserve/model/`)

**Config** (`qwen2_config.py`): read `config.json` and `generation_config.json`. Reject (with a clear
error) `model_type != "qwen2"`, `use_sliding_window == true`, or a non-null `rope_scaling`.

**Weight mapping** (HF name → TuringServe):
```
model.embed_tokens.weight                         → embed            [V, hidden]   (also the lm head: tied)
model.layers.{i}.input_layernorm.weight           → L[i].w_in        [hidden]
model.layers.{i}.self_attn.{q,k,v}_proj.weight    → L[i].w_qkv = cat([q, k, v], dim 0)   [(H+2·KVH)·D, hidden]
model.layers.{i}.self_attn.{q,k,v}_proj.bias      → L[i].b_qkv = cat([q, k, v])          [(H+2·KVH)·D]
model.layers.{i}.self_attn.o_proj.weight          → L[i].w_o         [hidden, H·D]
model.layers.{i}.post_attention_layernorm.weight  → L[i].w_post      [hidden]
model.layers.{i}.mlp.{gate,up}_proj.weight        → L[i].w_gu = cat([gate, up], dim 0)   [2·I, hidden]
model.layers.{i}.mlp.down_proj.weight             → L[i].w_down      [hidden, I]
model.norm.weight                                 → w_norm           [hidden]
lm_head.weight                                    → absent when tied; if present and tied, ignore it
```
Load each tensor on the CPU (`safetensors.safe_open`), fuse on the CPU, convert to fp16, check
`isfinite`, then move to the GPU. **Every checkpoint tensor must be used exactly once** (a test checks this).

**Forward (exact op order, matching HF Qwen2):**
```
h = embed[input_ids]                                  # [T, hidden] fp16
residual = None
for each layer l:
    if residual is None: residual = h; x = rmsnorm(h, w_in)
    else:                x, residual = rmsnorm_residual(h, residual, w_in)   # residual = h + residual; x = norm(residual)
    qkv = linear(x, w_qkv, b_qkv)                     # [T, (H+2·KVH)·D]
    rope_(qkv, positions)                             # in place on the q and k parts
    q, k, v = split(qkv)
    kv_write(k, v, k_cache[l], v_cache[l], slot_mapping)        # WRITE first
    a = attention(q, k_cache[l], v_cache[l], meta, scale = D ** -0.5)   # then ATTEND (reads cache)
    h = linear(a, w_o)
    x, residual = rmsnorm_residual(h, residual, w_post)
    h = linear(silu_mul(linear(x, w_gu)), w_down)
x, _ = rmsnorm_residual(h, residual, w_norm)          # final norm
return x                                              # hidden [T, hidden]

compute_logits(hidden, sample_indices) = (hidden[sample_indices] @ embed.T).float()   # [S, V] fp32
```
**HF numerics to copy exactly (for bit-close tests):**
- RMSNorm: `x32 = x.float(); x32 = x32 * rsqrt(mean(x32²) + eps); out = weight * x32.to(fp16)` (cast *before* multiplying by the weight).
- Residual adds happen in fp16.
- RoPE: `inv_freq = 1 / theta^(arange(0, D, 2) / D)` in fp32; `freqs = pos ⊗ inv_freq` (fp32);
  `emb = cat(freqs, freqs)`; cos/sin computed in fp32, then **cast to fp16**;
  `q' = q·cos + rotate_half(q)·sin`, with `rotate_half(x) = cat(−x[..., D/2:], x[..., :D/2])`.
- MLP: `silu(gate) * up` in fp16 (silu computed internally in fp32 and rounded).
- GQA: query head `h` uses KV head `h // G`.

### 6.8 Attention (`turingserve/ops/`)

**Dispatch:** rows `[0, num_decode)` → decode kernel; remaining rows → prefill kernel. In speculative
verify steps `num_decode = 0`, so every row uses the prefill kernel.

**Reference implementation** (`ops/reference.py`, CPU or GPU, fp32 math): for each sequence, gather its
blocks into dense `K, V [seq_len, KVH, D]`, repeat each KV head G times, and run SDPA with an
**explicit bottom-right causal mask**: query row i (absolute position `seq_len − q_len + i`) may see key j
iff `j ≤ seq_len − q_len + i`.

**Decode kernel — three versions, all kept and tested:**
- **v1 (simplest):** grid `(S, H)`. One program per (sequence, query head). Load q `[D]`. Loop over the
  sequence's blocks (16 tokens per iteration): `phys = block_tables[s, i]`; load K `[16, D]` (mask tokens
  ≥ seq_len); `scores = sum(q[None, :] * K, axis=1) * scale`; online softmax
  (`m_new = max(m, max(scores)); α = exp(m − m_new); p = exp(scores − m_new); l = l·α + Σp; acc = acc·α + Σ p[:, None]·V; m = m_new`);
  output `acc / l`. Uses no `tl.dot`, so it works for any group size.
- **v2 (production):** grid `(S, KVH)`. One program per KV head, handling all G query heads of its
  group together, so K/V are read **once per group**. Pad G (7 or 6) to **BLOCK_H = 16** (the minimum
  `tl.dot` size) and mask rows ≥ G. `scores = tl.dot(q[BLOCK_H, D], trans(K)[D, BLOCK_N])`;
  `acc += tl.dot(p.to(fp16), V)`. BLOCK_N ∈ {16, 32, 64}: each iteration covers BLOCK_N/16 blocks.
- **v3 (split-K, "flash-decoding"):** grid `(S, KVH, NUM_SPLITS)`. Each split handles a range of
  `partition` tokens (256/512/1024) and writes partial `(m, l, acc)` to scratch buffers. A second kernel
  (grid `(S, H)`) combines them: `M = max_j m_j`, `L = Σ_j l_j·e^(m_j − M)`, `out = Σ_j acc_j·e^(m_j − M) / L`.
  Empty splits use `m = −inf, l = 0` and must not produce NaN; rows with `seq_len = 0` (padding) write zeros.
  Used when `S × KVH < 2 × num_SMs` and the longest context > 1024 (the tuner refines the threshold).
  **Inside CUDA graphs,** NUM_SPLITS is fixed at `ceil(max_model_len / partition)`, and splits beyond the
  sequence exit immediately.

**Prefill kernel** (varlen, paged, causal, FlashAttention-style): grid `(ceil(max_query_len / BLOCK_M), P, H)`.
A program handles query rows `[m0, m0 + BLOCK_M)` of prefill sequence p and head h. With
`prefix = seq_len − q_len`, query row i has absolute position `prefix + i`. Loop over key tiles
`j ∈ [0, min(seq_len, prefix + m0 + BLOCK_M))` in steps of BLOCK_N. For each key, compute its physical
address from `block_tables[p, j // 16]` and `j % 16`. Mask: `j ≤ prefix + i` and `j < seq_len`. Online
softmax with `tl.dot` for Q·Kᵀ and P·V, fp32 accumulators, P cast to fp16 before the second dot.
- **Turing-safe configs only:** BLOCK_M ∈ {32, 64}, BLOCK_N ∈ {32, 64}, num_warps ∈ {4, 8},
  num_stages ∈ {1, 2}. Shared memory check for D = 128, BLOCK_M = BLOCK_N = 64: Q 16 KiB + K 16 KiB +
  V 16 KiB = 48 KiB ≤ 64 KiB. BLOCK_M = 128 with D = 128 reaches the limit, so it is excluded.

### 6.9 Other kernels (`turingserve/ops/triton_*.py`)

| Op | Signature | Grid / blocks | Numerics |
|---|---|---|---|
| `rmsnorm` / `rmsnorm_residual` | `(x[T,N], residual[T,N] or None, w[N], eps) → (out, new_residual)` | `(T,)`, BLOCK_N = next_pow2(N) with mask (896→1024, 1536→2048) | Add in fp32 then round to fp16 (= fp16 add); norm in fp32; cast; × w |
| `silu_mul` | `gu[T, 2I] → out[T, I]` | `(T, ceil(I/1024))` | silu in fp32 → fp16, × up |
| `rope_` | in place on `qkv[T, (H+2KVH)·D]`, `positions[T]`, cos/sin cache `[max_model_len, D/2]` fp32 | `(T, H+KVH)` | fp32 math, one rounding (≤ 1 ulp from HF) |
| `kv_write` | `(k[T,KVH,D], v, k_cache, v_cache, slot_mapping[T])` | `(T, KVH)` | Skip slot < 0; exact copy |
| `w8a16_linear` | `(x[M,K] fp16, Wq[N,K] int8, scale[N] fp32, bias[N] or None) → y[M,N] fp16` | `(ceil(M/BM), ceil(N/BN))` | int8 → fp16 in registers, `tl.dot`, fp32 acc, × scale at the end, + bias |
| fp16 linear | `torch.nn.functional.linear` (cuBLAS, tensor cores) | — | Not rewritten (cuBLAS is near peak) |

### 6.10 Model runner (`turingserve/runner/`)

**6.10.1 Input builder.** Persistent **pinned** CPU numpy buffers and persistent GPU buffers sized for
`max_num_batched_tokens` and `max_num_seqs`. Build with vectorized numpy: token ids, positions
(`num_computed + arange(n)` per sequence), slot mapping (block-table lookup by `pos // 16` and `pos % 16`),
`query_start_loc` (cumsum of n), `seq_lens`, block tables padded with 0, sample indices
(`query_start_loc[i+1] − 1` for sampling rows). Copy to the GPU with `non_blocking=True`.

**6.10.2 CUDA graphs** (pure-decode steps only):
```
sizes = graph batch sizes ≤ max_num_seqs
static GPU buffers sized for max(sizes): input_ids, positions, slot_mapping, seq_lens, block_tables,
                                         query_start_loc, logits_out [maxB, V] fp32
warm-up: run every size eagerly twice (compiles Triton kernels; the tuner must already have finished)
pool = torch.cuda.graph_pool_handle()
for B in sizes, largest first:
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool): logits_out[:B] = compute_logits(forward(static[:B]), arange(B))
replay(n): B = smallest size ≥ n; copy real rows into static[:n]; padding rows get input_id 0, position 0,
           slot −1, seq_len 0, block-table row of zeros; g.replay(); use logits_out[:n]
```
Forbidden during capture: `.item()`, `.cpu()`, `print(tensor)`, creating tensors from Python data,
Triton autotuning, and any host–device synchronization.

**6.10.3 Memory profiler** (`memory_profiler.py`), run once at startup:
```
free0, total = torch.cuda.mem_get_info()        # after CUDA init
baseline = total - free0                        # other processes + our CUDA context
load weights (and draft weights); allocate a temporary KV cache for max_num_batched_tokens tokens
torch.cuda.reset_peak_memory_stats()
dummy prefill of max_num_batched_tokens tokens + compute_logits for min(max_num_seqs, T) rows + sampler
peak = torch.cuda.max_memory_allocated(); free the temporary cache; empty_cache()
free1, _ = mem_get_info(); non_torch = max(0, (total - free1) - baseline - torch.cuda.memory_reserved())
graph_reserve = min(256 MiB, 0.04 * total)
kv_budget = total * gpu_memory_utilization - baseline - peak - non_torch - graph_reserve
bytes_per_block = 2 * L * KVH * 16 * D * 2  (+ the draft model's, if speculation uses a draft)
num_blocks = floor(kv_budget / bytes_per_block)
if num_blocks - 1 < ceil(max_model_len / 16): raise "Not enough GPU memory for one full-length sequence.
   Free GPU memory (is Ollama running?), lower --max-model-len, or use --quantization int8."
```
`--num-blocks N` skips the profiler and uses N blocks (tests only). On `--device cpu` the profiler is
skipped and `num_blocks` defaults to 2048.

### 6.11 Sampler (`turingserve/sampling/`)

Input: `logits [S, V] fp32` and the S sampling sequences. Steps, applied in this order:
1. **Penalties** (only if some row needs them; `penalties.py`): build token counts from the output
   (and, for the repetition penalty, the prompt) with `scatter_add` into `[S, V]`.
   - Repetition (HF): for tokens seen in prompt + output, `logit = logit / rp if logit > 0 else logit · rp`.
   - Frequency/presence (OpenAI): `logit −= frequency · count_out + presence · (count_out > 0)`.
2. **min_tokens:** if `num_output < min_tokens`, set the logits of the EOS ids and `stop_token_ids` to −inf.
3. **Greedy rows** (`temperature < 1e-5`): `argmax`.
4. **Other rows:** `logits / temperature` → top-k (keep the k largest; others −inf) → top-p (sort
   descending, `probs = softmax`, keep a token iff the cumulative probability *before* it is < top_p;
   always keep ≥ 1) → `probs = softmax(filtered)`.
5. **Sampling by exponential race:** `E = empty_like(probs).exponential_()`, using the row's own
   `generator` for seeded rows (a small loop over seeded rows only) and the global generator for the
   rest; `token = argmax(probs / E)`. This samples token i with probability `p_i`, keeps per-request seeds
   reproducible regardless of batch composition, and needs no per-row `multinomial`.
6. Also compute `bad = ~isfinite(logits).all(dim=−1)`. Copy `tokens` and `bad` to the CPU in **one** transfer.

### 6.12 Output processor (`turingserve/engine/`)

**Detokenizer** (`detokenizer.py`): one `tokenizers.decoders.DecodeStream(skip_special_tokens=True)`
per sequence; `piece = stream.step(backend_tokenizer, token_id)` returns new text or `None`, and it
handles UTF-8 characters split across tokens (Hindi, emoji). Fallback if `DecodeStream` is unavailable:
the prefix/read-offset algorithm (decode `ids[prefix:]` and `ids[prefix:read]`, emit the difference only
if it does not end with "�"). **Test:** concatenated deltas == `tokenizer.decode(all_output_ids, skip_special_tokens=True)`.

**Finish checks** after each new token (`stop.py`, in this order):
1. Token in EOS ids (and `ignore_eos` is false) or in `stop_token_ids` → finish "stop" (the token's text is not emitted).
2. Append the detokenized piece to `seq.text`. Search for each stop string in the region that could
   contain a new match (`text.find(s, max(0, emitted_len − len(s) + 1))`). If found at index k:
   truncate `text` to k, emit `text[emitted_len:k]`, finish "stop".
3. `num_output == max_tokens` or `len(token_ids) == max_model_len` → finish "length".
4. Not finished: emit `text[emitted_len : len(text) − (max_stop_len − 1)]` (**hold back** characters that
   could be the start of a stop string; with no stop strings, hold back nothing).
5. On finish: emit everything not yet emitted (except a matched stop string).

### 6.13 Speculative decoding (`turingserve/spec/`, `turingserve/sampling/rejection.py`)

**When it runs** (`should_run`): `--spec-method` is not `none`, Autopilot has it enabled, the plan is
pure decode (no prefill chunks), `len(plan) ≤ spec_max_batch`, and the request uses no penalties
(requests with penalties always take the normal path).

**Proposers**
- **n-gram (prompt lookup):** for n from `ngram_max` down to `ngram_min`: take the last n tokens of
  `token_ids`; find their **latest earlier occurrence** in `token_ids` (numpy search); if found at index i,
  propose `token_ids[i+n : i+n+k]` (it may be shorter than k). No match → propose nothing (k_i = 0).
- **Draft model:** a second Qwen2 model (0.5B) with its own KV tensor of the same `num_blocks`, **sharing
  the target's block tables** (same token positions), so it needs no second allocator.
  1. Catch-up and first proposal: feed `token_ids[draft_computed:]` (≥ 1 token) through the draft model
     (prefill path), and take `d1` and `q1` at the last position.
  2. For j = 2..k: feed `d_{j−1}` (decode path) to get `d_j` and `q_j`.
  3. Afterwards `draft_computed = old_len + (k − 1)`, adjusted after verification (below).

**Verification (target model, one forward pass, prefill kernel path):** each speculating sequence
contributes `1 + k_i` query tokens: `[last uncomputed token, d1 … d_k]` at consecutive positions, with KV
slots allocated for them. Logits are computed at all `1 + k_i` positions.

**Acceptance (`rejection.py`)**, using distributions after temperature/top-k/top-p, identical for p and q:
- Greedy: accept `d_i` iff `d_i == argmax p_i`. At the first rejection, output `argmax p_i`. If all k
  are accepted, also output the bonus `argmax p_{k+1}`.
- Sampling, draft model: accept with probability `min(1, p_i(d_i) / q_i(d_i))`. On rejection, sample from
  `normalize(max(0, p_i − q_i))` and stop. If all are accepted, sample the bonus from `p_{k+1}`.
- Sampling, n-gram (q is a point mass on d_i): accept with probability `p_i(d_i)`. On rejection, sample
  from p_i with d_i removed and renormalized.
- Random numbers come from the request's generator (seeded) or the global one.
- `min_tokens` masks are applied per position, using the output count at that position.

**After acceptance of m drafts:** the new tokens are `d1..dm` plus one more (the correction or the bonus).
Feed them one by one through the output processor; stop early on EOS/stop/length. Then
`num_computed_tokens = old_computed + 1 + m`; `draft_computed = old_len + min(m, k − 1)`; free trailing
blocks that are no longer needed (`kv.trim(seq)`); register newly full blocks with the prefix cache.

**Metrics:** drafted tokens, accepted tokens, acceptance rate, mean accepted length per verify step.

### 6.14 INT8 weight-only quantization (`turingserve/quant/int8.py`)

- Quantize at load: for each linear weight `W [N, K]` (qkv, o, gate-up, down):
  `scale[n] = max(|W[n, :]|) / 127` (clamped ≥ 1e-8, stored fp32), `Wq = round(W / scale[:, None]).clamp(−127, 127).to(int8)`.
  The embedding (= lm head), the norms and the biases stay fp16.
- **Kernel choice per call:** if `M ≤ M_cross` (tuned per (N, K); default 256), use the Triton `w8a16_linear`.
  Otherwise dequantize the weight to a temporary fp16 tensor (`Wq.to(fp16) * scale`) and use cuBLAS:
  large M is compute-bound, so the extra memory traffic is small.
- `--int8-lm-head`: keep an extra INT8 copy of the embedding for the logits matmul. It costs ~233 MB on
  the 1.5B, and the embedding lookup still uses fp16. This is an ablation option.
- **Quality gates (§10.6):** WikiText-2 perplexity increase ≤ 2% (target) and GSM8K accuracy change,
  reported with sample size and noise.

### 6.15 Autopilot (`turingserve/autopilot/`) — the self-tuning part

**Startup (`--autopilot startup` or `full`)**, cached in
`<tuning-dir>/<gpu>__<model>__<dtype|int8>__mml<max_model_len>__tok<max_num_batched_tokens>__torch<v>__triton<v>.json`:
1. **Kernel tuner (`tuner.py`):** for each kernel and shape bucket, time each candidate config with
   `triton.testing.do_bench(fn, warmup=10, rep=50)` and keep the fastest. Configs that fail to compile
   (for example, shared-memory overflow) are skipped and logged.
   - Buckets: decode `S ∈ {1, 2, 4, …, 128}` × `max_seq_len ∈ {256, 512, …, max_model_len}`;
     prefill `max_query_len ∈ {16, …, max_num_batched_tokens}`; w8a16 `M ∈ {1, 16, 64, 256, 1024, 4096}`
     for each (N, K) of the model, plus `M_cross`.
   - Candidates: prefill as listed in §6.8; decode BLOCK_N ∈ {16, 32, 64}, num_warps ∈ {2, 4}, split
     partition ∈ {256, 512, 1024}. Elementwise kernels (rmsnorm, silu_mul, rope, kv_write) are not tuned:
     num_warps = 4 if their BLOCK ≤ 1024, else 8. w8a16 (BM, BN, BK) ∈ {(16,64,64), (16,128,64), (32,64,64),
     (64,64,32), (64,128,32), (128,64,32)}, warps {4, 8}, stages {1, 2}.
   - At runtime a kernel looks up its bucket (rounding up). On a miss it uses the first candidate and logs once.
2. **Memory profile** (§6.10.3) → number of KV blocks.
3. **CUDA graph capture** (§6.10.2).
4. **Chunk-size sweep** (only if not cached): run the engine offline for 30 s per candidate
   `max_num_batched_tokens ∈ {512, 1024, 2048}` (capped by the flag), with 16 concurrent synthetic
   requests (prompt 1024, output 128). Keep the candidate with the highest throughput that meets the
   TTFT/TPOT targets at p90; if none meets them, keep the one with the lowest TPOT p90.

**Runtime controller (`controller.py`, `--autopilot full`):** a **pure function**
`decide(state, signals) → (new_state, decisions)`, called every **2 s**. Signals use a 10 s sliding
window: TTFT p90, TPOT p90, running decode batch size, preemptions, KV usage, and an EMA of accepted
drafts per verify step (ā). A rule fires only after its condition holds in **3 consecutive windows**.
After any change, that setting cools down for **10 s**.

| Rule | Condition | Action | Bounds |
|---|---|---|---|
| R0 OOM recovery | CUDA OOM in a step (§6.17) | halve max_num_batched_tokens, immediately | ≥ 256 |
| R1 smoother decoding | TPOT p90 > target | chunk size one level down | levels 256 / 512 / 1024 / 2048 / 4096 (≤ flag) |
| R2 faster first token | TTFT p90 > target **and** TPOT p90 < 0.7 × target | chunk size one level up | same |
| R3 speculation off (hard) | decode batch > spec_max_batch | skip speculation for this step | immediate, no hysteresis |
| R4 fewer drafts | ā < 0.6 | k − 1; if k is already 1, speculation off for 60 s, then re-probe with k = 2 | k ∈ [1, 8] |
| R5 more drafts | ā > 0.8 × k | k + 1 | k ≤ `--spec-k` |
| R6 less memory pressure | preemptions > 5 per window | max_num_seqs × 0.75 | ≥ 8 |
| R7 more concurrency | 0 preemptions and KV usage < 70% for 6 windows | max_num_seqs × 1.25 | ≤ flag value |

The 0.7 band between R1 and R2 prevents oscillation. Every decision is logged as one JSON line
(time, rule, signal values, old → new) and exported as a metric.
**Why rules and not reinforcement learning:** rules are explainable, testable and stable. Say this in interviews.

### 6.16 Observability (`turingserve/metrics/`)

| Metric | Type |
|---|---|
| `turingserve_request_ttft_seconds` (buckets 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30) | Histogram |
| `turingserve_request_tpot_seconds` (buckets 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5) | Histogram |
| `turingserve_request_e2e_seconds` | Histogram |
| `turingserve_prompt_tokens_total`, `turingserve_generation_tokens_total` | Counter |
| `turingserve_requests_total{finish_reason}` | Counter |
| `turingserve_num_running`, `turingserve_num_waiting`, `turingserve_kv_usage_ratio` | Gauge |
| `turingserve_prefix_cache_queried_tokens_total`, `turingserve_prefix_cache_hit_tokens_total` | Counter |
| `turingserve_preemptions_total` | Counter |
| `turingserve_spec_draft_tokens_total`, `turingserve_spec_accepted_tokens_total` | Counter |
| `turingserve_step_seconds{phase=schedule\|prepare\|forward\|sample\|output}` | Histogram |
| `turingserve_engine_cpu_seconds_per_step` | Histogram |
| `turingserve_autopilot_decisions_total{rule}`, `turingserve_autopilot_setting{name}` | Counter, Gauge |

- JSON-lines logs (`ts, level, event, request_id, …`). With `--log-requests`, one line per finished
  request (timings, token counts, cached tokens, finish reason).
- Prometheus and Grafana run as **plain binaries in WSL** (no Docker), scraping `localhost:8000/metrics` every 2 s.
- Profiling: `POST /debug/profile?steps=N` writes a `torch.profiler` Chrome trace (open it at ui.perfetto.dev).

### 6.17 Robustness

- **CUDA OOM during a step:** log it, preempt every running sequence, `torch.cuda.empty_cache()`, halve the
  current `max_num_batched_tokens` (an Autopilot decision, R0), and continue. Three OOMs within 60 s →
  unhealthy (503) and abort everything.
- **NaN/inf logits** (the `bad` flag): finish that request with "error" and log the prompt length and
  position. Never sample from NaN.
- **Engine thread exception:** log the stack trace, fail all requests (500 or a stream error), report 503 on `/health`.
- **Shutdown** (SIGINT/SIGTERM via uvicorn lifespan): stop accepting (503), abort running requests,
  join the engine thread (10 s timeout), exit.
- **Memory hygiene:** finished and aborted requests are removed from every dict (engine, sinks,
  generators). The soak test checks `num_requests_tracked` returns to 0.

---

## 7. Request lifecycle (example)

```
1. POST /v1/chat/completions, stream=true, max_tokens=200. Template + tokenize → 900 prompt tokens.
2. AsyncEngine queues it. Step k: the scheduler admits it. The prefix cache hits 40 blocks (640 tokens),
   so 260 tokens are uncomputed. The budget left after decodes is 256 → chunk 1 = 256 tokens, no sampling.
3. Step k+1: 4 remaining prompt tokens. The chunk reaches the end, so it samples token 1. First SSE delta → TTFT.
4. Steps k+2…: 1 token per step, mixed with other requests' decodes and prefills (or up to 1 + k tokens
   per step with speculation, when the batch is small and pure decode).
5. A stop string appears → text truncated, finish_reason "stop", final chunk, usage chunk, [DONE].
6. kv.free(seq): its hashed full blocks become evictable LRU entries, ready for the next request with the same prefix.
7. Metrics record TTFT/TPOT/E2E. Autopilot sees the new samples in its next window.
```

---

## 8. Repository layout

```
turingserve/                                 (git repo, inside WSL at ~/code/turingserve)
├── pyproject.toml · requirements.lock · README.md · LICENSE (Apache-2.0) · .gitignore
├── docs/ ARCHITECTURE.md · BUILD_GUIDE.md · day1_gate.md · baselines.md · lab_notes.md · design/*.md
├── turingserve/
│   ├── __init__.py · cli.py · config.py · simple_generate.py (Phase 2 helper)
│   ├── utils/        timing.py · logging.py · gpu.py
│   ├── api/          server.py · protocol.py · streaming.py · async_engine.py
│   ├── engine/       engine.py · request.py · sequence.py · sampling_params.py · output_processor.py · detokenizer.py · stop.py ·
│   │                 static_batch.py (Phase 3 helper)
│   ├── scheduler/    scheduler.py
│   ├── kv/           block_allocator.py · kv_manager.py · prefix_cache.py
│   ├── model/        qwen2_config.py · loader.py · layers.py · qwen2.py
│   ├── ops/          __init__.py (dispatch) · reference.py · triton_rmsnorm.py · triton_silu_mul.py · triton_rope.py ·
│   │                 triton_kv_write.py · triton_decode_attention.py · triton_prefill_attention.py · triton_w8a16.py
│   ├── runner/       attention_metadata.py · input_builder.py · model_runner.py · cuda_graphs.py · memory_profiler.py ·
│   │                 contiguous_cache.py (Phase 2; kept as a test reference)
│   ├── sampling/     sampler.py · penalties.py · rejection.py
│   ├── spec/         ngram.py · draft.py · spec_decoder.py
│   ├── quant/        int8.py
│   ├── autopilot/    tuner.py · startup.py · controller.py
│   └── metrics/      prometheus.py · tracing.py
├── tests/  conftest.py · fakes.py · unit/ · correctness/ · kernels/ · e2e/ · golden/ (generated, git-ignored)
├── scripts/  probe_triton_mma.py · measure_bandwidth.py · day1_gate.py · perplexity.py · make_tiny_model.py ·
│             make_golden.py · check_versions.py
├── benchmarks/  serve_bench.py · datasets.py · offline_bench.py · chaos.py · soak.py · make_report.py ·
│                compare_rag_answers.py · micro/{decode_speed,bench_*}.py · baselines/hf_server.py ·
│                results/*.json · figures/*.png · data/ (git-ignored) · REPORT.md
├── deploy/observability/  prometheus.yml · grafana/provisioning/{datasources,dashboards}/*.yaml · dashboard.json · run.sh
├── notebooks/kaggle/  00_setup · 10_turingserve_bench · 20_vllm_latest · 21_vllm_v0 · 22_llamacpp · 30_quality (.ipynb)
└── .github/workflows/ci.yml
```

---

## 9. Testing strategy

```
          ▲  few, slow      e2e/: API, streaming, errors, cancel, overload; soak + chaos (scripts)
         ▲▲▲                correctness/: vs HF golden files; paged == contiguous; batch invariance;
        ▲▲▲▲▲                             preemption equivalence; graphs == eager; spec == plain; int8 quality
       ▲▲▲▲▲▲▲  many, fast  unit/ + kernels/: allocator, prefix cache, scheduler, sampler, detokenizer,
                                         stop strings, controller; every kernel vs reference (random shapes)
```

**Test models**
- **Tiny random Qwen2** (CPU, CI): `Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
  num_attention_heads=4, num_key_value_heads=2, vocab_size=151936, tie_word_embeddings=True,
  rope_theta=1e6, max_position_embeddings=4096, initializer_range=0.2)`, created by
  `scripts/make_tiny_model.py` with a fixed seed and saved with the real Qwen tokenizer files. The larger
  initializer range spreads the logits, so greedy ties are rare. A second tiny config uses 14 heads / 2 KV heads (G = 7).
- **Real models** (GPU): the 0.5B on the laptop; the 1.5B on Kaggle.
- **FakeModelRunner** (`tests/fakes.py`): a deterministic next token
  `(sum(last 3 token ids) + position) % 1000`. Used for **exact-text** scheduler/preemption/spec/prefix tests.
- **Golden files** (`scripts/make_golden.py` → `tests/golden/*.pt`, git-ignored): HF outputs computed once
  per model and dtype. For 20 prompts: teacher-forced top-20 logits (values and indices) at every position;
  full logits at 8 positions of 3 prompts; greedy 64-token outputs (`do_sample=False, repetition_penalty=1.0`);
  per-layer hidden states for 2 prompts. Also the **noise floor**: max |logit| difference between HF fp16 and HF fp32.

**Tolerance policy** (use exactly this; never loosen a threshold without writing the measured numbers in `lab_notes.md`)

| Comparison | Pass condition |
|---|---|
| fp32, tiny model, CPU, ours vs HF | max abs logit diff ≤ 1e-4 |
| fp32, real model, GPU, ours vs HF | max abs logit diff ≤ 2e-3; greedy 64 tokens identical |
| fp16, ours vs HF fp32 | max abs diff ≤ 1.5 × noise floor + 0.05; greedy may diverge only at a **near tie** (top-2 gap < 0.1 in our logits at the first differing position) |
| Kernel vs reference (attention) | `assert_close(atol=1e-2, rtol=1e-2)` **plus needle tests** |
| Kernel vs reference (rmsnorm, silu_mul) | `assert_close(atol=1e-3, rtol=1e-3)` |
| rope | `assert_close(atol=2e-3, rtol=2e-3)` |
| kv_write, graphs vs eager | exact (`torch.equal`) |
| w8a16 | relative Frobenius error ‖y − ref‖ / ‖ref‖ < 1e-3 |
| Sampler distributions | chi-square statistic below the 99.9% critical value (df = 3 → 16.27) |

**Needle tests** (they catch off-by-one mask bugs that a 1e-2 tolerance would hide): set one key equal to
`100 × q` at a boundary position (first token, last token, 15/16/17, a block boundary, the last token
of the prefix), so its attention weight is ≈ 1. The output must equal that token's V within 1e-2.

**Markers:** `gpu` (needs CUDA), `model` (needs a downloaded real model), `slow` (> 30 s), `kaggle`
(needs the T4 or the 1.5B). CI runs `pytest -m "not gpu and not model and not slow and not kaggle"`.
Kernel tests can also run in CI with `TRITON_INTERPRET=1` on tiny shapes (optional; if an op is
unsupported by the interpreter, mark that test `gpu`).

**Debug ladder for any numeric mismatch**
1. `--kernels reference`: if this fixes it, the bug is in a Triton kernel → add that shape to its unit test.
2. `--dtype float32`: if this fixes it, it is precision → apply the tolerance policy; otherwise it is a logic bug.
3. Layer-by-layer vs golden → the first layer that diverges.
4. Inside the layer: compare q/k after RoPE, the attention output and the MLP output with HF forward hooks.
5. Scheduler bugs: FakeModelRunner + Hypothesis shrinking → the smallest failing sequence of events.
6. Illegal memory access: `CUDA_LAUNCH_BLOCKING=1`, then the kernel's test with `TRITON_INTERPRET=1` on that shape.

---

## 10. Benchmark methodology

### 10.1 Engines (same GPU, same weights, same prompts)

| Engine | Where | Notes |
|---|---|---|
| **TuringServe** | Kaggle T4 | `--dtype float16 --max-model-len 4096` (8192 for prefill tests) |
| HF sequential | Kaggle T4 | `benchmarks/baselines/hf_server.py`: one request at a time, `generate()` + streamer |
| HF static batch (offline only) | Kaggle T4 | batch 16, left padding, `generate()` |
| vLLM latest | Kaggle T4, separate notebook | `--dtype half --max-model-len 4096 --gpu-memory-utilization 0.85`; record the auto-selected attention backend |
| vLLM 0.6.x (V0, xformers) | Kaggle T4, separate notebook | Optional; only if it installs. Older engine that uses tensor-core attention on Turing |
| llama.cpp `llama-server` | Kaggle T4, separate notebook | CUDA build for sm_75, f16 GGUF of the same model, `-ngl 99 -np 16 -c 65536`, flash attention on |

### 10.2 Workloads
- **ShareGPT** (`ShareGPT_V3_unfiltered_cleaned_split.json`): prompt = first human turn, target output
  length = token count of the first assistant turn. Keep `4 ≤ prompt ≤ 1024`, `output ≥ 4`,
  `prompt + output ≤ 2048`. Sample 500 with seed 0. Send `max_tokens = target` and `ignore_eos = true`.
- **Synthetic fixed:** WikiText-2 text cut to exactly 512 prompt tokens, 128 output tokens, `ignore_eos`.
- **Prefill:** prompts of 2,048, 4,096 and 8,000 tokens, `max_tokens = 1`, concurrency 1. TTFT is the
  result (the comparison with vLLM issue #60821).
- **RAG:** real prompts captured from your RAG project (55 dev + 27 held-out questions), greedy,
  `max_tokens = 512`, natural stopping. This shows prefix caching and prompt lookup.

### 10.3 Fairness rules
- Prompts are sent to **`/v1/completions`** as text with the chat template already applied, so every
  engine sees the same tokens. The harness compares each response's `usage.prompt_tokens` with its own
  count and warns if they differ by > 1%.
- Same fp16 weights, same `max_model_len`, same output lengths (`ignore_eos`), streaming on.
- 10 warm-up requests (not recorded), then **3 repetitions; report the median**.
- Log GPU clocks, temperature and power once per second during each run
  (`nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv -l 1`).
- One engine at a time on the GPU; with "T4 ×2", set `CUDA_VISIBLE_DEVICES=0`.

### 10.4 Metric definitions
- **TTFT** = time of the first non-empty streamed chunk − send time (client side).
- **TPOT** = (end − first-token time) / (output tokens − 1), for requests with > 1 output token.
- **ITL** = gaps between consecutive chunks (with speculation, one chunk can contain several tokens; TPOT is the headline).
- **E2E latency** = end − send. **Output throughput** = total output tokens / (last end − first send).
- **Goodput** = requests per second that meet **both SLOs: TTFT ≤ 2 s and TPOT ≤ 50 ms**.
- Percentiles: p50, p90, p99 and the mean.

### 10.5 Load patterns
- **Fixed concurrency:** 1, 4, 16, 32, 64 clients, each sending the next request when the last finishes.
- **Poisson rate:** exponential inter-arrival times with a fixed seed.
- **Ramp:** 0.5, 1, 2, 4, 8, 12, 16 req/s, 60 s each → the maximum rate with ≥ 90% goodput.
- **Profile:** piecewise-constant Poisson rates, written `seconds:rate`, e.g. `120:0.5,120:4,120:0.5` (the Autopilot experiment).

### 10.6 Quality
- **Perplexity:** WikiText-2 (`Salesforce/wikitext`, `wikitext-2-raw-v1`, test split), text joined with
  "\n\n", window 1024, stride 512. Loss counted only on the new tokens of each window; `ppl = exp(Σ nll / Σ tokens)`.
  The same script runs HF and TuringServe (fp16, int8).
- **GSM8K** with lm-eval `local-chat-completions` against the server, 5-shot, chat template, multi-turn
  few-shot, temperature 0, the full 1,319 questions. Report `flexible-extract` (headline) and `strict-match`.

### 10.7 Targets (to check that your numbers are sensible; not promises)

| Metric (T4, Qwen2.5-1.5B fp16) | Target |
|---|---|
| Batch-1 decode | ≥ 60% of speed of light (measured bandwidth / weight bytes) |
| 8,000-token prefill TTFT | ≤ 2.0 s (vLLM-latest default: 10.2 s; FlashInfer: 1.1 s [V]) |
| Output throughput at 32 concurrent ShareGPT clients | ≥ 5× HF sequential |
| vs llama.cpp and vLLM | Report honestly, including where you lose |
| INT8 | Batch-1 decode ≥ 1.4× fp16; perplexity +≤ 2% |

### 10.8 Ablations
Turn off one feature at a time: CUDA graphs · Triton kernels (→ reference) · chunked prefill
(`--max-num-batched-tokens = max_model_len`) · prefix caching · speculation · INT8 · Autopilot.
Report throughput at 32 concurrency, TTFT p90, TPOT p90 and batch-1 tokens/s.

### 10.9 Result files
Each run writes `benchmarks/results/<date>_<engine>_<workload>_<label>.json`:
`meta` (git commit, engine name and version, GPU name, driver, torch/triton versions, model, dtype, all
flags, dataset, seed, start time), `requests` (per request: send, first token, chunk times, end, prompt
and output tokens, error), `summary` (all metrics of §10.4), `gpu_log` (clocks/power samples).
`benchmarks/make_report.py` builds `benchmarks/REPORT.md` and the charts **only** from these files.

---

## 11. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Something upgrades torch/Triton (Triton ≥ 3.3 → no tensor cores) | Medium | `requirements.lock`; `scripts/check_versions.py` runs at server start and warns |
| MMA probe fails on Triton 3.2 | Low | Fallback stack: torch 2.5.1 / Triton 3.1.0 (confirmed MMA [V]) |
| Qwen2.5 overflows in fp16 | Low–medium | Day-1 gate; fallback SmolLM2 |
| GTX 1650 runs tensor-core code slowly | High | Laptop = correctness; T4 = performance |
| Ollama occupies the laptop GPU | **High (seen today)** | Unload step in every session; error message names Ollama |
| C: drive fills up | **High (16.5 GB free)** | Move WSL to D: first |
| vLLM / llama.cpp do not install on Kaggle | Medium | Separate notebooks; report what works |
| vLLM enables fast attention on SM75 before you publish | Medium | Re-measure; the claim becomes "close to vLLM, written from scratch" |
| Kaggle quota | Low | ~15 GPU-hours planned in total (Guide, Kaggle plan) |
| Scope creep | **High** | Milestones; M2 is already a complete project; stretch items only after v1.0 |
| Python overhead dominates small models | High | CUDA graphs, vectorized input building, `DecodeStream`, step-time metrics |

---

## 12. Methods: what works, what does not

| Method | Verdict | Reason |
|---|---|---|
| Paged KV cache | ✅ Use | No padding waste; industry standard [S] |
| Continuous batching + chunked prefill (token budget) | ✅ Use | Main throughput gain; smooth streaming [S] |
| Preemption by recompute | ✅ Use | Simple; cheap with the prefix cache; swap over PCIe is slow [S] |
| Hash-chain prefix caching + LRU | ✅ Use | Fits the block allocator; a radix tree (SGLang) is more code for little gain here [S] |
| Write-then-attend | ✅ Use | One memory path for decode and prefill [S] |
| Logits only at sampling rows | ✅ Use (mandatory) | 2,048 × 151,936 fp32 logits = 1.24 GB [S] |
| CUDA graphs for decode | ✅ Use | Removes launch overhead; largest gain at small batch [V] |
| Own Triton kernels on Triton 3.2 | ✅ Use | Tensor cores on Turing; ≥ 3.3 does not use them [V] |
| FlashAttention-2 / FlashInfer libraries | ❌ Not available | sm80+ only (FlashInfer gated in vLLM) [V] |
| bf16 / fp8 | ❌ Not available | Not supported by Turing [V] |
| `torch.compile` | ❌ Skip | Adds a second compiler stack on a pinned old Triton; the hand-written path is the point of the project [S] |
| Rewriting fp16 GEMMs in Triton | ❌ Skip | cuBLAS already uses tensor cores near peak [S] |
| W8A16 INT8 | ✅ Use | Decode is bandwidth-bound; simple kernel [S] |
| W8A8 INT8 tensor cores (SmoothQuant) | ⏸ Stretch | Needs activation calibration [S] |
| INT4 (AWQ/GPTQ) | ⏸ Stretch | Kernel and checkpoint work [S] |
| Prompt-lookup speculation | ✅ Use | No extra model; strong for RAG (answers copy the context) [S] |
| Draft-model speculation, adaptive | ✅ Use | Helps at low batch, hurts at high batch → Autopilot [S] |
| EAGLE / Medusa | ⏸ Stretch | Needs trained heads [S] |
| Per-request seeds via exponential race | ✅ Use | Reproducible and batched [S] |
| `DecodeStream` detokenizer | ✅ Use | Fast (Rust), handles split UTF-8 [S] |
| Swapping KV to CPU | ❌ Skip | Slow on PCIe laptops; recompute is enough [S] |
| Multi-process API/engine (ZeroMQ) | ⏸ Only if measured | Add only if profiling shows GIL contention [S] |
| RL-based tuning | ❌ Skip | Rules are explainable and testable [S] |
| Tensor parallel on T4 ×2 | ⏸ Stretch | Small models don't need it [S] |
| Locust / k6 for load | ❌ Not needed | Own async client covers fixed concurrency, Poisson, ramp, soak [S] |
| lm-eval loglikelihood tasks (HellaSwag) | ❌ Not possible via chat API | No prompt logprobs [V]; use GSM8K + perplexity |
| Native Windows development | ⚠ Avoid | WSL2 matches Kaggle Linux [S] |
| Docker for Prometheus/Grafana | ⚠ Avoid | Networking from Docker Desktop to WSL is fiddly; plain binaries are simpler [S] |

---

## 13. Out of scope and stretch

**Out of scope for v1.0:** multi-GPU, LoRA, multimodal, tool calling, JSON-schema constrained decoding,
logprobs, `n > 1` / beam search, embeddings endpoint, native Windows, AMD GPUs, models other than Qwen2,
pipeline parallelism, disaggregated prefill, KV offload to CPU.

**Stretch (only after v1.0), in order of value:**
1. **Llama 3.2 architecture**, so the RAG's real model (`llama3.2:3b`) can run on TuringServe. Needs:
   no QKV bias, `rope_theta` 500,000 and the **"llama3" RoPE frequency scaling** (factor 32,
   low_freq_factor 1, high_freq_factor 4, original context 8192; read the exact values from `config.json`).
   The 3B needs ~6.4 GB in fp16 → T4 only.
2. JSON-schema structured output (grammar-constrained decoding), which would let the RAG's `v3` prompt run.
3. INT4 AWQ (W4A16) · 4. EAGLE heads · 5. Overlap scheduling (CPU prepares step n+1 while the GPU runs step n) ·
6. Engine process split · 7. W8A8 SmoothQuant · 8. Tensor parallelism on T4 ×2.

---

## 14. Review log

### v1 → v2 (first review)
Own Triton kernels instead of flash-attn · fp16 instead of bf16 · pinned Triton · WSL2 · laptop for
correctness, T4 for numbers · recompute instead of swap · hash prefix cache instead of a radix tree ·
single process first · adaptive speculation · prompt lookup first · a reference path for every kernel ·
honest goals instead of "beat vLLM" · Autopilot · speed-of-light estimates · milestone cut-off at M2.

### v2 → v3 (this review)

What to do about each problem (the suggestion, how to check it, and the fallback), plus the items that are
not verified yet, is written in the Build Guide section **"Problems found in the review, and what to do about each"** (P1–P55).

| # | Found | Change |
|---|---|---|
| 1 | Triton ≥ 3.3 emits no tensor-core MMA on sm < 80; vLLM's default T4 attention is 9× slower at 8k prefill [V] | Pin Triton 3.2; MMA probe in Phase 0 (PTX must contain `mma.sync`); fallback torch 2.5.1/Triton 3.1.0; this is now the headline differentiator |
| 2 | Current vLLM *does* run on a T4 (TRITON_ATTN/FLEX) [V] | Baselines: vLLM latest + optional vLLM 0.6.x + llama.cpp + HF |
| 3 | Your WSL is Ubuntu 26.04 with Python 3.14 [M]; torch 2.6 has no 3.14 wheels | uv-managed Python 3.12 |
| 4 | C: has 16.5 GB free [M] | Move WSL to D: (`wsl --manage --move`, with export/import fallback) |
| 5 | Ollama holds 2.3 GB of the 4 GB GPU [M] | Unload step every session; startup error mentions Ollama |
| 6 | Qwen's generation_config sets repetition_penalty 1.1, which HF applies even in greedy mode [V] | Golden files and tests pass `repetition_penalty=1.0`, `do_sample=False` |
| 7 | HF `output_hidden_states`: last entry is *after* the final norm [S] | Layer-wise test compares entries 0..L−1 to layer inputs/outputs and the last entry to the final norm |
| 8 | SDPA `is_causal` is top-left aligned [S] | Explicit bottom-right masks in the reference path |
| 9 | KV layout `[blocks, 2, 16, KVH, D]` gives strided per-head loads | `[L, 2, blocks, KVH, 16, D]` |
| 10 | Separate prefill/decode memory paths duplicate code | Write-then-attend |
| 11 | Logits for every token would need 1.24 GB per 2k-token step | Logits only at sampling rows |
| 12 | GQA groups are 7 and 6, not powers of 2; `tl.dot` needs ≥ 16 rows | Decode v1 without `tl.dot`; v2 pads to 16 with masks |
| 13 | Turing limits (64 KiB shared memory, no cp.async, 40 SMs) were not in the plan | Turing-safe tile lists; split-K threshold uses SM count |
| 14 | Per-request torch.Generator + multinomial does not batch | Exponential-race sampling |
| 15 | Detokenization and stop strings were underspecified | `DecodeStream`, stop search window, hold-back rule, finish order |
| 16 | v2 said the final stream chunk always has usage | OpenAI semantics: only with `stream_options.include_usage` |
| 17 | fp16 "batch invariance" and "match HF" cannot be exact | Tolerance policy with fp32 strict tests, a measured noise floor, the near-tie rule and golden files |
| 18 | A 1e-2 tolerance hides off-by-one mask bugs | Needle tests |
| 19 | Exact-text tests on random tiny models are flaky (near-ties) | FakeModelRunner for exact tests; tiny model with initializer_range 0.2 |
| 20 | No CPU CI path for the model | Tiny random Qwen2 + real tokenizer; reference kernels on the CPU; optional Triton interpreter |
| 21 | HellaSwag needs prompt logprobs | GSM8K via `local-chat-completions` + WikiText-2 perplexity |
| 22 | INT8 savings were overstated (tied embedding stays fp16) | Real numbers: 0.63 GB / 1.78 GB; optional INT8 lm-head copy; dequant + cuBLAS for large M |
| 23 | Speculative decoding details were missing | Exact algorithm: shared block tables, draft catch-up, n-gram acceptance rule, no penalties, pure-decode only |
| 24 | Prefix cache details were missing | blake2b chain, register only computed full blocks, last-block rule, null block 0 |
| 25 | CUDA graph pitfalls (autotune during capture, padding) | Tuner before capture; slot −1; fixed split count; shared pool, largest first |
| 26 | `triton.autotune` cannot run inside graphs or persist reliably | Own tuner with a JSON cache per GPU/model/version |
| 27 | Memory budget formula was vague | Exact profiler formula with baseline, non-torch memory, graph reserve and a minimum check |
| 28 | Scheduler preemption could undo already-scheduled work | Preempt only in the decode pass, newest first; no admissions after a preemption |
| 29 | Benchmark fairness was underspecified | Pre-templated `/v1/completions`, `ignore_eos`, token-count check, warm-up, 3 runs, clock logging, defined SLOs |
| 30 | Your RAG uses **Ollama** (`ollama` package, `llama3.2:3b`), not an OpenAI client; `src.generator` imports `generate` by name [M] | New stdlib-HTTP provider in the RAG; same model (Qwen2.5-1.5B) on both backends; prompt capture patches `src.generator.generate` |
| 31 | Docker networking between Docker Desktop and WSL is uncertain | Prometheus/Grafana as plain binaries in WSL |
| 32 | Locust/k6 and our own client overlapped | Own async client only |

---

## 15. Sources

**Verified during the reviews**
- vLLM issue #60821 — Triton attention without tensor cores on Turing since Triton 3.3; T4 numbers: https://github.com/vllm-project/vllm/issues/60821
- triton-windows README (Turing supported by Triton ≤ 3.2): https://github.com/woct0rdho/triton-windows
- triton-turing (community fork restoring Turing MMA): https://github.com/Chennesxu/triton-turing
- PyTorch 2.6.0 ↔ Triton 3.2.0: https://github.com/woct0rdho/triton-windows/issues/50 · release dates: https://www.piwheels.org/project/triton/
- vLLM GPU requirements (CC 7.0+ incl. T4, v0.7): https://docs.vllm.ai/en/v0.7.0/getting_started/installation/gpu/index.html
- vLLM attention backends: https://docs.vllm.ai/en/stable/design/attention_backends/
- vLLM forum, T4 issues: https://discuss.vllm.ai/t/build-issues-when-serving-gpt-oss-20b-on-tesla-t4-gpus-with-vllm/1673
- nano-vllm: https://github.com/coderonion/nano-vllm
- CUDA graphs measurements: https://arxiv.org/pdf/2605.30571 · https://www.spheron.network/blog/torch-compile-cuda-graphs-llm-inference-pytorch-2-6/
- mini-sglang CUDA graphs: https://www.mintlify.com/sgl-project/mini-sglang/configuration/cuda-graph
- Qwen2.5 configs: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json · https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/config.json · https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/raw/main/generation_config.json
- Triton debugging / interpreter: https://triton-lang.org/main/programming-guide/chapter-3/debugging.html
- lm-eval with OpenAI-compatible servers: https://www.morphllm.com/llm-eval-harness · https://docs.openvino.ai/2026/model-server/ovms_demos_continuous_batching_accuracy.html
- GGUF files: https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF · Ollama + Qwen: https://qwen.readthedocs.io/en/latest/run_locally/ollama.html
- Moving WSL: https://zenn.dev/mfmfuyu/articles/fac34e5d1e3ba7 · https://pureinfotech.com/?p=174231 · access-denied reports: https://www.elevenforum.com/t/access-denied-after-moving-my-ubuntu-distribution-to-another-drive.42378/latest
- Kaggle quota: https://gpuperhour.com/blog/free-cloud-gpus-and-credits
- fp16 overflow background: https://huggingface.co/docs/transformers/main/ko/debugging

**Papers and posts to read (background)**
- PagedAttention / vLLM (SOSP 2023): https://arxiv.org/abs/2309.06180
- Orca, iteration-level scheduling (OSDI 2022): https://www.usenix.org/conference/osdi22/presentation/yu
- Sarathi-Serve, chunked prefill (OSDI 2024): https://arxiv.org/abs/2403.02310
- FlashAttention: https://arxiv.org/abs/2205.14135 · FlashAttention-2: https://arxiv.org/abs/2307.08691
- Flash-Decoding: https://crfm.stanford.edu/2023/10/12/flashdecoding.html
- Speculative decoding: https://arxiv.org/abs/2211.17192 · speculative sampling: https://arxiv.org/abs/2302.01318
- Prompt lookup decoding: https://github.com/apoorvumang/prompt-lookup-decoding
- Batch invariance and nondeterminism: https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
