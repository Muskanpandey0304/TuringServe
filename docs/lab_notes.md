# Lab notes

## Measurements

### 2026-11-01 - Phases 1-6, 9, 11, 12 (CPU) - correctness suite on the dev machine
Machine: Windows 11, CPU only (torch 2.10.0+cpu, Python 3.14.0, transformers 5.12.1 - NOT the pinned stack)   Commit: none (no git repo yet)
Command: `python -m pytest tests -q`
Result: all CPU tests pass (unit, correctness vs Hugging Face on tiny g2/g7 Qwen2, e2e against a real
server subprocess). GPU tests (`tests/kernels`, CUDA-graph test) are skipped: no CUDA build of torch / no Triton here.
Tiny-model agreement with HF (fp32, CPU): logits <= 1e-4, every hidden-state entry <= 1e-4, 32 greedy tokens identical.
Decision / next step: run `pytest tests/kernels -m gpu` on the laptop (GTX 1650) and on the Kaggle T4 under the
pinned stack; write the microbenchmark tables here.

### 2026-11-01 - Phase 14 (CPU) - chaos script against the tiny CPU server
Command: `python benchmarks/chaos.py --base-url http://localhost:8123 --model tiny`
Result: PASS - no crash, only 200/400 statuses, `/debug/state` clean afterwards.

## Compatibility findings (not in the design docs)

- transformers 5 writes `rope_parameters: {rope_theta, rope_type}` instead of a top-level `rope_theta`.
  `Qwen2Config` reads both; without this fix every position > 0 silently used theta = 10000 (caught by the
  logits-vs-HF test). The pinned stack (transformers 4.x) uses the top-level key.
- `tokenizer.apply_chat_template(..., tokenize=True)` returns a dict in transformers 5; the server renders text
  with `tokenize=False` and encodes with `add_special_tokens=False`, which is identical in both versions.
- Paged-vs-dense logits differ by ~1e-5 (fp32) because the dense path uses SDPA; the paged-vs-*contiguous*
  comparison (same math) is the one held to 1e-5.

## Failed ideas
(what you tried, the numbers, why it failed)

## Threshold changes
(test, old threshold, new threshold, measured values, reason)

- `test_int8_reference_path_agrees_with_fp32_on_tiny`: top-1 agreement >= 80% (the Build Guide's >= 95% gate is for the
  real 0.5B on the golden prompts - `tests/correctness` for that needs the model; the tiny random 2-layer model has
  near-tied logits).

## Proposals
(ideas not in ARCHITECTURE.md; decide later)

## Differences vs vLLM / nano-vllm
(component, their approach, my approach, why - fill in after reading their code, never copy it)

### 2026-11-01 - Phase 0 - MMA probe on the laptop
Machine: laptop GTX 1650
Command: python scripts/probe_triton_mma.py
Result: mma.sync present: True; result matches torch.matmul (atol=rtol=1e-2); triton 0.69 TFLOPS, torch 0.41 TFLOPS (GTX 1650 has no tensor cores — expected; correctness confirmed, not speed)
Decision / next step: keep the pinned stack (torch 2.6.0 / Triton 3.2.0). Measure real TFLOPS on the Kaggle T4.
