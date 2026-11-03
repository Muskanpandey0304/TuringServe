# Baselines (Build Guide K.5-K.6)

Each baseline runs in its OWN Kaggle notebook (each installs its own torch). Fill this table by running
`notebooks/kaggle/20_vllm_latest` (and `21_vllm_prefill_cold`), `22_llamacpp`; re-run the vLLM row right before publishing (P55).

| Engine | Version | Install command | Launch command | Attention backend | Works? | Notes |
|---|---|---|---|---|---|---|
| vLLM latest | 0.33.0 (torch 2.14.0+cu130) | `pip install -q vllm` | `vllm serve ... --dtype half --max-model-len 4096 --gpu-memory-utilization 0.85` | TRITON_ATTN | Yes (T4, 0 errors) | measured 2026-11-03, see lab_notes.md; cold prefill (prefix caching off) re-measured: 8000 tokens 11.1 s vs TuringServe 1.7 s, see lab_notes.md |
| llama.cpp | | cmake `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75` | `llama-server -ngl 99 -np 16 -c 65536 -fa on` | | | f16 GGUF |
| Hugging Face (sequential) | | - | `python -m benchmarks.baselines.hf_server` | sdpa | | one request at a time |
