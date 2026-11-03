# Benchmark methodology and known limitations

Hardware for every result unless stated: Kaggle Tesla T4 (sm_75, 15 GiB), pinned stack torch 2.6.0+cu124 and
Triton 3.2.0, TuringServe at tag `m3-rc`, Qwen2.5-1.5B-Instruct fp16. Raw per-request data is in
`benchmarks/results/`; analysis notes are in `docs/lab_notes.md`.

## How a run is measured
- Client: `python -m benchmarks.serve_bench`, OpenAI-style streaming requests; TTFT, TPOT, ITL, E2E and output
  tokens per second come from the per-request timestamps stored in each JSON.
- Workload: ShareGPT prompts filtered to 4 <= prompt <= 1024 tokens, output >= 4, prompt + output <= 2048;
  `ignore_eos` is always on so every request generates its full reference length.
- The reported value per level is the median over repetitions; every repetition is kept in the JSON (`all`).
- Server and client run in the same notebook, the server on GPU 0 only.

## Run sizes (these are different on purpose)
| Experiment | Prompts | Reps | Levels |
|---|---|---|---|
| Main TuringServe sweep | 500 | 3 | c = 1, 4, 16, 32, 64 |
| Speculative decoding sweep | 40 | 2 | c = 1, 4, 8, 16 |
| Autopilot profile | 400 | 3 | Poisson 0.5 / 4 / 0.5 req/s, 120 s each |
| HF naive baseline | 10 | 1 | c = 1 |

## Known limitations
1. **HF baseline is a smoke test.** The sequential server holds one lock and `ignore_eos` forces full-length
   generation, so a 200-prompt sweep took over an hour and froze the kernel more than once. Its 27.8 tok/s is a
   10-prompt, single-rep indicator, not a sweep, and its concurrency levels would not change the result.
2. **Speculative sweep is small.** 40 prompts and 2 reps, no confidence intervals. Differences of a few percent
   (n-gram at c=4 and c=8, about -1% to -10%) are within run-to-run noise and are described as neutral, not as
   regressions. The 0.5B draft model was slower than no speculation at every level (-20% to -57%); that is the
   measured result for this implementation on ShareGPT chat text, not a claim about draft-model speculation in general.
3. **Autopilot test did not saturate the GPU.** At a 4 req/s peak the T4 is arrival-limited (KV usage under 4%, no
   preemptions), so throughput was 443.6 / 450.5 / 444.2 tok/s for off / off+ngram / full: no measurable benefit is
   claimed. The controller also toggled `spec_on` 51 times in one run instead of settling. A heavier profile and a
   cooldown in the controller are planned follow-ups.
4. **Single GPU model, single run day.** Results are one T4 on one date (2026-11-02/03); Kaggle machines vary between
   sessions, so absolute tok/s should be read with that in mind and relative comparisons are the intended use.
5. **Accuracy is reported separately** (`docs/quality.md`: WikiText-2 perplexity, GSM8K fp16 vs INT8).
6. **Prefix caching flatters repeated prompts.** TuringServe and vLLM both enable prefix caching by default, and
   `serve_bench` replays the same prompts in the warm-up and in every repetition. The ShareGPT sweeps are dominated by
   decoding (`ignore_eos`, about 266 output tokens per request), so throughput is barely affected, but TTFT in
   repetitions 2 and later can be optimistic. The prefill sweep is affected completely: it replays only 15 prompts, so
   its TTFT is a cache hit, not cold prefill (8000 tokens in 64 ms is physically impossible on a T4). It must be rerun
   with prefix caching disabled on both engines before any prefill claim is made. That was done on 2026-11-03
   (`*_prefill_t4-1.5b-prefill-cold.json`); the earlier `*_prefill_t4-1.5b-prefill.json` files are cache hits and
   must not be cited as prefill speed.

## Reproducibility notes (mistakes that cost runs, and the fix)
- Results are written to `/kaggle/working`, never `/tmp`: `/tmp` is lost when the kernel restarts.
- After pinning torch the session must be restarted before anything imports torch; otherwise the old torch is
  still loaded and runs misbehave.
- `serve_bench --out` is a directory; the file inside is `<date>_<engine>_<workload>_<label>.json`.
- Kaggle `!` lines inside an indented loop do not substitute `{variables}`, which once made three runs overwrite
  one file named `{name}.json`. Notebook 40 calls `serve_bench` through `subprocess.run` for this reason.
- Download the result JSON files, not the server logs; check the JSON `meta.args.label` and `num_prompts` before
  committing.
- Run each baseline engine in its own notebook (vLLM installs its own torch).

## Provenance notes
- **Commit hashes in older result files.** The git history was rewritten on 2026-11-03 to change commit messages only (file contents are identical). Result files recorded before that carry the old hashes: `349b3f3` is now `b6dc052` (CI lint fixes) and `edd9b7d` is now `1e7c73a` (tag `m3-rc`). Files from 2026-11-03 carry `1e7c73a`.
- **Hugging Face smoke file metadata.** `2026-11-02_hf_baseline_t4-1.5b_smoke.json` was recorded without `--engine` and `--label`, so its metadata said `turingserve` / `run`. Its `base_url` (port 8003) shows it measured the Hugging Face server; the metadata was corrected to `hf` / `hf-baseline-smoke` on 2026-11-03. No measured value was changed.
