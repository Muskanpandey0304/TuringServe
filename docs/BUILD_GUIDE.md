# TuringServe — Complete Build Guide (v3)

This guide is the step-by-step procedure for building TuringServe. **ARCHITECTURE.md** (same folder)
defines the design; this guide refers to it as "ARCH §n". Read ARCHITECTURE.md completely once before Phase 0.

Every phase has the same parts: **Goal · Read first · Files · Steps · Tests · Done when · Pitfalls**.

---

## How to use this guide

### Rules (for you, and for any AI agent that helps you)
1. **One phase at a time, in order.** Do not start the next phase until every "Done when" box is checked.
2. **Write tests together with the code.** Never commit failing tests to `main`.
3. **Never loosen a test threshold silently.** If a threshold must change, write the measured numbers and
   the reason in `docs/lab_notes.md` first (ARCH §9, tolerance policy).
4. **Never upgrade torch or Triton** (ARCH §2.5). Install nothing outside the pinned list without
   updating `requirements.lock` and writing why in the lab notes.
5. **Every measurement goes into `docs/lab_notes.md`** (date, machine, commit, command, result).
6. **Build only what ARCHITECTURE.md describes.** Write new ideas under "Proposals" in the lab notes.
7. **An AI agent must ask the human before:** pushing to GitHub, running anything on Kaggle, deleting
   files, or installing software.
8. Before Phase 0, read **"Problems found in the review"** (below the timeline); each problem has a
   suggested fix. When something breaks: first that section and the **Troubleshooting table (§T)**, then
   the **debug ladder (ARCH §9)**.

### Daily routine
```
Windows PowerShell:  ollama ps      → ollama stop <each model listed>      → nvidia-smi  (memory < 500 MiB)
WSL terminal:        cd ~/code/turingserve && source .venv/bin/activate
                     git pull && pytest -m "not slow" -q          # start from a green state
                     ... work ...
                     ruff check . && pytest -q                    # before every commit
                     git add -A && git commit -m "phase N: <what>" && git push
```

### Git workflow
- One branch per phase: `phase-01-model`, `phase-02-kvcache`, …. Merge into `main` when the phase is done.
- Commit messages: `phase N: <what changed>`.
- Milestone tags: `m1` (Phase 2), `m2` (Phase 6), `m3` (Phase 8), `m4` (Phase 11), `m5` (Phase 14), `v1.0` (Phase 15).
  Push tags with `git push --tags`.

### Timeline (3–4 hours per day)

| Week | Phases | Milestone |
|---|---|---|
| 1 | 0, 0K, 1 | Environment ready; model matches HF |
| 2 | 2 | **M1**: KV cache, greedy output identical to HF |
| 3 | 3, 4 | Paged KV cache, scheduler |
| 4–5 | 5, 6 | **M2**: OpenAI API with streaming (**complete, showable project**) |
| 5–6 | 7 | Triton kernels |
| 6–7 | 8 | **M3**: CUDA graphs, first T4 benchmark |
| 7 | 9 | Prefix caching |
| 8 | 10 | INT8 + quality |
| 8–9 | 11 | **M4**: speculative decoding |
| 9–10 | 12, 13 | Autopilot, observability |
| 11 | 14 | **M5**: load/soak/chaos, full report |
| 12 | 15 | **M6 / v1.0**: RAG integration, docs, blog, resume |

---

## Problems found in the review, and what to do about each

These problems were found by checking the plan against your real machine (27 Oct 2026), your RAG
project and current tools. Each one has a suggested fix, and the phases below already follow it.
**Read this section once before Phase 0.** Labels: **[M]** measured on your machine, **[V]** verified from
a source (ARCH §15), **[S]** engineering judgement.

### A. Problems on your machine (you must act on these)

**P1. Ollama is holding 2.3 GB of your 4 GB GPU** [M]
- *Why it matters:* TuringServe needs almost all of the 4 GB. With Ollama loaded, the memory profiler stops
  with "not enough GPU memory", tests run out of memory, or the KV cache becomes tiny.
- *What to do:* before every session run `ollama ps`, then `ollama stop <model>` for each model listed, or
  quit Ollama from the system tray (§0.1). Never run Ollama and TuringServe on the laptop GPU at the same
  time (Phase 15 runs them one after the other). Optional: let Ollama unload sooner by itself with
  `setx OLLAMA_KEEP_ALIVE 1m` in PowerShell, then restart Ollama (its default is 5 minutes).
- *Check:* `nvidia-smi` shows less than ~500 MiB used.

**P2. Drive C: has only 16.5 GB free, and WSL keeps its disk on C:** [M]
- *Why it matters:* the PyTorch + CUDA libraries (~5 GB), the two models (~4 GB), ShareGPT (~0.7 GB) and the
  pip/uv/HF/Triton caches (several GB) all go into the WSL disk file. It grows and does not shrink by itself.
  A full C: drive breaks Windows and WSL.
- *What to do:* move WSL to D: **before** installing anything (§0.2): export a backup,
  `wsl --manage Ubuntu --move D:\WSL\Ubuntu`, and use the import fallback if the move fails. If C: is still
  low afterwards: move Docker Desktop's disk (Docker Desktop → Settings → Resources → Advanced → Disk image
  location) or clean it (`docker system prune`), and clear the uv cache with `uv cache clean`.
- *Check:* `D:\WSL\Ubuntu\ext4.vhdx` exists, and C:'s free space does not drop while you install packages in WSL.

**P3. Your WSL Ubuntu is 26.04, with Python 3.14** [M]
- *Why it matters:* PyTorch 2.6 and Triton 3.2 have wheels only up to Python 3.13. Installing with the
  system Python fails or picks the wrong versions.
- *What to do:* install Python 3.12 with `uv` and use only the project's `.venv` (§0.4, §0.7). Never use
  `sudo pip install`, and never use the system `python3` for this project.
- *Check:* `.venv/bin/python --version` prints 3.12.x and `python scripts/check_versions.py` passes.

**P4. The laptop GPU is small and has no tensor cores (GTX 1650, 4 GB)** [M][S]
- *Why it matters:* laptop speed numbers do not represent the T4, and the 1.5B in fp16 (3.1 GB of weights) barely fits.
- *What to do:* develop and test correctness on the laptop with the 0.5B; publish only Kaggle T4 numbers. On
  the laptop, run the 1.5B in INT8 (after Phase 10), or in fp16 with a small KV cache only for the RAG
  comparison (Phase 15 B).
- *Check:* every result file records the GPU name; the README labels laptop numbers separately.

**P5. The laptop GPU runs hot (69 °C during the review)** [M]
- *Why it matters:* laptop GPUs lower their clocks when hot, so laptop timings change from run to run.
- *What to do:* for laptop microbenchmarks, plug in the charger, choose the Windows "Best performance" power
  mode, run 3 times and take the median, and compare only numbers from the same session. Use the T4 for every published number.

### B. Problems with tools and libraries (the guide works around them)

**P6. Triton 3.3 and newer do not use tensor cores on Turing GPUs** [V]
- *Why it matters:* your attention kernels would silently run on slow FMA units. vLLM measured about 9× slower
  long-prompt prefill on a T4 because of this, and you would wrongly conclude that your kernels are bad.
- *What to do:* pin `torch==2.6.0` (it brings Triton 3.2.0). Run the MMA probe on the laptop (§0.8) and the T4
  (K.4); it checks that the compiled code contains `mma.sync`. If it fails, use the fallback `torch==2.5.1`
  (Triton 3.1.0, confirmed to emit `mma.sync`). Never run `pip install -U` on torch-related packages. vLLM and
  lm-eval may upgrade torch, so they live in separate notebooks or virtual environments. `check_versions.py`
  runs at server start and warns when Triton ≥ 3.3.
- *Opportunity:* this is the project's headline. Measure your 8,000-token prefill against vLLM's default
  backend on the same T4 (Phases 8 and 14).

**P7. FlashAttention-2 and FlashInfer are not available on Turing** [V]
- *Why it matters:* most mini engines (for example nano-vllm) depend on them.
- *What to do:* do not try to install `flash-attn` (it fails, or compiles for a very long time). Write your own
  Triton attention kernels, each with a PyTorch reference (Phases 3 and 7).

**P8. Turing has no bf16, but Qwen2.5 was trained in bf16** [V]
- *Why it matters:* fp16 has a much smaller range, so activations can overflow to inf/NaN.
- *What to do:* run the Day-1 gate for both models (§0.11, K.4); accumulate in fp32 inside every kernel;
  compute RMSNorm in fp32. If the gate fails, switch to the SmolLM2 family.
- *Check:* gates G1 and G3 pass; no test ever finishes a request with "error" because of NaN logits.

**P9. Hugging Face `generate()` silently applies Qwen's generation settings** [V]
- *Why it matters:* Qwen's `generation_config.json` sets `do_sample: true`, temperature 0.7, top-p 0.8, top-k 20
  and `repetition_penalty: 1.1`. Without overrides, HF samples randomly; even with `do_sample=False` it still
  applies the repetition penalty. Every "matches HF" test would fail for no real reason.
- *What to do:* in every HF reference call, pass `do_sample=False, repetition_penalty=1.0, temperature=None,
  top_p=None, top_k=None`. The golden files store the settings they were made with.

**P10. HF's `hidden_states` do not line up one-to-one with your layer outputs** [S]
- *Why it matters:* the last entry comes after the final norm, and your fused design adds the residual later (inside the next norm).
- *What to do:* compare entry i with `h + residual` after layer i−1, and the last entry with your final-norm output (Phase 1, step 5).

**P11. PyTorch SDPA's `is_causal=True` aligns the mask to the top-left** [S]
- *Why it matters:* for chunked prefill (query shorter than the keys) that mask is wrong, yet most tests would still pass.
- *What to do:* build an explicit bottom-right mask whenever query length ≠ key length (Phase 2, step 2; ARCH §6.8).

**P12. transformers v5 changed its APIs, and transformers 4.x needs `huggingface_hub` < 1.0** [S]
- *What to do:* install `transformers>=4.45,<5`; never upgrade `huggingface_hub` on its own; freeze everything
  in `requirements.lock`; use `hf download`, or `huggingface-cli download` if the `hf` command is missing.

**P13. Kaggle traps** [S][V]
- *Why it matters:* downgrading torch breaks the preinstalled torchvision (`operator torchvision::nms does not
  exist`); a new torch needs a session restart; GPU and Internet need phone verification; with "T4 ×2" both
  GPUs are visible; the free quota is limited (~30 h/week).
- *What to do:* follow K.2 exactly (uninstall torchvision and torchaudio, restart, set
  `CUDA_VISIBLE_DEVICES=0` before importing torch); verify your phone number; follow the ~15-hour Kaggle plan;
  run each baseline engine in its own notebook.

**P14. WSL traps** [S]
- *Why it matters:* a repo under `/mnt/d` is many times slower. Triton compiles a small C launcher the first
  time a kernel runs, so it needs `gcc` and `Python.h`. Triton may not find the WSL driver library.
- *What to do:* keep the repo in `~/code`; install `build-essential`; use the uv Python (it includes
  `Python.h`); set `TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib` (§0.10).

**P15. `wsl --manage --move` can fail with `E_ACCESSDENIED`** [V]
- *What to do:* always export a backup first. If the move fails, unregister, import from the backup, and set the default user (§0.2).

**P16. Windows programs must reach the server that runs inside WSL** [S]
- *Why it matters:* your RAG app runs in Windows Python and calls TuringServe in WSL (Phase 15).
- *What to do:* keep `--host 0.0.0.0` (the default). From PowerShell check `curl.exe http://localhost:8000/health`.
  If that fails, make sure `C:\Users\msi\.wslconfig` does not contain `localhostForwarding=false`, run
  `wsl --shutdown`, and start the server again.

**P17. lm-eval cannot run likelihood tasks (such as HellaSwag) through a chat API** [V]
- *What to do:* measure quality with GSM8K (a generation task, run through the API) and WikiText-2 perplexity
  (computed directly from the model's logits) (Phase 10).

**P18. Benchmarks across engines are easy to make unfair** [S][V]
- *Why it matters:* engines may apply chat templates differently, stop at different lengths, or run at
  different GPU clocks (the T4's 70 W power cap causes throttling).
- *What to do:* send pre-templated prompts to `/v1/completions`; fix output lengths with `ignore_eos`; check
  that `usage.prompt_tokens` matches your own count; warm up; repeat 3 times and report the median; log GPU
  clocks and power (ARCH §10.3, Phase 8).

**P19. The captured RAG prompts contain text from arXiv papers** [S]
- *Why it matters:* publishing them in a public repo may break the papers' licenses.
- *What to do:* keep `benchmarks/data/` git-ignored and publish only the script that regenerates the prompts (Phase 9, step 2).

### C. Problems in the earlier plan (already fixed in v3; know the reasons for interviews)

| # | Problem in the earlier plan | Fix in v3 | Where |
|---|---|---|---|
| P20 | Logits for every token need 1.24 GB per 2,048-token step (out of memory on 4 GB) | Logits only for the rows that sample | ARCH §6.7 |
| P21 | Qwen's GQA groups are 7 and 6 (not powers of 2); `tl.dot` needs ≥ 16 rows | Decode v1 without `tl.dot`; v2 pads the group to 16 and masks | ARCH §6.8 |
| P22 | Turing has 64 KiB of shared memory per block and no `cp.async` | Only Turing-safe tile sizes; `num_stages` 1–2 | ARCH §2.3, §6.8 |
| P23 | fp16 results can never match HF exactly | fp32 strict tests, a measured noise floor, the near-tie rule, golden files | ARCH §9 |
| P24 | A 1e-2 tolerance hides off-by-one mask bugs | Needle tests at every boundary | ARCH §9 |
| P25 | Random tiny models create near-ties, so exact-text tests become flaky | FakeModelRunner for exact tests; tiny model with `initializer_range=0.2` | ARCH §9 |
| P26 | Tuning or synchronizing inside CUDA-graph capture crashes | Tuner runs before capture; no `.item()`; padding slot −1; fixed split count | ARCH §6.10.2 |
| P27 | `@triton.autotune` cannot run inside graphs and does not persist reliably | Own tuner with a JSON cache per GPU/model/version | ARCH §6.15 |
| P28 | Per-request seeds with `multinomial` cannot be batched | Exponential-race sampling | ARCH §6.11 |
| P29 | Hindi/emoji characters split across tokens; stop strings split across chunks | `DecodeStream` detokenizer; stop-string hold-back | ARCH §6.12 |
| P30 | The last stream chunk was planned to always carry usage | OpenAI's rule: only with `stream_options.include_usage` | ARCH §6.1 |
| P31 | INT8 savings were overstated (the tied embedding stays fp16) | Real sizes (0.63 / 1.78 GB); optional INT8 lm-head copy; dequant + cuBLAS for large M | ARCH §6.14 |
| P32 | A preemption could undo work already scheduled in the same step | Preempt only in the decode pass, newest first; no admissions after a preemption | ARCH §6.4 |
| P33 | Speculative-decoding details were missing | Exact algorithm: shared block tables, draft catch-up, n-gram acceptance rule, off for requests with penalties, pure-decode steps only | ARCH §6.13 |
| P34 | A fully cached prompt would leave nothing to compute, so no logits | Drop the last cached block in that case | ARCH §6.6 |
| P35 | The memory budget formula was vague | Exact profiler formula; a clear error message that mentions Ollama | ARCH §6.10.3 |
| P36 | Docker Desktop ↔ WSL networking for Prometheus is uncertain | Prometheus and Grafana as plain binaries in WSL | Phase 13 |
| P37 | Scope creep (never finishing) | Milestones; M2 is already a complete project; stretch items only after v1.0 | Timeline |

### D. Problems in connecting your RAG project (Phase 15)

**P38. Your RAG talks to Ollama through the `ollama` package, not an OpenAI-style API** [M]
- *What to do:* add an `openai_compat` provider to `src/llm.py` that uses only the standard library
  (`urllib.request`, `json`), so the RAG spec's "one LLM SDK" rule still holds (Phase 15 A).

**P39. Your RAG's model is `llama3.2:3b`, which TuringServe v1.0 cannot run** [M]
- *Why it matters:* TuringServe v1.0 supports only the Qwen2 architecture, and the 3B in fp16 (~6.4 GB) does not fit in 4 GB.
- *What to do:* compare the two backends with the **same** model on both, Qwen2.5-1.5B-Instruct (Phase 15 B).
  Llama 3.2 support is stretch item 1 (Phase 16), running on the T4.

**P40. Your RAG's structured-output prompt (`PROMPT_VERSION=v3`) needs JSON-schema decoding** [M]
- *What to do:* use the default `v1r` prompt for the comparison. The new provider raises a clear
  `NotImplementedError` if a schema is passed. Grammar-constrained decoding is stretch item 2.

**P41. `src/generator.py` imports `generate` by name** [M]
- *Why it matters:* patching `src.llm.generate` does not affect it, so a prompt-capture script would silently capture nothing.
- *What to do:* patch `src.generator.generate` (the script in Phase 9, step 2 does this).

**P42. Every eval run writes to the same file names, and `.env` may hold other settings** [M]
- *What to do:* move the result files into `eval/backend_compare/` after every run (Phase 15 C).
  `load_dotenv()` does not override variables already set in the shell, so the `$env:...` lines in Phase 15
  take priority over `.env`. Still look at `.env` for `LLM_PROVIDER` and `LLM_MODEL` before you start.

**P43. Ollama's default repetition penalty may differ from TuringServe's default (1.0)** [S]
- *Why it matters:* different decoding settings make the answer comparison unfair.
- *What to do:* set `LLM_REPEAT_PENALTY=1.0` for both runs (Phase 15 A and C).

### E. Not verified yet: how to settle each one

| # | Open question | How to settle it | If the answer is "no" |
|---|---|---|---|
| P44 | Does Triton 3.2 emit tensor-core `mma.sync` on sm_75? (inferred; the source tested 3.1.0) | MMA probe (§0.8, K.4) | Fallback torch 2.5.1 / Triton 3.1.0 (confirmed) |
| P45 | Does the GTX 1650 run tensor-core instructions at all, and how fast? | MMA probe correctness and TFLOPS on the laptop | The laptop is for correctness only anyway; if `tl.dot` fails there, run those kernel tests only on Kaggle |
| P46 | Is Qwen2.5 safe in fp16? | Day-1 gate (§0.11, K.4) | SmolLM2 family |
| P47 | Do the Ollama tags `qwen2.5:1.5b-instruct-fp16` / `qwen2.5:1.5b-instruct-q8_0` exist? | https://ollama.com/library/qwen2.5/tags | Create the model from the bartowski GGUF with a Modelfile (Phase 15 B) |
| P49 | Is the f16 file in `bartowski/Qwen2.5-1.5B-Instruct-GGUF` named `*f16.gguf`? | `--include "*f16.gguf"`, then `ls` (K.5) | Convert with llama.cpp's `convert_hf_to_gguf.py --outtype f16` |
| P50 | Are the lm-eval flag names unchanged? | `lm_eval --help` | Use the current names; note them in the lab notes |
| P51 | Is `router` the config behind the README's held-out result? | `CONFIGS` in `src/eval.py` and the README | Use the config the README names |
| P52 | Is `tokenizers.decoders.DecodeStream` available? | `check_versions.py` (§0.7) | Use the offset-based fallback detokenizer (ARCH §6.12) |
| P53 | What is your real Kaggle quota, and is "T4 ×2" offered? | Kaggle settings and the accelerator menu | Use "GPU T4"; spread the sessions across weeks |
| P54 | What are the current Prometheus and Grafana versions? | Their download pages, at Phase 13 | Use the current ones; write the versions into `run.sh` |
| P55 | Will vLLM enable fast attention on SM75 (PR #55380) before you publish? | Re-run the vLLM baseline right before publishing | Change the headline to "close to vLLM, written from scratch"; everything else stays valid |

---

## Phase 0 — Laptop setup and the Day-1 gate (2–3 days)

**Goal:** a working, pinned environment on the laptop; proof that Triton uses tensor-core instructions;
proof that the model is safe in fp16; CI running on GitHub.
**Read first:** ARCH §2 (all of it).

### 0.1 Free the GPU from Ollama (every session)
Ollama keeps models on the GPU (2.3 GB of your 4 GB was in use during the review). In **Windows PowerShell**:
```
ollama ps                       # lists loaded models
ollama stop llama3.2:3b         # repeat for every model listed
nvidia-smi                      # "Memory-Usage" must now be below ~500MiB
```
If memory stays high: right-click the Ollama icon in the system tray → **Quit Ollama**, then run `nvidia-smi` again.

### 0.2 Move WSL to drive D: (once)
C: has only 16.5 GB free; this project needs ~20 GB. In **PowerShell as Administrator**:
```
wsl -d Ubuntu -- whoami                           # write down the Linux user name it prints
wsl --shutdown
New-Item -ItemType Directory -Force D:\WSL | Out-Null
wsl --export Ubuntu D:\WSL\ubuntu-backup.tar      # backup first (several minutes, no progress shown)
wsl --manage Ubuntu --move D:\WSL\Ubuntu          # several minutes, no progress shown
wsl -d Ubuntu -- echo ok
```
**Check:** it prints `ok` and `D:\WSL\Ubuntu\ext4.vhdx` exists.

**If the last command fails** (for example `E_ACCESSDENIED`), restore from the backup:
```
wsl --unregister Ubuntu                           # removes the broken entry; the backup tar is safe
wsl --import Ubuntu D:\WSL\Ubuntu D:\WSL\ubuntu-backup.tar
wsl -d Ubuntu -u root -- cat /etc/wsl.conf        # look at the file first
```
If `/etc/wsl.conf` has **no** `[user]` section, add one (replace `<USER>`):
```
wsl -d Ubuntu -u root -- bash -c "printf '\n[user]\ndefault=<USER>\n' >> /etc/wsl.conf"
wsl --terminate Ubuntu
wsl -d Ubuntu -- whoami                           # must print <USER>
```
If it already has a `[user]` section, edit its `default=` line with `wsl -d Ubuntu -u root -- nano /etc/wsl.conf`.

**Memory for WSL:** create `C:\Users\msi\.wslconfig` with:
```
[wsl2]
memory=10GB
swap=8GB
```
Then run `wsl --shutdown`. **Check** inside WSL: `free -g` shows about 10 GB total.
Delete `D:\WSL\ubuntu-backup.tar` only after Phase 0 is complete.

### 0.3 Ubuntu packages (WSL terminal)
```
sudo apt update
sudo apt install -y build-essential git curl wget cmake jq htop unzip
gcc --version                                     # must print a version
nvidia-smi                                        # must show the GTX 1650
```
Do **not** install any NVIDIA driver inside WSL. The Windows driver provides the GPU.

### 0.4 uv and Python 3.12
The system Python is 3.14, and PyTorch 2.6 has no wheels for it (ARCH §2.1).
```
curl -LsSf https://astral.sh/uv/install.sh | sh
exec bash                                         # reload PATH
uv --version
uv python install 3.12
uv python find 3.12                               # prints the interpreter path
```

### 0.5 GitHub repository (code lives in the Linux file system)
1. On github.com create a new **empty** public repository named `turingserve` (no README, no .gitignore).
2. In WSL:
```
sudo apt install -y gh
gh auth login                                     # GitHub.com → HTTPS → log in with a web browser
mkdir -p ~/code && cd ~/code
gh repo clone <github-user>/turingserve
cd turingserve
git config user.name "<your name>"
git config user.email "<your email>"
mkdir -p docs
cp /mnt/d/TuringServe/ARCHITECTURE.md docs/ARCHITECTURE.md
cp "/mnt/d/TuringServe/TuringServe - Complete Build Guide.md" docs/BUILD_GUIDE.md
```
3. VS Code: install the **WSL** extension on Windows, then run `code .` in the repo folder (WSL terminal).
4. From now on, edit the copies in `docs/`. `D:\TuringServe` keeps the original plan.

**Why `~/code` and not `/mnt/d`:** access to Windows drives from WSL is many times slower for many
small files (git, pip, Python imports).

### 0.6 Project skeleton
Create every folder in ARCH §8, each with an empty `__init__.py` inside `turingserve/`. Create these files:

**`pyproject.toml`**
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "turingserve"
version = "0.1.0"
description = "LLM inference server for NVIDIA Turing GPUs"
requires-python = ">=3.10,<3.14"
license = {text = "Apache-2.0"}
dependencies = []        # everything is pinned in requirements.lock; torch is installed separately

[project.scripts]
turingserve = "turingserve.cli:main"

[tool.setuptools.packages.find]
include = ["turingserve*"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP"]

[tool.pytest.ini_options]
testpaths = ["tests"]
timeout = 600
asyncio_mode = "auto"
markers = [
  "gpu: needs a CUDA GPU",
  "model: needs a downloaded real model (TS_MODEL_05 / TS_MODEL_15)",
  "slow: takes more than 30 seconds",
  "kaggle: needs the T4 or the 1.5B model",
]
```

**`.gitignore`**
```
.venv/
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
.ruff_cache/
tests/golden/
tests/assets/
benchmarks/data/
benchmarks/results/tmp/
*.trace.json
```
**`LICENSE`**: the Apache-2.0 text from https://www.apache.org/licenses/LICENSE-2.0.txt.
**`docs/lab_notes.md`**: start from Appendix A.
**`tests/conftest.py`** (fill in step by step):
- auto-skip tests marked `gpu` when `torch.cuda.is_available()` is false;
- auto-skip tests marked `model` when the env var `TS_MODEL_05` or `TS_MODEL_15` (as needed) is missing;
- auto-skip tests marked `kaggle` unless the env var `TS_KAGGLE=1` is set (Kaggle notebooks set it);
- session fixtures `tiny_model_dir` and `tiny_model_g7_dir`, which run `scripts/make_tiny_model.py` once into `tests/assets/`.

### 0.7 Virtual environment and pinned installs
```
cd ~/code/turingserve
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install "transformers>=4.45,<5" safetensors numpy fastapi "uvicorn[standard]" pydantic \
    httpx openai prometheus-client psutil pytest pytest-asyncio pytest-timeout hypothesis ruff \
    datasets pandas matplotlib
uv pip install -e . --no-deps
uv pip freeze > requirements.lock
```
Write **`scripts/check_versions.py`**. It prints torch, triton, transformers and tokenizers versions,
`torch.version.cuda`, `torch.cuda.is_available()`, the device name and capability, and
`torch.cuda.mem_get_info()` in GiB. It checks that `from tokenizers.decoders import DecodeStream` works.
It **exits with code 1** if torch is not `2.6.0` or Triton is not `3.2.0`, unless the environment
variable `TS_FALLBACK_STACK=1` is set and the versions are `2.5.1` / `3.1.0`.
```
python scripts/check_versions.py
```
**Expected:** `torch 2.6.0+cu124`, `triton 3.2.0`, `NVIDIA GeForce GTX 1650`, capability `(7, 5)`, DecodeStream OK.

### 0.8 Tensor-core (MMA) probe — `scripts/probe_triton_mma.py`
This proves that Triton emits tensor-core instructions on Turing (ARCH §2.4).
1. Write a matmul kernel exactly as in the official Triton tutorial "03-matrix-multiplication": fp16
   inputs, fp32 accumulator, `tl.dot`, BLOCK_M = 64, BLOCK_N = 64, BLOCK_K = 32, GROUP_M = 8, num_warps = 4, num_stages = 2.
2. Random fp16 A and B on the GPU (seed 0), M = N = K = 2048.
3. `handle = kernel[grid](...)`; `ptx = handle.asm["ptx"]`. If `.asm` is missing, open the newest `*.ptx`
   file under `~/.triton/cache/` that contains the kernel name.
4. Print `mma.sync present: <True/False>`.
5. Check `torch.testing.assert_close(C, A @ B, atol=1e-2, rtol=1e-2)`.
6. Time the kernel and `torch.matmul` with `triton.testing.do_bench`. Print TFLOPS = 2·M·N·K / seconds / 1e12.

**Decision:**

| Result | Action |
|---|---|
| `mma.sync` present, correct | Keep the stack. Write the TFLOPS in the lab notes |
| `mma.sync` missing | Switch to the fallback: `uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124`, set `TS_FALLBACK_STACK=1`, re-run 0.7's check and this probe |
| Results incorrect | Stop. See §T. Do not continue until it is correct |

On the GTX 1650 the TFLOPS may be low (no tensor cores). That is expected; presence and correctness are what matter here.

### 0.9 Memory bandwidth — `scripts/measure_bandwidth.py`
Allocate two fp16 tensors of 256 MiB each on the GPU. Warm up with 3 copies `dst.copy_(src)`. Time 20
copies with CUDA events. Bandwidth = 2 × bytes × 20 / seconds / 1e9 GB/s (read + write).
Print it and record it as **B_meas (laptop)**. Every "% of speed of light" number later uses this value.

### 0.10 Downloads and environment variables
```
mkdir -p ~/models ~/datasets/sharegpt
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir ~/models/Qwen2.5-0.5B-Instruct
hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir ~/models/Qwen2.5-1.5B-Instruct
hf download anon8231489123/ShareGPT_Vicuna_unfiltered ShareGPT_V3_unfiltered_cleaned_split.json \
    --repo-type dataset --local-dir ~/datasets/sharegpt
```
If `hf` is not found, use `huggingface-cli download` with the same arguments.
**Do not** run `pip install -U huggingface_hub`: transformers 4.x needs a version below 1.0.

Add to the end of `~/.bashrc`, then run `source ~/.bashrc`:
```
export TS_MODEL_05=~/models/Qwen2.5-0.5B-Instruct
export TS_MODEL_15=~/models/Qwen2.5-1.5B-Instruct
export TS_SHAREGPT=~/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json
export TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib
```
**Check:** `ls $TS_MODEL_05` shows `config.json`, `generation_config.json`, `model.safetensors`,
`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`.
WikiText-2 downloads automatically the first time it is used (`datasets`, ID `Salesforce/wikitext`).

### 0.11 Perplexity script and the Day-1 gate

**`scripts/perplexity.py`** (the HF engine now; `--engine turingserve` is added in Phase 10):
- Args: `--model PATH --dtype float16|float32 --engine hf|turingserve --window 1024 --stride 512 --max-windows 0 --out FILE`.
- Text: `load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")`, joined with `"\n\n"`, tokenized once.
- Loop: `prev_end = 0`; for `begin` in `range(0, N, stride)`: `end = min(begin + window, N)`;
  `trg_len = end − prev_end`; input = `ids[begin:end]`; target = a copy with `target[:-trg_len] = −100`.
  Get logits for every position. Compute the NLL **manually**: logits `[:-1]` against target `[1:]`, ignore −100,
  in chunks of 256 positions (memory); add the summed NLL and the count of counted tokens.
  `prev_end = end`; stop when `end == N`.
- Print and save `ppl = exp(total_nll / total_count)`.

**`scripts/day1_gate.py`** — is the model safe in fp16?
- Args: `--model PATH --num-prompts 200 --out docs/day1_gate_<name>.json`.
- Prompts: the first 180 ShareGPT first human turns (in file order, skipping empty ones) whose chat-templated
  length is ≤ 1024 tokens, plus 20 long prompts made of consecutive 3,000-token slices of WikiText-2. Each
  prompt is wrapped as a chat user message with the chat template.
- Run **fp32 first, then fp16** (one model in memory at a time; delete the model and call
  `torch.cuda.empty_cache()` between). For each prompt: `generate(max_new_tokens=32, do_sample=False,
  repetition_penalty=1.0, temperature=None, top_p=None, top_k=None)`, one prompt at a time.
- Forward hooks on every `model.model.layers[i]`: record the max absolute value and `isfinite` of the
  layer output (the output may be a tensor or a tuple; take element 0 if it is a tuple).
- Gates:

| Gate | Check | Pass rule |
|---|---|---|
| G1 | Any inf/NaN in fp16 layer outputs or logits | Must be none |
| G2 | Max \|activation\| in fp16 | < 30,000 passes; 30,000–65,504 is a warning |
| G3 | WikiText-2 perplexity, fp16 vs fp32 (`perplexity.py`) | Relative difference ≤ 1% |
| G4 | Fraction of prompts whose first 32 greedy tokens are identical in fp16 and fp32 | ≥ 80% (information only) |

- **Verdict: pass if G1 and G3 pass.** Write `docs/day1_gate.md` (date, model, gate table, verdict).
- **If it fails:** switch to the SmolLM2 family (ARCH §2.6), run the gate for it, and adapt Phase 1 to the
  Llama architecture (no QKV bias; read the heads and rope theta from `config.json`).
```
python scripts/day1_gate.py --model $TS_MODEL_05 --out docs/day1_gate_0.5b.json
```

### 0.12 Tiny test models and CI
**`scripts/make_tiny_model.py`** (`--out DIR --variant g2|g7`):
- g2: `Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
  num_key_value_heads=2, vocab_size=151936, tie_word_embeddings=True, rope_theta=1e6,
  max_position_embeddings=4096, initializer_range=0.2, rms_norm_eps=1e-6)`.
- g7: the same but `hidden_size=224, intermediate_size=448, num_attention_heads=14` (head_dim 16, G = 7).
- `torch.manual_seed(0)`; `Qwen2ForCausalLM(config)`; `save_pretrained(DIR, safe_serialization=True)`.
- Copy `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` and `generation_config.json`
  from `$TS_MODEL_05`, or, if that is not set (CI), download only those five files with
  `huggingface_hub.hf_hub_download("Qwen/Qwen2.5-0.5B-Instruct", filename, local_dir=DIR)`.

**`tests/unit/test_smoke.py`**: `import turingserve`.

**`.github/workflows/ci.yml`**
```yaml
name: ci
on: [push, pull_request]
jobs:
  cpu-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install uv
      - run: uv venv .venv --python 3.12
      - run: |
          . .venv/bin/activate
          uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
          uv pip install "transformers>=4.45,<5" safetensors numpy fastapi "uvicorn[standard]" pydantic \
            httpx openai prometheus-client psutil pytest pytest-asyncio pytest-timeout hypothesis ruff
          uv pip install -e . --no-deps
      - run: . .venv/bin/activate && ruff check .
      - run: . .venv/bin/activate && pytest -m "not gpu and not model and not slow and not kaggle" -q
```
```
git add -A && git commit -m "phase 0: skeleton, scripts, CI" && git push -u origin main
```
**Check:** the Actions tab on GitHub shows a green run.

### Phase 0 — Done when
- [ ] Before GPU work, `ollama ps` lists nothing and `nvidia-smi` shows < 500 MiB used
- [ ] `D:\WSL\Ubuntu\ext4.vhdx` exists; `free -g` shows ~10 GB
- [ ] `check_versions.py` passes (torch 2.6.0+cu124, Triton 3.2.0, GTX 1650 (7, 5), DecodeStream OK)
- [ ] MMA probe: `mma.sync` present and the result is correct
- [ ] B_meas (laptop) recorded
- [ ] Models and ShareGPT downloaded; environment variables set
- [ ] Day-1 gate for the 0.5B passed; `docs/day1_gate.md` written
- [ ] CI is green on GitHub
- [ ] `docs/lab_notes.md` has the Phase 0 entries

**Pitfalls:** installing torch without `--index-url` (you get a different build); running `pip install -U`
on anything torch-related; working inside `/mnt/d`; forgetting to unload Ollama.

---

## Phase 0K — Kaggle setup, 1.5B gate, baseline probes (1 day)

**Goal:** the same pinned stack on a T4, the 1.5B approved for fp16, and the baseline engines known to work (or not).

### K.1 Account
kaggle.com → sign in → **Settings → Phone verification** (required for GPU and Internet). Note your
weekly GPU quota from the settings page.

### K.2 Setup notebook (`notebooks/kaggle/00_setup.ipynb`)
New Notebook → right panel **Session options**: Accelerator = **GPU T4 ×2** (or GPU T4); Internet = **On**.

**First cell of every TuringServe notebook:**
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          # must run before torch is imported
```
**Install cell:**
```
!nvidia-smi
!pip uninstall -y -q torchvision torchaudio
!pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
!pip install -q "transformers>=4.45,<5" safetensors fastapi "uvicorn[standard]" httpx openai \
    prometheus-client psutil hypothesis pytest pytest-asyncio pytest-timeout datasets pandas matplotlib
```
Then **Run → Restart session**, run the first cell again, and continue:
```
!git clone https://github.com/<github-user>/turingserve.git /kaggle/working/turingserve
%cd /kaggle/working/turingserve
!pip install -q -e . --no-deps
!python scripts/check_versions.py
```
**Expected:** torch 2.6.0+cu124, Triton 3.2.0, Tesla T4, (7, 5).

### K.3 Models and data
```
!hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir /tmp/models/Qwen2.5-1.5B-Instruct
!hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /tmp/models/Qwen2.5-0.5B-Instruct
!hf download anon8231489123/ShareGPT_Vicuna_unfiltered ShareGPT_V3_unfiltered_cleaned_split.json --repo-type dataset --local-dir /tmp/datasets/sharegpt
```
```python
os.environ["TS_MODEL_15"] = "/tmp/models/Qwen2.5-1.5B-Instruct"
os.environ["TS_MODEL_05"] = "/tmp/models/Qwen2.5-0.5B-Instruct"
os.environ["TS_SHAREGPT"] = "/tmp/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
os.environ["TS_KAGGLE"] = "1"
```
(Models go to `/tmp`, so they are not saved into the notebook output.)

### K.4 Probes and gate on the T4
```
!python scripts/probe_triton_mma.py                  # expect mma.sync True and far more TFLOPS than the laptop
!python scripts/measure_bandwidth.py                 # record B_meas (T4)
!python scripts/day1_gate.py --model $TS_MODEL_15 --out /kaggle/working/day1_gate_1.5b.json
```
Download the outputs (right panel → Output) and commit them into `docs/`. Download the notebook
(File → Download notebook) into `notebooks/kaggle/`.

### K.5 Baseline probes (each in its OWN new notebook, because each installs its own torch)

**`20_vllm_latest.ipynb`**
```
!pip install -q vllm
!python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"
!hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir /tmp/models/Qwen2.5-1.5B-Instruct
```
```python
import subprocess, time, requests
log = open("/kaggle/working/vllm.log", "w")
p = subprocess.Popen(["vllm", "serve", "/tmp/models/Qwen2.5-1.5B-Instruct", "--served-model-name", "qwen",
                      "--dtype", "half", "--max-model-len", "4096", "--gpu-memory-utilization", "0.85",
                      "--port", "8001"], stdout=log, stderr=subprocess.STDOUT)
for _ in range(180):
    try:
        if requests.get("http://localhost:8001/health", timeout=2).status_code == 200:
            break
    except Exception:
        pass
    time.sleep(5)
print(requests.post("http://localhost:8001/v1/completions",
                    json={"model": "qwen", "prompt": "Hello", "max_tokens": 16}).json())
```
```
!grep -i -E "backend|attention" /kaggle/working/vllm.log | head -20
```
Record: vLLM version, torch version, the attention backend it chose, works yes/no.

**`22_llamacpp.ipynb`**
```
!git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
!cmake -S /tmp/llama.cpp -B /tmp/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
!cmake --build /tmp/llama.cpp/build --config Release -j 4 --target llama-server
!hf download bartowski/Qwen2.5-1.5B-Instruct-GGUF --include "*f16.gguf" --local-dir /tmp/gguf
!ls -la /tmp/gguf
```
- If cmake cannot find CUDA, add `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`. The build takes 10–20 minutes.
- If no f16 file was downloaded, convert from the HF model:
  `pip install -q gguf sentencepiece` and `python /tmp/llama.cpp/convert_hf_to_gguf.py <HF dir> --outtype f16 --outfile /tmp/gguf/qwen2.5-1.5b-f16.gguf`.
- Start it in the background (as in the vLLM cell):
  `/tmp/llama.cpp/build/bin/llama-server -m /tmp/gguf/<file>.gguf -ngl 99 -c 65536 -np 16 --port 8002 -fa on`
  (if `-fa on` is rejected, use `-fa`). Check `/health` and one `/v1/completions` request.

### K.6 Record `docs/baselines.md`
| Engine | Version | Install command | Launch command | Attention backend | Works? | Notes |

### Kaggle GPU-hour plan (about 15 h in total)
| Session | Phase | Hours |
|---|---|---|
| Setup, probes, 1.5B gate | 0K | 1.5 |
| Baseline probes | 0K | 1.5 |
| First benchmark | 8 | 2 |
| INT8 quality (perplexity, GSM8K ×2) | 10 | 2 |
| Speculation measurements | 11 | 1.5 |
| Autopilot experiment | 12 | 1 |
| Full matrix, baselines, ablations | 14 | 5 |

**Done when:** `check_versions` passes on the T4; MMA probe passes; B_meas (T4) recorded; 1.5B gate
passed; `docs/baselines.md` filled in; notebooks committed.

---

## Phase 1 — Qwen2 forward pass (week 1)

**Goal:** your own Qwen2 model gives the same logits as Hugging Face: tiny model on the CPU (fp32), and the 0.5B on the GPU (fp32 and fp16).
**Read first:** ARCH §2.6, §6.7, §9. Read HF `modeling_qwen2.py` once, slowly, then close it.

**Files**
- `turingserve/model/qwen2_config.py`, `turingserve/model/loader.py`, `turingserve/model/layers.py`, `turingserve/model/qwen2.py`
- `turingserve/ops/__init__.py`, `turingserve/ops/reference.py`
- `scripts/make_golden.py`
- `tests/unit/test_qwen2_config.py`, `tests/correctness/test_loader.py`, `tests/correctness/test_logits_vs_hf.py`, `tests/correctness/test_layerwise_vs_hf.py`

**Steps**
1. **`Qwen2Config`** (frozen dataclass): `hidden_size, intermediate_size, num_layers, num_heads,
   num_kv_heads, head_dim (= hidden / heads), group_size, vocab_size, rms_norm_eps, rope_theta,
   max_position_embeddings, tie_word_embeddings, eos_token_ids: tuple[int, ...], bos_token_id`.
   `from_dir(path)` reads `config.json` and `generation_config.json`. `eos_token_id` may be an int or a
   list; store a tuple; if `generation_config.json` is missing, use the config's `eos_token_id`.
   Reject the unsupported cases listed in ARCH §6.7.
2. **`ops/reference.py`** (pure PyTorch, works on CPU and GPU, HF numerics from ARCH §6.7):
   `rmsnorm(x, w, eps)`, `rmsnorm_residual(x, residual, w, eps) -> (out, new_residual)`, `silu_mul(gu)`,
   `rope_(qkv, positions, cos, sin, H, KVH, D)`, `linear(x, w, b=None)` (= `F.linear`), and, for this phase
   only, `dense_causal_attention(q[T,H,D], k[T,KVH,D], v[T,KVH,D], scale) -> [T, H·D]` (repeat KV heads with
   `repeat_interleave(G, dim=1)`, SDPA with `is_causal=True`, which is valid because q_len == kv_len here).
   `ops/__init__.py`: `get_ops("reference" | "triton")` returns a namespace with these functions
   (only "reference" exists for now).
3. **`RotaryCache`** (`layers.py`): `inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.int64).float() / D))`;
   `freqs = torch.outer(torch.arange(max_len).float(), inv_freq)`; store `cos` and `sin` in fp32
   `[max_len, D/2]`. The reference RoPE builds `cat(cos[pos], cos[pos])`, casts it to the model dtype and
   applies the HF formula.
4. **Loader** (`load_weights(path, cfg, device, dtype) -> Weights`): open every `*.safetensors` file
   (follow `model.safetensors.index.json` if present) with `safe_open(..., framework="pt", device="cpu")`.
   Map names exactly as ARCH §6.7; fuse on the CPU; `.to(dtype)`; check `torch.isfinite(...).all()`; move
   to the device. Track used names and **raise if any tensor is unused** (except `lm_head.weight` when tied).
5. **`Qwen2Model(cfg, weights, ops)`**: `forward_dense(input_ids[T], positions[T], return_hidden=False) -> hidden[T, hidden]`
   follows the exact op order of ARCH §6.7, but uses `dense_causal_attention` instead of kv_write + paged
   attention. `compute_logits(hidden, sample_indices) -> [S, V] fp32`.
   With `return_hidden=True`, also return a list matching HF's `hidden_states`:
   - entry 0 = the embedding output;
   - entry i (1 ≤ i ≤ L−1) = the **full** hidden state after layer i−1, which in our fused structure is
     **`h + residual`**, computed explicitly in this debug mode (we normally defer that add into the next norm);
   - entry L = the final-norm output.
6. **`scripts/make_golden.py`** (`--model PATH --out tests/golden/<name>.pt --num-prompts 20`):
   - Prompts: 10 ShareGPT first turns (chat-templated, ≤ 512 tokens), 5 WikiText slices of 256 tokens, and
     5 special prompts: Hindi text, emoji text, a code snippet, a list of numbers, the single word "Hello".
   - With HF in fp32: greedy-generate 64 tokens (`do_sample=False, repetition_penalty=1.0`), then run one
     forward over prompt + output with `output_hidden_states=True`.
   - Save: the prompt ids, the greedy output ids, teacher-forced top-20 logits (values and indices, fp32)
     at every position, full logits at 8 evenly spaced positions of the first 3 prompts, and the hidden
     states of prompts 0 and 1 (fp16).
   - Then repeat the forward in fp16 and save the **noise floor**: the max absolute difference between
     HF fp16 and HF fp32 logits, over the same comparison set (top-20 indices of the fp32 reference plus the 8 full positions).
   - Load one model at a time (fp32, then delete, then fp16).
7. Run: `python scripts/make_golden.py --model $TS_MODEL_05 --out tests/golden/qwen05.pt`.

**Tests**
| Test | Model / device | Pass condition |
|---|---|---|
| `test_qwen2_config` | tiny + 0.5B (model marker) | Values equal ARCH §2.6 |
| `test_loader` | tiny + 0.5B | Every tensor used once; fused shapes `[(H+2KVH)·D, hidden]` and `[2I, hidden]`; dtype; finite |
| `test_logits_vs_hf` (tiny g2 and g7, CPU, fp32) | HF `Qwen2ForCausalLM` from the same directory, random 37-token input | Max abs diff over all positions ≤ 1e-4 |
| `test_logits_vs_hf` (0.5B, GPU, fp32) | Golden file | ≤ 2e-3 |
| `test_logits_vs_hf` (0.5B, GPU, fp16) | Golden file | ≤ 1.5 × noise floor + 0.05 |
| `test_layerwise_vs_hf` | tiny (CPU, fp32) + 0.5B (golden) | Each entry: fp32 ≤ 1e-4 (tiny); prints the first diverging layer on failure |

**Done when:** every test above passes; the golden file for the 0.5B exists.

**Pitfalls:** rotate-half layout; forgetting the QKV bias; the RMSNorm cast order; tied embeddings (no
`lm_head.weight` in the file); cos/sin computed in fp32; GQA mapping `h // G`; comparing HF hidden states
with our *deferred* residual (use `h + residual`); HF's `repetition_penalty` 1.1 default.

---

## Phase 2 — Contiguous KV cache and greedy generation (week 2) → Milestone M1

**Goal:** fast generation with a KV cache, identical to HF greedy output.
**Read first:** ARCH §2.7, §6.7.

**Files:** `turingserve/runner/contiguous_cache.py` (kept as a test reference later), `turingserve/simple_generate.py`,
`tests/correctness/test_cache_consistency.py`, `tests/correctness/test_generate_vs_hf.py`, `benchmarks/micro/decode_speed.py`.

**Steps**
1. `ContiguousCache(cfg, max_len, device, dtype)`: per layer `k, v [max_len, KVH, D]`; `write(layer, start, k, v)`; `view(layer, length)`.
2. `ops/reference.py`: add `attention_with_mask(q[Tq,H,D], k[Tk,KVH,D], v, scale)` with the **explicit
   bottom-right causal mask** `allowed[i, j] = j ≤ (Tk − Tq) + i` (ARCH §6.8). Never use `is_causal=True` when Tq ≠ Tk.
3. `Qwen2Model.forward_contiguous(input_ids, positions, cache)`: write K/V at their positions, then attend to `cache.view(...)`.
4. `simple_generate.generate_greedy(model, prompt_ids, max_new_tokens, eos_ids) -> list[int]`: prefill the
   whole prompt → argmax of the last position → loop one token at a time with position = current length;
   stop at EOS or `max_new_tokens`.
5. `benchmarks/micro/decode_speed.py`: 0.5B fp16, prompt of 128 tokens, generate 256; tokens/s excluding the
   first token; speed of light = B_meas / weight bytes; print the % reached.

**Tests**
| Test | Pass condition |
|---|---|
| `test_cache_consistency` (tiny, CPU, fp32) | Last-position logits: full dense forward vs (prefill half + decode the rest one by one), ≤ 1e-4 |
| `test_generate_vs_hf` (tiny, CPU, fp32) | 32 greedy tokens **identical** to HF `generate(do_sample=False, repetition_penalty=1.0, max_new_tokens=32, eos_token_id=list(eos))` |
| `test_generate_vs_hf` (0.5B, GPU, fp32) | 64 tokens identical to golden |
| `test_generate_vs_hf` (0.5B, GPU, fp16) | Identical, or the first difference is a near tie (ARCH §9: top-2 gap < 0.1 in our teacher-forced logits at that position). Log every near tie |

**Done when:** all pass; decode tokens/s and % of speed of light recorded.
`git tag m1 && git push --tags`. Optional: a short post ("I wrote Qwen2 from scratch; it matches HF token for token").

**Pitfalls:** decode position = current length; mask alignment; wrap inference in `torch.inference_mode()`.

---

## Phase 3 — Paged KV cache and the flat batch (week 3)

**Goal:** many sequences in one flat token batch with paged memory, with the same results as Phase 2.
**Read first:** ARCH §3 (principles), §5 (`AttentionMetadata`, `ModelInput`), §6.5, §6.8 (reference).
PagedAttention paper, sections 3–4.

**Files:** `turingserve/kv/block_allocator.py`, `turingserve/kv/kv_manager.py`,
`turingserve/runner/attention_metadata.py`, `turingserve/runner/input_builder.py` (simple Python version),
`turingserve/ops/reference.py` (+ `kv_write`, `paged_attention`), `turingserve/model/qwen2.py` (final
`forward`), `turingserve/engine/static_batch.py` (temporary), `tests/unit/test_block_allocator.py`,
`tests/unit/test_kv_manager.py`, `tests/correctness/test_paged_vs_contiguous.py`, `tests/correctness/test_batch_invariance.py`.

**Steps**
1. `BlockAllocator(num_blocks)` exactly as ARCH §6.5. Block 0 is never handed out. Create the `evictable`
   structure now; it stays empty until Phase 9.
2. `KVCacheManager` exactly as ARCH §6.5.
3. The KV tensor `[L, 2, num_blocks, KVH, 16, D]`, with a fixed `num_blocks` for now (laptop 0.5B: 2048 =
   384 MiB; tiny model on the CPU: 512). The memory profiler replaces this in Phase 8.
4. `ops.reference.kv_write(k[T,KVH,D], v, k_cache, v_cache, slot_mapping)`: `m = slot >= 0`;
   `b = slot[m] // 16`; `o = slot[m] % 16`; `k_cache[b, :, o, :] = k[m]`; same for v.
5. `ops.reference.paged_attention(q[T,H,D], k_cache, v_cache, meta, scale) -> [T, H·D]`: for each sequence,
   take its query rows from `query_start_loc`, gather its `ceil(seq_len/16)` blocks, reshape to
   `[seq_len, KVH, D]`, repeat KV heads G times, apply the bottom-right mask, run attention in fp32, write the rows.
6. `Qwen2Model.forward(input_ids, positions, meta, kv_cache)`: the exact ARCH §6.7 order (write-then-attend).
   `compute_logits` only at `sample_indices`.
7. `input_builder.build(items) -> ModelInput` (Python loops are fine for now): positions
   `num_computed + arange(n)`; slots from the block table; `query_start_loc`; `seq_lens = num_computed + n`;
   block tables padded with 0; `sample_indices = query_start_loc[i+1] − 1` for sampling rows.
8. `static_batch.generate_many(model, prompts, max_new_tokens)`: one flat prefill step for all prompts,
   then decode steps for all unfinished sequences together; greedy.

**Tests**
| Test | Pass condition |
|---|---|
| Allocator (Hypothesis, random allocate/free/incref/decref) | No id is both free and owned; refs ≥ 0; free + owned = num_blocks − 1; `NoFreeBlocks` when empty; block 0 never returned |
| KV manager, lengths 1, 15, 16, 17, 32, 33 | Number of blocks and slot values correct |
| Paged vs contiguous (tiny, CPU, fp32) | Logits ≤ 1e-5 |
| Batch invariance (tiny, CPU, fp32): 8 prompts of lengths 5, 16, 17, 100, 333, 512, 1000, 2047, alone vs together, teacher-forced | ≤ 1e-4 at every position |
| Batch invariance (0.5B, GPU, fp16) | ≤ 1.5 × noise floor + 0.05 |
| Shuffled free list (non-contiguous block ids) | Outputs unchanged |

**Done when:** all pass.
**Pitfalls:** off-by-one at block boundaries; `seq_lens` means the context *after* this step; forgetting to free blocks.

---

## Phase 4 — Scheduler and engine (weeks 3–4)

**Goal:** requests join and leave at every step; chunked prefill; preemption; an offline engine.
**Read first:** ARCH §5, §6.3, §6.4. Orca paper; Sarathi-Serve paper.

**Files:** `turingserve/engine/sampling_params.py` (only `max_tokens`, `ignore_eos` and greedy for now),
`turingserve/engine/sequence.py`, `turingserve/engine/request.py`, `turingserve/scheduler/scheduler.py`,
`turingserve/runner/model_runner.py`, `turingserve/engine/engine.py`, `tests/fakes.py`,
`tests/unit/test_scheduler.py`, `tests/correctness/test_preemption_equivalence.py`,
`tests/correctness/test_engine_vs_hf.py`, `benchmarks/offline_bench.py`.

**Steps**
1. `Sequence` with the fields and derived values of ARCH §5.
2. `Scheduler` **exactly** as the ARCH §6.4 pseudocode (prefix-cache hooks are no-ops until Phase 9).
3. `ModelRunner.execute(plan) -> (logits[S, V], sampling_seqs)`: input builder → model forward → `compute_logits`.
4. `LLMEngine` as in ARCH §6.3, with a temporary argmax sampler and finish rules (EOS, `max_tokens`,
   `max_model_len`). Output token ids only (no text yet). Include `add_request`, `abort`, `has_unfinished`, `step`, `generate`.
5. `tests/fakes.py: FakeModelRunner` with the same interface: the logits row is one-hot on
   `(sum(last 3 token ids) + position) % 1000`, where position is the index of the row's last input token.
6. `benchmarks/offline_bench.py`: `--engine turingserve|hf-seq|hf-static --model PATH --num-prompts 200`.
   ShareGPT prompts, **fixed output of 128 tokens** for every engine (HF: `min_new_tokens=128,
   max_new_tokens=128`; TuringServe: `max_tokens=128, ignore_eos=True`). HF static: batches of 16, `padding_side="left"`, pad id 151643. Prints total time and output tokens/s; saves JSON.

**Tests**
| Test | Pass condition |
|---|---|
| Scheduler property test (FakeModelRunner, Hypothesis): random arrivals, prompt 1–300, max_tokens 1–50, budget ∈ {16, 64, 256}, num_blocks 20–200, max_num_seqs ∈ {1, 4, 32} | Every step: token count ≤ budget; decodes first; block ownership matches `num_computed`; no admission in a step with a preemption. At the end: all finished, all blocks free, outputs **identical** to each request run alone with plenty of memory |
| Chunked prefill: 1000-token prompt, budget 64 | Prefill takes ceil(1000/64) = 16 steps; exactly one sample, at the end |
| Preemption equivalence (FakeModelRunner exact; tiny model within tolerance): num_blocks 30 vs 2000 | Same outputs |
| Abort mid-generation | Blocks freed; `finish_reason == "abort"` |
| Engine vs golden (0.5B, GPU) | 16 prompts, near-tie rule |

**Done when:** all pass; offline numbers for TuringServe vs HF (laptop, 0.5B) recorded.
**Pitfalls:** decodes must come first in the plan; a re-admitted preempted sequence recomputes prompt +
outputs; sample only when the chunk reaches the end; never preempt in pass 2.

---

## Phase 5 — Sampler and output processor (week 4)

**Goal:** OpenAI-style sampling and correct streamed text.
**Read first:** ARCH §5 (`SamplingParams`), §6.11, §6.12.

**Files:** `turingserve/engine/sampling_params.py` (complete, with validation), `turingserve/sampling/sampler.py`,
`turingserve/sampling/penalties.py`, `turingserve/engine/detokenizer.py`, `turingserve/engine/stop.py`,
`turingserve/engine/output_processor.py`, `tests/unit/test_sampler.py`, `tests/unit/test_penalties.py`,
`tests/unit/test_detokenizer.py`, `tests/unit/test_stop.py`.

**Steps:** implement exactly ARCH §6.11 and §6.12, then replace the temporary argmax in the engine.
`SamplingParams` validation errors raise `ValueError` with the field name (the API turns them into 400).
Each sequence gets a `torch.Generator(device)` seeded with `params.seed` if given.

**Tests**
| Test | Pass condition |
|---|---|
| Greedy equivalences | temperature 0 == top_k 1 == argmax |
| Seeds | Same seed → identical tokens, both alone and in a batch with 7 other requests |
| Distribution: logits = log([0.5, 0.3, 0.15, 0.05]), temperature 1, 20,000 samples | Chi-square < 16.27 |
| top_k = 2 | Only tokens 0 and 1; frequencies 0.625 / 0.375 (chi-square, df 1, < 10.83) |
| top_p = 0.75 | Only tokens 0 and 1 |
| temperature = 0.5 | Frequencies ∝ p² (chi-square < 16.27) |
| Penalties | repetition 2.0: logit 4 → 2 and −4 → −8; frequency 0.5 with count 2 → −1.0; presence 0.3 → −0.3 |
| min_tokens | EOS never appears before `min_tokens`, even when its logit is highest |
| ignore_eos | Generation continues past EOS until `max_tokens` |
| Detokenizer: English; Hindi "नमस्ते दुनिया, यह एक परीक्षण है"; emoji "🙂👍🏽🚀"; Chinese "你好世界"; code with tabs and newlines | Concatenated deltas == `tokenizer.decode(all, skip_special_tokens=True)`; no delta contains "\ufffd" |
| Stop strings | Truncated before the stop; a stop split across two tokens is caught; no prefix of a stop string is ever emitted early; a stop at the first token works; with several stops, the earliest match wins |
| Finish reasons | "stop" (EOS), "stop" (stop string), "length" (`max_tokens`), "length" (`max_model_len`) |

**Done when:** all pass; the engine returns text deltas.

---

## Phase 6 — OpenAI-compatible API server (weeks 4–5) → Milestone M2

**Goal:** a real server that the official `openai` SDK can use with only `base_url` changed.
**Read first:** ARCH §4, §6.1, §6.2, §6.17.

**Files:** `turingserve/config.py`, `turingserve/cli.py`, `turingserve/api/protocol.py`,
`turingserve/api/async_engine.py`, `turingserve/api/streaming.py`, `turingserve/api/server.py`,
`tests/e2e/conftest.py`, `tests/e2e/test_api_basic.py`, `test_api_streaming.py`, `test_api_errors.py`,
`test_api_cancel.py`, `test_api_overload.py`, `test_api_concurrency.py`.

**Steps**
1. `EngineConfig` with every flag in ARCH §4, plus `--num-blocks N` (testing only: forces the number of KV
   blocks and skips the memory profiler) and validation. `parse_args(argv) -> EngineConfig`. On
   `--device cpu`, CUDA graphs are off and `num_blocks` defaults to 2048.
2. `protocol.py`: pydantic models for both requests (with `extra="allow"`, then an explicit check that
   rejects the fields listed in ARCH §6.1), responses, stream chunks and errors. A function converts a
   request into `(prompt_token_ids, SamplingParams)` or raises a 400.
3. `async_engine.py` exactly as ARCH §6.2.
4. `server.py`: `create_app(cfg) -> FastAPI`, with a lifespan that starts/stops the `AsyncEngine`. Routes as in
   ARCH §6.1. Chat: `tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)`
   (join text parts). Streaming via `StreamingResponse` with the chunk order of ARCH §6.1 and the
   cancellation `try/finally`. Return 429 when the waiting count ≥ `max_waiting_requests`. Add `/health`,
   `/v1/models`, and `/debug/state` (when enabled). Add `usage.prompt_tokens_details.cached_tokens`
   (0 until Phase 9).
5. `cli.py`: `turingserve serve ...` → `uvicorn.run(create_app(cfg), host=..., port=..., workers=1, log_level=...)`.
6. `tests/e2e/conftest.py`: a fixture starts `turingserve serve --model <tiny> --device cpu --kernels reference
   --dtype float32 --port <free port> --debug-endpoints` as a subprocess, waits for `/health` (60 s timeout),
   and kills it afterwards.
7. README "Quick start" (install; `turingserve serve --model ~/models/Qwen2.5-0.5B-Instruct --kernels reference`; this curl):
```
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Say hi in five words"}],"max_tokens":32}'
```

**Tests**
| Test | Pass condition |
|---|---|
| Basic (openai SDK) | Chat and completions (string and token-list prompts) return text; `usage.prompt_tokens` equals the templated length |
| Streaming | Chunk order exactly as ARCH §6.1; usage chunk only with `include_usage`; concatenated stream == non-stream text (greedy) |
| Errors | `n=2`, `logprobs`, `tools`, 17 stop strings, temperature 3, empty prompt, too long → 400; wrong model → 404; error JSON shape correct |
| Cancel | Close a stream (`max_tokens` 500) after 3 chunks → within 2 s `/debug/state` shows 0 running and `num_free_blocks + num_cached_blocks == num_total_blocks − 1` |
| Overload | `--max-waiting-requests 2`, 20 concurrent requests → some 429 with `Retry-After`; all others complete |
| Concurrency | 100 parallel requests succeed; greedy outputs equal the sequential outputs |
| GPU variant (model marker) | Basic + streaming against the 0.5B with `--kernels reference --dtype float16` |

**Done when:** all pass; the README Quick start works from a fresh terminal. `git tag m2 && git push --tags`.
Record a 30-second demo of streaming output. **This is already a complete, showable project.**

**Pitfalls:** doing engine work inside the event loop (it must stay in the engine thread); forgetting to
remove sinks after finish; more than one uvicorn worker.

---

## Phase 7 — Triton kernels and the tuner (weeks 5–6)

**Goal:** fast kernels, each proven equal to its reference.
**Read first:** ARCH §2.3, §6.8, §6.9, §6.15 (part 1), §9 (tolerances, needle tests). Official Triton
tutorials 01 (vector add), 02 (fused softmax), 03 (matmul), 06 (fused attention); the FlashAttention
paper; the Flash-Decoding post.

**Files:** `turingserve/ops/triton_rmsnorm.py`, `triton_silu_mul.py`, `triton_rope.py`, `triton_kv_write.py`,
`triton_decode_attention.py` (v1, v2, v3 + reduce), `triton_prefill_attention.py`; `turingserve/ops/__init__.py`
(the triton namespace reads tuned configs); `turingserve/autopilot/tuner.py`; `tests/kernels/test_*.py`;
`benchmarks/micro/bench_*.py`.

**Order:** rmsnorm → silu_mul → rope → kv_write → decode v1 → decode v2 → prefill → decode v3 → tuner →
switch the engine to `--kernels triton` → end-to-end checks.

**For every kernel**
1. Write it with explicit `tl.constexpr` config parameters. **Do not use `@triton.autotune`** (ARCH §6.15).
2. Test against the reference with the shapes below and the tolerances in ARCH §9.
3. Microbenchmark with `triton.testing.do_bench`: report GB/s for memory-bound kernels (bytes moved /
   time) and TFLOPS for prefill. Compare with the reference and with PyTorch where possible.
4. Write the numbers in the lab notes.

**Test shapes**
| Kernel | Shapes |
|---|---|
| rmsnorm (± residual) | T ∈ {1, 7, 128, 2049} × N ∈ {64, 224, 896, 1536} |
| silu_mul | T ∈ {1, 33, 2048} × I ∈ {128, 4864, 8960} |
| rope | T ∈ {1, 17, 2048} × (H, KVH, D) ∈ {(4,2,16), (14,2,64), (12,2,128)}; random positions in [0, 4095] |
| kv_write | T ∈ {1, 17, 300}; random unique slots, some −1 → exact equality; untouched slots unchanged |
| decode (v1, v2, v3) | S ∈ {1, 3, 64}; seq_len per sequence drawn from {1, 2, 15, 16, 17, 100, 1000, 4096}; (H, KVH, D) ∈ {(4,2,16), (14,2,64), (12,2,128), (8,8,64)}; shuffled block ids; one padding row with seq_len 0 (expect zeros) |
| prefill | q_len ∈ {1, 15, 16, 17, 300, 2048} × prefix ∈ {0, 16, 37, 1000} × P ∈ {1, 3}; same head configs; also "prefill with q_len 1 == decode kernel" |
| needle tests | ARCH §9, for decode and prefill |

**Kernel notes:** fp32 accumulators; cast block ids to int64 before multiplying by strides; masks for every
non-multiple-of-16 length; `tl.dot` operands must both be fp16 and every dot dimension ≥ 16; decode v2 pads
the GQA group to BLOCK_H = 16; only the Turing-safe configs of ARCH §6.8 and §6.15; num_stages 1–2.

**Tuner** (`autopilot/tuner.py`), exactly as ARCH §6.15 part 1: candidate lists, buckets, `do_bench`, skip
configs that fail to compile, a JSON cache in `--tuning-dir`. Add `turingserve tune --model PATH` to run
only the tuner and print a table of winners.

**Switch the engine:** `--kernels triton` becomes the default on CUDA. Re-run all correctness tests
(golden comparisons for the 0.5B in fp16, E2E basic on the GPU) with Triton.

**Done when:** all kernel tests pass on the laptop **and** on the Kaggle T4 (`pytest tests/kernels -m gpu`
in a notebook cell); microbenchmark tables recorded for both GPUs; the engine passes the correctness tests with Triton.

**Pitfalls:** "out of resource: shared memory" (the tile is too big for 64 KiB; the tuner skips it); a first
call is slow (compilation, cached in `~/.triton/cache`); illegal memory access (missing mask, see §T).

---

## Phase 8 — Memory profiler, CUDA graphs, benchmark harness, first T4 numbers (weeks 6–7) → Milestone M3

**Goal:** remove CPU overhead, size memory properly, and publish the first honest T4 benchmark.
**Read first:** ARCH §6.10, all of §10.

**Files:** `turingserve/runner/memory_profiler.py`, `turingserve/runner/cuda_graphs.py`,
`turingserve/runner/input_builder.py` (vectorized, pinned), `turingserve/metrics/tracing.py` (step timers),
`benchmarks/datasets.py`, `benchmarks/serve_bench.py`, `benchmarks/baselines/hf_server.py`,
`benchmarks/make_report.py` (first version), `notebooks/kaggle/10_turingserve_bench.ipynb`,
`tests/correctness/test_cuda_graphs.py`, `tests/unit/test_input_builder.py`.

**Steps**
1. **Profile first** (laptop, 0.5B, 16 concurrent decodes): wrap 20 steps in `torch.profiler.profile(activities=[CPU, CUDA])`,
   export a Chrome trace, open it at ui.perfetto.dev, and record the GPU-busy percentage of a decode step.
2. **Vectorized input builder** with persistent pinned buffers (ARCH §6.10.1). Pinned buffers are reused
   only after the step's sampler synchronization, so they are never overwritten while a copy is still running.
   Unit test: identical output to the Phase 3 Python builder on 200 random plans.
3. **Memory profiler** exactly as ARCH §6.10.3; log every number at startup (baseline, weights, peak,
   non-torch, budget, blocks, max tokens).
4. **CUDA graphs** exactly as ARCH §6.10.2, for pure-decode steps only.
5. Profile again; record the GPU-busy percentage and the decode tokens/s gain.
6. **`benchmarks/datasets.py`**: ShareGPT, synthetic, prefill and (in Phase 9) RAG loaders, as in ARCH §10.2.
   Each returns `(prompt_text_with_template, max_tokens, ignore_eos)` items; seed 0.
7. **`benchmarks/serve_bench.py`**:
   - Args: `--base-url --model --tokenizer PATH --dataset sharegpt|synthetic|prefill|rag --num-prompts 500
     --mode concurrency|rate|ramp|profile --concurrency 1,4,16,32,64 --rate R --ramp 0.5,1,2,4,8,12,16
     --profile "120:0.5,120:4,120:0.5" --step-seconds 60 --ignore-eos --seed 0 --warmup 10 --repeat 3
     --label TEXT --out benchmarks/results --gpu-log`.
   - `httpx.AsyncClient(timeout=None)`. POST `/v1/completions` with `{"model", "prompt", "max_tokens",
     "temperature": 0, "stream": true, "stream_options": {"include_usage": true}, "ignore_eos"}`. Parse SSE
     lines; record `time.perf_counter()` for send, every chunk and the end. Token counts come from `usage`, or
     from re-tokenizing the output if usage is missing. Warn when `usage.prompt_tokens` differs from the
     local count by > 1%.
   - `--gpu-log` runs `nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv -l 1` alongside.
   - Compute the ARCH §10.4 metrics; write the ARCH §10.9 JSON; print a summary table.
8. **`benchmarks/baselines/hf_server.py`**: FastAPI, HF model in fp16 on the GPU; `/health` and streaming
   `/v1/completions`; one request at a time (`asyncio.Lock`); `generate(max_new_tokens=max_tokens,
   min_new_tokens=max_tokens if ignore_eos else 0, do_sample=False, repetition_penalty=1.0)` in a thread,
   with `TextIteratorStreamer`; OpenAI-format chunks and a usage chunk.
9. **`benchmarks/make_report.py`** (v1): read the results JSON files; write markdown tables per workload and
   engine (throughput, TTFT p50/p90/p99, TPOT p50/p90/p99, goodput) and PNG charts (throughput vs
   concurrency, TTFT p90 vs concurrency) into `benchmarks/REPORT.md` and `benchmarks/figures/`.
10. **Kaggle session** (`10_turingserve_bench.ipynb`):
    - On the laptop first: `git tag m3-rc && git push --tags` (the exact commit you will benchmark).
      In Kaggle: install as in K.2, `git checkout m3-rc`, download the 1.5B and ShareGPT.
    - Start TuringServe in the background: `turingserve serve --model $TS_MODEL_15 --served-model-name qwen --port 8000`.
      Wait for `/health`. Run serve_bench: ShareGPT with concurrency 1, 4, 16, 32, 64.
    - Restart with `--max-model-len 8192` and run the prefill dataset (2,048 / 4,096 / 8,000 tokens).
    - Stop it. Start `python -m benchmarks.baselines.hf_server --model $TS_MODEL_15 --port 8003` and run
      ShareGPT with concurrency 1, 4, 16 and 200 prompts (HF is slow; label the runs).
    - In the baseline notebooks (20, 22): clone the repo, `pip install httpx transformers`, start the
      engine as in K.5, and run the **same** serve_bench commands (and the prefill set with `--max-model-len 8192`).
    - Download every result JSON and commit them.
11. README: a first results section (chart, prefill table next to the vLLM issue numbers, honest notes).

**Tests**
| Test | Pass condition |
|---|---|
| Graphs vs eager: batch sizes 1, 3, 8, 31 (padding cases) with different seq_lens | Logits exactly equal (`torch.equal`) |
| End to end | Greedy outputs identical with and without graphs |
| Memory profiler on the laptop with Ollama loaded | Fails with the clear message that mentions Ollama |

**Done when:** tests pass; first T4 report with at least 3 engines; prefill numbers next to vLLM's. `git tag m3`.
**Pitfalls:** capturing before warm-up/tuning; any synchronization inside the captured region; padding
rows without slot −1; T4 throttling (warm up, repeat 3×, check the GPU log).

---

## Phase 9 — Prefix caching and the RAG workload (week 7)

**Goal:** reuse the KV of repeated prefixes and measure it on real RAG prompts.
**Read first:** ARCH §6.5, §6.6.

**Files:** `turingserve/kv/prefix_cache.py`, allocator eviction, scheduler hooks,
`tests/unit/test_prefix_cache.py`, `tests/correctness/test_prefix_cache_equivalence.py`,
the RAG loader in `benchmarks/datasets.py`, and **in the RAG repo**: `scripts/dump_llm_prompts.py`.

**Steps**
1. Implement the prefix cache exactly as ARCH §6.6, plus eviction in the allocator. Report
   `usage.prompt_tokens_details.cached_tokens` and the prefix-cache metrics.
2. **Capture the real RAG prompts** (Windows, in `D:\Rag`). Create `scripts\dump_llm_prompts.py`:
```python
"""Record every prompt the RAG pipeline would send to the LLM, without calling an LLM."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.generator as generator          # generator.py did `from src.llm import generate`
from src import eval as rag_eval
from src.llm import LLMResult

records = []

def recording_generate(prompt, system=None, schema=None):
    records.append({"system": system, "prompt": prompt, "uses_schema": schema is not None})
    return LLMResult(text="This is a placeholder answer. [1]", input_tokens=0, output_tokens=0, latency_s=0.0)

generator.generate = recording_generate     # patch the name generator.py actually calls

for path in ["eval/questions.json", "eval/heldout_questions.json"]:
    rag_eval.run_config("router", rag_eval.load_questions(Path(path)))

unique = list({json.dumps(r, sort_keys=True): r for r in records}.values())
Path("eval/llm_prompts.jsonl").write_text("\n".join(json.dumps(r) for r in unique), encoding="utf-8")
print(f"wrote {len(unique)} prompts")
```
   Run `python scripts\dump_llm_prompts.py` from `D:\Rag`. If `run_config` or `load_questions` has changed,
   read `src/eval.py: main()` and call what it calls. Copy `eval/llm_prompts.jsonl` to
   `~/code/turingserve/benchmarks/data/rag_prompts.jsonl`. **That folder is git-ignored on purpose:** the
   prompts contain text from arXiv papers, so do not publish them. Document how to regenerate the file.
3. RAG loader: JSONL → messages (system if present + user) → chat template → prompt text; greedy;
   `max_tokens` 512; natural stopping (no `ignore_eos`).
4. Benchmark (laptop 0.5B is fine for relative numbers; T4 for the report): RAG with concurrency 1 and 4,
   prefix caching on vs off → TTFT p50/p90 and hit rate. ShareGPT on vs off → no regression.

**Tests**
| Test | Pass condition |
|---|---|
| Same prompt of length L twice | Second request's cached tokens = 16 × (floor(L/16) − (1 if L % 16 == 0 else 0)) |
| Two prompts sharing exactly the first 32 tokens | 2 block hits |
| Equivalence (FakeModelRunner exact; tiny model within tolerance) | Outputs with cache == without |
| Eviction property test (small num_blocks, random workload) | A block with ref > 0 is never evictable; evicted hashes are removed from the map; running sequences are unaffected |
| Preemption + cache | A re-admitted sequence hits its own earlier blocks |

**Done when:** tests pass; the RAG TTFT improvement and hit rate are recorded.

---

## Phase 10 — INT8 weights and quality evaluation (week 8)

**Goal:** W8A16 with measured speed, memory and quality.
**Read first:** ARCH §6.14, §10.6.

**Files:** `turingserve/quant/int8.py`, `turingserve/ops/triton_w8a16.py` (+ dequant path), linear dispatch in
`ops/__init__.py`, loader support for `--quantization int8`, `scripts/perplexity.py --engine turingserve`,
`tests/kernels/test_w8a16.py`, `tests/correctness/test_int8_model.py`, `notebooks/kaggle/30_quality.ipynb`.

**Steps**
1. `quantize(W) -> (Wq, scale)` as ARCH §6.14. Unit test: |dequant − W| ≤ scale/2 per element.
2. Kernel + dequant path. Test shapes: M ∈ {1, 7, 16, 64, 300, 2048} × the model's (N, K):
   0.5B qkv (1152, 896), o (896, 896), gate-up (9728, 896), down (896, 4864); 1.5B qkv (2048, 1536),
   o (1536, 1536), gate-up (17920, 1536), down (1536, 8960). Relative error < 1e-3 against dequant + matmul in fp32.
3. Add the w8a16 buckets and `M_cross` to the tuner.
4. `test_int8_model`: teacher-forced top-1 agreement with the fp16 model ≥ 95% on the golden prompts
   (below that, investigate before going on).
5. Perplexity: laptop 0.5B (HF fp16, TuringServe fp16, TuringServe int8); Kaggle 1.5B (the same three).
   TuringServe fp16 must be within 0.1% of HF fp16.
6. **GSM8K** (Kaggle notebook 30): after the K.2 install, run `pip install -q "lm-eval[api]"`, then
   `python scripts/check_versions.py` (if torch changed, reinstall the pin and restart). Start TuringServe
   (fp16, `--served-model-name qwen`) and run:
```
lm_eval --model local-chat-completions \
  --model_args model=qwen,base_url=http://localhost:8000/v1/chat/completions,num_concurrent=16,max_retries=3,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 5 --apply_chat_template --fewshot_as_multiturn \
  --gen_kwargs temperature=0 \
  --output_path /kaggle/working/results/gsm8k_fp16 --log_samples
```
   Restart with `--quantization int8` and repeat (`gsm8k_int8`). If a flag is rejected, check
   `lm_eval --help` for its current name and note the change. Laptop alternative: a separate venv
   (`uv venv --python 3.12 ~/venvs/lmeval`, then `uv pip install "lm-eval[api]"`).
7. Speed: batch-1 decode tokens/s for fp16, int8 and int8 + `--int8-lm-head`; memory used; maximum
   concurrent users; the effect on prefill.

**Done when:** a README table with memory, batch-1 tokens/s, % of speed of light, perplexity and GSM8K
(fp16 vs int8, both GSM8K filters, sample size 1,319).

---

## Phase 11 — Speculative decoding (weeks 8–9) → Milestone M4

**Goal:** more tokens per step at small batch, with the output distribution unchanged.
**Read first:** ARCH §6.13. Leviathan et al.; Chen et al.; prompt lookup decoding.

**Files:** `turingserve/sampling/rejection.py`, `turingserve/spec/ngram.py`, `turingserve/spec/draft.py`,
`turingserve/spec/spec_decoder.py`, `KVCacheManager.trim`, `tests/unit/test_rejection.py`,
`tests/unit/test_ngram.py`, `tests/correctness/test_spec_equivalence.py`.

**Steps**
1. `rejection.py`: `accept_greedy(drafts, target_logits) -> (m, next_token)` and
   `accept_sampling(drafts, p, q_or_None, generator) -> (m, next_token)`, exactly as ARCH §6.13.
2. `ngram.py`: `propose(token_ids: np.ndarray, k, n_max, n_min) -> list[int]`.
3. `spec_decoder.py`: `should_run(plan)` and `step(plan)` for n-gram (verify only) first, then the draft
   model (`draft.py`: second model, its own KV tensor, the target's block tables, catch-up). The memory
   profiler counts the draft weights and the draft KV.
4. Measure (Kaggle T4, 1.5B target, 0.5B draft): batch 1, 4, 8, 16 × workloads {ShareGPT, RAG} × methods
   {none, ngram k = 2/4/6, draft k = 2/4/6} → tokens/s per user, TPOT p50 and acceptance rate.

**Tests**
| Test | Pass condition |
|---|---|
| `accept_greedy` on constructed logits | Correct m, correction and bonus |
| Distribution: p = [0.4, 0.3, 0.2, 0.1], q = [0.1, 0.2, 0.3, 0.4], 50,000 trials of "d ~ q, accept/reject, take the first output" | Output frequencies ≈ p (chi-square < 16.27). Repeat with an n-gram point mass on token 2 |
| `propose` | No match → []; the latest match is used; the proposal is cut at the end of the sequence |
| Greedy equivalence (FakeModelRunner, Hypothesis, 50 prompts) | Speculative output == normal output, exactly (n-gram and a fake draft) |
| Real 0.5B, n-gram | Near-tie rule against normal output |
| Kaggle: 1.5B + 0.5B draft | Near-tie rule |
| EOS/stop inside accepted drafts | Correct truncation; never more than `max_tokens` |

**Done when:** tests pass; the speedup table is in the README. `git tag m4`.

---

## Phase 12 — Autopilot (weeks 9–10)

**Goal:** the engine tunes itself at startup and adapts while running.
**Read first:** ARCH §6.15.

**Files:** `turingserve/autopilot/startup.py`, `turingserve/autopilot/controller.py`, engine integration,
metrics, `tests/unit/test_controller.py`, `tests/unit/test_tuner_cache.py`.

**Steps**
1. `startup.py`: tuner → memory profile → graphs → chunk sweep, with the cache key of ARCH §6.15. A second
   start with the same key logs "tuning cache hit" and takes seconds. `--retune` forces a new measurement.
2. `controller.py`: the pure `decide(state, signals)` function, plus a window aggregator (TTFT samples by
   first-token time in the window; TPOT samples from token gaps in the window). The engine applies decisions
   to `max_num_batched_tokens`, `max_num_seqs`, spec on/off and k, logs them as JSON lines and exports metrics.
3. Experiment (Kaggle T4): `serve_bench --mode profile --profile "120:0.5,120:4,120:0.5"` (ShareGPT, Poisson)
   for three configurations: (a) defaults with `--autopilot off`, (b) the best fixed configuration from the
   sweep with `--autopilot off`, (c) `--autopilot full --spec-method ngram`.
   Chart: settings over time (from the decision log) and TTFT/TPOT p90 over time. Table: goodput, TTFT p90, TPOT p90.

**Tests** (`test_controller.py`, pure function, synthetic signal traces)
| Case | Expected |
|---|---|
| Stable signals inside targets | No decisions |
| Violation for 2 windows | No decision |
| Violation for 3 windows | Exactly one decision |
| Right after a change | Cooldown respected (10 s) |
| Values at the bounds | Never beyond them |
| TTFT high but TPOT at 0.8 × target | R2 does **not** fire (hysteresis band) |
| Decode batch > spec_max_batch | R3 immediately |
| ā low / high | R4 / R5 as specified, including "off for 60 s, then re-probe" |
| Preemptions / idle memory | R6 / R7 as specified |
| CUDA OOM signal | R0 halves the chunk size immediately, never below 256 |

`test_tuner_cache.py`: the key includes every listed field; corrupt or missing files fall back to re-tuning.

**Done when:** the chart and table are in the README, and every decision can be explained from the log.

---

## Phase 13 — Observability (week 10)

**Goal:** a live dashboard, structured logs and profiler traces.
**Read first:** ARCH §6.16.

**Files:** `turingserve/metrics/prometheus.py`, `turingserve/utils/logging.py`, `turingserve/metrics/tracing.py`,
the `/debug/profile` endpoint, `deploy/observability/prometheus.yml`, `deploy/observability/run.sh`,
`deploy/observability/grafana/provisioning/datasources/prometheus.yaml`,
`deploy/observability/grafana/provisioning/dashboards/turingserve.yaml`, `deploy/observability/dashboard.json`.

**Steps**
1. Every metric in ARCH §6.16, mounted at `/metrics` (`prometheus_client.make_asgi_app()`). E2E test: after
   one request, `/metrics` contains every metric name.
2. JSON-lines logging and `--log-requests`.
3. **Prometheus and Grafana as binaries inside WSL** (`deploy/observability/run.sh`):
   - Prometheus: download the current LTS `linux-amd64` tarball from https://prometheus.io/download/ into
     `~/tools/prometheus` (write the version into `run.sh`), then run
     `./prometheus --config.file=<repo>/deploy/observability/prometheus.yml --storage.tsdb.path=$HOME/tools/prometheus-data --web.listen-address=:9090`.
   - `prometheus.yml`:
```yaml
global:
  scrape_interval: 2s
scrape_configs:
  - job_name: turingserve
    static_configs:
      - targets: ["localhost:8000"]
```
   - Grafana OSS: download the Linux standalone tarball from https://grafana.com/grafana/download?edition=oss
     into `~/tools/grafana`, then run
     `GF_PATHS_PROVISIONING=<repo>/deploy/observability/grafana/provisioning ./bin/grafana server --homepath $HOME/tools/grafana`
     and open http://localhost:3000 (first login admin/admin).
   - `datasources/prometheus.yaml`:
```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://localhost:9090
    isDefault: true
```
   - `dashboards/turingserve.yaml`:
```yaml
apiVersion: 1
providers:
  - name: turingserve
    type: file
    options:
      path: <absolute repo path>/deploy/observability
```
   - Build the dashboard in the Grafana UI with these panels, then export its JSON into `dashboard.json`:

| Panel | PromQL |
|---|---|
| TTFT p90 | `histogram_quantile(0.9, sum(rate(turingserve_request_ttft_seconds_bucket[1m])) by (le))` |
| TPOT p90 | `histogram_quantile(0.9, sum(rate(turingserve_request_tpot_seconds_bucket[1m])) by (le))` |
| Output tokens/s | `rate(turingserve_generation_tokens_total[30s])` |
| Running / waiting | `turingserve_num_running`, `turingserve_num_waiting` |
| KV usage | `turingserve_kv_usage_ratio` |
| Prefix hit rate | `rate(turingserve_prefix_cache_hit_tokens_total[1m]) / rate(turingserve_prefix_cache_queried_tokens_total[1m])` |
| Preemptions per minute | `60 * rate(turingserve_preemptions_total[1m])` |
| Spec acceptance | `rate(turingserve_spec_accepted_tokens_total[1m]) / rate(turingserve_spec_draft_tokens_total[1m])` |
| Step time by phase | `sum(rate(turingserve_step_seconds_sum[1m])) by (phase) / sum(rate(turingserve_step_seconds_count[1m])) by (phase)` |
| Autopilot settings | `turingserve_autopilot_setting` |

4. `POST /debug/profile?steps=N` (with `--profile-dir`); document how to open traces in Perfetto.

**Done when:** a screenshot of the live dashboard under load is in the README.

---

## Phase 14 — Load, soak, chaos, full benchmark and report (week 11) → Milestone M5

**Goal:** production-style evidence.
**Files:** `benchmarks/chaos.py`, `benchmarks/soak.py`, ramp/profile modes in `serve_bench.py`, final `make_report.py`, `benchmarks/REPORT.md`.

**Steps**
1. **Ramp** (Kaggle T4, 1.5B): `serve_bench --mode ramp` → the maximum rate with ≥ 90% goodput, for
   TuringServe, vLLM and llama.cpp.
2. **Soak** (laptop, 0.5B; costs no Kaggle time): first a short ramp on the laptop to find its maximum rate.
   Then `soak.py` runs Poisson load at 70% of that rate for **2 hours** and samples every 10 s: server process
   RSS (psutil), `/debug/state` and the error count.
   **Pass:** RSS growth < 5% after the first 10 minutes; at the end `num_requests_tracked == 0`, all blocks
   are free or cached, and there were zero 5xx responses.
3. **Chaos** (laptop): `chaos.py` cases: 20% of streaming clients disconnect after a random 0–2 s; prompts of
   length `max_model_len − 1`; a burst of 200 simultaneous requests; `max_tokens=1`; empty prompts; invalid
   parameters; Hindi/emoji-heavy prompts; stop strings that match immediately.
   **Pass:** no crash; every error is the 4xx defined in ARCH §6.1; afterwards `/debug/state` shows a clean state.
4. **Full matrix** (Kaggle, 2–3 sessions): every engine × {ShareGPT concurrency sweep, synthetic, prefill
   2k/4k/8k, RAG}; the ablation table (ARCH §10.8, TuringServe only); INT8 and speculation results.
5. **`make_report.py` final** → `REPORT.md`: environment table, main chart, prefill table next to vLLM
   issue #60821, ablations, quality, Autopilot chart, limitations.

**Done when:** `REPORT.md` is generated entirely from the JSON files. `git tag m5`.

---

## Phase 15 — RAG integration, documentation, release (week 12) → Milestone M6 / v1.0

**Goal:** your RAG app runs on TuringServe, and the project is published.

### A. Changes in the RAG repo (`D:\Rag`, Windows; follow its DECISIONS.md style)
1. `src/config.py`: add `LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")` and
   `LLM_REPEAT_PENALTY` (a float from the env var, or `None` when it is unset).
2. `src/llm.py`: `generate()` dispatches `"ollama"` → the existing function and `"openai_compat"` → a new
   `_generate_openai_compat(prompt, system, schema)` that uses **only `urllib.request` and `json`** (standard
   library, so the RAG spec's "one LLM SDK" rule still holds):
   - if `schema` is not None → raise `NotImplementedError("structured output (PROMPT_VERSION=v3) is not supported by the openai_compat provider")`;
   - POST `{LLM_BASE_URL}/chat/completions` with `model`, `messages`, `max_tokens=LLM_MAX_TOKENS`,
     `temperature=LLM_TEMPERATURE`, and `repetition_penalty` when `LLM_REPEAT_PENALTY` is set; 300 s timeout;
     retry once (like the Ollama path); return `LLMResult(text, usage.prompt_tokens, usage.completion_tokens, latency_s)`.
   - The Ollama path: add `"repeat_penalty": LLM_REPEAT_PENALTY` to `options` when it is set. Update the module docstring.
3. `tests/test_llm.py`: a test that monkeypatches `urllib.request.urlopen` and checks the payload and the
   `LLMResult` mapping; a test that `schema` raises `NotImplementedError`.
4. A DECISIONS.md entry. `pytest` passes in the RAG repo.

### B. Model and memory (laptop)
Both backends use **Qwen2.5-1.5B-Instruct** for the comparison.
- **Ollama:** look at https://ollama.com/library/qwen2.5/tags for `1.5b-instruct-fp16` and `1.5b-instruct-q8_0`,
  and `ollama pull` the one you need. If the tag does not exist, download the GGUF from
  `bartowski/Qwen2.5-1.5B-Instruct-GGUF` and create it with a Modelfile (`FROM ./<file>.gguf`, then
  `ollama create qwen2.5-1.5b-f16 -f Modelfile`).
- **TuringServe** (WSL), fp16 first:
  `turingserve serve --model $TS_MODEL_15 --served-model-name qwen2.5-1.5b --max-model-len 8192 --gpu-memory-utilization 0.95 --max-num-seqs 4 --max-num-batched-tokens 1024`.
  If it stops with the "not enough GPU memory" message, add `--quantization int8` and use Ollama's
  `q8_0` model for the comparison instead of fp16.

### C. Runs (only one backend on the GPU at a time)
```powershell
# 1) Ollama (PowerShell, in D:\Rag)
$env:LLM_PROVIDER="ollama"; $env:LLM_MODEL="qwen2.5:1.5b-instruct-fp16"; $env:LLM_REPEAT_PENALTY="1.0"
python -m src.eval --questions eval/heldout_questions.json --configs router
New-Item -ItemType Directory -Force eval\backend_compare | Out-Null
Move-Item eval\results_heldout_questions_router.md eval\backend_compare\ollama_results.md
Move-Item eval\to_grade_heldout_questions_router.csv eval\backend_compare\ollama_to_grade.csv
ollama stop qwen2.5:1.5b-instruct-fp16

# 2) TuringServe (server running in WSL, see B)
$env:LLM_PROVIDER="openai_compat"; $env:LLM_MODEL="qwen2.5-1.5b"; $env:LLM_BASE_URL="http://localhost:8000/v1"; $env:LLM_REPEAT_PENALTY="1.0"
python -m src.eval --questions eval/heldout_questions.json --configs router
Move-Item eval\results_heldout_questions_router.md eval\backend_compare\turingserve_results.md
Move-Item eval\to_grade_heldout_questions_router.csv eval\backend_compare\turingserve_to_grade.csv

# 3) Restart TuringServe with --spec-method ngram, run the same eval, save as turingserve_ngram_*
```
- Use the config name the README's held-out result used (`CONFIGS` in `src/eval.py`; the help text's example is `router`).
- Grade the to-grade CSVs by hand with the same rubric as before (same person, same day).
- `benchmarks/compare_rag_answers.py` (TuringServe repo): read two to-grade CSVs, report the exact-match
  rate after normalizing whitespace, and list the differing questions.
- **Expected:** correct/refused counts within ±2 of each other (the engines' fp16 numerics differ). A larger
  gap → inspect the differing answers first, then the debug ladder.
- Latency: mean and p95 from the results files; TTFT and prefix hit rate from TuringServe's metrics.

### D. Documentation and release
1. **README** order: one-line pitch → demo GIF → architecture diagram → headline results (chart, prefill
   table, ablations) → how it works (one paragraph per component) → correctness and testing → how to run →
   limitations and future work.
2. `docs/design/*.md`: one short note per component, in your own words (this is your interview preparation).
3. Blog post: "Building an LLM inference engine for GPUs that FlashAttention doesn't support": what
   worked, what failed (from the lab notes), the numbers.
4. Resume bullets (put in your measured numbers):
   - "Built TuringServe, an LLM serving engine from scratch (Python/Triton) for Turing GPUs: paged KV cache,
     continuous batching, chunked prefill; **X× throughput vs Hugging Face** on a T4 at 32 concurrent users."
   - "Wrote Triton tensor-core kernels for paged attention, RoPE, RMSNorm and INT8 GEMM; **8k-token prefill in
     Y s vs 10.2 s** for vLLM's default T4 backend; verified by 100+ randomized tests in CI."
   - "Designed a rule-based self-tuning controller that adapts chunk size and speculative decoding to load,
     improving goodput by **Z%** over the best fixed configuration."
   - "Served my RAG system on it with prefix caching and prompt-lookup decoding: **W% lower TTFT**, same answer accuracy."
5. `git tag v1.0 && git push --tags`; create a GitHub release with the summary and the report link.

**Done when:** RAG comparison done and documented; README, report, blog and resume bullets published; `v1.0` tagged.

---

## Phase 16 — Stretch (only after v1.0)
Follow ARCH §13 in order. First item, **Llama 3.2 support**:
1. Add `model_type == "llama"`: no QKV bias; read `rope_theta` and `rope_scaling` from `config.json`.
2. Implement the "llama3" RoPE frequency adjustment: with `wavelen = 2π / inv_freq`,
   `low = original_max / low_freq_factor`, `high = original_max / high_freq_factor`:
   if `wavelen > low` → `inv_freq / factor`; if `wavelen < high` → unchanged; otherwise blend with
   `smooth = (original_max / wavelen − low_freq_factor) / (high_freq_factor − low_freq_factor)`:
   `(1 − smooth) · inv_freq / factor + smooth · inv_freq`.
3. Repeat the Phase 1 golden tests for `meta-llama/Llama-3.2-1B-Instruct` (accept the license on Hugging Face).
4. Serve `Llama-3.2-3B-Instruct` on the T4 and connect the RAG through a tunnel
   (`cloudflared tunnel --url http://localhost:8000` in the Kaggle notebook).

---

## §T — Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is False in WSL | WSL or driver issue | `wsl --update` in Windows; update the NVIDIA driver; `nvidia-smi` must work in WSL; never install Linux NVIDIA drivers in WSL |
| First Triton call fails: `Python.h: No such file` or "failed to find C compiler" | Missing headers or compiler | `sudo apt install build-essential`; use the uv Python 3.12 venv |
| `libcuda.so cannot found` | Triton cannot find the WSL driver library | `export TRITON_LIBCUDA_PATH=/usr/lib/wsl/lib` (it is in `.bashrc` from §0.10) |
| `out of resource: shared memory, Required: …, Hardware limit: 65536` | Tile too large for Turing | Remove that config; the tuner skips it automatically |
| `CUDA error: an illegal memory access` | Missing mask or bad slot/block index | `CUDA_LAUNCH_BLOCKING=1`; run the kernel's test for that shape with `TRITON_INTERPRET=1`; check slot −1 and block-table padding; int64 offsets |
| `operation not permitted when stream is capturing` | Sync, allocation or tuning inside graph capture | Warm up and tune before capture; remove `.item()`, `.cpu()`, `print`, and tensors built from Python lists in the captured code |
| NaN/inf logits in fp16 | Overflow | Re-run the Day-1 gate on that prompt; check RMSNorm in fp32, attention accumulators in fp32, the max subtraction in softmax |
| HF greedy differs from yours at the very first token | HF applied `repetition_penalty` 1.1 from `generation_config` | Pass `repetition_penalty=1.0, do_sample=False` |
| Output text contains "�" | Incomplete UTF-8 emitted | Use `DecodeStream`; add the text to the detokenizer test |
| Triton kernels slow on the T4 and the PTX has no `mma.sync` | Triton ≥ 3.3 was installed by something | Reinstall the pins; run `check_versions.py`; find which package upgraded torch |
| `tl.dot` error about shape ≥ 16 | A dot dimension < 16 | Pad (GQA group → 16) and mask |
| Server hangs; requests never finish | The engine thread died | Check the logs; `/health` must return 503; every exception in the loop must be logged and fail the requests |
| Memory grows during the soak test | Aborted or finished requests are not removed | Check `/debug/state` `num_requests_tracked`; remove entries from every dict on finish/abort |
| Kaggle: `operator torchvision::nms does not exist` | torchvision does not match torch after the downgrade | `pip uninstall -y torchvision torchaudio`, restart the session |
| Kaggle: no GPU or no Internet | Account not verified or settings off | Phone verification; choose the accelerator; Internet On |
| Kaggle: OOM at startup with "T4 ×2" | Both GPUs visible, wrong device | Set `CUDA_VISIBLE_DEVICES=0` before importing torch |
| Kaggle: vLLM install replaced torch | Expected | Baselines always run in their own notebooks |
| WSL: git/pip/imports very slow | Repo is under `/mnt/…` | Keep the repo in `~/code` |
| Startup error "not enough GPU memory … Ollama?" | Ollama is holding the GPU | §0.1 |
| Throughput lower than expected, GPU busy % low | CPU overhead | Profile; CUDA graphs; vectorized input building; check the detokenizer's step time |
| T4 numbers vary > 10% between runs | Power/thermal throttling | Look at the GPU log; warm up; median of 3 runs |
| `wsl --manage --move` gives `E_ACCESSDENIED` | Known WSL issue | Use the export/import fallback in §0.2 |
| Hugging Face download fails (rate limit) | Anonymous limits | `hf auth login` (or `huggingface-cli login`) with a free token; retry |

---

## §I — Interview preparation (you must be able to answer all of these)
1. Why is decode memory-bound and prefill compute-bound? Compute the speed of light for your setup.
2. How does paged attention remove memory waste? What are a block table and a slot mapping?
3. Why is continuous batching so much faster than static batching?
4. What does chunked prefill trade off, and what does the token budget control?
5. Recompute vs swap for preemption: when would you choose swap?
6. Why can't you use FlashAttention-2 on a T4, and what did you do instead?
7. Why does Triton ≥ 3.3 make attention on Turing slow, and how did you prove your kernels use tensor cores?
8. How does online softmax work? How does split-K decoding help, and when?
9. Why does write-then-attend simplify the design?
10. Why compute logits only for sampling rows? How much memory does that save?
11. Why do CUDA graphs help small models and small batches most? What can't run inside a graph?
12. How does the exponential race give per-request reproducible sampling in a batch?
13. How do you stream text correctly when a character spans two tokens, and when a stop string spans two chunks?
14. Prove that speculative decoding keeps the output distribution unchanged. How does it work for n-gram proposals?
15. When does speculation hurt, and how does Autopilot decide?
16. Why INT8 weights but fp16 activations? Why is the saving smaller with tied embeddings?
17. How did you prove your engine is correct? Explain the tolerance policy and needle tests.
18. How did you make the benchmark fair across engines?
19. What are the differences between your design and vLLM / nano-vllm?
20. What would you build next, and why?

---

## §F — Final deliverables checklist
- [ ] GitHub repo with a green CI, tags m1–m5 and v1.0
- [ ] README: pitch, demo GIF, diagram, headline results, how it works, testing, how to run, limitations
- [ ] `benchmarks/REPORT.md` generated from committed result JSON files
- [ ] Ablation table, quality table (perplexity, GSM8K), Autopilot chart, dashboard screenshot
- [ ] `docs/design/*.md` for every component; `docs/lab_notes.md` complete
- [ ] RAG backend comparison documented in both repos
- [ ] Blog post published; resume bullets with real numbers

---

## Appendix A — `docs/lab_notes.md` template
```
# Lab notes

## Measurements
### YYYY-MM-DD — Phase N — <topic>
Machine: laptop GTX 1650 | Kaggle T4      Commit: <hash>
Command: <exact command>
Result: <numbers>
Decision / next step: <text>

## Failed ideas
(what you tried, the numbers, why it failed)

## Threshold changes
(test, old threshold, new threshold, measured values, reason)

## Proposals
(ideas not in ARCHITECTURE.md; decide later)

## Differences vs vLLM / nano-vllm
(component, their approach, my approach, why)
```

## Appendix B — Command cheat sheet
```
# environment
source ~/code/turingserve/.venv/bin/activate && python scripts/check_versions.py
# tests
pytest -m "not gpu and not model and not slow and not kaggle" -q      # what CI runs
pytest tests/kernels -m gpu -q                                         # kernels on this GPU
pytest -q                                                               # everything available here
# server
turingserve serve --model $TS_MODEL_05                                  # laptop
turingserve serve --model $TS_MODEL_05 --kernels reference --no-cuda-graphs   # debug path
turingserve tune --model $TS_MODEL_05 --retune                          # re-run the tuner
# benchmark
python -m benchmarks.serve_bench --base-url http://localhost:8000 --model Qwen2.5-0.5B-Instruct \
  --tokenizer $TS_MODEL_05 --dataset sharegpt --num-prompts 200 --mode concurrency --concurrency 1,4,16 \
  --ignore-eos --label laptop-test --out benchmarks/results
python -m benchmarks.make_report
```
