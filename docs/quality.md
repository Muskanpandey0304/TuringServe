# Quality: fp16 vs INT8 (W8A16)

Qwen2.5-1.5B-Instruct on a Kaggle Tesla T4, repo commit 349b3f3 (now b6dc052; engine code identical to tag m3-rc), 2026-11-02.
Raw files: `benchmarks/results/quality/`.

## WikiText-2 perplexity (test split, 299,077 tokens)

| Engine | Precision | Perplexity | vs HF fp16 |
|---|---|---|---|
| Hugging Face | fp16 | 9.2122 | baseline |
| TuringServe | fp16 | 9.2123 | +0.0003% |
| TuringServe | INT8 W8A16 | 9.2270 | +0.16% |

TuringServe fp16 matches the HF reference, and INT8 is well inside the +2% target (ARCHITECTURE 10.6).

## GSM8K (5-shot, chat template, temperature 0, 1319 questions, lm-eval local-chat-completions)

| Precision | flexible-extract | strict-match |
|---|---|---|
| fp16 | 0.6050 +/- 0.0135 | 0.3465 +/- 0.0131 |
| INT8 | 0.6012 +/- 0.0135 | 0.3541 +/- 0.0132 |

The fp16-to-INT8 difference (-0.4 pt flexible, +0.8 pt strict) is smaller than one standard error, so no measurable accuracy change.
Strict-match is low for both because the model does not reliably emit the exact "#### answer" format; flexible-extract is the headline number.
