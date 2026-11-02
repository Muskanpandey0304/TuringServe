"""Day-1 gate: is the model safe in fp16? (Build Guide 0.11, P8/P46)

Runs 200 prompts (180 ShareGPT first turns + 20 long WikiText slices) in fp32 then fp16 with HF,
records max |activation| and finiteness of every layer output, and compares WikiText-2 perplexity.

  G1 no inf/NaN in fp16 layer outputs or logits               (must hold)
  G2 max |activation| in fp16 < 30,000 (30,000-65,504 = warning)
  G3 WikiText-2 perplexity fp16 vs fp32 within 1%             (must hold)
  G4 >= 80% of prompts have identical first 32 greedy tokens  (information only)
Verdict: pass if G1 and G3 pass.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import date

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_prompts(tok, n_total: int):
    from datasets import load_dataset

    path = os.environ["TS_SHAREGPT"]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    prompts = []
    for conv in data:
        turns = conv.get("conversations", [])
        if not turns or turns[0].get("from") != "human" or not turns[0].get("value", "").strip():
            continue
        ids = tok.apply_chat_template([{"role": "user", "content": turns[0]["value"]}],
                                      tokenize=False, add_generation_prompt=True)
        ids = tok.encode(ids, add_special_tokens=False)
        if len(ids) <= 1024:
            prompts.append(ids)
        if len(prompts) == n_total - 20:
            break
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wt_ids = tok("\n\n".join(wt["text"]), return_tensors="pt").input_ids[0].tolist()
    for i in range(20):
        sl = wt_ids[i * 3000:(i + 1) * 3000]
        text = tok.decode(sl)
        t = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                    add_generation_prompt=True)
        prompts.append(tok.encode(t, add_special_tokens=False))
    return prompts


def run_dtype(model_path, dtype, prompts):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).cuda().eval()
    stats = {"max_abs": 0.0, "finite": True}

    def hook(_, __, out):
        t = out[0] if isinstance(out, tuple) else out
        stats["max_abs"] = max(stats["max_abs"], float(t.abs().max()))
        stats["finite"] &= bool(torch.isfinite(t).all())

    for layer in model.model.layers:
        layer.register_forward_hook(hook)
    outs = []
    with torch.inference_mode():
        for p in prompts:
            ids = torch.tensor([p]).cuda()
            g = model.generate(ids, max_new_tokens=32, do_sample=False, repetition_penalty=1.0,
                               temperature=None, top_p=None, top_k=None, pad_token_id=151643)
            outs.append(g[0, len(p):].tolist())
            logits = model(ids).logits
            stats["finite"] &= bool(torch.isfinite(logits).all())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outs, stats


def ppl(model_path, dtype):
    out = subprocess.check_output([sys.executable, os.path.join(ROOT, "scripts", "perplexity.py"),
                                   "--model", model_path, "--dtype", dtype, "--engine", "hf"],
                                  text=True)
    return float(out.strip().rsplit("ppl=", 1)[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    prompts = build_prompts(tok, a.num_prompts)
    out32, _ = run_dtype(a.model, torch.float32, prompts)
    out16, s16 = run_dtype(a.model, torch.float16, prompts)
    p32, p16 = ppl(a.model, "float32"), ppl(a.model, "float16")
    rel = abs(p16 - p32) / p32
    same = sum(x == y for x, y in zip(out32, out16, strict=False)) / len(prompts)
    g1 = s16["finite"]
    g2 = "pass" if s16["max_abs"] < 30000 else ("warning" if s16["max_abs"] <= 65504 else "fail")
    g3 = rel <= 0.01
    verdict = "PASS" if g1 and g3 else "FAIL"
    res = {"date": str(date.today()), "model": a.model, "G1_finite": g1,
           "G2_max_abs_fp16": s16["max_abs"], "G2": g2, "G3_ppl_fp32": p32, "G3_ppl_fp16": p16,
           "G3_rel_diff": rel, "G3": g3, "G4_identical_first32": same, "verdict": verdict}
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        with open(os.path.join(os.path.dirname(a.out) or ".", "day1_gate.md"), "a") as f:
            f.write(f"\n## {res['date']} - {os.path.basename(a.model.rstrip('/'))}\n\n"
                    f"| Gate | Result |\n|---|---|\n| G1 finite | {g1} |\n"
                    f"| G2 max abs | {s16['max_abs']:.1f} ({g2}) |\n"
                    f"| G3 ppl fp32/fp16 | {p32:.4f} / {p16:.4f} (rel {rel:.4%}) |\n"
                    f"| G4 identical first 32 | {same:.1%} |\n\n**Verdict: {verdict}**\n")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
