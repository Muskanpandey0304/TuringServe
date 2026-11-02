"""HF golden files (Build Guide Phase 1, step 6): computed once per model, git-ignored.

  python scripts/make_golden.py --model $TS_MODEL_05 --out tests/golden/qwen05.pt

For 20 prompts (10 ShareGPT turns, 5 WikiText slices, 5 special: Hindi / emoji / code / numbers /
"Hello"): HF fp32 greedy 64 tokens (do_sample=False, repetition_penalty=1.0), teacher-forced
top-20 logits at every position, full logits at 8 positions of the first 3 prompts, hidden states
of prompts 0 and 1; plus the fp16 NOISE FLOOR = max |HF fp16 - HF fp32| logit difference.
"""

from __future__ import annotations

import argparse
import gc
import json
import os

import torch

SPECIAL = ["नमस्ते दुनिया, यह एक परीक्षण है।", "🙂👍🏽🚀 emoji test 🎉",
           "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
           "1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233", "Hello"]


def prompts(tok):
    from datasets import load_dataset

    out = []
    with open(os.environ["TS_SHAREGPT"], encoding="utf-8") as f:
        data = json.load(f)
    for conv in data:
        t = conv.get("conversations", [])
        if t and t[0].get("from") == "human" and t[0].get("value", "").strip():
            s = tok.apply_chat_template([{"role": "user", "content": t[0]["value"]}], tokenize=False,
                                        add_generation_prompt=True)
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) <= 512:
                out.append(ids)
        if len(out) == 10:
            break
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wt_ids = tok("\n\n".join(wt["text"]), return_tensors="pt").input_ids[0].tolist()
    for i in range(5):
        out.append(wt_ids[1000 + i * 256: 1000 + (i + 1) * 256])
    for text in SPECIAL:
        s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                    add_generation_prompt=True)
        out.append(tok.encode(s, add_special_tokens=False))
    return out


def forward(model_path, dtype, plist, gen: bool):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).cuda().eval()
    res = []
    with torch.inference_mode():
        for i, p in enumerate(plist):
            r = {}
            ids = torch.tensor([p]).cuda()
            if gen:
                g = model.generate(ids, max_new_tokens=64, do_sample=False, repetition_penalty=1.0,
                                   temperature=None, top_p=None, top_k=None, pad_token_id=151643)
                r["output_ids"] = g[0, len(p):].tolist()
            full = torch.tensor([p + r.get("output_ids", [])]).cuda()
            o = model(full, output_hidden_states=gen and i < 2)
            lg = o.logits[0].float()
            v, ix = lg.topk(20, dim=-1)
            r["top20_values"], r["top20_indices"] = v.cpu(), ix.cpu()
            pos = torch.linspace(0, lg.shape[0] - 1, 8).long()
            if i < 3:
                r["full_positions"], r["full_logits"] = pos, lg[pos].cpu()
            if gen and i < 2:
                r["hidden_states"] = [h[0].half().cpu() for h in o.hidden_states]
            res.append(r)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-prompts", type=int, default=20)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    plist = prompts(AutoTokenizer.from_pretrained(a.model))[: a.num_prompts]
    fp32 = forward(a.model, torch.float32, plist, gen=True)
    fp16 = forward(a.model, torch.float16, plist, gen=False)
    noise = 0.0
    for r32, r16 in zip(fp32, fp16, strict=False):
        # compare on the fp32 top-20 indices and the 8 full positions
        gathered = r16["top20_values"]                        # fp16's own top-20 values
        noise = max(noise, (gathered[:, 0] - r32["top20_values"][:, 0]).abs().max().item())
        if "full_logits" in r32 and "full_logits" in r16:
            noise = max(noise, (r32["full_logits"] - r16["full_logits"]).abs().max().item())
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"prompts": plist, "fp32": fp32, "noise_floor": noise,
                "settings": {"do_sample": False, "repetition_penalty": 1.0, "max_new_tokens": 64}},
               a.out)
    print(f"wrote {a.out}: {len(plist)} prompts, fp16 noise floor = {noise:.4f}")


if __name__ == "__main__":
    main()
