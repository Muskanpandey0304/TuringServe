"""Offline throughput: TuringServe vs HF sequential vs HF static batch (Build Guide Phase 4).

Every engine produces exactly 128 output tokens per ShareGPT prompt (HF: min_new_tokens =
max_new_tokens = 128; TuringServe: max_tokens=128, ignore_eos=True).

  python -m benchmarks.offline_bench --engine turingserve --model $TS_MODEL_05 --num-prompts 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_LEN = 128


def prompts(tok, n):
    from benchmarks.datasets import load_sharegpt

    return [tok.encode(i.prompt, add_special_tokens=False)
            for i in load_sharegpt(os.environ["TS_SHAREGPT"], tok, n)]


def run_turingserve(args, tok, plist):
    from turingserve.config import EngineConfig
    from turingserve.engine.builder import build_engine
    from turingserve.engine.sampling_params import SamplingParams

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = EngineConfig(model=args.model, device=dev, kernels="triton" if dev == "cuda" else "reference",
                       dtype="float16" if dev == "cuda" else "float32", max_model_len=4096,
                       max_num_seqs=32 if dev == "cuda" else 16, autopilot="off",
                       gpu_memory_utilization=args.gpu_memory_utilization)
    eng = build_engine(cfg, tok)
    sp = SamplingParams(temperature=0.0, max_tokens=OUT_LEN, ignore_eos=True)
    eng.generate(plist[:2], sp)                                   # warm-up (compiles kernels)
    t0 = time.perf_counter()
    outs = eng.generate(plist, sp)
    return time.perf_counter() - t0, sum(len(o.token_ids) for o in outs)


def run_hf(args, tok, plist, static: bool):
    from transformers import AutoModelForCausalLM

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    kw = dict(do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
              min_new_tokens=OUT_LEN, max_new_tokens=OUT_LEN, pad_token_id=151643)
    total = 0
    t0 = time.perf_counter()
    with torch.inference_mode():
        if static:
            tok.padding_side = "left"
            for i in range(0, len(plist), 16):
                batch = plist[i:i + 16]
                width = max(map(len, batch))
                ids = torch.tensor([[151643] * (width - len(p)) + p for p in batch]).to(dev)
                mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in batch]).to(dev)
                out = model.generate(ids, attention_mask=mask, **kw)
                total += (out.shape[1] - width) * len(batch)
        else:
            for p in plist:
                out = model.generate(torch.tensor([p]).to(dev), **kw)
                total += out.shape[1] - len(p)
    return time.perf_counter() - t0, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["turingserve", "hf-seq", "hf-static"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    ap.add_argument("--out", default="benchmarks/results")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    plist = prompts(tok, args.num_prompts)
    if args.engine == "turingserve":
        secs, toks = run_turingserve(args, tok, plist)
    else:
        secs, toks = run_hf(args, tok, plist, args.engine == "hf-static")
    print(f"{args.engine}: {toks} output tokens in {secs:.1f} s = {toks / secs:.1f} tok/s")
    os.makedirs(args.out, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    with open(os.path.join(args.out, f"offline_{args.engine}_{os.path.basename(args.model)}.json"),
              "w") as f:
        json.dump({"engine": args.engine, "model": args.model, "gpu": gpu,
                   "num_prompts": len(plist), "seconds": secs, "output_tokens": toks,
                   "tok_per_s": toks / secs}, f, indent=1)


if __name__ == "__main__":
    main()
