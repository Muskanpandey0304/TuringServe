"""Async benchmark client (ARCH 10, Build Guide Phase 8).

POSTs /v1/completions with a pre-templated prompt, streaming on, temperature 0 and ignore_eos;
parses SSE; records send / every chunk / end with time.perf_counter(). Modes:
  concurrency  fixed number of clients, each sends the next request when the last finishes
  rate         Poisson arrivals at --rate req/s (seed fixed)
  ramp         0.5,1,2,4,8,12,16 req/s for --step-seconds each -> max rate with >= 90% goodput
  profile      piecewise-constant Poisson rates "120:0.5,120:4,120:0.5"
Writes benchmarks/results/<date>_<engine>_<workload>_<label>.json (ARCH 10.9).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.datasets import BenchItem, load  # noqa: E402

TTFT_SLO, TPOT_SLO = 2.0, 0.050


@dataclass
class Record:
    send: float = 0.0
    first: float | None = None
    chunks: list = field(default_factory=list)
    end: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    local_prompt_tokens: int = 0
    error: str | None = None


def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    pos = (len(xs) - 1) * q / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


async def send_one(client: httpx.AsyncClient, base_url: str, model: str, item: BenchItem,
                   ignore_eos_flag: bool, tok=None) -> Record:
    rec = Record(local_prompt_tokens=item.prompt_tokens)
    body = {"model": model, "prompt": item.prompt, "max_tokens": item.max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    if item.ignore_eos or ignore_eos_flag:
        body["ignore_eos"] = True
    rec.send = time.perf_counter()
    text = ""
    try:
        async with client.stream("POST", base_url + "/v1/completions", json=body) as r:
            if r.status_code != 200:
                rec.error = f"HTTP {r.status_code}"
                await r.aread()
            else:
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    j = json.loads(data)
                    if j.get("usage"):
                        rec.prompt_tokens = j["usage"]["prompt_tokens"]
                        rec.output_tokens = j["usage"]["completion_tokens"]
                    for ch in j.get("choices", []):
                        if ch.get("text"):
                            now = time.perf_counter()
                            if rec.first is None:
                                rec.first = now
                            rec.chunks.append(now)
                            text += ch["text"]
    except Exception as e:  # noqa: BLE001
        rec.error = repr(e)
    rec.end = time.perf_counter()
    if rec.output_tokens == 0 and tok is not None and text:
        rec.output_tokens = len(tok.encode(text, add_special_tokens=False))
    return rec


def summarize(recs: list[Record], duration: float) -> dict:
    ok = [r for r in recs if r.error is None and r.first is not None]
    ttft = [r.first - r.send for r in ok]
    tpot = [(r.end - r.first) / (r.output_tokens - 1) for r in ok if r.output_tokens > 1]
    itl = [b - a for r in ok for a, b in zip(r.chunks, r.chunks[1:])]
    e2e = [r.end - r.send for r in ok]
    good = sum(1 for r in ok if r.first - r.send <= TTFT_SLO and (
        r.output_tokens <= 1 or (r.end - r.first) / (r.output_tokens - 1) <= TPOT_SLO))
    out_tokens = sum(r.output_tokens for r in ok)

    def block(xs):
        return {"mean": sum(xs) / len(xs) if xs else float("nan"), "p50": pct(xs, 50),
                "p90": pct(xs, 90), "p99": pct(xs, 99)}

    mism = sum(1 for r in ok if r.prompt_tokens and r.local_prompt_tokens and
               abs(r.prompt_tokens - r.local_prompt_tokens) > 0.01 * r.local_prompt_tokens)
    return {"num_requests": len(recs), "num_ok": len(ok), "errors": len(recs) - len(ok),
            "duration_s": duration, "output_tokens": out_tokens,
            "output_throughput_tok_s": out_tokens / duration if duration else 0.0,
            "request_throughput_rps": len(ok) / duration if duration else 0.0,
            "goodput_rps": good / duration if duration else 0.0,
            "goodput_fraction": good / len(recs) if recs else 0.0,
            "ttft": block(ttft), "tpot": block(tpot), "itl": block(itl), "e2e": block(e2e),
            "prompt_token_mismatch_requests": mism}


async def run_concurrency(client, args, items, level, tok):
    queue = list(items)
    recs: list[Record] = []

    async def worker():
        while queue:
            recs.append(await send_one(client, args.base_url, args.model, queue.pop(0),
                                       args.ignore_eos, tok))

    t0 = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(level)])
    return recs, time.perf_counter() - t0


async def run_poisson(client, args, items, schedule, tok):
    """schedule: list of (seconds, rate). Arrival times are exponential with a fixed seed."""
    rng = random.Random(args.seed)
    tasks, t_off, i = [], 0.0, 0
    t0 = time.perf_counter()
    for seconds, rate in schedule:
        t = 0.0
        while True:
            t += rng.expovariate(rate)
            if t >= seconds:
                break
            delay = t_off + t - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(send_one(client, args.base_url, args.model,
                                                      items[i % len(items)], args.ignore_eos, tok)))
            i += 1
        t_off += seconds
    recs = await asyncio.gather(*tasks)
    return list(recs), time.perf_counter() - t0


def gpu_logger(path):
    return subprocess.Popen(
        ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
         "--format=csv", "-l", "1"], stdout=open(path, "w"), stderr=subprocess.DEVNULL)


async def amain(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    items = load(args.dataset, tok, args.num_prompts)
    meta = {"git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                         text=True).stdout.strip(),
            "engine": args.engine, "base_url": args.base_url, "model": args.model,
            "dataset": args.dataset, "seed": args.seed, "label": args.label,
            "start_time": datetime.now().isoformat(), "args": vars(args)}
    try:
        meta["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                                      "--format=csv,noheader"], capture_output=True,
                                     text=True).stdout.strip()
    except FileNotFoundError:
        meta["gpu"] = "unknown"
    os.makedirs(args.out, exist_ok=True)
    results = []
    async with httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=1000)) as client:
        if args.warmup:
            await run_concurrency(client, args, items[: args.warmup], min(4, args.warmup), tok)
        if args.mode == "concurrency":
            runs = [("concurrency", int(c), None) for c in args.concurrency.split(",")]
        elif args.mode == "rate":
            runs = [("rate", args.rate, [(args.step_seconds, args.rate)])]
        elif args.mode == "ramp":
            runs = [("rate", float(r), [(args.step_seconds, float(r))]) for r in args.ramp.split(",")]
        else:
            sched = [(float(a), float(b)) for a, b in (p.split(":") for p in args.profile.split(","))]
            runs = [("profile", None, sched)]
        for kind, level, sched in runs:
            reps = []
            for rep in range(args.repeat):
                log = gpu_logger(os.path.join(args.out, "tmp_gpu.csv")) if args.gpu_log else None
                if kind == "concurrency":
                    recs, dur = await run_concurrency(client, args, items, level, tok)
                else:
                    recs, dur = await run_poisson(client, args, items, sched, tok)
                if log:
                    log.terminate()
                s = summarize(recs, dur)
                s["level"] = level
                reps.append({"summary": s, "requests": [asdict(r) for r in recs],
                             "gpu_log": open(os.path.join(args.out, "tmp_gpu.csv")).read()
                             if log else None})
                print(f"[{kind} {level}] rep {rep + 1}/{args.repeat}: "
                      f"{s['output_throughput_tok_s']:.1f} tok/s  TTFT p50/p90 "
                      f"{s['ttft']['p50']:.3f}/{s['ttft']['p90']:.3f}s  TPOT p50/p90 "
                      f"{s['tpot']['p50'] * 1000:.1f}/{s['tpot']['p90'] * 1000:.1f}ms  "
                      f"goodput {s['goodput_fraction']:.0%}  errors {s['errors']}")
            reps.sort(key=lambda r: r["summary"]["output_throughput_tok_s"])
            median = reps[len(reps) // 2]
            results.append({"level": level, "kind": kind, "median": median["summary"],
                            "all": [r["summary"] for r in reps], "requests": median["requests"],
                            "gpu_log": median["gpu_log"]})
    fn = f"{datetime.now():%Y-%m-%d}_{args.engine}_{args.dataset}_{args.label}.json"
    path = os.path.join(args.out, fn)
    with open(path, "w") as f:
        json.dump({"meta": meta, "runs": results}, f)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--engine", default="turingserve")
    ap.add_argument("--dataset", default="sharegpt", choices=["sharegpt", "synthetic", "prefill", "rag"])
    ap.add_argument("--num-prompts", type=int, default=500)
    ap.add_argument("--mode", default="concurrency", choices=["concurrency", "rate", "ramp", "profile"])
    ap.add_argument("--concurrency", default="1,4,16,32,64")
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--ramp", default="0.5,1,2,4,8,12,16")
    ap.add_argument("--profile", default="120:0.5,120:4,120:0.5")
    ap.add_argument("--step-seconds", type=float, default=60.0)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="benchmarks/results")
    ap.add_argument("--gpu-log", action="store_true")
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
