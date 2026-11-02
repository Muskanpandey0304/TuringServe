"""Soak test (Build Guide Phase 14, step 2): Poisson load at a fixed rate for hours, sampling every
10 s the server's RSS (psutil), /debug/state and the error count.

PASS: RSS growth < 5% after the first 10 minutes; at the end num_requests_tracked == 0, all blocks
free or cached, zero 5xx responses.

  python -m benchmarks.soak --server-pid <pid> --rate 2.0 --hours 2 --model tiny
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time

import httpx
import psutil

PROMPTS = ["Explain what a hash table is.", "Write a haiku about GPUs.",
           "Summarize the plot of Hamlet in two sentences.", "What is 17 * 23?",
           "नमस्ते, आप कैसे हैं?", "List three uses of binary search."]


async def main_async(a) -> int:
    proc = psutil.Process(a.server_pid)
    rng = random.Random(0)
    stats = {"sent": 0, "ok": 0, "e4xx": 0, "e5xx": 0, "exc": 0}
    samples = []
    end = time.time() + a.hours * 3600

    async def req(client):
        stats["sent"] += 1
        try:
            r = await client.post(a.base_url + "/v1/completions", json={
                "model": a.model, "prompt": rng.choice(PROMPTS), "max_tokens": a.max_tokens,
                "temperature": 0.8})
            if r.status_code == 200:
                stats["ok"] += 1
            elif r.status_code >= 500:
                stats["e5xx"] += 1
            else:
                stats["e4xx"] += 1
        except httpx.HTTPError:
            stats["exc"] += 1

    async def sampler(client):
        while time.time() < end:
            st = (await client.get(a.base_url + "/debug/state")).json()
            samples.append({"t": time.time(), "rss": proc.memory_info().rss, **st, **stats})
            await asyncio.sleep(10)

    async with httpx.AsyncClient(timeout=300, limits=httpx.Limits(max_connections=500)) as client:
        sam = asyncio.create_task(sampler(client))
        tasks = []
        while time.time() < end:
            tasks.append(asyncio.create_task(req(client)))
            await asyncio.sleep(rng.expovariate(a.rate))
        await asyncio.gather(*tasks)
        await sam
        await asyncio.sleep(3)
        final = (await client.get(a.base_url + "/debug/state")).json()
    t0 = samples[0]["t"]
    base = [s["rss"] for s in samples if s["t"] - t0 >= 600] or [samples[0]["rss"]]
    growth = (samples[-1]["rss"] - base[0]) / base[0]
    ok = (growth < 0.05 and final["num_requests_tracked"] == 0 and stats["e5xx"] == 0
          and final["num_free_blocks"] + final["num_cached_blocks"] == final["num_total_blocks"] - 1)
    json.dump({"stats": stats, "samples": samples, "final": final, "rss_growth": growth},
              open(a.out, "w"))
    print(f"requests {stats}  RSS growth after 10 min: {growth:+.1%}  final {final}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--server-pid", type=int, required=True)
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", default="benchmarks/results/soak.json")
    sys.exit(asyncio.run(main_async(ap.parse_args())))
