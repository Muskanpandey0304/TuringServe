"""Chaos test (Build Guide Phase 14, step 3). Needs a server started with --debug-endpoints.

Cases: 20% of streaming clients disconnect after 0-2 s; prompts of length max_model_len-1; a burst
of 200 simultaneous requests; max_tokens=1; empty prompts; invalid parameters; Hindi/emoji-heavy
prompts; stop strings that match immediately.
PASS: no crash, every error is a 4xx from ARCH 6.1, and /debug/state is clean afterwards.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys

import httpx

ALLOWED_ERRORS = {400, 404, 429}


async def one(client, base, model, body, stream=False, disconnect_after=None):
    try:
        if stream:
            async with client.stream("POST", base + "/v1/completions", json={**body, "stream": True}) as r:
                if r.status_code != 200:
                    await r.aread()
                    return r.status_code
                t = 0
                async for _ in r.aiter_lines():
                    t += 1
                    if disconnect_after is not None and t >= disconnect_after:
                        return 200
                return 200
        r = await client.post(base + "/v1/completions", json=body)
        return r.status_code
    except httpx.HTTPError as e:
        return f"EXC {e!r}"


async def main_async(a) -> int:
    base, model = a.base_url, a.model
    rng = random.Random(0)
    bad = []
    async with httpx.AsyncClient(timeout=120) as c:
        mml = (await c.get(base + "/v1/models")).json()["data"][0]["max_model_len"]
        cases = {
            "disconnects": [one(c, base, model, {"model": model, "prompt": "Tell me a story " * 3,
                                                 "max_tokens": 300}, True,
                                disconnect_after=rng.randint(1, 10) if rng.random() < 0.2 else None)
                            for _ in range(50)],
            "near_max_len": [one(c, base, model, {"model": model, "prompt": [1000 + i % 50] * (mml - 1),
                                                  "max_tokens": 1}) for i in range(3)],
            "burst_200": [one(c, base, model, {"model": model, "prompt": f"hello {i}", "max_tokens": 4})
                          for i in range(200)],
            "max_tokens_1": [one(c, base, model, {"model": model, "prompt": "hi", "max_tokens": 1})
                             for _ in range(10)],
            "empty_prompt": [one(c, base, model, {"model": model, "prompt": ""})],
            "invalid_params": [one(c, base, model, {"model": model, "prompt": "x", "temperature": 9}),
                               one(c, base, model, {"model": model, "prompt": "x", "n": 3}),
                               one(c, base, model, {"model": model, "prompt": "x", "top_p": 0})],
            "unicode": [one(c, base, model, {"model": model, "max_tokens": 16,
                                             "prompt": "नमस्ते 🙂👍🏽🚀 你好世界 " * 20}) for _ in range(10)],
            "immediate_stop": [one(c, base, model, {"model": model, "prompt": "abc", "max_tokens": 8,
                                                    "stop": ["a", "b", " "]}) for _ in range(5)],
        }
        for name, coros in cases.items():
            codes = await asyncio.gather(*coros)
            for code in codes:
                if code != 200 and code not in ALLOWED_ERRORS:
                    bad.append((name, code))
            print(f"{name:<16} {len(codes):>4} requests  statuses: "
                  f"{ {str(k): codes.count(k) for k in set(codes)} }")
        await asyncio.sleep(3)
        st = (await c.get(base + "/debug/state")).json()
        health = (await c.get(base + "/health")).status_code
    print("final state:", st, "health:", health)
    clean = (st["num_running"] == 0 and st["num_waiting"] == 0 and st["num_requests_tracked"] == 0
             and st["num_free_blocks"] + st["num_cached_blocks"] == st["num_total_blocks"] - 1)
    ok = clean and health == 200 and not bad
    print("PASS" if ok else f"FAIL (unexpected: {bad[:5]}, clean={clean}, health={health})")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    sys.exit(asyncio.run(main_async(ap.parse_args())))
