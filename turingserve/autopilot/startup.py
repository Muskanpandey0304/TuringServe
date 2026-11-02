"""Autopilot startup phase (ARCH 6.15): tuner -> memory profile -> CUDA graphs -> chunk sweep.

The tuning result is cached; a second start with the same key logs "tuning cache hit" and takes
seconds. ``--retune`` forces a new measurement.
"""

from __future__ import annotations

import time

import torch

from turingserve.autopilot import tuner
from turingserve.engine.sampling_params import SamplingParams
from turingserve.utils.gpu import device_name
from turingserve.utils.logging import log
from turingserve.utils.timing import percentile


def tuning_key(cfg, mcfg) -> str:
    import triton

    return tuner.cache_key(
        device_name(cfg.device), cfg.served_model_name or cfg.model,
        "int8" if cfg.quantization == "int8" else cfg.dtype, cfg.max_model_len,
        cfg.max_num_batched_tokens, torch.__version__, triton.__version__)


def run_tuner(cfg, mcfg) -> tuner.TuningTable:
    """Load the cached table or measure (needs CUDA + Triton). Installs it as the active table."""
    key = tuning_key(cfg, mcfg)
    table = None if cfg.retune else tuner.load_cache(cfg.tuning_dir, key)
    if table is not None:
        log("tuning cache hit", key=key)
    else:
        log("tuning cache miss; measuring kernels (this takes a few minutes)", key=key)
        t0 = time.time()
        table = tuner.tune(mcfg, cfg, cfg.device, quantized=cfg.quantization == "int8")
        table.meta.update({"key": key, "seconds": round(time.time() - t0, 1)})
        log("tuning saved", path=tuner.save_cache(cfg.tuning_dir, key, table))
    tuner.set_active(table)
    return table


def chunk_sweep(engine, cfg, table: tuner.TuningTable, seconds: float = 30.0, concurrency: int = 16,
                prompt_len: int = 1024, out_len: int = 128, save: bool = True) -> int:
    """Run the engine offline for ``seconds`` per candidate chunk size; keep the highest-throughput
    candidate that meets the TTFT/TPOT targets at p90 (else the lowest TPOT p90)."""
    cached = table.meta.get("chunk")
    if cached and not cfg.retune:
        engine.scheduler.max_num_batched_tokens = min(cfg.max_num_batched_tokens, int(cached))
        log("chunk sweep cache hit", chunk=cached)
        return int(cached)
    cands = [c for c in (512, 1024, 2048) if c <= cfg.max_num_batched_tokens] or [
        cfg.max_num_batched_tokens]
    results = []
    for c in cands:
        engine.scheduler.max_num_batched_tokens = c
        results.append((c, _measure(engine, seconds, concurrency, prompt_len, out_len)))
    ok = [(c, r) for c, r in results
          if r["ttft_p90"] * 1000 <= cfg.ttft_target_ms and r["tpot_p90"] * 1000 <= cfg.tpot_target_ms]
    best = max(ok, key=lambda x: x[1]["throughput"])[0] if ok else min(
        results, key=lambda x: x[1]["tpot_p90"])[0]
    engine.scheduler.max_num_batched_tokens = best
    table.meta["chunk"] = best
    table.meta["chunk_sweep"] = {str(c): r for c, r in results}
    if save:
        tuner.save_cache(cfg.tuning_dir, tuning_key(cfg, None), table)
    log("chunk sweep", chosen=best, results={str(c): r for c, r in results})
    return best


def _measure(engine, seconds: float, concurrency: int, prompt_len: int, out_len: int) -> dict:
    g = torch.Generator().manual_seed(0)
    params = SamplingParams(temperature=0.0, max_tokens=out_len, ignore_eos=True)
    live: dict[str, tuple[float, float | None]] = {}      # rid -> (start, first_token_time)
    ttft, tpot, done_tokens = [], [], 0
    t_start = time.perf_counter()
    n = 0

    def top_up():
        nonlocal n
        while len(live) < concurrency:
            rid = f"sweep-{n}"
            n += 1
            ids = torch.randint(1000, 100000, (prompt_len,), generator=g).tolist()
            engine.add_request(rid, ids, params)
            live[rid] = (time.perf_counter(), None)

    while time.perf_counter() - t_start < seconds:
        top_up()
        for out in engine.step():
            if out.request_id not in live:
                continue
            start, first = live[out.request_id]
            now = time.perf_counter()
            if first is None and out.delta_token_ids:
                first = now
                ttft.append(now - start)
                live[out.request_id] = (start, first)
            done_tokens += len(out.delta_token_ids)
            if out.finished:
                if first is not None and out.completion_tokens > 1:
                    tpot.append((now - first) / (out.completion_tokens - 1))
                del live[out.request_id]
    for rid in list(live):                                  # drain
        engine.abort(rid)
    while engine.has_unfinished():
        engine.step()
    elapsed = time.perf_counter() - t_start
    return {"throughput": done_tokens / elapsed, "ttft_p90": percentile(ttft, 90),
            "tpot_p90": percentile(tpot, 90) if tpot else float("inf")}
