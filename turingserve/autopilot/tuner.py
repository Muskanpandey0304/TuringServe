"""Kernel tuner and tuning cache (ARCH 6.15, part 1).

Not ``@triton.autotune``: that cannot run inside CUDA graphs and does not persist reliably. Instead:
time each candidate config with ``triton.testing.do_bench`` per shape bucket, keep the fastest, and
store the winners in a JSON file keyed by GPU / model / dtype / lengths / torch / triton versions.
At runtime a kernel looks its bucket up (rounding up); on a miss it uses the first candidate and
logs once.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from turingserve.utils.logging import log

# ---- candidate lists (Turing-safe, ARCH 6.8 / 6.15) ---------------------------------------------
PREFILL_CANDIDATES = [
    {"block_m": bm, "block_n": bn, "num_warps": w, "num_stages": st}
    for bm in (32, 64) for bn in (32, 64) for w in (4, 8) for st in (1, 2)]
DECODE_CANDIDATES = [
    {"block_n": bn, "num_warps": w} for bn in (16, 32, 64) for w in (2, 4)]
SPLIT_PARTITIONS = (256, 512, 1024)
W8A16_CANDIDATES = [
    {"bm": bm, "bn": bn, "bk": bk, "num_warps": w, "num_stages": st}
    for (bm, bn, bk) in ((16, 64, 64), (16, 128, 64), (32, 64, 64), (64, 64, 32), (64, 128, 32),
                         (128, 64, 32))
    for w in (4, 8) for st in (1, 2)]

DEFAULT_PREFILL = {"block_m": 64, "block_n": 64, "num_warps": 4, "num_stages": 2}
DEFAULT_DECODE = {"block_n": 32, "num_warps": 4}
DEFAULT_W8A16 = {"bm": 16, "bn": 64, "bk": 64, "num_warps": 4, "num_stages": 2}
DEFAULT_M_CROSS = 256

W8A16_M_BUCKETS = (1, 16, 64, 256, 1024, 4096)


def pow2_buckets(lo: int, hi: int) -> list[int]:
    out, v = [], lo
    while v < hi:
        out.append(v)
        v *= 2
    out.append(hi)
    return out


def round_up_bucket(buckets: list[int], value: int) -> int:
    for b in buckets:
        if value <= b:
            return b
    return buckets[-1]


def cache_key(gpu: str, model: str, dtype_or_int8: str, max_model_len: int, max_batched: int,
              torch_v: str, triton_v: str) -> str:
    safe = lambda s: re.sub(r"[^A-Za-z0-9_.+-]", "-", s)  # noqa: E731
    return (f"{safe(gpu)}__{safe(model)}__{dtype_or_int8}__mml{max_model_len}__tok{max_batched}"
            f"__torch{safe(torch_v)}__triton{safe(triton_v)}.json")


@dataclass
class TuningTable:
    """kernel -> {bucket-string: config}. ``meta`` carries the cache key fields."""

    tables: dict[str, dict[str, dict]] = field(default_factory=dict)
    m_cross: dict[str, int] = field(default_factory=dict)       # "N,K" -> M_cross
    meta: dict = field(default_factory=dict)
    _missed: set = field(default_factory=set, repr=False)

    def set(self, kernel: str, bucket: str, cfg: dict) -> None:
        self.tables.setdefault(kernel, {})[bucket] = cfg

    def lookup(self, kernel: str, bucket: str, default: dict) -> dict:
        cfg = self.tables.get(kernel, {}).get(bucket)
        if cfg is None:
            if (kernel, bucket) not in self._missed:
                self._missed.add((kernel, bucket))
                log("tuning_miss", level="debug", kernel=kernel, bucket=bucket)
            return default
        return cfg

    def lookup_rounded(self, kernel: str, key: tuple[int, ...], default: dict) -> dict:
        """Tables are keyed 'a,b'; find the smallest recorded bucket >= key in each dimension."""
        table = self.tables.get(kernel)
        if not table:
            return default
        best = None
        for name, cfg in table.items():
            dims = tuple(int(x) for x in name.split(","))
            if all(d >= k for d, k in zip(dims, key, strict=False)) and (best is None or dims < best[0]):
                best = (dims, cfg)
        return best[1] if best else next(iter(table.values()))

    def to_json(self) -> str:
        return json.dumps({"tables": self.tables, "m_cross": self.m_cross, "meta": self.meta},
                          indent=1, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> TuningTable:
        d = json.loads(text)
        return cls(tables=d["tables"], m_cross=d.get("m_cross", {}), meta=d.get("meta", {}))


ACTIVE = TuningTable()       # consulted by ops.triton_ops


def set_active(table: TuningTable) -> None:
    global ACTIVE
    ACTIVE = table


def load_cache(tuning_dir: str, key: str) -> TuningTable | None:
    path = os.path.join(os.path.expanduser(tuning_dir), key)
    try:
        with open(path, encoding="utf-8") as f:
            table = TuningTable.from_json(f.read())
        if not isinstance(table.tables, dict):
            raise ValueError("bad table")
        return table
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None                       # missing or corrupt -> re-tune


def save_cache(tuning_dir: str, key: str, table: TuningTable) -> str:
    d = os.path.expanduser(tuning_dir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, key)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(table.to_json())
    os.replace(tmp, path)
    return path


# ---- the measuring part (needs a GPU) -------------------------------------------------------------
def _bench(fn) -> float:
    import triton

    return triton.testing.do_bench(fn, warmup=10, rep=50)


def _best(candidates: list[dict], run) -> dict | None:
    """``run(cfg)`` returns a zero-arg callable; configs that fail to compile are skipped."""
    best, best_t = None, float("inf")
    for cfg in candidates:
        try:
            t = _bench(run(cfg))
        except Exception as e:  # e.g. "out of resource: shared memory"
            log("tuner_skip", level="debug", cfg=cfg, error=str(e).splitlines()[0][:120])
            continue
        if t < best_t:
            best, best_t = cfg, t
    return best


def tune(mcfg, cfg_engine, device="cuda", quantized: bool = False) -> TuningTable:
    """Measure every kernel/bucket for this model on this GPU. Takes minutes; result is cached."""
    import torch

    from turingserve.ops import triton_decode_attention as dec
    from turingserve.ops import triton_prefill_attention as pre
    from turingserve.ops.triton_w8a16 import w8a16_linear
    from turingserve.runner.attention_metadata import AttentionMetadata
    from turingserve.utils.gpu import num_sms

    table = TuningTable()
    H, KVH, D = mcfg.num_heads, mcfg.num_kv_heads, mcfg.head_dim
    dtype = torch.float16
    scale = D ** -0.5
    nsm = num_sms(device)
    max_len = cfg_engine.max_model_len
    nb_total = 4 * (max_len // 16) + 8
    kc = torch.randn(nb_total, KVH, 16, D, device=device, dtype=dtype)
    vc = torch.randn_like(kc)

    # decode: S x max_seq_len buckets
    for S in [s for s in pow2_buckets(1, 128) if s <= cfg_engine.max_num_seqs]:
        for L in pow2_buckets(256, max(256, max_len)):
            nb = -(-L // 16)
            bt = (torch.arange(S * nb, device=device, dtype=torch.int32) % (nb_total - 1) + 1
                  ).view(S, nb)
            sl = torch.full((S,), L, device=device, dtype=torch.int32)
            q = torch.randn(S, H, D, device=device, dtype=dtype)
            out = torch.empty(S, H * D, device=device, dtype=dtype)
            run = lambda c: (lambda: dec.decode_attention_v2(  # noqa: E731
                q, out, kc, vc, bt, sl, scale, S, c["block_n"], c["num_warps"]))
            best = _best(DECODE_CANDIDATES, run)
            if best:
                table.set("decode_v2", f"{S},{L}", best)
            # split-K candidates
            for part in SPLIT_PARTITIONS:
                splits = -(-max_len // part)
                runs = lambda c, part=part, splits=splits: (lambda: dec.decode_attention_v3(  # noqa: E731
                    q, out, kc, vc, bt, sl, scale, S, splits, part, c["block_n"], c["num_warps"]))
                b3 = _best(DECODE_CANDIDATES, runs)
                if b3:
                    t3 = _bench(runs(b3))
                    prev = table.tables.get("decode_v3", {}).get(f"{S},{L}")
                    if prev is None or t3 < prev["time"]:
                        table.set("decode_v3", f"{S},{L}", {**b3, "partition": part, "time": t3})
    table.meta["split_threshold_programs"] = 2 * nsm

    # prefill: max_query_len buckets
    for Q in pow2_buckets(16, cfg_engine.max_num_batched_tokens):
        T = Q
        q = torch.randn(T, H, D, device=device, dtype=dtype)
        out = torch.empty(T, H * D, device=device, dtype=dtype)
        nb = -(-T // 16)
        meta = AttentionMetadata(
            query_start_loc=torch.tensor([0, T], device=device, dtype=torch.int32),
            seq_lens=torch.tensor([T], device=device, dtype=torch.int32),
            block_tables=torch.arange(1, nb + 1, device=device, dtype=torch.int32)[None],
            slot_mapping=torch.zeros(T, device=device, dtype=torch.int64),
            num_decode=0, max_query_len=T, max_seq_len=T)
        run = lambda c: (lambda: pre.prefill_attention(  # noqa: E731
            q, out, kc, vc, meta, scale, c["block_m"], c["block_n"], c["num_warps"],
            c["num_stages"]))
        best = _best(PREFILL_CANDIDATES, run)
        if best:
            table.set("prefill", str(Q), best)

    # w8a16 (only needed when quantized)
    if quantized:
        shapes = {(mcfg.qkv_dim, mcfg.hidden_size), (mcfg.hidden_size, mcfg.num_heads * D),
                  (2 * mcfg.intermediate_size, mcfg.hidden_size),
                  (mcfg.hidden_size, mcfg.intermediate_size)}
        for (N, K) in sorted(shapes):
            wq = torch.randint(-127, 127, (N, K), device=device, dtype=torch.int8)
            sc = torch.rand(N, device=device, dtype=torch.float32) / 127
            wdeq = (wq.to(dtype) * sc[:, None].to(dtype))
            m_cross = W8A16_M_BUCKETS[-1]
            for M in W8A16_M_BUCKETS:
                x = torch.randn(M, K, device=device, dtype=dtype)
                run = lambda c: (lambda: w8a16_linear(  # noqa: E731
                    x, wq, sc, None, (c["bm"], c["bn"], c["bk"], c["num_warps"], c["num_stages"])))
                best = _best(W8A16_CANDIDATES, run)
                if best:
                    table.set("w8a16", f"{N},{K},{M}", best)
                    t_w8 = _bench(run(best))
                    t_cublas = _bench(lambda: torch.nn.functional.linear(x, wdeq)) + _bench(
                        lambda: (wq.to(dtype) * sc[:, None].to(dtype)))
                    if t_cublas < t_w8 and m_cross == W8A16_M_BUCKETS[-1]:
                        m_cross = M
            table.m_cross[f"{N},{K}"] = m_cross
    log("tuning_done", kernels={k: len(v) for k, v in table.tables.items()})
    return table
