"""The Triton op namespace: thin wrappers that pick kernel versions and tuned configs."""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F

from turingserve.autopilot import tuner
from turingserve.model.layers import LinearWeight
from turingserve.ops.triton_decode_attention import (
    decode_attention_v1,
    decode_attention_v2,
    decode_attention_v3,
)
from turingserve.ops.triton_kv_write import kv_write
from turingserve.ops.triton_prefill_attention import prefill_attention
from turingserve.ops.triton_rmsnorm import rmsnorm, rmsnorm_residual
from turingserve.ops.triton_rope import rope_
from turingserve.ops.triton_silu_mul import silu_mul
from turingserve.ops.triton_w8a16 import w8a16_linear
from turingserve.quant.int8 import dequantize
from turingserve.utils.gpu import num_sms

__all__ = ["rmsnorm", "rmsnorm_residual", "silu_mul", "rope_", "kv_write", "linear",
           "paged_attention", "lm_head_int8"]


@functools.lru_cache(maxsize=8)
def _sms(device_index: int) -> int:
    return num_sms(f"cuda:{device_index}")


def _decode(q, out, k_cache, v_cache, meta, scale, version: str | None = None):
    S = meta.num_decode
    KVH = k_cache.shape[1]
    max_len = meta.max_seq_len
    qd, od = q[:S], out[:S]
    bt, sl = meta.block_tables, meta.seq_lens
    if version == "v1":
        decode_attention_v1(qd, od, k_cache, v_cache, bt, sl, scale, S)
        return
    nsm = _sms(q.device.index or 0)
    use_split = version == "v3" or (version is None and S * KVH < 2 * nsm and max_len > 1024)
    if use_split:
        c3 = tuner.ACTIVE.lookup_rounded("decode_v3", (S, max_len),
                                         {**tuner.DEFAULT_DECODE, "partition": 512})
        part = c3.get("partition", 512)
        # inside CUDA graphs the split count is fixed by max_model_len (max_seq_len == it there)
        splits = -(-max_len // part)
        decode_attention_v3(qd, od, k_cache, v_cache, bt, sl, scale, S, splits, part,
                            c3.get("block_n", 32), c3.get("num_warps", 4))
    else:
        c = tuner.ACTIVE.lookup_rounded("decode_v2", (S, max_len), tuner.DEFAULT_DECODE)
        decode_attention_v2(qd, od, k_cache, v_cache, bt, sl, scale, S,
                            c["block_n"], c["num_warps"])


def paged_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, meta,
                    scale: float) -> torch.Tensor:
    """q [T,H,D] -> [T, H*D]. Rows [0, num_decode) use the decode kernel, the rest prefill."""
    T, H, D = q.shape
    out = torch.empty(T, H * D, dtype=q.dtype, device=q.device)
    if meta.num_decode > 0:
        _decode(q, out, k_cache, v_cache, meta, scale)
    if meta.seq_lens.shape[0] > meta.num_decode:
        c = tuner.ACTIVE.lookup_rounded("prefill", (meta.max_query_len,), tuner.DEFAULT_PREFILL)
        prefill_attention(q, out, k_cache, v_cache, meta, scale, c["block_m"], c["block_n"],
                          c["num_warps"], c["num_stages"])
    return out


def linear(x: torch.Tensor, lw: LinearWeight) -> torch.Tensor:
    if not lw.quantized:
        return F.linear(x, lw.w, lw.b)          # cuBLAS: tensor cores, near peak; not rewritten
    N, K = lw.w.shape
    M = x.shape[0]
    m_cross = tuner.ACTIVE.m_cross.get(f"{N},{K}", tuner.DEFAULT_M_CROSS)
    if M <= m_cross:
        mb = tuner.round_up_bucket(list(tuner.W8A16_M_BUCKETS), M)
        c = tuner.ACTIVE.tables.get("w8a16", {}).get(f"{N},{K},{mb}", tuner.DEFAULT_W8A16)
        return w8a16_linear(x, lw.w, lw.scale, lw.b,
                            (c["bm"], c["bn"], c["bk"], c["num_warps"], c["num_stages"]))
    return F.linear(x, dequantize(lw.w, lw.scale, x.dtype), lw.b)    # large M: compute-bound


def lm_head_int8(h: torch.Tensor, wq: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if h.shape[0] <= tuner.DEFAULT_M_CROSS:
        return w8a16_linear(h, wq, scale, None)
    return F.linear(h, dequantize(wq, scale, h.dtype))
