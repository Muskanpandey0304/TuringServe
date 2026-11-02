"""Triton KV-cache write: exact copy of new K/V into their paged slots; slot < 0 is skipped."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _kv_write_kernel(k_ptr, v_ptr, kc_ptr, vc_ptr, slot_ptr, stride_kt, stride_kh, stride_vt,
                     stride_vh, KVH: tl.constexpr, D: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    slot = tl.load(slot_ptr + t)
    if slot < 0:
        return
    block = slot // 16
    off = slot % 16
    d = tl.arange(0, D)
    dst = ((block * KVH + h) * 16 + off) * D + d
    k = tl.load(k_ptr + t * stride_kt + h * stride_kh + d)
    v = tl.load(v_ptr + t * stride_vt + h * stride_vh + d)
    tl.store(kc_ptr + dst, k)
    tl.store(vc_ptr + dst, v)


def kv_write(k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
             slot_mapping: torch.Tensor) -> None:
    """k, v: [T, KVH, D] (may be strided views of the qkv tensor); caches [NB, KVH, 16, D]."""
    T, KVH, D = k.shape
    if T == 0:
        return
    _kv_write_kernel[(T, KVH)](k, v, k_cache, v_cache, slot_mapping, k.stride(0), k.stride(1),
                               v.stride(0), v.stride(1), KVH=KVH, D=D, num_warps=1)
