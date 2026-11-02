"""Triton in-place rotate-half RoPE on the q and k parts of qkv. ARCH 6.9."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(qkv_ptr, pos_ptr, cos_ptr, sin_ptr, stride_t, D: tl.constexpr, HALF: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1)
    pos = tl.load(pos_ptr + t).to(tl.int64)
    offs = tl.arange(0, HALF)
    base = qkv_ptr + t * stride_t + head * D
    x1 = tl.load(base + offs)
    x2 = tl.load(base + HALF + offs)
    # the cos/sin tables are fp32; HF casts them to the model dtype before use
    c = tl.load(cos_ptr + pos * HALF + offs).to(x1.dtype).to(tl.float32)
    s = tl.load(sin_ptr + pos * HALF + offs).to(x1.dtype).to(tl.float32)
    x1f, x2f = x1.to(tl.float32), x2.to(tl.float32)
    tl.store(base + offs, (x1f * c - x2f * s).to(x1.dtype))
    tl.store(base + HALF + offs, (x2f * c + x1f * s).to(x1.dtype))


def rope_(qkv: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
          H: int, KVH: int, D: int) -> None:
    T = qkv.shape[0]
    _rope_kernel[(T, H + KVH)](qkv, positions, cos, sin, qkv.stride(0), D=D, HALF=D // 2,
                               num_warps=1)
