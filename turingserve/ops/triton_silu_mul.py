"""Triton fused SiLU(gate) * up. ARCH 6.9."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_mul_kernel(gu_ptr, out_ptr, I, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < I
    g = tl.load(gu_ptr + row * 2 * I + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * 2 * I + I + offs, mask=mask, other=0.0)
    silu = (g * tl.sigmoid(g)).to(gu_ptr.dtype.element_ty)          # silu in fp32, rounded
    tl.store(out_ptr + row * I + offs, silu * u, mask=mask)


def silu_mul(gu: torch.Tensor) -> torch.Tensor:
    gu = gu.contiguous()
    T, two_i = gu.shape
    I = two_i // 2
    out = torch.empty(T, I, dtype=gu.dtype, device=gu.device)
    BLOCK = 1024
    _silu_mul_kernel[(T, triton.cdiv(I, BLOCK))](gu, out, I, BLOCK=BLOCK, num_warps=4)
    return out
