"""Triton RMSNorm (+ fused residual add). ARCH 6.9."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, res_ptr, w_ptr, out_ptr, res_out_ptr, N, eps,
                    HAS_RES: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    if HAS_RES:
        r = tl.load(res_ptr + row * N + offs, mask=mask, other=0.0)
        # fp32 add, rounded back to the model dtype (== an fp16 add, as HF does)
        x = (x.to(tl.float32) + r.to(tl.float32)).to(x_ptr.dtype.element_ty)
        tl.store(res_out_ptr + row * N + offs, x, mask=mask)
    x32 = x.to(tl.float32)
    var = tl.sum(x32 * x32, axis=0) / N
    y = (x32 * tl.rsqrt(var + eps)).to(x_ptr.dtype.element_ty)      # cast BEFORE the weight
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    out = (w.to(tl.float32) * y.to(tl.float32)).to(x_ptr.dtype.element_ty)
    tl.store(out_ptr + row * N + offs, out, mask=mask)


def rmsnorm_residual(x: torch.Tensor, residual: torch.Tensor | None, w: torch.Tensor, eps: float
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """(out, new_residual). ``residual=None`` -> new_residual is ``x`` itself (no copy)."""
    x = x.contiguous()
    T, N = x.shape
    out = torch.empty_like(x)
    block = triton.next_power_of_2(N)
    warps = 4 if block <= 1024 else 8
    if residual is None:
        _rmsnorm_kernel[(T,)](x, x, w, out, out, N, eps, HAS_RES=False, BLOCK_N=block,
                              num_warps=warps)
        return out, x
    residual = residual.contiguous()
    res_out = torch.empty_like(x)
    _rmsnorm_kernel[(T,)](x, residual, w, out, res_out, N, eps, HAS_RES=True, BLOCK_N=block,
                          num_warps=warps)
    return out, res_out


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return rmsnorm_residual(x, None, w, eps)[0]
