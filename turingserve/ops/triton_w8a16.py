"""W8A16 GEMM: y = x @ (Wq * scale)^T + bias. int8 -> fp16 in registers, tl.dot, fp32 accumulator,
the per-output-row scale applied once at the end. ARCH 6.9 / 6.14."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _w8a16_kernel(x_ptr, w_ptr, scale_ptr, bias_ptr, y_ptr, M, N, K,
                  HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    mmask = offs_m < M
    nmask = offs_n < N
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + offs_k
        kmask = kk < K
        a = tl.load(x_ptr + offs_m[:, None].to(tl.int64) * K + kk[None, :],
                    mask=mmask[:, None] & kmask[None, :], other=0.0)               # [BM, BK] fp16
        w = tl.load(w_ptr + offs_n[None, :].to(tl.int64) * K + kk[:, None],
                    mask=nmask[None, :] & kmask[:, None], other=0)                 # [BK, BN] int8
        acc += tl.dot(a, w.to(a.dtype))
    scale = tl.load(scale_ptr + offs_n, mask=nmask, other=0.0)
    acc = acc * scale[None, :]
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=nmask, other=0.0).to(tl.float32)[None, :]
    tl.store(y_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :],
             acc.to(y_ptr.dtype.element_ty), mask=mmask[:, None] & nmask[None, :])


def w8a16_linear(x: torch.Tensor, wq: torch.Tensor, scale: torch.Tensor,
                 bias: torch.Tensor | None, cfg: tuple[int, int, int, int, int] = (16, 64, 64, 4, 2)
                 ) -> torch.Tensor:
    """cfg = (BM, BN, BK, num_warps, num_stages)."""
    x = x.contiguous()
    M, K = x.shape
    N = wq.shape[0]
    BM, BN, BK, warps, stages = cfg
    y = torch.empty(M, N, dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _w8a16_kernel[grid](x, wq, scale, bias if bias is not None else x, y, M, N, K,
                        HAS_BIAS=bias is not None, BM=BM, BN=BN, BK=BK, num_warps=warps,
                        num_stages=stages)
    return y
