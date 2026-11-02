"""Triton varlen paged causal prefill attention, FlashAttention-style (ARCH 6.8).

Grid (ceil(max_query_len / BLOCK_M), P, H). Query row i of prefill sequence p has absolute position
prefix + i, with prefix = seq_len - q_len. Mask: j <= prefix + i and j < seq_len. Online softmax;
tl.dot for Q.K^T and P.V; fp32 accumulators; P cast to the model dtype before the second dot.

Turing-safe configs only (ARCH 6.8): BLOCK_M, BLOCK_N in {32, 64}, warps in {4, 8}, stages in {1, 2}.
D = 128 with BLOCK_M = BLOCK_N = 64 uses Q 16 + K 16 + V 16 = 48 KiB <= 64 KiB of shared memory.
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _prefill_kernel(q_ptr, out_ptr, kc_ptr, vc_ptr, bt_ptr, sl_ptr, qsl_ptr, scale,
                    stride_qt, stride_ot, stride_bt, num_decode,
                    H: tl.constexpr, KVH: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    mblk = tl.program_id(0)
    p = tl.program_id(1)
    h = tl.program_id(2)
    b = (num_decode + p).to(tl.int64)
    kvh = h // G
    q_start = tl.load(qsl_ptr + b).to(tl.int64)
    q_len = tl.load(qsl_ptr + b + 1).to(tl.int64) - q_start
    seq_len = tl.load(sl_ptr + b).to(tl.int64)
    m0 = mblk * BLOCK_M
    if m0 >= q_len:
        return
    prefix = seq_len - q_len
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    mmask = offs_m < q_len
    q = tl.load(q_ptr + (q_start + offs_m)[:, None] * stride_qt + h * D + offs_d[None, :],
                mask=mmask[:, None], other=0.0)                         # [BLOCK_M, D]
    m = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    hi = tl.minimum(seq_len, prefix + m0 + BLOCK_M)
    for j0 in range(0, hi, BLOCK_N):
        j = j0 + offs_n
        jmask = j < seq_len
        phys = tl.load(bt_ptr + b * stride_bt + j // 16, mask=jmask, other=0).to(tl.int64)
        row = ((phys * KVH + kvh) * 16 + j % 16) * D
        k = tl.load(kc_ptr + row[None, :] + offs_d[:, None], mask=jmask[None, :], other=0.0)  # [D, N]
        scores = tl.dot(q, k) * scale                                  # [BLOCK_M, BLOCK_N]
        allowed = (j[None, :] <= (prefix + offs_m)[:, None]) & jmask[None, :]
        scores = tl.where(allowed, scores, float("-inf"))
        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - m_new)
        pr = tl.exp(scores - m_new[:, None])
        v = tl.load(vc_ptr + row[:, None] + offs_d[None, :], mask=jmask[:, None], other=0.0)
        l = l * alpha + tl.sum(pr, axis=1)
        acc = acc * alpha[:, None] + tl.dot(pr.to(v.dtype), v)
        m = m_new
    l = tl.where(l == 0.0, 1.0, l)
    out = (acc / l[:, None]).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + (q_start + offs_m)[:, None] * stride_ot + h * D + offs_d[None, :], out,
             mask=mmask[:, None])


def prefill_attention(q, out, k_cache, v_cache, meta, scale,
                      block_m: int = 64, block_n: int = 64, num_warps: int = 4, num_stages: int = 2):
    """q [T,H,D] (flat), out [T,H*D]; handles sequences num_decode .. B-1 of the batch."""
    _, H, D = q.shape
    KVH = k_cache.shape[1]
    P = meta.seq_lens.shape[0] - meta.num_decode
    if P <= 0:
        return
    grid = (triton.cdiv(meta.max_query_len, block_m), P, H)
    _prefill_kernel[grid](q, out, k_cache, v_cache, meta.block_tables, meta.seq_lens,
                          meta.query_start_loc, scale, q.stride(0), out.stride(0),
                          meta.block_tables.stride(0), meta.num_decode,
                          H=H, KVH=KVH, G=H // KVH, D=D, BLOCK_M=block_m, BLOCK_N=block_n,
                          num_warps=num_warps, num_stages=num_stages)
