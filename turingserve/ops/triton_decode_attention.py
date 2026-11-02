"""Triton paged decode attention (ARCH 6.8). Three versions, all kept and tested:

v1  grid (S, H)           one program per (sequence, query head); no tl.dot; any group size
v2  grid (S, KVH)         one program per KV head handles its whole GQA group (padded to 16 rows)
                          with tl.dot -> tensor cores; K/V are read once per group  [production]
v3  grid (S, KVH, SPLITS) split-K ("flash-decoding") + a reduce kernel; for few sequences and
                          long contexts. SPLITS is fixed inside CUDA graphs.

All use fp32 accumulators and an online softmax. Block ids are cast to int64 before multiplying
by strides. seq_len == 0 rows (graph padding) write zeros.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ------------------------------------------------------------------------------------------------
# v1
# ------------------------------------------------------------------------------------------------
@triton.jit
def _decode_v1_kernel(q_ptr, out_ptr, kc_ptr, vc_ptr, bt_ptr, sl_ptr, scale,
                      stride_qs, stride_os, stride_bt,
                      H: tl.constexpr, KVH: tl.constexpr, G: tl.constexpr, D: tl.constexpr):
    s = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    kvh = h // G
    seq_len = tl.load(sl_ptr + s)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, 16)
    q = tl.load(q_ptr + s * stride_qs + h * D + offs_d).to(tl.float32)
    zero = tl.sum(tl.zeros([16], dtype=tl.float32), axis=0)     # a scalar tensor (loop-carried)
    m = zero - float("inf")
    l = zero
    acc = tl.zeros([D], dtype=tl.float32)
    for i in range(0, tl.cdiv(seq_len, 16)):
        phys = tl.load(bt_ptr + s * stride_bt + i).to(tl.int64)
        tok = i * 16 + offs_n
        mask = tok < seq_len
        base = ((phys * KVH + kvh) * 16 + offs_n[:, None]) * D + offs_d[None, :]
        k = tl.load(kc_ptr + base, mask=mask[:, None], other=0.0).to(tl.float32)
        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(mask, scores, float("-inf"))
        m_new = tl.maximum(m, tl.max(scores, axis=0))
        alpha = tl.exp(m - m_new)
        p = tl.exp(scores - m_new)
        v = tl.load(vc_ptr + base, mask=mask[:, None], other=0.0).to(tl.float32)
        l = l * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m = m_new
    l = tl.where(l == 0.0, 1.0, l)
    tl.store(out_ptr + s * stride_os + h * D + offs_d, (acc / l).to(out_ptr.dtype.element_ty))


# ------------------------------------------------------------------------------------------------
# v2
# ------------------------------------------------------------------------------------------------
@triton.jit
def _decode_v2_kernel(q_ptr, out_ptr, kc_ptr, vc_ptr, bt_ptr, sl_ptr, scale,
                      stride_qs, stride_os, stride_bt,
                      H: tl.constexpr, KVH: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                      BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr):
    s = tl.program_id(0).to(tl.int64)
    kvh = tl.program_id(1)
    seq_len = tl.load(sl_ptr + s)
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    hmask = offs_h < G
    head = kvh * G + offs_h
    q = tl.load(q_ptr + s * stride_qs + head[:, None] * D + offs_d[None, :],
                mask=hmask[:, None], other=0.0)                      # [BLOCK_H, D]
    m = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)
    for start in range(0, seq_len, BLOCK_N):
        tok = start + offs_n
        tmask = tok < seq_len
        phys = tl.load(bt_ptr + s * stride_bt + tok // 16, mask=tmask, other=0).to(tl.int64)
        row = ((phys * KVH + kvh) * 16 + tok % 16) * D                 # [BLOCK_N]
        k = tl.load(kc_ptr + row[None, :] + offs_d[:, None], mask=tmask[None, :], other=0.0)  # [D, N]
        scores = tl.dot(q.to(k.dtype), k) * scale                       # [BLOCK_H, BLOCK_N] fp32
        scores = tl.where(tmask[None, :], scores, float("-inf"))
        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(scores - m_new[:, None])
        v = tl.load(vc_ptr + row[:, None] + offs_d[None, :], mask=tmask[:, None], other=0.0)  # [N, D]
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new
    l = tl.where(l == 0.0, 1.0, l)
    out = (acc / l[:, None]).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + s * stride_os + head[:, None] * D + offs_d[None, :], out, mask=hmask[:, None])


# ------------------------------------------------------------------------------------------------
# v3 (split-K) + reduce
# ------------------------------------------------------------------------------------------------
@triton.jit
def _decode_v3_kernel(q_ptr, m_ptr, l_ptr, acc_ptr, kc_ptr, vc_ptr, bt_ptr, sl_ptr, scale,
                      stride_qs, stride_bt,
                      H: tl.constexpr, KVH: tl.constexpr, G: tl.constexpr, D: tl.constexpr,
                      BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, PARTITION: tl.constexpr,
                      NUM_SPLITS: tl.constexpr):
    s = tl.program_id(0).to(tl.int64)
    kvh = tl.program_id(1)
    split = tl.program_id(2)
    seq_len = tl.load(sl_ptr + s)
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    hmask = offs_h < G
    head = kvh * G + offs_h
    q = tl.load(q_ptr + s * stride_qs + head[:, None] * D + offs_d[None, :],
                mask=hmask[:, None], other=0.0)
    m = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)   # empty split: m = -inf, l = 0
    l = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)
    lo = split * PARTITION
    hi = tl.minimum(lo + PARTITION, seq_len)
    for start in range(lo, hi, BLOCK_N):
        tok = start + offs_n
        tmask = tok < hi
        phys = tl.load(bt_ptr + s * stride_bt + tok // 16, mask=tmask, other=0).to(tl.int64)
        row = ((phys * KVH + kvh) * 16 + tok % 16) * D
        k = tl.load(kc_ptr + row[None, :] + offs_d[:, None], mask=tmask[None, :], other=0.0)
        scores = tl.dot(q.to(k.dtype), k) * scale
        scores = tl.where(tmask[None, :], scores, float("-inf"))
        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(scores - m_new[:, None])
        v = tl.load(vc_ptr + row[:, None] + offs_d[None, :], mask=tmask[:, None], other=0.0)
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new
    base = (s * H + head) * NUM_SPLITS + split                         # [BLOCK_H]
    tl.store(m_ptr + base, m, mask=hmask)
    tl.store(l_ptr + base, l, mask=hmask)
    tl.store(acc_ptr + base[:, None] * D + offs_d[None, :], acc, mask=hmask[:, None])


@triton.jit
def _decode_reduce_kernel(m_ptr, l_ptr, acc_ptr, out_ptr, stride_os,
                          H: tl.constexpr, D: tl.constexpr, NUM_SPLITS: tl.constexpr,
                          SPLITS_P2: tl.constexpr):
    s = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1)
    offs_j = tl.arange(0, SPLITS_P2)
    jmask = offs_j < NUM_SPLITS
    offs_d = tl.arange(0, D)
    base = (s * H + h) * NUM_SPLITS + offs_j
    m = tl.load(m_ptr + base, mask=jmask, other=float("-inf"))
    l = tl.load(l_ptr + base, mask=jmask, other=0.0)
    M = tl.max(m, axis=0)
    M = tl.where(M == float("-inf"), 0.0, M)              # no valid split (seq_len 0): avoid NaN
    w = tl.where(m == float("-inf"), 0.0, tl.exp(m - M))
    L = tl.sum(l * w, axis=0)
    acc = tl.load(acc_ptr + base[:, None] * D + offs_d[None, :], mask=jmask[:, None], other=0.0)
    out = tl.sum(acc * w[:, None], axis=0)
    L = tl.where(L == 0.0, 1.0, L)
    tl.store(out_ptr + s * stride_os + h * D + offs_d, (out / L).to(out_ptr.dtype.element_ty))


# ------------------------------------------------------------------------------------------------
# launch helpers
# ------------------------------------------------------------------------------------------------
def decode_attention_v1(q, out, k_cache, v_cache, block_tables, seq_lens, scale, S: int):
    _, H, D = q.shape
    KVH = k_cache.shape[1]
    _decode_v1_kernel[(S, H)](q, out, k_cache, v_cache, block_tables, seq_lens, scale,
                              q.stride(0), out.stride(0), block_tables.stride(0),
                              H=H, KVH=KVH, G=H // KVH, D=D, num_warps=2)


def decode_attention_v2(q, out, k_cache, v_cache, block_tables, seq_lens, scale, S: int,
                        block_n: int = 32, num_warps: int = 4):
    _, H, D = q.shape
    KVH = k_cache.shape[1]
    _decode_v2_kernel[(S, KVH)](q, out, k_cache, v_cache, block_tables, seq_lens, scale,
                                q.stride(0), out.stride(0), block_tables.stride(0),
                                H=H, KVH=KVH, G=H // KVH, D=D, BLOCK_H=16, BLOCK_N=block_n,
                                num_warps=num_warps, num_stages=1)


def decode_attention_v3(q, out, k_cache, v_cache, block_tables, seq_lens, scale, S: int,
                        num_splits: int, partition: int = 512, block_n: int = 32,
                        num_warps: int = 4):
    _, H, D = q.shape
    KVH = k_cache.shape[1]
    m = torch.empty(S, H, num_splits, dtype=torch.float32, device=q.device)
    l = torch.empty_like(m)
    acc = torch.empty(S, H, num_splits, D, dtype=torch.float32, device=q.device)
    _decode_v3_kernel[(S, KVH, num_splits)](
        q, m, l, acc, k_cache, v_cache, block_tables, seq_lens, scale, q.stride(0),
        block_tables.stride(0), H=H, KVH=KVH, G=H // KVH, D=D, BLOCK_H=16, BLOCK_N=block_n,
        PARTITION=partition, NUM_SPLITS=num_splits, num_warps=num_warps, num_stages=1)
    _decode_reduce_kernel[(S, H)](m, l, acc, out, out.stride(0), H=H, D=D, NUM_SPLITS=num_splits,
                                  SPLITS_P2=triton.next_power_of_2(num_splits), num_warps=2)
