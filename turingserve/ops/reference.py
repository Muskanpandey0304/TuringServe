"""Reference (pure PyTorch) ops. Slow and obviously correct; every Triton kernel is tested
against these. HF numerics are copied exactly (ARCH 6.7)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from turingserve.model.layers import LinearWeight
from turingserve.quant.int8 import dequantize


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return w * x32.to(x.dtype)                       # cast BEFORE multiplying by the weight


def rmsnorm_residual(x: torch.Tensor, residual: torch.Tensor | None, w: torch.Tensor,
                     eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """residual = x + residual (fp16 add); out = norm(residual). residual None -> residual = x."""
    new_res = x if residual is None else x + residual
    return rmsnorm(new_res, w, eps), new_res


def silu_mul(gu: torch.Tensor) -> torch.Tensor:
    i = gu.shape[-1] // 2
    return F.silu(gu[..., :i]) * gu[..., i:]


def rope_(qkv: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
          H: int, KVH: int, D: int) -> None:
    """In-place rotate-half RoPE on the q and k parts of qkv [T, (H+2KVH)*D]."""
    T = qkv.shape[0]
    c = cos[positions].to(qkv.dtype)[:, None, :]       # fp32 table, cast to model dtype like HF
    s = sin[positions].to(qkv.dtype)[:, None, :]
    qk = qkv[:, : (H + KVH) * D].view(T, H + KVH, D)
    x1, x2 = qk[..., : D // 2], qk[..., D // 2:]
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    qk[..., : D // 2] = o1
    qk[..., D // 2:] = o2


def linear(x: torch.Tensor, lw: LinearWeight) -> torch.Tensor:
    w = dequantize(lw.w, lw.scale, x.dtype) if lw.quantized else lw.w
    return F.linear(x, w, lw.b)


def kv_write(k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
             slot_mapping: torch.Tensor) -> None:
    m = slot_mapping >= 0
    slots = slot_mapping[m]
    b = slots // 16
    o = slots % 16
    k_cache[b, :, o, :] = k[m]
    v_cache[b, :, o, :] = v[m]


def attention_with_mask(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        scale: float) -> torch.Tensor:
    """q [Tq,H,D], k/v [Tk,KVH,D] -> [Tq, H*D]. Explicit bottom-right causal mask (never
    ``is_causal=True`` when Tq != Tk)."""
    Tq, H, D = q.shape
    Tk, KVH, _ = k.shape
    G = H // KVH
    qf = q.float().permute(1, 0, 2)                                   # [H, Tq, D]
    kf = k.float().repeat_interleave(G, dim=1).permute(1, 0, 2)      # [H, Tk, D]
    vf = v.float().repeat_interleave(G, dim=1).permute(1, 0, 2)
    scores = torch.matmul(qf, kf.transpose(1, 2)) * scale            # [H, Tq, Tk]
    i = torch.arange(Tq, device=q.device)[:, None]
    j = torch.arange(Tk, device=q.device)[None, :]
    allowed = j <= (Tk - Tq) + i
    scores = scores.masked_fill(~allowed, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    out = torch.matmul(p, vf).permute(1, 0, 2)                        # [Tq, H, D]
    return out.reshape(Tq, H * D).to(q.dtype)


def dense_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           scale: float) -> torch.Tensor:
    """Phase-1 attention: q_len == kv_len, so SDPA ``is_causal=True`` is valid."""
    T, H, D = q.shape
    KVH = k.shape[1]
    G = H // KVH
    kk = k.repeat_interleave(G, dim=1)
    vv = v.repeat_interleave(G, dim=1)
    out = F.scaled_dot_product_attention(
        q.permute(1, 0, 2)[None], kk.permute(1, 0, 2)[None], vv.permute(1, 0, 2)[None],
        is_causal=True, scale=scale)
    return out[0].permute(1, 0, 2).reshape(T, H * D)


def paged_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, meta,
                    scale: float) -> torch.Tensor:
    """q [T,H,D]; caches [num_blocks, KVH, 16, D] -> [T, H*D]. Padding rows (seq_len 0) give 0."""
    T, H, D = q.shape
    KVH = k_cache.shape[1]
    out = torch.zeros(T, H * D, dtype=q.dtype, device=q.device)
    qsl = meta.query_start_loc.tolist()
    seq_lens = meta.seq_lens.tolist()
    for b, seq_len in enumerate(seq_lens):
        qs, qe = qsl[b], qsl[b + 1]
        if seq_len == 0 or qe == qs:
            continue
        nb = -(-seq_len // 16)
        blocks = meta.block_tables[b, :nb].long()
        kk = k_cache[blocks].permute(0, 2, 1, 3).reshape(nb * 16, KVH, D)[:seq_len]
        vv = v_cache[blocks].permute(0, 2, 1, 3).reshape(nb * 16, KVH, D)[:seq_len]
        out[qs:qe] = attention_with_mask(q[qs:qe], kk, vv, scale)
    return out
