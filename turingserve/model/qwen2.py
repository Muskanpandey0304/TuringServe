"""Qwen2 forward pass on a flat token batch (ARCH 6.7)."""

from __future__ import annotations

import torch

from turingserve.model.layers import RotaryCache, Weights
from turingserve.model.qwen2_config import Qwen2Config


class Qwen2Model:
    def __init__(self, cfg: Qwen2Config, weights: Weights, ops, device,
                 dtype: torch.dtype = torch.float16, max_model_len: int | None = None) -> None:
        self.cfg = cfg
        self.w = weights
        self.ops = ops
        self.device = device
        self.dtype = dtype
        self.scale = cfg.head_dim ** -0.5
        self.rotary = RotaryCache(cfg.head_dim, cfg.rope_theta,
                                  max_model_len or cfg.max_position_embeddings, device)

    # -- shared layer pieces -------------------------------------------------------------------
    def _qkv(self, x, lw, positions):
        c = self.cfg
        qkv = self.ops.linear(x, lw.qkv)
        self.ops.rope_(qkv, positions, self.rotary.cos, self.rotary.sin,
                       c.num_heads, c.num_kv_heads, c.head_dim)
        T = qkv.shape[0]
        q = qkv[:, : c.num_heads * c.head_dim].view(T, c.num_heads, c.head_dim)
        k = qkv[:, c.num_heads * c.head_dim: (c.num_heads + c.num_kv_heads) * c.head_dim]
        v = qkv[:, (c.num_heads + c.num_kv_heads) * c.head_dim:]
        return q, k.view(T, c.num_kv_heads, c.head_dim), v.view(T, c.num_kv_heads, c.head_dim)

    def _mlp(self, x, lw):
        return self.ops.linear(self.ops.silu_mul(self.ops.linear(x, lw.gu)), lw.down)

    def _run(self, input_ids, positions, attn, return_hidden=False):
        """Exact ARCH 6.7 op order. ``attn(layer, q, k, v) -> [T, H*D]`` does the cache work."""
        c, ops, w = self.cfg, self.ops, self.w
        h = w.embed[input_ids]
        residual = None
        hiddens = [h] if return_hidden else None
        for l, lw in enumerate(w.layers):
            if residual is None:
                residual = h
                x = ops.rmsnorm(h, lw.w_in, c.rms_norm_eps)
            else:
                x, residual = ops.rmsnorm_residual(h, residual, lw.w_in, c.rms_norm_eps)
                if return_hidden:
                    hiddens.append(residual.clone())     # == full hidden state after layer l-1
            q, k, v = self._qkv(x, lw, positions)
            a = attn(l, q, k, v)
            h = ops.linear(a, lw.o)
            x, residual = ops.rmsnorm_residual(h, residual, lw.w_post, c.rms_norm_eps)
            h = self._mlp(x, lw)
        x, _ = ops.rmsnorm_residual(h, residual, w.w_norm, c.rms_norm_eps)
        if return_hidden:
            hiddens.append(x)
            return x, hiddens
        return x

    # -- Phase 1: dense, no cache ----------------------------------------------------------------
    @torch.inference_mode()
    def forward_dense(self, input_ids, positions, return_hidden=False):
        def attn(l, q, k, v):
            return self.ops.dense_causal_attention(q, k, v, self.scale)

        return self._run(input_ids, positions, attn, return_hidden)

    # -- Phase 2: contiguous cache ---------------------------------------------------------------
    @torch.inference_mode()
    def forward_contiguous(self, input_ids, positions, cache, start: int):
        n = input_ids.shape[0]

        def attn(l, q, k, v):
            cache.write(l, start, k, v)
            kk, vv = cache.view(l, start + n)
            return self.ops.attention_with_mask(q, kk, vv, self.scale)

        return self._run(input_ids, positions, attn)

    # -- Phase 3+: paged KV cache, write-then-attend ---------------------------------------------
    @torch.inference_mode()
    def forward(self, input_ids, positions, meta, kv_cache):
        """kv_cache: [L, 2, num_blocks, KVH, 16, D]."""
        def attn(l, q, k, v):
            k_cache, v_cache = kv_cache[l, 0], kv_cache[l, 1]
            self.ops.kv_write(k, v, k_cache, v_cache, meta.slot_mapping)       # WRITE first
            return self.ops.paged_attention(q, k_cache, v_cache, meta, self.scale)  # then ATTEND

        return self._run(input_ids, positions, attn)

    def compute_logits(self, hidden: torch.Tensor, sample_indices: torch.Tensor) -> torch.Tensor:
        """Logits only for the rows that sample a token: [S, V] fp32."""
        h = hidden[sample_indices]
        if self.w.embed_q is not None and hasattr(self.ops, "lm_head_int8"):
            return self.ops.lm_head_int8(h, *self.w.embed_q).float()
        return (h @ self.w.embed.T).float()
