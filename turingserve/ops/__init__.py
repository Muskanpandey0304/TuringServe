"""Op dispatch: ``get_ops("reference" | "triton")`` returns a namespace of functions.

Every op has a pure-PyTorch twin in ``reference.py``; the Triton implementations are checked
against it. Importing the Triton namespace needs a CUDA build of Triton.
"""

from __future__ import annotations

from types import SimpleNamespace


def _reference() -> SimpleNamespace:
    from turingserve.ops import reference as r

    return SimpleNamespace(
        name="reference", rmsnorm=r.rmsnorm, rmsnorm_residual=r.rmsnorm_residual,
        silu_mul=r.silu_mul, rope_=r.rope_, linear=r.linear, kv_write=r.kv_write,
        paged_attention=r.paged_attention, dense_causal_attention=r.dense_causal_attention,
        attention_with_mask=r.attention_with_mask)


def _triton() -> SimpleNamespace:
    from turingserve.ops import triton_ops as t

    ns = _reference()          # dense/masked attention helpers stay reference-only
    ns.name = "triton"
    ns.rmsnorm = t.rmsnorm
    ns.rmsnorm_residual = t.rmsnorm_residual
    ns.silu_mul = t.silu_mul
    ns.rope_ = t.rope_
    ns.linear = t.linear
    ns.kv_write = t.kv_write
    ns.paged_attention = t.paged_attention
    ns.lm_head_int8 = t.lm_head_int8
    return ns


def get_ops(kernels: str) -> SimpleNamespace:
    if kernels == "reference":
        return _reference()
    if kernels == "triton":
        return _triton()
    raise ValueError(f"unknown kernels backend {kernels!r}")
