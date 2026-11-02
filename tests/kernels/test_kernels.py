"""Every Triton kernel vs its PyTorch reference (Phase 7). Tolerances: ARCH 9. Run on a CUDA GPU:
    pytest tests/kernels -m gpu -q
"""

import pytest
import torch

pytestmark = pytest.mark.gpu
triton = pytest.importorskip("triton")

from turingserve.ops import reference as R  # noqa: E402
from turingserve.runner.attention_metadata import AttentionMetadata  # noqa: E402

DEV = "cuda"
HEADS = [(4, 2, 16), (14, 2, 64), (12, 2, 128), (8, 8, 64)]


def close(a, b, atol, rtol):
    torch.testing.assert_close(a.float(), b.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("T", [1, 7, 128, 2049])
@pytest.mark.parametrize("N", [64, 224, 896, 1536])
@pytest.mark.parametrize("res", [False, True])
def test_rmsnorm(T, N, res):
    from turingserve.ops import triton_rmsnorm as K

    x = torch.randn(T, N, device=DEV, dtype=torch.float16)
    r = torch.randn(T, N, device=DEV, dtype=torch.float16) if res else None
    w = torch.randn(N, device=DEV, dtype=torch.float16)
    o1, r1 = K.rmsnorm_residual(x, r, w, 1e-6)
    o2, r2 = R.rmsnorm_residual(x, r, w, 1e-6)
    close(o1, o2, 1e-3, 1e-3)
    close(r1, r2, 0, 0)


@pytest.mark.parametrize("T", [1, 33, 2048])
@pytest.mark.parametrize("I", [128, 4864, 8960])
def test_silu_mul(T, I):
    from turingserve.ops import triton_silu_mul as K

    gu = torch.randn(T, 2 * I, device=DEV, dtype=torch.float16)
    close(K.silu_mul(gu), R.silu_mul(gu), 1e-3, 1e-3)


@pytest.mark.parametrize("T", [1, 17, 2048])
@pytest.mark.parametrize("H,KVH,D", [(4, 2, 16), (14, 2, 64), (12, 2, 128)])
def test_rope(T, H, KVH, D):
    from turingserve.model.layers import RotaryCache
    from turingserve.ops import triton_rope as K

    rc = RotaryCache(D, 1e6, 4096, DEV)
    qkv = torch.randn(T, (H + 2 * KVH) * D, device=DEV, dtype=torch.float16)
    pos = torch.randint(0, 4096, (T,), device=DEV)
    a, b = qkv.clone(), qkv.clone()
    K.rope_(a, pos, rc.cos, rc.sin, H, KVH, D)
    R.rope_(b, pos, rc.cos, rc.sin, H, KVH, D)
    close(a, b, 2e-3, 2e-3)


@pytest.mark.parametrize("T", [1, 17, 300])
def test_kv_write_exact(T):
    from turingserve.ops import triton_kv_write as K

    KVH, D, NB = 2, 64, 64
    k = torch.randn(T, KVH, D, device=DEV, dtype=torch.float16)
    v = torch.randn(T, KVH, D, device=DEV, dtype=torch.float16)
    slots = torch.randperm(NB * 16, device=DEV)[:T].long()
    slots[::5] = -1
    kc1 = torch.randn(NB, KVH, 16, D, device=DEV, dtype=torch.float16)
    vc1 = torch.randn_like(kc1)
    kc2, vc2 = kc1.clone(), vc1.clone()
    K.kv_write(k, v, kc1, vc1, slots)
    R.kv_write(k, v, kc2, vc2, slots)
    assert torch.equal(kc1, kc2) and torch.equal(vc1, vc2)


def make_paged(seq_lens, H, KVH, D, q_lens=None, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    B = len(seq_lens)
    q_lens = q_lens or [1] * B
    nbs = [-(-s // 16) for s in seq_lens]
    total = sum(nbs) + 1
    perm = torch.randperm(total - 1, device=DEV, generator=g) + 1          # shuffled block ids
    kc = torch.randn(total, KVH, 16, D, device=DEV, dtype=torch.float16, generator=g)
    vc = torch.randn(total, KVH, 16, D, device=DEV, dtype=torch.float16, generator=g)
    bt = torch.zeros(B, max(nbs), dtype=torch.int32, device=DEV)
    off = 0
    for b, nb in enumerate(nbs):
        bt[b, :nb] = perm[off:off + nb].int()
        off += nb
    qsl = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32, device=DEV)
    T = sum(q_lens)
    q = torch.randn(T, H, D, device=DEV, dtype=torch.float16, generator=g)
    return q, kc, vc, qsl, torch.tensor(seq_lens, dtype=torch.int32, device=DEV), bt


def meta_of(qsl, sl, bt, num_decode, q_lens, T):
    return AttentionMetadata(qsl, sl, bt, torch.zeros(T, dtype=torch.int64, device=DEV),
                             num_decode, max(q_lens), int(sl.max()))


@pytest.mark.parametrize("H,KVH,D", HEADS)
@pytest.mark.parametrize("S", [1, 3, 64])
@pytest.mark.parametrize("version", ["v1", "v2", "v3"])
def test_decode_versions(H, KVH, D, S, version):
    from turingserve.ops import triton_ops

    lens = [[1, 2, 15, 16, 17, 100, 1000, 4096][(i * 3) % 8] for i in range(S)] + [0]
    q, kc, vc, qsl, sl, bt = make_paged(lens, H, KVH, D)
    n = len(lens)
    q = q[:n]
    meta = meta_of(qsl, sl, bt, n, [1] * n, n)
    out = torch.empty(n, H * D, device=DEV, dtype=torch.float16)
    triton_ops._decode(q, out, kc, vc, meta, D**-0.5, version)
    ref = R.paged_attention(q, kc, vc, meta, D**-0.5)
    close(out, ref, 1e-2, 1e-2)
    assert (out[-1] == 0).all()                                   # padding row (seq_len 0)


@pytest.mark.parametrize("H,KVH,D", HEADS)
@pytest.mark.parametrize("q_len,prefix", [(1, 0), (15, 16), (16, 37), (17, 1000), (300, 0),
                                          (2048, 16)])
@pytest.mark.parametrize("P", [1, 3])
def test_prefill(H, KVH, D, q_len, prefix, P):
    from turingserve.ops import triton_ops

    q_lens = [q_len] * P
    q, kc, vc, qsl, sl, bt = make_paged([prefix + q_len] * P, H, KVH, D, q_lens)
    meta = meta_of(qsl, sl, bt, 0, q_lens, q.shape[0])
    close(triton_ops.paged_attention(q, kc, vc, meta, D**-0.5),
          R.paged_attention(q, kc, vc, meta, D**-0.5), 1e-2, 1e-2)


@pytest.mark.parametrize("H,KVH,D", HEADS[:3])
@pytest.mark.parametrize("pos", [0, 14, 15, 16, 17, 31, 32, 99])
def test_needle_decode_and_prefill(H, KVH, D, pos):
    """A key equal to 100*q at ``pos`` gets weight ~1, so the output must equal that token's V."""
    from turingserve.ops import triton_ops

    L = 100
    q, kc, vc, qsl, sl, bt = make_paged([L], H, KVH, D)
    G = H // KVH
    for h in range(H):
        kvh = h // G
        blk = int(bt[0, pos // 16])
        kc[blk, kvh, pos % 16] = (100 * q[0, h]).to(torch.float16) if G == 1 else kc[blk, kvh, pos % 16]
    if G != 1:
        pytest.skip("needle construction needs one q per KV head (G=1)")
    meta = meta_of(qsl, sl, bt, 1, [1], 1)
    out = triton_ops.paged_attention(q, kc, vc, meta, 1.0)
    for h in range(H):
        want = vc[int(bt[0, pos // 16]), h // G, pos % 16]
        close(out[0, h * D:(h + 1) * D], want, 1e-2, 1e-2)


@pytest.mark.parametrize("M", [1, 7, 16, 64, 300, 2048])
@pytest.mark.parametrize("N,K", [(1152, 896), (896, 896), (9728, 896), (896, 4864),
                                 (2048, 1536), (1536, 1536), (17920, 1536), (1536, 8960)])
def test_w8a16(M, N, K):
    from turingserve.ops.triton_w8a16 import w8a16_linear
    from turingserve.quant.int8 import dequantize, quantize

    w = torch.randn(N, K, device=DEV) * 0.05
    wq, sc = quantize(w)
    x = torch.randn(M, K, device=DEV, dtype=torch.float16)
    y = w8a16_linear(x, wq, sc, None).float()
    ref = x.float() @ dequantize(wq, sc, torch.float32).T
    assert (y - ref).norm() / ref.norm() < 1e-3
