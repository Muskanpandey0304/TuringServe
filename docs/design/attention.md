# Attention kernels (turingserve/ops/triton_*attention.py)
Write-then-attend: each layer writes K/V into the paged cache first, then attention reads prefix + new keys from the
cache, so decode and prefill share one memory path. Online softmax keeps (m, l, acc) in fp32.
Decode v1: grid (S, H), no tl.dot. v2: grid (S, KVH), the GQA group (7 or 6 heads) padded to 16 rows so tl.dot runs
on tensor cores and K/V are read once per group. v3: split-K across the sequence + a reduce kernel (m, l, acc
combined with exp(m_j - M)); fixed split count inside CUDA graphs. Prefill: FlashAttention-style tiles over a paged
cache with a bottom-right causal mask (j <= prefix + i). Turing limits: 64 KiB shared memory, no cp.async.
