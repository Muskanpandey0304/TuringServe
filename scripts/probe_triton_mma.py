"""Does Triton emit tensor-core instructions on this GPU? (ARCH 2.4, Build Guide 0.8)

Matmul exactly as in the official Triton tutorial 03: fp16 inputs, fp32 accumulator, tl.dot,
BLOCK 64x64x32, GROUP_M 8, 4 warps, 2 stages, M=N=K=2048. Checks the PTX for ``mma.sync``,
the result against torch, and prints TFLOPS for the kernel and for torch.matmul.
"""

from __future__ import annotations

import sys

import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                  stride_cm, stride_cn, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_in_group = GROUP_M * num_pid_n
    group_id = pid // num_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_in_group) % group_size_m)
    pid_n = (pid % num_in_group) // group_size_m
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


def main() -> int:
    torch.manual_seed(0)
    M = N = K = 2048
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = torch.empty(M, N, device="cuda", dtype=torch.float16)
    grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)

    def run():
        return matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0),
                                   b.stride(1), c.stride(0), c.stride(1), BLOCK_M=64, BLOCK_N=64,
                                   BLOCK_K=32, GROUP_M=8, num_warps=4, num_stages=2)

    handle = run()
    ptx = handle.asm["ptx"] if hasattr(handle, "asm") else ""
    has_mma = "mma.sync" in ptx
    print(f"triton {triton.__version__}  device {torch.cuda.get_device_name(0)}")
    print(f"mma.sync present: {has_mma}")
    torch.testing.assert_close(c, a @ b, atol=1e-2, rtol=1e-2)
    print("result matches torch.matmul (atol=rtol=1e-2)")
    flops = 2 * M * N * K
    t_triton = triton.testing.do_bench(run) / 1e3
    t_torch = triton.testing.do_bench(lambda: a @ b) / 1e3
    print(f"triton   {flops / t_triton / 1e12:7.2f} TFLOPS")
    print(f"torch    {flops / t_torch / 1e12:7.2f} TFLOPS")
    return 0 if has_mma else 1


if __name__ == "__main__":
    sys.exit(main())
