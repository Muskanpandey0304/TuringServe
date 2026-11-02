"""Measured memory bandwidth B_meas (Build Guide 0.9): two fp16 tensors of 256 MiB, 20 copies."""

from __future__ import annotations

import torch


def measure() -> float:
    n = 256 * 1024**2 // 2
    src = torch.randn(n, device="cuda", dtype=torch.float16)
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()
    seconds = start.elapsed_time(end) / 1e3
    return 2 * src.numel() * 2 * 20 / seconds / 1e9        # read + write


if __name__ == "__main__":
    print(f"{torch.cuda.get_device_name(0)}: B_meas = {measure():.1f} GB/s")
