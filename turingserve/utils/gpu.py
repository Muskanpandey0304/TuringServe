"""GPU helpers: device info, memory in GiB, version check (ARCH 2.5)."""

from __future__ import annotations

import os
import warnings

GiB = 1024**3


def device_name(device) -> str:
    import torch

    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(device)
    return "cpu"


def num_sms(device) -> int:
    import torch

    if str(device).startswith("cuda"):
        return torch.cuda.get_device_properties(device).multi_processor_count
    return 1


def check_stack(strict: bool = False) -> list[str]:
    """Warn (or fail, if strict) when torch/triton differ from the pinned stack."""
    import torch

    problems: list[str] = []
    fallback = os.environ.get("TS_FALLBACK_STACK") == "1"
    want_torch, want_triton = ("2.5.1", "3.1.0") if fallback else ("2.6.0", "3.2.0")
    if not torch.__version__.startswith(want_torch):
        problems.append(f"torch {torch.__version__} != pinned {want_torch}")
    try:
        import triton

        if not triton.__version__.startswith(want_triton):
            msg = f"triton {triton.__version__} != pinned {want_triton}"
            if tuple(int(x) for x in triton.__version__.split(".")[:2]) >= (3, 3):
                msg += " (Triton >= 3.3 emits no tensor-core MMA on Turing, ARCH 2.4)"
            problems.append(msg)
    except ImportError:
        pass
    for p in problems:
        warnings.warn(p, stacklevel=2)
    if strict and problems:
        raise RuntimeError("; ".join(problems))
    return problems
