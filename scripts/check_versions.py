"""Print the stack and fail (exit 1) unless it is the pinned one (ARCH 2.5, Build Guide 0.7).

Pinned: torch 2.6.0 + Triton 3.2.0. Fallback (TS_FALLBACK_STACK=1): torch 2.5.1 + Triton 3.1.0.
"""

from __future__ import annotations

import os
import sys

GiB = 1024**3


def main() -> int:
    import torch

    fallback = os.environ.get("TS_FALLBACK_STACK") == "1"
    want_torch, want_triton = ("2.5.1", "3.1.0") if fallback else ("2.6.0", "3.2.0")
    try:
        import triton
        triton_v = triton.__version__
    except ImportError:
        triton_v = "not installed"
    import tokenizers
    import transformers

    print(f"torch          {torch.__version__}")
    print(f"triton         {triton_v}")
    print(f"transformers   {transformers.__version__}")
    print(f"tokenizers     {tokenizers.__version__}")
    print(f"torch.cuda     {torch.version.cuda}  available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"device         {torch.cuda.get_device_name(0)}  capability={torch.cuda.get_device_capability(0)}")
        print(f"memory         free {free / GiB:.2f} GiB / total {total / GiB:.2f} GiB")
    ok = True
    try:
        from tokenizers.decoders import DecodeStream  # noqa: F401
        print("DecodeStream   OK")
    except ImportError:
        print("DecodeStream   MISSING (the offset-based fallback detokenizer is used)")
    if not torch.__version__.startswith(want_torch):
        print(f"FAIL: torch must be {want_torch}", file=sys.stderr)
        ok = False
    if not triton_v.startswith(want_triton):
        print(f"FAIL: triton must be {want_triton}"
              + (" (Triton >= 3.3 emits no MMA on Turing)" if triton_v[:3] >= "3.3" else ""),
              file=sys.stderr)
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
