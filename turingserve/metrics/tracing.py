"""Step timers and the on-demand torch profiler (ARCH 6.16)."""

from __future__ import annotations

import os
import threading

from turingserve.utils.timing import StepTimer

__all__ = ["StepTimer", "ProfilerHook"]


class ProfilerHook:
    """``POST /debug/profile?steps=N`` arms this; the engine calls ``on_step()`` once per step."""

    def __init__(self, out_dir: str | None) -> None:
        self.out_dir = out_dir
        self.remaining = 0
        self.prof = None
        self.lock = threading.Lock()

    def arm(self, steps: int) -> str:
        if not self.out_dir:
            raise RuntimeError("start the server with --profile-dir to enable profiling")
        with self.lock:
            if self.remaining:
                raise RuntimeError("a profile is already running")
            os.makedirs(self.out_dir, exist_ok=True)
            self.remaining = steps
        return self.out_dir

    def before_step(self) -> None:
        if self.remaining and self.prof is None:
            import torch

            acts = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                acts.append(torch.profiler.ProfilerActivity.CUDA)
            self.prof = torch.profiler.profile(activities=acts)
            self.prof.__enter__()

    def after_step(self) -> None:
        if self.prof is None:
            return
        self.remaining -= 1
        if self.remaining <= 0:
            self.prof.__exit__(None, None, None)
            path = os.path.join(self.out_dir, "turingserve.trace.json")
            self.prof.export_chrome_trace(path)       # open at ui.perfetto.dev
            self.prof = None
            self.remaining = 0
