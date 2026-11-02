"""Starts a real ``turingserve serve`` subprocess on the tiny CPU model."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, model: str, extra: list[str] | None = None, name: str = "tiny") -> None:
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.name = name
        cmd = [sys.executable, "-m", "turingserve.cli", "serve", "--model", model,
               "--served-model-name", name, "--device", "cpu", "--kernels", "reference",
               "--dtype", "float32", "--port", str(self.port), "--host", "127.0.0.1",
               "--debug-endpoints", "--num-blocks", "512", "--max-model-len", "2048",
               "--log-level", "warning"] + (extra or [])
        env = {**os.environ, "PYTHONPATH": ROOT}
        self.proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("server died:\n" + self.proc.stdout.read().decode(errors="replace"))
            try:
                if httpx.get(self.url + "/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        self.stop()
        raise RuntimeError("server did not become healthy in 120 s")

    def state(self) -> dict:
        return httpx.get(self.url + "/debug/state").json()

    def stop(self) -> None:
        self.proc.kill()
        self.proc.wait(10)


@pytest.fixture(scope="session")
def server(tiny_model_dir):
    s = Server(tiny_model_dir)
    yield s
    s.stop()


@pytest.fixture(scope="session")
def overload_server(tiny_model_dir):
    s = Server(tiny_model_dir, ["--max-waiting-requests", "2", "--max-num-seqs", "1"])
    yield s
    s.stop()
