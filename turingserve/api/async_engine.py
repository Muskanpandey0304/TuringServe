"""Async bridge (ARCH 6.2): the engine runs in its own thread; the API talks to it through
thread-safe queues. Tokenization and chat templates run in the event-loop thread; detokenization
runs in the engine thread."""

from __future__ import annotations

import asyncio
import queue
import threading
import traceback
from collections.abc import AsyncIterator

from turingserve.engine.request import Request
from turingserve.engine.sequence import RequestOutput
from turingserve.utils.logging import log


class AsyncEngine:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.input_q: queue.Queue[Request] = queue.Queue()
        self.abort_q: queue.Queue[str] = queue.Queue()
        self.sinks: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.healthy = True
        self.thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------------------------------
    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, name="engine", daemon=True)
        self.thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout)

    @property
    def alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive() and self.healthy

    @property
    def num_waiting(self) -> int:
        return self.input_q.qsize() + len(self.engine.scheduler.waiting)

    # -- engine thread ----------------------------------------------------------------------------
    def _deliver(self, out: RequestOutput) -> None:
        with self._lock:
            sink = self.sinks.get(out.request_id)
            if sink is not None and out.finished:
                del self.sinks[out.request_id]
        if sink is not None:
            loop, q = sink
            try:
                loop.call_soon_threadsafe(q.put_nowait, out)
            except RuntimeError:        # loop closed during shutdown
                pass

    def _fail_all(self) -> None:
        for out in self.engine.fail_all():
            self._deliver(out)
        with self._lock:
            pending = list(self.sinks)
        for rid in pending:
            self._deliver(RequestOutput(rid, "", [], True, "error", 0, 0))

    def _add(self, req: Request) -> None:
        try:
            self.engine.add_request(req.request_id, req.prompt_token_ids, req.params,
                                    req.arrival_time)
        except Exception:
            log("add_request_failed", level="warning", error=traceback.format_exc())
            self._deliver(RequestOutput(req.request_id, "", [], True, "error", 0, 0))

    def _drain(self) -> None:
        while True:
            try:
                self.engine.abort(self.abort_q.get_nowait())
            except queue.Empty:
                break
        while True:
            try:
                req = self.input_q.get_nowait()
            except queue.Empty:
                break
            self._add(req)

    def _loop(self) -> None:
        import torch

        from turingserve.engine.engine import EngineUnhealthy

        eng = self.engine
        oom = getattr(torch.cuda, "OutOfMemoryError", MemoryError)
        while not self._stop.is_set():
            self._drain()
            if not eng.has_unfinished():
                try:
                    self._add(self.input_q.get(timeout=0.05))      # idle: block up to 50 ms
                except queue.Empty:
                    pass
                continue
            try:
                outputs = eng.step()
            except oom:
                try:
                    eng.recover_from_oom()
                except EngineUnhealthy:
                    log("engine_unhealthy", level="warning", reason="repeated CUDA OOM")
                    self.healthy = False
                    self._fail_all()
                    break
                continue
            except Exception:
                log("engine_exception", level="warning", stack=traceback.format_exc())
                self.healthy = False
                self._fail_all()
                break
            for out in outputs:
                self._deliver(out)
        if self._stop.is_set():                      # graceful shutdown: abort what is running
            for rid in list(eng.seqs):
                eng.abort(rid)
            for out in eng.step() if eng.has_unfinished() else []:
                self._deliver(out)

    # -- API side -------------------------------------------------------------------------------------
    async def generate(self, request: Request) -> AsyncIterator[RequestOutput]:
        if not self.alive:
            raise RuntimeError("engine is not running")
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self.sinks[request.request_id] = (loop, q)
        self.input_q.put(request)
        try:
            while True:
                out = await q.get()
                yield out
                if out.finished:
                    return
        finally:
            with self._lock:
                still_registered = self.sinks.pop(request.request_id, None) is not None
            if still_registered:                      # client went away before the end
                self.abort(request.request_id)

    def abort(self, request_id: str) -> None:
        self.abort_q.put(request_id)               # thread-safe and idempotent
