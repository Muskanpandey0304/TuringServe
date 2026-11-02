"""Prometheus metrics (ARCH 6.16).

Uses ``prometheus_client`` when installed; otherwise a tiny in-process implementation with the
same text exposition format so the server and its tests keep working.
"""

from __future__ import annotations

import math
import threading
import time

try:  # pragma: no cover - exercised when the package is installed
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
    HAVE_PROM = True
except ImportError:  # pragma: no cover
    HAVE_PROM = False

TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
TPOT_BUCKETS = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5)
E2E_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
STEP_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5)


class _Metric:
    def __init__(self, name, doc, labels=(), kind="gauge", buckets=()):
        self.name, self.doc, self.labels, self.kind, self.buckets = name, doc, tuple(labels), kind, buckets
        self.values: dict[tuple, float] = {}
        self.hist: dict[tuple, list] = {}
        self.lock = threading.Lock()

    def labels_(self, *vals, **kw):
        key = tuple(kw[l] for l in self.labels) if kw else tuple(vals)
        return _Child(self, key)

    # unlabeled convenience
    def inc(self, v=1.0):
        _Child(self, ()).inc(v)

    def set(self, v):
        _Child(self, ()).set(v)

    def observe(self, v):
        _Child(self, ()).observe(v)


class _Child:
    def __init__(self, m, key):
        self.m, self.key = m, key

    def inc(self, v=1.0):
        with self.m.lock:
            self.m.values[self.key] = self.m.values.get(self.key, 0.0) + v

    def set(self, v):
        with self.m.lock:
            self.m.values[self.key] = float(v)

    def observe(self, v):
        with self.m.lock:
            h = self.m.hist.setdefault(self.key, [[0] * len(self.m.buckets), 0, 0.0])
            for i, b in enumerate(self.m.buckets):
                if v <= b:
                    h[0][i] += 1
            h[1] += 1
            h[2] += v


def _fmt_labels(names, key, extra=""):
    parts = [f'{n}="{v}"' for n, v in zip(names, key, strict=False)]
    if extra:
        parts.append(extra)
    return "{" + ",".join(parts) + "}" if parts else ""


def _le(b) -> str:
    return f'le="{b}"'


def _render(metrics: list[_Metric]) -> bytes:
    out = []
    for m in metrics:
        out.append(f"# HELP {m.name} {m.doc}")
        out.append(f"# TYPE {m.name} {m.kind}")
        if m.kind == "histogram":
            for key, (counts, n, s) in m.hist.items():
                for b, c in zip(m.buckets, counts, strict=False):
                    out.append(f"{m.name}_bucket{_fmt_labels(m.labels, key, _le(b))} {c}")
                out.append(f"{m.name}_bucket{_fmt_labels(m.labels, key, _le('+Inf'))} {n}")
                out.append(f"{m.name}_sum{_fmt_labels(m.labels, key)} {s}")
                out.append(f"{m.name}_count{_fmt_labels(m.labels, key)} {n}")
            if not m.hist:
                for b in m.buckets:
                    out.append(f"{m.name}_bucket{_fmt_labels(m.labels, (), _le(b))} 0")
                out.append(f"{m.name}_bucket{_fmt_labels(m.labels, (), _le('+Inf'))} 0")
                out.append(f"{m.name}_sum 0.0")
                out.append(f"{m.name}_count 0")
        else:
            sample_name = m.name + "_total" if m.kind == "counter" and not m.name.endswith("_total") else m.name
            if not m.values and not m.labels:
                out.append(f"{sample_name} 0.0")
            for key, v in m.values.items():
                out.append(f"{sample_name}{_fmt_labels(m.labels, key)} {v}")
    return ("\n".join(out) + "\n").encode()


class Metrics:
    """All engine metrics. Cheap to call from the engine thread."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry() if HAVE_PROM else None
        mk = self._make
        self.ttft = mk("histogram", "turingserve_request_ttft_seconds", "Time to first token", buckets=TTFT_BUCKETS)
        self.tpot = mk("histogram", "turingserve_request_tpot_seconds", "Time per output token", buckets=TPOT_BUCKETS)
        self.e2e = mk("histogram", "turingserve_request_e2e_seconds", "End-to-end latency", buckets=E2E_BUCKETS)
        self.prompt_tokens = mk("counter", "turingserve_prompt_tokens_total", "Prompt tokens processed")
        self.generation_tokens = mk("counter", "turingserve_generation_tokens_total", "Tokens generated")
        self.requests = mk("counter", "turingserve_requests_total", "Finished requests", ["finish_reason"])
        self.num_running = mk("gauge", "turingserve_num_running", "Running sequences")
        self.num_waiting = mk("gauge", "turingserve_num_waiting", "Waiting sequences")
        self.kv_usage = mk("gauge", "turingserve_kv_usage_ratio", "KV blocks in use / total")
        self.prefix_queried = mk("counter", "turingserve_prefix_cache_queried_tokens_total", "Prompt tokens looked up in the prefix cache")
        self.prefix_hit = mk("counter", "turingserve_prefix_cache_hit_tokens_total", "Prompt tokens served from the prefix cache")
        self.preemptions = mk("counter", "turingserve_preemptions_total", "Preemptions")
        self.spec_draft = mk("counter", "turingserve_spec_draft_tokens_total", "Speculative tokens proposed")
        self.spec_accepted = mk("counter", "turingserve_spec_accepted_tokens_total", "Speculative tokens accepted")
        self.step_seconds = mk("histogram", "turingserve_step_seconds", "Engine step time by phase", ["phase"], STEP_BUCKETS)
        self.cpu_per_step = mk("histogram", "turingserve_engine_cpu_seconds_per_step", "Engine CPU time per step", buckets=STEP_BUCKETS)
        self.autopilot_decisions = mk("counter", "turingserve_autopilot_decisions_total", "Autopilot decisions", ["rule"])
        self.autopilot_setting = mk("gauge", "turingserve_autopilot_setting", "Current Autopilot setting", ["name"])
        self._last_prefix = (0, 0)
        self._last_preempt = 0

    def _make(self, kind, name, doc, labels=(), buckets=()):
        if HAVE_PROM:
            base = name[:-6] if kind == "counter" and name.endswith("_total") else name
            if kind == "counter":
                m = Counter(base, doc, labels, registry=self.registry)
            elif kind == "gauge":
                m = Gauge(name, doc, labels, registry=self.registry)
            else:
                m = Histogram(name, doc, labels, buckets=buckets, registry=self.registry)
            return _PromWrap(m, labels)
        m = _Metric(name, doc, labels, kind, buckets)
        self._fallback = getattr(self, "_fallback", []) + [m]
        return _FallbackWrap(m, labels)

    def render(self) -> bytes:
        if HAVE_PROM:
            return generate_latest(self.registry)
        return _render(self._fallback)

    # -- recording helpers -----------------------------------------------------------------------
    def record_finished(self, seq) -> None:
        self.requests.labels(finish_reason=seq.finish_reason or "unknown").inc()
        self.prompt_tokens.inc(seq.num_prompt_tokens)
        self.generation_tokens.inc(seq.num_output)
        if seq.first_token_time is not None:
            self.ttft.observe(seq.first_token_time - seq.arrival_time)
            if seq.num_output > 1 and seq.finish_time is not None:
                self.tpot.observe((seq.finish_time - seq.first_token_time) / (seq.num_output - 1))
        if seq.finish_time is not None:
            self.e2e.observe(seq.finish_time - seq.arrival_time)

    def record_step(self, engine, phases: dict[str, float], cpu_seconds: float) -> None:
        for k, v in phases.items():
            self.step_seconds.labels(phase=k).observe(v)
        self.cpu_per_step.observe(cpu_seconds)
        sch = engine.scheduler
        self.num_running.set(len(sch.running))
        self.num_waiting.set(len(sch.waiting))
        self.kv_usage.set(engine.kv.usage)
        pc = engine.prefix_cache
        q, h = pc.queried_tokens, pc.hit_tokens
        if q > self._last_prefix[0]:
            self.prefix_queried.inc(q - self._last_prefix[0])
        if h > self._last_prefix[1]:
            self.prefix_hit.inc(h - self._last_prefix[1])
        self._last_prefix = (q, h)
        if sch.num_preemptions > self._last_preempt:
            self.preemptions.inc(sch.num_preemptions - self._last_preempt)
            self._last_preempt = sch.num_preemptions


class _PromWrap:
    def __init__(self, m, labels):
        self.m, self.labelnames = m, tuple(labels)

    def labels(self, **kw):
        return self.m.labels(**kw)

    def __getattr__(self, item):
        return getattr(self.m, item)


class _FallbackWrap:
    def __init__(self, m, labels):
        self.m = m

    def labels(self, **kw):
        return self.m.labels_(**kw)

    def inc(self, v=1.0):
        self.m.inc(v)

    def set(self, v):
        self.m.set(v)

    def observe(self, v):
        self.m.observe(v)


def now() -> float:
    return time.perf_counter()


__all__ = ["Metrics", "math"]
