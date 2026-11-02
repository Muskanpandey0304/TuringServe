"""Autopilot runtime controller (ARCH 6.15): rules, not reinforcement learning.

``decide(state, signals, now, limits)`` is a PURE function: no clocks, no globals. It is called every
2 s over a 10 s sliding window. A rule fires only after its condition holds in 3 consecutive
windows; afterwards that setting cools down for 10 s. The 0.7 band between R1 and R2 prevents
oscillation. Every decision is logged as one JSON line and exported as a metric.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from turingserve.utils.logging import log
from turingserve.utils.timing import percentile

CHUNK_LEVELS = (256, 512, 1024, 2048, 4096)
WINDOW_S = 10.0
INTERVAL_S = 2.0
CONSECUTIVE = 3
COOLDOWN_S = 10.0
SPEC_OFF_S = 60.0


@dataclass(frozen=True)
class Limits:
    ttft_target_s: float
    tpot_target_s: float
    max_chunk: int            # --max-num-batched-tokens
    max_num_seqs: int         # --max-num-seqs
    spec_k_max: int           # --spec-k
    spec_max_batch: int
    spec_enabled: bool = False
    min_chunk: int = 256
    min_seqs: int = 8


@dataclass(frozen=True)
class State:
    chunk: int
    max_num_seqs: int
    spec_k: int = 4
    spec_on: bool = True
    spec_off_until: float = 0.0
    spec_skipping: bool = False                               # R3 edge trigger
    streaks: tuple[tuple[str, int], ...] = ()
    cooldown_until: tuple[tuple[str, float], ...] = ()

    def streak(self, rule: str) -> int:
        return dict(self.streaks).get(rule, 0)

    def cooldown(self, setting: str) -> float:
        return dict(self.cooldown_until).get(setting, 0.0)


@dataclass(frozen=True)
class Signals:
    ttft_p90: float | None = None
    tpot_p90: float | None = None
    decode_batch: int = 0
    preemptions: int = 0
    kv_usage: float = 0.0
    accepted_ema: float | None = None       # mean accepted drafts per verify step
    oom: bool = False


@dataclass(frozen=True)
class Decision:
    rule: str
    setting: str
    old: object
    new: object
    signals: dict = field(default_factory=dict, compare=False)


def _level_down(chunk: int, lo: int) -> int:
    lower = [c for c in CHUNK_LEVELS if c < chunk and c >= lo]
    return lower[-1] if lower else max(lo, min(chunk, CHUNK_LEVELS[0]))


def _level_up(chunk: int, hi: int) -> int:
    upper = [c for c in CHUNK_LEVELS if c > chunk and c <= hi]
    return upper[0] if upper else min(chunk, hi)


def decide(state: State, sig: Signals, now: float, lim: Limits) -> tuple[State, list[Decision]]:
    decisions: list[Decision] = []
    streaks = dict(state.streaks)
    cooldown = dict(state.cooldown_until)
    chunk, seqs = state.chunk, state.max_num_seqs
    k, spec_on, off_until = state.spec_k, state.spec_on, state.spec_off_until
    skipping = state.spec_skipping
    sd = {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in vars(sig).items()}

    def fire(rule: str, setting: str, old, new) -> None:
        decisions.append(Decision(rule, setting, old, new, sd))
        cooldown[setting] = now + COOLDOWN_S
        streaks[rule] = 0

    def cond(rule: str, holds: bool, setting: str, need: int = CONSECUTIVE) -> bool:
        """Update the streak; True when the rule may fire now."""
        streaks[rule] = streaks.get(rule, 0) + 1 if holds else 0
        return holds and streaks[rule] >= need and now >= cooldown.get(setting, 0.0)

    # R0: OOM recovery, immediate
    if sig.oom:
        new = max(lim.min_chunk, chunk // 2)
        if new != chunk:
            fire("R0", "chunk", chunk, new)
            chunk = new
    else:
        tpot_hi = sig.tpot_p90 is not None and sig.tpot_p90 > lim.tpot_target_s
        # R1 smoother decoding
        if cond("R1", tpot_hi, "chunk"):
            new = _level_down(chunk, lim.min_chunk)
            if new != chunk:
                fire("R1", "chunk", chunk, new)
                chunk = new
            else:
                streaks["R1"] = 0
        # R2 faster first token (only with headroom on TPOT: the 0.7 band)
        ttft_hi = sig.ttft_p90 is not None and sig.ttft_p90 > lim.ttft_target_s
        tpot_low = sig.tpot_p90 is not None and sig.tpot_p90 < 0.7 * lim.tpot_target_s
        if cond("R2", ttft_hi and tpot_low, "chunk"):
            new = _level_up(chunk, lim.max_chunk)
            if new != chunk:
                fire("R2", "chunk", chunk, new)
                chunk = new
            else:
                streaks["R2"] = 0

    # R3: speculation off for a step when the decode batch is too large (edge-triggered record)
    if lim.spec_enabled:
        over = sig.decode_batch > lim.spec_max_batch
        if over and not skipping:
            decisions.append(Decision("R3", "spec_skip", False, True, sd))
        skipping = over

        # re-probe after a 60 s pause
        if not spec_on and off_until and now >= off_until:
            spec_on, k, off_until = True, min(2, lim.spec_k_max), 0.0
            decisions.append(Decision("R4", "spec_on", False, True, sd))
        a = sig.accepted_ema
        if spec_on and a is not None:
            # R4 fewer drafts
            if cond("R4", a < 0.6, "spec_k"):
                if k > 1:
                    fire("R4", "spec_k", k, k - 1)
                    k -= 1
                else:
                    fire("R4", "spec_on", True, False)
                    spec_on, off_until = False, now + SPEC_OFF_S
            # R5 more drafts
            if cond("R5", a > 0.8 * k, "spec_k") and k < lim.spec_k_max:
                fire("R5", "spec_k", k, k + 1)
                k += 1

    # R6 less memory pressure / R7 more concurrency
    if cond("R6", sig.preemptions > 5, "max_num_seqs"):
        new = max(lim.min_seqs, int(seqs * 0.75))
        if new != seqs:
            fire("R6", "max_num_seqs", seqs, new)
            seqs = new
        else:
            streaks["R6"] = 0
    calm = sig.preemptions == 0 and sig.kv_usage < 0.7
    if cond("R7", calm, "max_num_seqs", need=6):
        new = min(lim.max_num_seqs, math.ceil(seqs * 1.25))
        if new != seqs:
            fire("R7", "max_num_seqs", seqs, new)
            seqs = new
        else:
            streaks["R7"] = 0

    new_state = replace(state, chunk=chunk, max_num_seqs=seqs, spec_k=k, spec_on=spec_on,
                        spec_off_until=off_until, spec_skipping=skipping,
                        streaks=tuple(sorted(streaks.items())),
                        cooldown_until=tuple(sorted(cooldown.items())))
    return new_state, decisions


# ---------------------------------------------------------------------------------------------
# Window aggregation + engine glue (impure part)
# ---------------------------------------------------------------------------------------------
class Autopilot:
    """Collects TTFT/TPOT samples from engine outputs, calls ``decide`` every 2 s, applies the result."""

    def __init__(self, engine, limits: Limits, clock=None) -> None:
        import time

        self.clock = clock or time.monotonic
        self.limits = limits
        self.state = State(chunk=engine.scheduler.max_num_batched_tokens,
                           max_num_seqs=engine.scheduler.max_num_seqs,
                           spec_k=engine.spec.k if engine.spec else limits.spec_k_max)
        self.ttft: list[tuple[float, float]] = []       # (time, seconds)
        self.tpot: list[tuple[float, float]] = []
        self.last_tok: dict[str, float] = {}
        self.arrivals: dict[str, float] = {}
        self.preempt_log: list[tuple[float, int]] = []
        self.accept_ema: float | None = None
        self.oom_pending = False
        self.next_eval = self.clock() + INTERVAL_S
        self._last_preempts = engine.scheduler.num_preemptions
        self.decisions: list[Decision] = []

    def note_oom(self, engine) -> None:
        self.oom_pending = True
        self.state = replace(self.state, chunk=engine.scheduler.max_num_batched_tokens)

    def observe(self, engine, outputs) -> None:
        now = self.clock()
        for out in outputs:
            n = len(out.delta_token_ids)
            if n == 0:
                continue
            rid = out.request_id
            if rid not in self.last_tok:
                self.ttft.append((now, now - self.arrivals.get(rid, now)))
            else:
                self.tpot.append((now, (now - self.last_tok[rid]) / n))
            self.last_tok[rid] = now
            if out.finished:
                self.last_tok.pop(rid, None)
                self.arrivals.pop(rid, None)
        spec = engine.spec
        if spec is not None and spec.last_accept_mean is not None:
            a = spec.last_accept_mean
            self.accept_ema = a if self.accept_ema is None else 0.8 * self.accept_ema + 0.2 * a
            spec.last_accept_mean = None
        if now >= self.next_eval:
            self.next_eval = now + INTERVAL_S
            self.evaluate(engine, now)

    def signals(self, engine, now: float) -> Signals:
        lo = now - WINDOW_S
        self.ttft = [x for x in self.ttft if x[0] >= lo]
        self.tpot = [x for x in self.tpot if x[0] >= lo]
        pre = engine.scheduler.num_preemptions
        self.preempt_log.append((now, pre - self._last_preempts))
        self._last_preempts = pre
        self.preempt_log = [x for x in self.preempt_log if x[0] >= lo]
        decode_batch = sum(1 for s in engine.scheduler.running if s.num_uncomputed == 1)
        sig = Signals(
            ttft_p90=percentile([v for _, v in self.ttft], 90) if self.ttft else None,
            tpot_p90=percentile([v for _, v in self.tpot], 90) if self.tpot else None,
            decode_batch=decode_batch, preemptions=sum(c for _, c in self.preempt_log),
            kv_usage=engine.kv.usage, accepted_ema=self.accept_ema, oom=self.oom_pending)
        self.oom_pending = False
        return sig

    def evaluate(self, engine, now: float) -> None:
        sig = self.signals(engine, now)
        self.state, decisions = decide(self.state, sig, now, self.limits)
        for d in decisions:
            self.apply(engine, d)
        m = engine.metrics
        for name, val in (("chunk", self.state.chunk), ("max_num_seqs", self.state.max_num_seqs),
                          ("spec_k", self.state.spec_k)):
            m.autopilot_setting.labels(name=name).set(val)

    def apply(self, engine, d: Decision) -> None:
        sch = engine.scheduler
        if d.setting == "chunk":
            sch.max_num_batched_tokens = d.new
        elif d.setting == "max_num_seqs":
            sch.max_num_seqs = d.new
        elif d.setting == "spec_k" and engine.spec is not None:
            engine.spec.k = d.new
        elif d.setting == "spec_on" and engine.spec is not None:
            engine.spec.enabled = bool(d.new)
            if not d.new:
                import time

                engine.spec.off_until = time.monotonic() + SPEC_OFF_S
            else:
                engine.spec.off_until = 0.0
                engine.spec.k = self.state.spec_k
        self.decisions.append(d)
        engine.metrics.autopilot_decisions.labels(rule=d.rule).inc()
        log("autopilot_decision", rule=d.rule, setting=d.setting, old=d.old, new=d.new,
            signals=d.signals)
