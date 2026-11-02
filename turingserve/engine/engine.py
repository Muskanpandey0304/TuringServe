"""LLMEngine (ARCH 6.3): one step = one forward pass over a flat token batch."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import torch

from turingserve.config import EngineConfig
from turingserve.engine.output_processor import OutputProcessor
from turingserve.engine.sampling_params import SamplingParams
from turingserve.engine.sequence import RequestOutput, Sequence, SeqStatus
from turingserve.kv.kv_manager import KVCacheManager
from turingserve.kv.prefix_cache import PrefixCache
from turingserve.metrics.prometheus import Metrics
from turingserve.metrics.tracing import ProfilerHook
from turingserve.sampling.sampler import Sampler
from turingserve.scheduler.scheduler import Scheduler
from turingserve.utils.logging import log
from turingserve.utils.timing import StepTimer


@dataclass
class FinalOutput:
    request_id: str
    text: str
    token_ids: list[int]
    finish_reason: str | None
    prompt_tokens: int
    cached_prompt_tokens: int = 0
    pieces: list[str] = field(default_factory=list)


class EngineUnhealthy(RuntimeError):
    pass


class LLMEngine:
    def __init__(self, cfg: EngineConfig, runner, kv: KVCacheManager, prefix_cache: PrefixCache,
                 scheduler: Scheduler, sampler: Sampler, output_processor: OutputProcessor,
                 eos_ids: tuple[int, ...], metrics: Metrics | None = None, device="cpu") -> None:
        self.cfg, self.runner, self.kv, self.prefix_cache = cfg, runner, kv, prefix_cache
        self.scheduler, self.sampler, self.output_processor = scheduler, sampler, output_processor
        self.eos_ids = eos_ids
        self.metrics = metrics or Metrics()
        self.device = device
        self.seqs: dict[str, Sequence] = {}
        self._aborts: list[str] = []
        self.timer = StepTimer()
        self.spec = None             # SpecDecoder, set by from_config / tests
        self.autopilot = None        # Controller wrapper, set by from_config
        self.profiler = ProfilerHook(cfg.profile_dir)
        self.oom_times: deque[float] = deque()
        self.step_count = 0

    # -- construction ---------------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: EngineConfig) -> LLMEngine:
        from turingserve.engine.builder import build_engine

        return build_engine(cfg)

    # -- requests -----------------------------------------------------------------------------------
    def add_request(self, request_id: str, prompt_token_ids: list[int], params: SamplingParams,
                    arrival_time: float | None = None) -> Sequence:
        if request_id in self.seqs:
            raise ValueError(f"duplicate request id {request_id}")
        gen = None
        if params.seed is not None:
            gen = torch.Generator(device=self.device)
            gen.manual_seed(params.seed)
        seq = Sequence(request_id=request_id, token_ids=list(prompt_token_ids),
                       num_prompt_tokens=len(prompt_token_ids), params=params, generator=gen,
                       detok=self.output_processor.new_detokenizer())
        if arrival_time is not None:
            seq.arrival_time = arrival_time
        self.seqs[request_id] = seq
        self.scheduler.add(seq)
        if self.autopilot is not None:
            self.autopilot.arrivals[request_id] = seq.arrival_time
        return seq

    def abort(self, request_id: str) -> None:
        self._aborts.append(request_id)

    def has_unfinished(self) -> bool:
        return bool(self._aborts) or self.scheduler.has_unfinished()

    @property
    def num_requests_tracked(self) -> int:
        return len(self.seqs)

    # -- one step -----------------------------------------------------------------------------------
    def step(self) -> list[RequestOutput]:
        cpu0 = time.process_time()
        t = self.timer
        t.reset()
        self.profiler.before_step()
        outputs = self._process_aborts()

        t.start("schedule")
        plan = self.scheduler.schedule()
        t.stop("schedule")
        if not plan.items:
            return outputs + self._fail_unschedulable()

        if self.spec is not None and self.spec.should_run(plan):
            outputs += self.spec.step(plan)
        else:
            outputs += self._normal_step(plan)

        self.step_count += 1
        self.metrics.record_step(self, dict(t.t), time.process_time() - cpu0)
        if self.autopilot is not None:
            self.autopilot.observe(self, outputs)
        self.profiler.after_step()
        return outputs

    def _normal_step(self, plan) -> list[RequestOutput]:
        t = self.timer
        for it in plan.items:
            if it.seq.first_scheduled_time is None:
                it.seq.first_scheduled_time = time.perf_counter()
        t.start("forward")
        logits = self.runner.execute(plan)                       # [S, V] fp32 or None
        t.stop("forward")
        sampling = [it.seq for it in plan.items if it.samples]
        tokens = bad = None
        t.start("sample")
        if sampling:
            tokens, bad = self.sampler.sample(logits, sampling)  # ONE GPU->CPU copy
        t.stop("sample")

        t.start("output")
        outputs: list[RequestOutput] = []
        for it in plan.items:
            it.seq.num_computed_tokens += it.num_tokens
            self.prefix_cache.register_full_blocks(it.seq)
        for row, seq in enumerate(sampling):
            if bad[row]:
                log("nan_logits", level="warning", request_id=seq.request_id,
                    prompt_len=seq.num_prompt_tokens, position=len(seq.token_ids))
                outputs.append(self._finish_with_error(seq))
                continue
            outputs.append(self.commit_token(seq, int(tokens[row])))
        self._finalize_finished(outputs)
        t.stop("output")
        return outputs

    def commit_token(self, seq: Sequence, token: int) -> RequestOutput:
        """Append a sampled token and run the output processor (stop checks, detok, finish)."""
        now = time.perf_counter()
        if seq.first_token_time is None:
            seq.first_token_time = now
        seq.last_token_time = now
        seq.token_ids.append(token)
        return self.output_processor.process(seq, token)

    def _finish_with_error(self, seq: Sequence) -> RequestOutput:
        seq.status = SeqStatus.FINISHED
        seq.finish_reason = "error"
        return self.output_processor.make_output(seq, "", [])

    def _finalize_finished(self, outputs: list[RequestOutput]) -> None:
        for out in outputs:
            if not out.finished:
                continue
            seq = self.seqs.pop(out.request_id, None)
            if seq is None:
                continue
            seq.finish_time = time.perf_counter()
            self.scheduler.finish(seq)
            self.metrics.record_finished(seq)
            seq.detok = None
            seq.generator = None
            if self.cfg.log_requests:
                log("request_finished", request_id=seq.request_id,
                    prompt_tokens=seq.num_prompt_tokens, completion_tokens=seq.num_output,
                    cached_tokens=seq.num_cached_prompt_tokens, finish_reason=seq.finish_reason,
                    ttft=None if seq.first_token_time is None
                    else round(seq.first_token_time - seq.arrival_time, 4),
                    e2e=round(seq.finish_time - seq.arrival_time, 4))

    def _process_aborts(self) -> list[RequestOutput]:
        outs: list[RequestOutput] = []
        ids, self._aborts = self._aborts, []
        for rid in ids:
            seq = self.seqs.get(rid)
            if seq is None or seq.finished:
                continue
            seq.status = SeqStatus.FINISHED
            seq.finish_reason = "abort"
            outs.append(self.output_processor.make_output(seq, "", []))
        self._finalize_finished(outs)
        return outs

    def _fail_unschedulable(self) -> list[RequestOutput]:
        """Nothing can be scheduled and nothing is running: the head request can never fit."""
        if self.scheduler.running or not self.scheduler.waiting:
            return []
        seq = self.scheduler.waiting[0]
        log("request_unschedulable", level="warning", request_id=seq.request_id,
            tokens=len(seq.token_ids))
        out = self._finish_with_error(seq)
        self._finalize_finished([out])
        return [out]

    # -- robustness ---------------------------------------------------------------------------------
    def recover_from_oom(self) -> None:
        """ARCH 6.17 / Autopilot R0: preempt everything, free the cache, halve the chunk size."""
        now = time.monotonic()
        self.oom_times.append(now)
        while self.oom_times and now - self.oom_times[0] > 60:
            self.oom_times.popleft()
        if len(self.oom_times) >= 3:
            raise EngineUnhealthy("3 CUDA OOMs within 60 s")
        self.scheduler.preempt_all()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        old = self.scheduler.max_num_batched_tokens
        self.scheduler.max_num_batched_tokens = max(256, old // 2)
        log("oom_recovery", level="warning", old_chunk=old,
            new_chunk=self.scheduler.max_num_batched_tokens)
        if self.autopilot is not None:
            self.autopilot.note_oom(self)

    def fail_all(self) -> list[RequestOutput]:
        outs = []
        for seq in list(self.seqs.values()):
            outs.append(self._finish_with_error(seq))
        self._finalize_finished(outs)
        return outs

    # -- offline helper -----------------------------------------------------------------------------
    def generate(self, prompts: list[list[int]], params: SamplingParams | list[SamplingParams]
                 ) -> list[FinalOutput]:
        plist = params if isinstance(params, list) else [params] * len(prompts)
        ids = [f"gen-{id(self)}-{i}-{time.monotonic_ns()}" for i in range(len(prompts))]
        for rid, p, sp in zip(ids, prompts, plist):
            self.add_request(rid, p, sp)
        finals = {rid: FinalOutput(rid, "", [], None, len(p)) for rid, p in zip(ids, prompts)}
        steps = 0
        while self.has_unfinished():
            for out in self.step():
                f = finals.get(out.request_id)
                if f is None:
                    continue
                f.text += out.delta_text
                f.token_ids += out.delta_token_ids
                f.pieces.append(out.delta_text)
                f.cached_prompt_tokens = out.cached_prompt_tokens
                if out.finished:
                    f.finish_reason = out.finish_reason
            steps += 1
            if steps > 10_000_000:
                raise RuntimeError("generate() did not terminate")
        return [finals[r] for r in ids]
