"""Sequence, Scheduled, BatchPlan, RequestOutput (ARCH 5)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from turingserve.engine.sampling_params import SamplingParams


class SeqStatus(Enum):
    WAITING = 0
    RUNNING = 1
    FINISHED = 2


@dataclass
class Sequence:
    request_id: str
    token_ids: list[int]
    num_prompt_tokens: int
    params: SamplingParams
    num_computed_tokens: int = 0
    block_table: list[int] = field(default_factory=list)
    num_hashed_blocks: int = 0
    hash_chain: list[bytes] = field(default_factory=list)   # prefix-cache chain hash per full block
    draft_computed: int = 0
    status: SeqStatus = SeqStatus.WAITING
    generator: object | None = None
    detok: object | None = None
    text: str = ""
    emitted_len: int = 0
    finish_reason: str | None = None
    arrival_time: float = field(default_factory=time.perf_counter)
    first_scheduled_time: float | None = None
    first_token_time: float | None = None
    last_token_time: float | None = None
    finish_time: float | None = None
    num_cached_prompt_tokens: int = 0

    @property
    def num_uncomputed(self) -> int:
        return len(self.token_ids) - self.num_computed_tokens

    @property
    def num_output(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def output_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def finished(self) -> bool:
        return self.status == SeqStatus.FINISHED


@dataclass
class Scheduled:
    seq: Sequence
    num_tokens: int
    samples: bool          # the chunk reaches the end of token_ids


@dataclass
class BatchPlan:
    items: list[Scheduled] = field(default_factory=list)
    num_decode: int = 0
    num_tokens: int = 0
    preempted: list[Sequence] = field(default_factory=list)


@dataclass
class RequestOutput:
    request_id: str
    delta_text: str
    delta_token_ids: list[int]
    finished: bool
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int = 0
