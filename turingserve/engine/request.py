"""A request as it travels from the API to the engine."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from turingserve.engine.sampling_params import SamplingParams


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    params: SamplingParams
    arrival_time: float = field(default_factory=time.perf_counter)
