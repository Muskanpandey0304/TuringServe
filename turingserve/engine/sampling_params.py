"""SamplingParams (ARCH 5) with validation. Errors are ValueError that name the field."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    temperature: float = 1.0        # 0 -> greedy
    top_p: float = 1.0              # (0, 1]
    top_k: int = -1                 # -1 = off, else >= 1
    seed: int | None = None
    max_tokens: int = 16
    min_tokens: int = 0
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError("top_k must be -1 or >= 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.min_tokens < 0 or self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must be in [0, max_tokens]")
        if len(self.stop) > 16:
            raise ValueError("stop: at most 16 strings")
        if any((not isinstance(s, str)) or s == "" for s in self.stop):
            raise ValueError("stop: strings must be non-empty")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty must be in [-2, 2]")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty must be in [-2, 2]")

    @property
    def greedy(self) -> bool:
        return self.temperature < 1e-5

    @property
    def uses_penalties(self) -> bool:
        return (self.repetition_penalty != 1.0 or self.presence_penalty != 0.0
                or self.frequency_penalty != 0.0)

    @property
    def max_stop_len(self) -> int:
        return max((len(s) for s in self.stop), default=0)
