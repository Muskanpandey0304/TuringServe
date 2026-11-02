"""SSE helpers (ARCH 6.1). Each event is ``data: <json>\\n\\n``; the stream ends with ``data: [DONE]``."""

from __future__ import annotations

import json
import time


def sse(obj) -> str:
    return "data: " + (obj if isinstance(obj, str) else json.dumps(obj, separators=(",", ":"))) + "\n\n"


DONE = "data: [DONE]\n\n"


def chat_chunk(rid: str, model: str, created: int, delta: dict, finish: str | None) -> dict:
    return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def completion_chunk(rid: str, model: str, created: int, text: str, finish: str | None) -> dict:
    return {"id": rid, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": finish}]}


def usage_chunk(rid: str, model: str, created: int, usage: dict, chat: bool) -> dict:
    return {"id": rid, "object": "chat.completion.chunk" if chat else "text_completion",
            "created": created, "model": model, "choices": [], "usage": usage}


def now() -> int:
    return int(time.time())
