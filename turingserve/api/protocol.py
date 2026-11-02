"""OpenAI-compatible request/response models and request validation (ARCH 6.1)."""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict

from turingserve.engine.sampling_params import SamplingParams


class APIError(Exception):
    def __init__(self, status: int, message: str, code: str, param: str | None = None,
                 type_: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.status, self.message, self.code, self.param, self.type = status, message, code, param, type_

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.type, "param": self.param,
                          "code": self.code}}


def bad_request(message: str, param: str | None = None, code: str = "invalid_request") -> APIError:
    return APIError(400, message, code, param)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")
    include_usage: bool = False


class _Base(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    # vLLM-compatible extensions
    top_k: int | None = None
    repetition_penalty: float | None = None
    min_tokens: int | None = None
    ignore_eos: bool | None = None
    stop_token_ids: list[int] | None = None


class ChatRequest(_Base):
    messages: list[dict[str, Any]] | None = None
    max_completion_tokens: int | None = None
    response_format: dict[str, Any] | None = None


class CompletionRequest(_Base):
    prompt: str | list[int] | list[str] | list[list[int]] | None = None


REJECTED = ("logprobs", "top_logprobs", "echo", "best_of", "suffix", "tools", "tool_choice",
            "functions", "function_call")


def _reject_unsupported(req: _Base) -> None:
    extras = req.model_extra or {}
    for f in REJECTED:
        if f in extras and extras[f] not in (None, False):
            raise bad_request(f"'{f}' is not supported", f, "unsupported_parameter")
    if req.n not in (None, 1):
        raise bad_request("n must be 1", "n", "unsupported_parameter")
    if extras.get("logit_bias"):
        raise bad_request("'logit_bias' is not supported", "logit_bias", "unsupported_parameter")


def _sampling_params(req: _Base, max_tokens: int) -> SamplingParams:
    stop = [req.stop] if isinstance(req.stop, str) else list(req.stop or [])
    kw: dict[str, Any] = dict(
        temperature=1.0 if req.temperature is None else req.temperature,
        top_p=1.0 if req.top_p is None else req.top_p,
        top_k=-1 if req.top_k is None else req.top_k, seed=req.seed, max_tokens=max_tokens,
        min_tokens=req.min_tokens or 0, stop=stop, stop_token_ids=list(req.stop_token_ids or []),
        ignore_eos=bool(req.ignore_eos),
        repetition_penalty=1.0 if req.repetition_penalty is None else req.repetition_penalty,
        presence_penalty=req.presence_penalty or 0.0, frequency_penalty=req.frequency_penalty or 0.0)
    try:
        return SamplingParams(**kw)
    except ValueError as e:
        msg = str(e)
        param = next((k for k in kw if msg.startswith(k) or msg.split(":")[0] == k), None)
        raise bad_request(msg, param) from e


def _check_lengths(prompt_len: int, max_tokens: int | None, max_model_len: int) -> int:
    if prompt_len == 0:
        raise bad_request("the prompt is empty (0 tokens)", "messages")
    if prompt_len >= max_model_len:
        raise APIError(400, f"prompt has {prompt_len} tokens; max_model_len is {max_model_len}",
                       "context_length_exceeded", "messages")
    if max_tokens is None:
        return max_model_len - prompt_len
    if max_tokens < 1:
        raise bad_request("max_tokens must be >= 1", "max_tokens")
    if prompt_len + max_tokens > max_model_len:
        raise APIError(400, f"prompt ({prompt_len}) + max_tokens ({max_tokens}) exceeds "
                            f"max_model_len ({max_model_len})", "context_length_exceeded",
                       "max_tokens")
    return max_tokens


def _chat_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if not isinstance(p, dict) or p.get("type") != "text" or "text" not in p:
                raise bad_request("only {'type':'text'} content parts are supported", "messages")
            parts.append(p["text"])
        return "".join(parts)
    raise bad_request("message content must be a string or a list of text parts", "messages")


def chat_to_internal(req: ChatRequest, tokenizer, max_model_len: int
                     ) -> tuple[list[int], SamplingParams]:
    _reject_unsupported(req)
    if req.response_format and req.response_format.get("type", "text") != "text":
        raise bad_request("only response_format {'type':'text'} is supported", "response_format",
                          "unsupported_parameter")
    if not req.messages:
        raise bad_request("'messages' is required", "messages")
    msgs = []
    for m in req.messages:
        if m.get("role") not in ("system", "user", "assistant"):
            raise bad_request(f"unknown role {m.get('role')!r}", "messages")
        msgs.append({"role": m["role"], "content": _chat_text(m.get("content", ""))})
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    requested = req.max_completion_tokens if req.max_completion_tokens is not None else req.max_tokens
    max_tokens = _check_lengths(len(ids), requested, max_model_len)
    return ids, _sampling_params(req, max_tokens)


def completion_to_internal(req: CompletionRequest, tokenizer, max_model_len: int
                           ) -> tuple[list[int], SamplingParams]:
    _reject_unsupported(req)
    p = req.prompt
    if p is None:
        raise bad_request("'prompt' is required", "prompt")
    if isinstance(p, str):
        ids = tokenizer.encode(p, add_special_tokens=False)
    elif isinstance(p, list) and p and all(isinstance(x, int) for x in p):
        ids = list(p)
    elif isinstance(p, list) and not p:
        ids = []
    else:
        raise bad_request("a list of prompts is not supported; send one string or one list of "
                          "token ids", "prompt")
    max_tokens = _check_lengths(len(ids), 16 if req.max_tokens is None else req.max_tokens,
                                max_model_len)
    return ids, _sampling_params(req, max_tokens)


# -- responses -----------------------------------------------------------------------------------
def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def usage(prompt: int, completion: int, cached: int) -> dict:
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "prompt_tokens_details": {"cached_tokens": cached}}


def finish_reason_of(reason: str | None) -> str | None:
    return "stop" if reason == "abort" else reason


def chat_response(rid: str, model: str, text: str, reason: str, p: int, c: int, cached: int) -> dict:
    return {"id": rid, "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": reason}], "usage": usage(p, c, cached)}


def completion_response(rid: str, model: str, text: str, reason: str, p: int, c: int,
                        cached: int) -> dict:
    return {"id": rid, "object": "text_completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": reason}],
            "usage": usage(p, c, cached)}
