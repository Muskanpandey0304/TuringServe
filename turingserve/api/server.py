"""FastAPI server (ARCH 6.1, 6.2, 6.17). One uvicorn worker: each worker would load its own engine."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from turingserve.api import protocol as P
from turingserve.api import streaming as S
from turingserve.api.async_engine import AsyncEngine
from turingserve.config import EngineConfig
from turingserve.engine.request import Request as EngineRequest
from turingserve.utils.logging import log


def create_app(cfg: EngineConfig, engine=None, tokenizer=None) -> FastAPI:
    """``engine``/``tokenizer`` can be injected (tests); otherwise they are built at startup."""
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        eng = engine
        tok = tokenizer
        if eng is None:
            from turingserve.engine.builder import build_engine, load_tokenizer

            tok = tok or load_tokenizer(cfg.model)
            eng = await asyncio.get_running_loop().run_in_executor(None, build_engine, cfg, tok)
        state["tokenizer"] = tok if tok is not None else eng.output_processor.tokenizer
        state["aeng"] = AsyncEngine(eng)
        state["aeng"].start()
        state["accepting"] = True
        yield
        state["accepting"] = False                    # shutdown: stop accepting, abort, join
        state["aeng"].stop(10.0)

    app = FastAPI(title="TuringServe", lifespan=lifespan)

    # -- errors --------------------------------------------------------------------------------------
    @app.exception_handler(P.APIError)
    async def _api_error(_: Request, e: P.APIError):
        headers = {"Retry-After": "1"} if e.status == 429 else None
        return JSONResponse(e.body(), status_code=e.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError):
        return JSONResponse(P.bad_request(str(e.errors()[0].get("msg", e)),
                                          ".".join(map(str, e.errors()[0].get("loc", [])[1:])) or None
                                          ).body(), status_code=400)

    def _check_ready(model: str | None) -> AsyncEngine:
        aeng: AsyncEngine = state["aeng"]
        if not state.get("accepting") or not aeng.alive:
            raise P.APIError(503, "the engine is not available", "engine_unhealthy",
                             type_="server_error")
        if model != cfg.served_model_name:
            raise P.APIError(404, f"model {model!r} does not exist; served model is "
                                  f"{cfg.served_model_name!r}", "model_not_found", "model")
        if aeng.num_waiting >= cfg.max_waiting_requests:
            raise P.APIError(429, "too many waiting requests", "server_overloaded",
                             type_="server_error")
        return aeng

    # -- routes --------------------------------------------------------------------------------------
    @app.get("/health")
    async def health():
        aeng = state.get("aeng")
        if aeng is not None and aeng.alive:
            return {"status": "ok"}
        return JSONResponse({"status": "unhealthy"}, status_code=503)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": cfg.served_model_name, "object": "model",
                                            "owned_by": "turingserve",
                                            "max_model_len": cfg.max_model_len}]}

    @app.get("/metrics")
    async def metrics():
        return Response(state["aeng"].engine.metrics.render(),
                        media_type="text/plain; version=0.0.4; charset=utf-8")

    if cfg.debug_endpoints:
        @app.get("/debug/state")
        async def debug_state():
            e = state["aeng"].engine
            return {"num_running": len(e.scheduler.running), "num_waiting": len(e.scheduler.waiting),
                    "num_free_blocks": e.kv.num_free, "num_cached_blocks": e.kv.num_cached,
                    "num_total_blocks": e.kv.num_blocks, "num_requests_tracked": e.num_requests_tracked}

    if cfg.profile_dir:
        @app.post("/debug/profile")
        async def debug_profile(steps: int = 10):
            try:
                out = state["aeng"].engine.profiler.arm(max(1, steps))
            except RuntimeError as e:
                raise P.bad_request(str(e)) from e
            return {"status": "armed", "steps": steps, "dir": out}

    async def _run(request: Request, req_cls, converter, chat: bool):
        try:
            body = req_cls.model_validate(await request.json())
        except (ValidationError, ValueError) as e:
            raise P.bad_request(f"invalid request body: {e}") from e
        aeng = _check_ready(body.model)
        ids, params = converter(body, state["tokenizer"], cfg.max_model_len)
        rid = P.new_id("chatcmpl" if chat else "cmpl")
        ereq = EngineRequest(rid, ids, params)
        include_usage = bool(body.stream_options and body.stream_options.include_usage)
        if body.stream:
            return StreamingResponse(_stream(aeng, ereq, rid, chat, include_usage),
                                     media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache"})
        return await _collect(aeng, ereq, rid, chat, len(ids))

    async def _collect(aeng, ereq, rid, chat: bool, n_prompt: int):
        text, last = "", None
        try:
            async for out in aeng.generate(ereq):
                text += out.delta_text
                last = out
        except asyncio.CancelledError:
            aeng.abort(rid)
            raise
        if last is None or last.finish_reason == "error":
            raise P.APIError(500, "internal error while generating", "internal_error",
                             type_="server_error")
        reason = P.finish_reason_of(last.finish_reason)
        fn = P.chat_response if chat else P.completion_response
        return JSONResponse(fn(rid, cfg.served_model_name, text, reason, last.prompt_tokens,
                               last.completion_tokens, last.cached_prompt_tokens))

    async def _stream(aeng, ereq, rid, chat: bool, include_usage: bool):
        created = S.now()
        model = cfg.served_model_name
        last = None
        try:
            if chat:
                yield S.sse(S.chat_chunk(rid, model, created, {"role": "assistant", "content": ""},
                                         None))
            async for out in aeng.generate(ereq):
                last = out
                if out.finish_reason == "error":
                    yield S.sse(P.APIError(500, "internal error while generating", "internal_error",
                                           type_="server_error").body())
                    yield S.DONE
                    return
                if out.delta_text:
                    yield S.sse(S.chat_chunk(rid, model, created, {"content": out.delta_text}, None)
                                if chat else S.completion_chunk(rid, model, created,
                                                                out.delta_text, None))
                if out.finished:
                    reason = P.finish_reason_of(out.finish_reason)
                    yield S.sse(S.chat_chunk(rid, model, created, {}, reason) if chat
                                else S.completion_chunk(rid, model, created, "", reason))
            if include_usage and last is not None:
                yield S.sse(S.usage_chunk(rid, model, created, P.usage(
                    last.prompt_tokens, last.completion_tokens, last.cached_prompt_tokens), chat))
            yield S.DONE
        finally:
            if last is None or not last.finished:      # client disconnected: free the sequence
                aeng.abort(rid)
                log("stream_aborted", level="debug", request_id=rid)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await _run(request, P.ChatRequest, P.chat_to_internal, True)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _run(request, P.CompletionRequest, P.completion_to_internal, False)

    return app
