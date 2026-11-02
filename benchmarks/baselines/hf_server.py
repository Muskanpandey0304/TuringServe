"""Hugging Face baseline server: one request at a time (asyncio.Lock), generate() in a thread with
TextIteratorStreamer, OpenAI-format /v1/completions chunks (Build Guide Phase 8, step 8).

  python -m benchmarks.baselines.hf_server --model $TS_MODEL_15 --port 8003
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


def create_app(model_path: str, name: str) -> FastAPI:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    lock = asyncio.Lock()
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        max_tokens = body.get("max_tokens", 16)
        ignore_eos = bool(body.get("ignore_eos"))
        ids = tok(body["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        rid, created = f"cmpl-{uuid.uuid4().hex}", int(time.time())

        def chunk(text, finish=None):
            return "data: " + json.dumps({"id": rid, "object": "text_completion", "created": created,
                                          "model": name, "choices": [{"index": 0, "text": text,
                                                                      "finish_reason": finish}]}) + "\n\n"

        async def stream():
            async with lock:
                streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
                kw = dict(input_ids=ids, streamer=streamer, max_new_tokens=max_tokens,
                          min_new_tokens=max_tokens if ignore_eos else 0, do_sample=False,
                          repetition_penalty=1.0, temperature=None, top_p=None, top_k=None,
                          pad_token_id=151643)
                out_holder = {}

                def run():
                    with torch.inference_mode():
                        out_holder["ids"] = model.generate(**kw)

                th = threading.Thread(target=run)
                th.start()
                loop = asyncio.get_running_loop()
                it = iter(streamer)
                while True:
                    piece = await loop.run_in_executor(None, next, it, None)
                    if piece is None:
                        break
                    if piece:
                        yield chunk(piece)
                th.join()
                n_out = out_holder["ids"].shape[1] - ids.shape[1]
                yield chunk("", "length" if n_out >= max_tokens else "stop")
                yield "data: " + json.dumps({"id": rid, "object": "text_completion", "created": created,
                                             "model": name, "choices": [],
                                             "usage": {"prompt_tokens": ids.shape[1],
                                                       "completion_tokens": n_out,
                                                       "total_tokens": ids.shape[1] + n_out}}) + "\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-model-name", default="qwen")
    ap.add_argument("--port", type=int, default=8003)
    a = ap.parse_args()
    uvicorn.run(create_app(a.model, a.served_model_name), host="0.0.0.0", port=a.port)
