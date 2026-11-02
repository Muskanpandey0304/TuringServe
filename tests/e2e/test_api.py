"""Phase 6 end-to-end tests against a real server subprocess (tiny model, CPU)."""

import json
import threading
import time

import httpx
import pytest

M = "tiny"
MSG = [{"role": "user", "content": "Say hi in five words"}]


def post(server, path, body, **kw):
    return httpx.post(server.url + path, json=body, timeout=120, **kw)


def sse_events(resp_text: str):
    return [line[6:] for line in resp_text.split("\n\n") if line.startswith("data: ")]


def test_openai_sdk_chat_and_completions(server, tokenizer):
    from openai import OpenAI

    c = OpenAI(base_url=server.url + "/v1", api_key="x")
    r = c.chat.completions.create(model=M, messages=MSG, max_tokens=8, temperature=0)
    assert r.choices[0].message.role == "assistant"
    templ = tokenizer.apply_chat_template(MSG, tokenize=False, add_generation_prompt=True)
    assert r.usage.prompt_tokens == len(tokenizer.encode(templ, add_special_tokens=False))
    assert r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens
    t = c.completions.create(model=M, prompt="Hello world", max_tokens=5, temperature=0)
    assert t.object == "text_completion" and t.id.startswith("cmpl-")
    ids = tokenizer.encode("Hello world", add_special_tokens=False)
    t2 = c.completions.create(model=M, prompt=ids, max_tokens=5, temperature=0)
    assert t2.choices[0].text == t.choices[0].text and t2.usage.prompt_tokens == len(ids)


def test_models_health_metrics(server):
    assert httpx.get(server.url + "/health").json() == {"status": "ok"}
    m = httpx.get(server.url + "/v1/models").json()["data"][0]
    assert m["id"] == M and m["owned_by"] == "turingserve" and m["max_model_len"] == 2048
    post(server, "/v1/chat/completions", {"model": M, "messages": MSG, "max_tokens": 4})
    text = httpx.get(server.url + "/metrics").text
    for name in ("turingserve_request_ttft_seconds", "turingserve_request_tpot_seconds",
                 "turingserve_request_e2e_seconds", "turingserve_prompt_tokens_total",
                 "turingserve_generation_tokens_total", "turingserve_requests_total",
                 "turingserve_num_running", "turingserve_num_waiting",
                 "turingserve_kv_usage_ratio", "turingserve_prefix_cache_queried_tokens_total",
                 "turingserve_prefix_cache_hit_tokens_total", "turingserve_preemptions_total",
                 "turingserve_spec_draft_tokens_total", "turingserve_spec_accepted_tokens_total",
                 "turingserve_step_seconds", "turingserve_engine_cpu_seconds_per_step",
                 "turingserve_autopilot_decisions_total", "turingserve_autopilot_setting"):
        assert name in text, name


def test_streaming_order_and_equals_nonstream(server):
    body = {"model": M, "messages": MSG, "max_tokens": 12, "temperature": 0}
    plain = post(server, "/v1/chat/completions", body).json()
    r = post(server, "/v1/chat/completions", {**body, "stream": True})
    assert r.headers["content-type"].startswith("text/event-stream")
    ev = sse_events(r.text)
    assert ev[-1] == "[DONE]"
    chunks = [json.loads(e) for e in ev[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert all("usage" not in c for c in chunks)                     # no usage without include_usage
    assert chunks[-1]["choices"][0]["delta"] == {} and chunks[-1]["choices"][0]["finish_reason"]
    assert all(c["choices"][0]["delta"].get("content", "x") != "" for c in chunks[1:-1])
    streamed = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert streamed == plain["choices"][0]["message"]["content"]
    r = post(server, "/v1/chat/completions",
             {**body, "stream": True, "stream_options": {"include_usage": True}})
    chunks = [json.loads(e) for e in sse_events(r.text)[:-1]]
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["prompt_tokens"] > 0
    assert "cached_tokens" in chunks[-1]["usage"]["prompt_tokens_details"]


@pytest.mark.parametrize("body,status,code", [
    ({"model": M, "messages": MSG, "n": 2}, 400, None),
    ({"model": M, "messages": MSG, "logprobs": True}, 400, None),
    ({"model": M, "messages": MSG, "tools": [{"type": "function"}]}, 400, None),
    ({"model": M, "messages": MSG, "stop": ["a"] * 17}, 400, None),
    ({"model": M, "messages": MSG, "temperature": 3}, 400, None),
    ({"model": M, "messages": [{"role": "robot", "content": "x"}]}, 400, None),
    ({"model": M, "messages": MSG, "max_tokens": 10**6}, 400, "context_length_exceeded"),
    ({"model": M, "messages": MSG, "response_format": {"type": "json_object"}}, 400, None),
    ({"model": "nope", "messages": MSG}, 404, "model_not_found"),
])
def test_errors(server, body, status, code):
    r = post(server, "/v1/chat/completions", body)
    assert r.status_code == status
    err = r.json()["error"]
    assert set(err) == {"message", "type", "param", "code"} and err["message"]
    if code:
        assert err["code"] == code


def test_empty_prompt_rejected(server):
    r = post(server, "/v1/completions", {"model": M, "prompt": ""})
    assert r.status_code == 400
    r = post(server, "/v1/completions", {"model": M, "prompt": ["a", "b"]})
    assert r.status_code == 400


def test_cancel_frees_everything(server):
    body = {"model": M, "prompt": "Once upon a time", "max_tokens": 500, "stream": True,
            "temperature": 1.0}
    with httpx.stream("POST", server.url + "/v1/completions", json=body, timeout=60) as r:
        n = 0
        for line in r.iter_lines():
            if line.startswith("data: "):
                n += 1
            if n >= 3:
                break
    deadline = time.time() + 2.5
    st = server.state()
    while time.time() < deadline and st["num_running"] != 0:
        time.sleep(0.1)
        st = server.state()
    assert st["num_running"] == 0 and st["num_requests_tracked"] == 0
    assert st["num_free_blocks"] + st["num_cached_blocks"] == st["num_total_blocks"] - 1


def test_concurrency_100_parallel_equals_sequential(server):
    prompts = [f"Request number {i}: tell me about the number {i * 7}." for i in range(100)]

    def ask(p):
        r = post(server, "/v1/completions", {"model": M, "prompt": p, "max_tokens": 6,
                                             "temperature": 0})
        assert r.status_code == 200
        return r.json()["choices"][0]["text"]

    seq = [ask(p) for p in prompts[:10]]
    results = [None] * 100

    def worker(i):
        results[i] = ask(prompts[i])

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert results[:10] == seq and all(r is not None for r in results)
    st = server.state()
    assert st["num_requests_tracked"] == 0


def test_overload_returns_429_with_retry_after(overload_server):
    s = overload_server
    codes = []

    def go():
        r = post(s, "/v1/completions", {"model": M, "prompt": "hello there", "max_tokens": 40})
        codes.append((r.status_code, r.headers.get("retry-after")))

    ts = [threading.Thread(target=go) for _ in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert any(c == 429 and ra == "1" for c, ra in codes)
    assert all(c in (200, 429) for c, _ in codes) and any(c == 200 for c, _ in codes)
