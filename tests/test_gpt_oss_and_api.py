"""gpt-oss wire format (recorded Ollama fixtures), OpenAI-compatible tool streaming, routing, and the REST API."""
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from riko.api.protocol import client_message
from riko.api.server import create_app
from riko.core.config import ConfigError, LLMConfig, RikoConfig, load_config
from riko.llm import ModelRouter, OllamaLLM, OpenAICompatibleLLM
from fakes import FakeLLM, make_services

FIX = Path(__file__).parent / "fixtures" / "ollama_gpt_oss"
SHOW = {"capabilities": ["completion", "tools", "thinking"], "thinking": {"values": ["low", "medium", "high"],
        "default": "medium"}, "model_info": {"gptoss.context_length": 131072}}


def ollama(fixture, seen=None, show=SHOW, **kw):
    def handler(req: httpx.Request):
        if req.url.path == "/api/show":
            return httpx.Response(200, json=show)
        body = json.loads(req.content)
        if seen is not None:
            seen.append(body)
        if not body["messages"]:  # preload
            return httpx.Response(200, json={"done": True, "done_reason": "load", "load_duration": 3_000_000})
        return httpx.Response(200, content=(FIX / f"{fixture}.ndjson").read_bytes())
    return OllamaLLM(base_url="http://o", model="gpt-oss:20b", transport=httpx.MockTransport(handler), **kw)


async def events(llm, **kw):
    return [e async for e in llm.stream_events([{"role": "user", "content": "hi"}], **kw)]


async def test_gpt_oss_fixture_separates_reasoning_and_reports_usage():
    seen = []
    evs = await events(ollama("chat_think_low", seen, reasoning_effort="low", num_ctx=8192))
    kinds = [e.kind for e in evs]
    assert "thinking" in kinds and "content" in kinds and kinds[-1] == "done"
    assert all(e.text == "" for e in evs if e.kind == "thinking")  # reasoning text never leaves the provider
    assert "redacted" not in "".join(e.text for e in evs)
    done = evs[-1]
    assert done.finish_reason == "stop" and done.usage["prompt_tokens"] and "cached_tokens" in done.usage
    assert seen[0]["think"] == "low" and seen[0]["options"]["num_ctx"] == 8192


async def test_gpt_oss_fixture_tool_call_and_length_starvation():
    evs = await events(ollama("tool_call"), tools=[{"type": "function", "function": {"name": "calculator"}}])
    call = next(e for e in evs if e.kind == "tool_calls").tool_calls[0]
    assert call.name == "calculator" and isinstance(call.arguments, dict) and call.arguments["expression"]
    starved = await events(ollama("length_starved"))
    assert starved[-1].finish_reason == "length" and not any(e.kind == "content" for e in starved)


async def test_think_parameter_is_translated_per_model():
    seen = []
    await events(ollama("chat_think_low", seen, reasoning_effort="high"))
    assert seen[-1]["think"] == "high"
    await events(ollama("chat_think_low", seen, reasoning_effort="off"))
    assert "think" not in seen[-1]  # gpt-oss ignores false; nothing is sent rather than something wrong
    bool_model = {"capabilities": ["completion", "thinking"]}
    await events(ollama("chat_think_low", seen, show=bool_model, reasoning_effort="off"))
    assert seen[-1]["think"] is False
    await events(ollama("chat_think_low", seen, show={"capabilities": ["completion"]}, reasoning_effort="low"))
    assert "think" not in seen[-1]


async def test_warmup_preloads_with_same_num_ctx_and_message_conversion():
    seen = []
    llm = ollama("chat_think_low", seen, num_ctx=8192)
    res = await llm.warmup()
    assert res["loaded"] and seen[0]["options"]["num_ctx"] == 8192 and seen[0]["messages"] == []
    from riko.llm.ollama import to_ollama_messages
    msgs = to_ollama_messages([
        {"role": "user", "content": "look", "images": ["data:image/png;base64,QUJD"]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "calculator", "arguments": {"x": 1}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "calculator", "content": "2"}])
    assert msgs[0]["images"] == ["QUJD"] and msgs[1]["tool_calls"][0]["function"]["arguments"] == {"x": 1}
    assert msgs[2] == {"role": "tool", "tool_name": "calculator", "content": "2"}


async def test_openai_compatible_reasoning_and_fragmented_tool_calls():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "private plan"}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "calc", "arguments": '{"expr'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'ession": "2+2"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 9,
                                       "prompt_tokens_details": {"cached_tokens": 32}}},
        ]
        sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=sse.encode())
    llm = OpenAICompatibleLLM(name="lmstudio", base_url="http://l/v1", model="openai/gpt-oss-20b",
                              reasoning_effort="low", transport=httpx.MockTransport(handler))
    evs = await events(llm, tools=[{"type": "function", "function": {"name": "calc"}}])
    assert [e.kind for e in evs] == ["thinking", "tool_calls", "done"] and evs[0].text == ""
    call = evs[1].tool_calls[0]
    assert (call.id, call.name, call.arguments) == ("call_1", "calc", {"expression": "2+2"})
    assert evs[2].usage == {"prompt_tokens": 50, "completion_tokens": 9, "cached_tokens": 32}
    assert seen["reasoning_effort"] == "low" and seen["tools"]
    from riko.llm.openai_compat import to_openai_messages
    m = to_openai_messages([{"role": "assistant", "content": "", "tool_calls": [{"id": "c", "name": "n", "arguments": {}}]}])
    assert m[0]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_model_roles_and_legacy_config(tmp_path):
    cfg = RikoConfig.model_validate({"models": {"vision": {"model": "qwen3.6:latest", "reasoning_effort": False},
                                                "smart": {"provider": "lmstudio", "model": "big"}}})
    assert cfg.model_for("fast") is cfg.llm and cfg.llm.model == "gpt-oss:20b"
    assert cfg.model_for("vision").reasoning_effort == "off"          # YAML `off` parses as False
    assert cfg.model_for("smart").base_url == "http://127.0.0.1:1234/v1"  # provider defaults, not Ollama's URL
    assert RikoConfig().model_for("vision") is None
    router = ModelRouter(RikoConfig.model_validate({"models": {"smart": {"temperature": 1.0}}}))
    assert router.get("fast") is router.get("smart")  # identical settings share one client / loaded model
    assert RikoConfig.model_validate({"llm": {"ollama_num_ctx": 16384}}).llm.num_ctx == 16384
    with pytest.raises(Exception, match="smaller than llm.num_ctx"):
        LLMConfig(num_ctx=2048, max_tokens=2048)
    (tmp_path / "config.yaml").write_text("models:\n  fast:\n    modle: x\n")
    with pytest.raises(ConfigError):
        load_config(tmp_path / "config.yaml", env_file=False)


# ------------------------------------------------------------------ REST API
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("RIKO_AUTH_TOKEN", raising=False)
    svc = make_services(tmp_path, llm=FakeLLM("[happy] Hi!"))
    with TestClient(create_app(svc.config, svc, warmup=False)) as c:
        c.svc = svc
        yield c


def test_memory_crud_export_clear_and_toggle(client):
    m = client.post("/api/memories", json={"content": "User likes tea", "layer": "semantic"}).json()["memory"]
    assert m["source"] == "user_edit" and m["confidence"] == 1.0
    p = client.patch(f"/api/memories/{m['id']}", json={"content": "User likes green tea", "pinned": True}).json()
    assert p["memory"]["content"] == "User likes green tea" and p["memory"]["pinned"]
    assert client.patch("/api/memories/nope", json={"pinned": True}).status_code == 404
    assert client.post("/api/memories", json={"content": "x"}).status_code == 422
    exp = client.get("/api/memories/export")
    assert "attachment" in exp.headers["content-disposition"] and exp.json()["memories"][0]["content"]
    assert client.delete("/api/memories").status_code == 400  # needs ?confirm=true
    assert client.delete("/api/memories?confirm=true").json()["deleted"] == 1
    assert client.get("/api/memories").json()["memories"] == []
    assert client.put("/api/memory/settings", json={"enabled": False}).json() == {"enabled": False}
    assert client.get("/api/memories").json()["enabled"] is False


def test_cross_origin_writes_are_refused(client):
    r = client.post("/api/memories", json={"content": "User likes tea"}, headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    assert client.delete("/api/memories?confirm=true", headers={"origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/memories", headers={"origin": "https://evil.example"}).status_code == 200  # reads ok


def test_images_relationship_mood_tasks_status_tools(client):
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 64
    r = client.post("/api/images", content=png, headers={"content-type": "image/png"})
    assert r.status_code == 200 and len(r.json()["id"]) == 16 and r.json()["vision_model"] is None
    assert client.post("/api/images", content=b"<script>", headers={"content-type": "image/png"}).status_code == 415
    assert client.post("/api/images", content=b"\x89PNG" + b"0" * (9 * 1024 * 1024)).status_code == 413
    rel = client.get("/api/relationship").json()
    assert rel["stage"] == 0 and "not a score" in rel["explanation"]
    mood = client.get("/api/mood").json()
    assert -1 <= mood["valence"] <= 1 and mood["description"] and client.delete("/api/mood").json()["reset"]
    svc = client.svc
    tid = svc.memory.store.add_task("riko", "Buy milk")
    assert client.get("/api/tasks").json()["tasks"][0]["title"] == "Buy milk"
    assert client.patch(f"/api/tasks/{tid}", json={"status": "done"}).status_code == 200
    assert client.get("/api/tasks").json()["tasks"] == []
    st = client.get("/api/status").json()
    assert st["llm"]["model"] == "fake-model" and "background" in st and "key" not in json.dumps(st).lower()
    tools = {t["name"]: t for t in client.get("/api/tools").json()["tools"]}
    assert tools["calculator"]["enabled"] and not tools["get_weather"]["enabled"]
    assert "permission 'network' not granted" in tools["get_weather"]["reason"]


def test_websocket_v2_messages(client):
    assert client_message.validate_json('{"type":"text","text":"","image_id":"0123456789abcdef"}').image_id
    for bad in ['{"type":"text","text":""}', '{"type":"text","text":"hi","image_id":"../x"}',
                '{"type":"settings","proactive_level":"loud"}']:
        with pytest.raises(Exception):
            client_message.validate_json(bad)
    with client.websocket_connect("/ws") as ws:
        hello = json.loads(ws.receive_text())
        assert hello["event"] == "conversation.started" and "happy" in hello["settings"]["emotions"]
        assert hello["settings"]["protocol"] == 2 and hello["mood"]
        ws.send_text('{"type":"ready"}')
        ws.send_text('{"type":"typing"}')
        ws.send_text('{"type":"settings","proactive_level":"off","memory_enabled":false}')
        for _ in range(50):
            ev = json.loads(ws.receive_text())
            if ev["type"] == "settings":
                assert ev["settings"]["proactive_level"] == "off" and ev["settings"]["memory_enabled"] is False
                break
        else:
            raise AssertionError("no settings event")
