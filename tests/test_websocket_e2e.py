"""WebSocket protocol + end-to-end smoke tests through the real FastAPI app."""
import json
import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from riko.api.protocol import client_message
from riko.api.server import create_app
from riko.core.events import decode_audio_frame
from fakes import FakeASR, FakeLLM, make_services, speech_audio, to_frames


@pytest.fixture
def app_and_services(tmp_path, monkeypatch):
    monkeypatch.delenv("RIKO_AUTH_TOKEN", raising=False)
    svc = make_services(tmp_path, llm=FakeLLM("[happy] Nice to meet you, Alex! I will remember that."),
                        asr=FakeASR("Hi Riko, my name is Alex"))
    return create_app(svc.config, svc, warmup=False), svc


def recv_until(ws, type_, limit=500):
    events, audio = [], []
    for _ in range(limit):
        m = ws.receive()
        if m.get("bytes") is not None:
            audio.append(decode_audio_frame(m["bytes"]))
            continue
        ev = json.loads(m["text"])
        events.append(ev)
        if ev["type"] == type_:
            return events, audio
    raise AssertionError(f"{type_} not received")


def test_protocol_validation():
    assert client_message.validate_json('{"type":"text","text":"hi"}').text == "hi"
    for bad in ['{"type":"text","text":""}', '{"type":"rm -rf"}', '{"type":"settings","preset":"../x"}',
                '{"type":"settings","model":"a b"}', '{"type":"ptt","action":"start","extra":1}']:
        with pytest.raises(Exception):
            client_message.validate_json(bad)


def test_text_conversation_end_to_end(app_and_services):
    app, svc = app_and_services
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "hello" and hello["settings"]["character_name"] == "Riko"
        assert "key" not in json.dumps(hello).lower()
        ws.send_text(json.dumps({"type": "text", "text": "Hi, my name is Alex"}))
        events, audio = recv_until(ws, "assistant_stopped")
        types = [e["type"] for e in events]
        for t in ["transcription", "llm_started", "llm_token", "llm_sentence", "emotion_changed", "tts_started",
                  "assistant_speaking", "tts_finished", "llm_finished"]:
            assert t in types, t
        assert audio and all(a[0] == 1 and a[2] == 24000 for a in audio)
        assert events[-1]["reason"] == "completed"
        for _ in range(100):  # background memory extraction runs once idle
            if svc.memory.store.list("riko"):
                break
            time.sleep(0.05)
        events, _ = recv_until(ws, "memory_created")
        assert events[-1]["content"] == "User's name is Alex"
        r = client.get("/api/memories")
        assert r.json()["memories"][0]["content"] == "User's name is Alex"
        mid = r.json()["memories"][0]["id"]
        assert client.delete(f"/api/memories/{mid}").status_code == 200
        assert client.delete("/api/memories/nope").status_code == 404
        hist = client.get("/api/history").json()["messages"]
        assert hist[0]["text"] == "Hi, my name is Alex" and not hist[1]["text"].startswith("[")


def test_voice_conversation_smoke(app_and_services):
    """Mic frames -> VAD -> ASR -> LLM -> TTS audio frames back."""
    app, _ = app_and_services
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_text()
        for f in to_frames(speech_audio()):
            ws.send_bytes(f)
        events, audio = recv_until(ws, "assistant_stopped")
        types = [e["type"] for e in events]
        assert "user_speech_started" in types and "user_speech_stopped" in types
        assert [e for e in events if e["type"] == "transcription"][0]["text"] == "Hi Riko, my name is Alex"
        assert audio


def test_bad_input_does_not_kill_connection(app_and_services):
    app, _ = app_and_services
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_text()
        ws.send_text("not json")
        assert recv_until(ws, "error")[0][-1]["component"] == "protocol"
        ws.send_bytes(b"\x00\x01\x02")  # odd-length audio
        assert recv_until(ws, "error")[0][-1]["message"] == "invalid audio frame"
        ws.send_text(json.dumps({"type": "settings", "preset": "friendly"}))
        assert recv_until(ws, "settings")[0][-1]["settings"]["preset"] == "friendly"
        ws.send_text('{"type":"ping"}')
        assert recv_until(ws, "pong")


def test_auth_token_and_origin(app_and_services, monkeypatch):
    app, _ = app_and_services
    monkeypatch.setenv("RIKO_AUTH_TOKEN", "s3cret-token")
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws") as ws:
                ws.receive_text()
        with client.websocket_connect("/ws?token=s3cret-token") as ws:
            assert json.loads(ws.receive_text())["type"] == "hello"
        assert client.get("/api/memories").status_code == 401
        assert client.get("/api/memories?token=s3cret-token").status_code == 200
    monkeypatch.delenv("RIKO_AUTH_TOKEN")
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws", headers={"origin": "https://evil.example"}) as ws:
                ws.receive_text()


def test_health_and_static(app_and_services):
    app, _ = app_and_services
    with TestClient(app) as client:
        h = client.get("/api/health").json()
        assert h["llm"]["ok"] and h["memory"]["ok"]
        assert client.get("/api/models").json()["current"] == "fake-model"
        page = client.get("/")
        assert page.status_code == 200 and "Start talking" in page.text
        assert client.get("/app.js").status_code == 200


def test_remote_bind_requires_token(monkeypatch):
    from riko.api.server import check_bind_security
    from riko.core.config import RikoConfig
    monkeypatch.delenv("RIKO_AUTH_TOKEN", raising=False)
    cfg = RikoConfig.model_validate({"server": {"host": "0.0.0.0"}})
    with pytest.raises(SystemExit):
        check_bind_security(cfg)
    check_bind_security(RikoConfig())
