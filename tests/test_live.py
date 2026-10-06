"""Live checks against your real local model. Skipped unless RIKO_LIVE=1 (never downloads models).

    RIKO_LIVE=1 pytest tests/test_live.py -q          # uses config.yaml (default: Ollama gpt-oss:20b)
"""
import json
import os

import pytest

from riko.core.config import load_config
from riko.core.session import ConversationSession, Services
from fakes import RecordingSink
from test_session import wait_for

pytestmark = pytest.mark.skipif(os.environ.get("RIKO_LIVE") != "1", reason="set RIKO_LIVE=1 to run live checks")


@pytest.fixture
async def live(tmp_path):
    cfg = load_config()
    cfg.memory.db_path = str(tmp_path / "live.db")
    cfg.tts.provider = "none"
    cfg.proactive.enabled = False
    svc = Services.from_config(cfg)
    ok, msg = await svc.llm.health()
    if not ok:
        await svc.aclose()
        pytest.skip(f"model not available: {msg}")
    yield svc
    await svc.aclose()


async def turn(svc, sink, s, text):
    n = len(sink.of("assistant_stopped"))
    await s.submit_text(text)
    await wait_for(lambda: len(sink.of("assistant_stopped")) > n, timeout=90)
    return " ".join(e["text"] for e in sink.of("llm_sentence")[-20:] if e["turn_id"] == s.turn_id)


async def test_live_identity_tools_and_no_reasoning_leak(live):
    sink = RecordingSink()
    s = ConversationSession(live, sink)
    reply = await turn(live, sink, s, "Which model are you running on? One sentence.")
    assert live.llm.model.split(":")[0].lower().replace("-", "") in reply.lower().replace("-", "").replace("‑", "")
    reply = await turn(live, sink, s, "Use your calculator: what is 1234.5 * 987 / 3?")
    assert "406" in reply.replace(",", "") and any(e["tool"] == "calculator" for e in sink.of("tool_finished"))
    blob = json.dumps(sink.events)
    assert "<|channel|>" not in blob and all(set(e) == {"type", "event", "ts", "turn_id"}
                                             for e in sink.of("assistant_thinking"))
    m = sink.of("metrics")[-1]
    assert m["llm_first_content_ms"] < live.config.llm.first_content_timeout_s * 1000
