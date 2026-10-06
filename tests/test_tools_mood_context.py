"""Tools (safety, validation), mood, context budgeting, time helpers, images, proactive policy, scheduler."""
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import pytest

from riko.core.config import RikoConfig
from riko.core.context import (ContextOverflow, ImageStore, estimate_tokens, fit_history, humanize_gap,
                               in_quiet_hours)
from riko.core.scheduler import Scheduler
from riko.core.session import ConversationSession
from riko.emotion import Mood, MoodModel, classify_user_affect, make_state
from riko.llm import ToolCall
from riko.tools import ToolContext, ToolError, default_registry
from riko.tools.builtin import safe_eval, safe_path
from fakes import RecordingSink, make_services, test_config


# ------------------------------------------------------------------ tools
@pytest.mark.parametrize("expr,value", [("1234.5 * 987 / 3", 406150.5), ("2^10", 1024), ("17% of 2450", 416.5),
                                        ("sqrt(16) + abs(-2)", 6.0), ("10 % 3", 1), ("15%", 0.15)])
def test_calculator(expr, value):
    assert safe_eval(expr) == pytest.approx(value)


@pytest.mark.parametrize("bad", ['__import__("os").system("x")', "(1).__class__", "9**9**9", "open('f')",
                                 "factorial(100000)", "a" * 300, "lambda: 1"])
def test_calculator_rejects_code(bad):
    with pytest.raises(ToolError):
        safe_eval(bad)


def test_safe_paths(tmp_path):
    root = (tmp_path / "shared").resolve()
    root.mkdir()
    (root / "notes.md").write_text("hello")
    assert safe_path([root], "notes.md") == root / "notes.md"
    for bad in ["../secret.txt", str(tmp_path / "x.txt"), ".env", "keys/server.pem", "sub/../../x"]:
        with pytest.raises(ToolError):
            safe_path([root], bad)
    with pytest.raises(ToolError, match="no folders"):
        safe_path([], "notes.md")


async def test_registry_permissions_validation_and_bounds(tmp_path):
    cfg = RikoConfig.model_validate({"tools": {"max_result_chars": 200, "disabled": ["add_note"]}})
    reg = default_registry(cfg)
    ctx = ToolContext(cfg)
    names = {t["function"]["name"] for t in reg.schemas(ctx)}
    assert {"calculator", "get_datetime", "add_task"} <= names
    assert not names & {"get_weather", "web_search", "read_file", "search_files", "add_note", "look_at_image"}
    task = next(t for t in reg.schemas(ctx) if t["function"]["name"] == "add_task")["function"]["parameters"]
    assert "title" in task["properties"] and task["required"] == ["title"]  # property named "title" survives
    # unknown tool, extra args, empty optional args, oversized results
    assert not (await reg.execute(ToolCall("rm_rf", {}), ctx)).ok
    assert not (await reg.execute(ToolCall("calculator", {"expression": "1", "x": 2}), ctx)).ok
    ok = await reg.execute(ToolCall("get_datetime", {"timezone": ""}), ctx)
    assert ok.ok and "datetime" in json.loads(ok.content)
    cfg.tools.permissions.append("files")
    cfg.tools.file_roots = [str(tmp_path)]
    (tmp_path / "big.txt").write_text("x" * 5000)
    res = await reg.execute(ToolCall("read_file", {"path": "big.txt"}), ToolContext(cfg))
    assert res.ok and len(res.content) <= 200 + 20 and res.content.endswith("[truncated]")


async def test_tool_timeout(tmp_path):
    from riko.tools import Tool, ToolRegistry

    class Slow(Tool):
        name, description = "slow", "sleeps"

        async def run(self, args, ctx):
            await asyncio.sleep(5)

    cfg = RikoConfig.model_validate({"tools": {"timeout_s": 0.1}})
    res = await ToolRegistry(cfg.tools, [Slow()]).execute(ToolCall("slow", {}), ToolContext(cfg))
    assert not res.ok and "timed out" in res.content


async def test_task_and_reminder_tools(tmp_path):
    svc = make_services(tmp_path)
    reg, ctx = svc.tools, ToolContext(svc.config, svc.memory)
    res = await reg.execute(ToolCall("add_task", {"title": "Call Mia", "in_minutes": 30}), ctx)
    data = json.loads(res.content)
    assert res.ok and data["reminder_at"] and "only while the app is open" in data["note"]
    past = await reg.execute(ToolCall("add_task", {"title": "Old thing", "due": "2001-01-01T09:00"}), ctx)
    assert not past.ok and "past" in past.content
    listed = json.loads((await reg.execute(ToolCall("list_tasks", {}), ctx)).content)["tasks"]
    done = await reg.execute(ToolCall("complete_task", {"task_id": listed[0]["id"]}), ctx)
    assert done.ok and svc.memory.store.list_tasks("riko") == []


# ------------------------------------------------------------------ mood & affect
def test_mood_decay_formula_and_bounds():
    mm = MoodModel(0.2, 0.4, half_life_minutes=20, max_step=0.15)
    m = Mood(1.0, 1.0, 0.0)
    half = mm.decayed(m, 20 * 60)
    assert half.valence == pytest.approx(0.2 + 0.8 * 0.5) and half.arousal == pytest.approx(0.4 + 0.6 * 0.5)
    assert mm.decayed(m, -500).valence == 1.0  # clock jumped backwards: no change, never amplified
    assert mm.decayed(mm.decayed(m, 600), 600).valence == pytest.approx(mm.decayed(m, 600).valence)  # reads don't decay
    after = mm.update(Mood(0.2, 0.4, 0), 0, [make_state("excited", 1.0)] * 5)
    assert after.valence - 0.2 <= 0.15 + 1e-9  # capped step per turn


def test_mood_has_no_negative_spiral_and_recovers_by_time():
    mm = MoodModel()
    m = None
    for i in range(50):
        m = mm.update(m, float(i), [make_state("sad", 1.0)], classify_user_affect("I feel miserable"))
    assert m.valence > -0.5  # own negative tags stop pushing once mood is low
    assert mm.decayed(m, 50 + 4 * 3600).valence == pytest.approx(mm.bv, abs=0.01)  # time alone restores baseline
    assert MoodModel.describe(m) and abs(MoodModel.speed_factor(m) - 1.0) <= 0.03


def test_user_affect_is_uncertain():
    assert classify_user_affect("I'm not happy").label == "unknown"
    assert classify_user_affect("ugh this is not working").label == "frustrated"
    assert classify_user_affect("estoy triste").confidence == 0.0
    assert classify_user_affect("I got the job!!").label == "positive"


# ------------------------------------------------------------------ context helpers
def test_token_budget_drops_oldest_history_never_the_request():
    static = {"role": "system", "content": "s" * 300}
    ctx = {"role": "system", "content": "c" * 300}
    user = {"role": "user", "content": "hello"}
    hist = [{"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 600} for i in range(10)]
    kept, est, dropped = fit_history(static, hist, ctx, user, budget=900)
    assert kept and kept[0]["role"] == "user" and dropped >= 2 and est <= 900
    with pytest.raises(ContextOverflow):
        fit_history(static, [], ctx, {"role": "user", "content": "y" * 10000}, budget=900)
    assert estimate_tokens("hello world") >= 3 and estimate_tokens("日本語") == 3


def test_time_helpers():
    assert humanize_gap(None) is None and humanize_gap(-5) is None
    assert humanize_gap(30) == "moments ago" and humanize_gap(3 * 86400) == "about 3 days ago"
    night, noon = datetime(2026, 1, 1, 23, 30), datetime(2026, 1, 1, 12, 0)
    assert in_quiet_hours("22:30-07:30", night) and not in_quiet_hours("22:30-07:30", noon)
    assert in_quiet_hours("09:00-17:00", noon) and not in_quiet_hours(None, night)
    with pytest.raises(Exception):
        RikoConfig.model_validate({"proactive": {"quiet_hours": "late"}})


def test_image_store_sniffs_and_bounds():
    s = ImageStore(max_items=2, ttl_s=60)
    a = s.put(b"\x89PNG\r\n\x1a\n" + b"0" * 10)
    assert s.get(a).startswith("data:image/png;base64,")
    with pytest.raises(ValueError):
        s.put(b"<html>not an image</html>")
    s.put(b"\xff\xd8\xff" + b"0" * 10)
    s.put(b"GIF89a" + b"0" * 10)
    assert s.get(a) is None  # evicted: bounded


# ------------------------------------------------------------------ proactive policy
async def test_greeting_once_after_ready_and_cancelled_by_user_speech(tmp_path):
    svc = make_services(tmp_path)
    sink = RecordingSink()
    s = ConversationSession(svc, sink)
    assert await s.proactive.tick() is None  # nothing before the client says it's ready
    await s.set_client_ready()
    assert await s.proactive.tick() == "greeting"
    assert s.proactive.greeting_due() is False  # once per session
    s2 = ConversationSession(svc, RecordingSink())
    await s2.set_client_ready()
    s2.proactive.user_spoke()
    assert await s2.proactive.tick() is None
    s3 = ConversationSession(svc, RecordingSink())
    s3.proactive_level = "off"
    await s3.set_client_ready()
    assert await s3.proactive.tick() is None


async def test_nudges_are_rare_and_need_a_user_turn_between(tmp_path):
    cfg = test_config(tmp_path)
    cfg.proactive.idle_nudge_s = 0.01
    cfg.proactive.nudge_cooldown_s = 0.0
    svc = make_services(tmp_path, cfg=cfg)
    s = ConversationSession(svc, RecordingSink())
    s.proactive.greeting_eligible = False
    await s.set_client_ready()
    assert await s.proactive.tick() is None  # no user turn yet in this session
    s.user_turns = 1
    s.last_activity -= 1
    assert await s.proactive.tick() == "nudge"
    await asyncio.sleep(0.05)
    while s.turn_running:
        await asyncio.sleep(0.01)
    s.last_activity -= 1
    assert await s.proactive.tick() is None  # no second nudge without an intervening user turn
    s.proactive.user_turn_completed()
    assert await s.proactive.tick() == "nudge"
    while s.turn_running:
        await asyncio.sleep(0.01)
    s.proactive.user_turn_completed()
    s.last_activity -= 1
    assert await s.proactive.tick() is None  # max 2 per session


async def test_due_reminder_is_delivered_once_across_tabs(tmp_path):
    svc = make_services(tmp_path)
    svc.memory.store.add_task("riko", "Stretch", time.time() - 1)
    a, b = ConversationSession(svc, RecordingSink()), ConversationSession(svc, RecordingSink())
    for s in (a, b):
        s.proactive.greeting_eligible = False
        await s.set_client_ready()
    assert await a.proactive.tick() == "reminder"
    assert await b.proactive.tick() is None
    while a.turn_running:
        await asyncio.sleep(0.01)
    assert any("Stretch" in json.dumps(m) for m in svc.llm.calls[-1])


async def test_proactive_turns_are_not_stored_as_user_messages(tmp_path):
    svc = make_services(tmp_path)
    s = ConversationSession(svc, RecordingSink())
    await s.proactive_turn("greeting", "Greet the user.")
    while s.turn_running:
        await asyncio.sleep(0.01)
    msgs = svc.memory.store.recent_messages("riko", 10)
    assert [m["role"] for m in msgs] == ["assistant"] and msgs[0]["source"] == "greeting"
    assert svc.memory.store.get_relationship("riko")["sessions"] == 0  # proactive output never counts


# ------------------------------------------------------------------ scheduler bounds
async def test_scheduler_bounds_coalescing_and_expiry():
    sch = Scheduler(max_pending=2, expiry_s=0.0)
    ran = []

    async def job():
        ran.append(1)
    sch._fg = 1  # hold jobs while "a turn is running"
    sch.submit("a", job)
    sch.submit("a", job)
    sch.submit("b", job)
    sch.submit("c", job)
    assert sch.backlog()["pending"] == 2 and sch.stats["dropped"] == 1 and sch.stats["coalesced"] == 1
    sch._fg = 0
    await asyncio.sleep(1.2)
    assert ran == [] and sch.stats["expired"] == 2  # expired jobs never run
    await sch.aclose()
