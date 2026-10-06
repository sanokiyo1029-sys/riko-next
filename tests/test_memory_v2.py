"""Memory v2: migration, evidence validation, corrections, deletion that sticks, isolation, familiarity."""
import json
import sqlite3
import time

import pytest

from riko.core.config import MemoryConfig
from riko.core.context import familiarity_stage
from riko.memory import (ExtractedItem, HashEmbedder, Layer, MemoryManager, MemoryStore, Status,
                         validate_evidence)


def mgr(tmp_path, complete=None, profile=None, character="riko", db="m.db"):
    return MemoryManager(MemoryStore(tmp_path / db), HashEmbedder(256), MemoryConfig(), character,
                         complete=complete, profile=profile)


def item(claim, evidence, speaker="user", **kw):
    return ExtractedItem(layer=kw.pop("layer", "semantic"), claim=claim, evidence=evidence, speaker=speaker, **kw)


USER = ("Hi, I'm Alex. I do not live in Delhi anymore, I moved to Pune. My friend says I love jazz. "
        "Would I like sushi? If I had a dog I would call it Rex. \"I hate tea\" is what my sister quotes.")


@pytest.mark.parametrize("claim,evidence,expected", [
    ("User's name is Alex", "I'm Alex", None),
    ("User moved to Pune", "I moved to Pune", None),
    ("User does not live in Delhi", "I do not live in Delhi", None),
    ("User lives in Delhi", "live in Delhi", "negation_mismatch"),
    ("User loves jazz", "I love jazz", "attributed_or_quoted"),
    ("User likes sushi", "like sushi", "question"),
    ("User has a dog", "I had a dog", "hypothetical"),
    ("User hates tea", "I hate tea", "attributed_or_quoted"),
    ("User likes cake", "I like cake", "evidence_not_found"),
])
def test_evidence_rules(claim, evidence, expected):
    assert validate_evidence(item(claim, evidence), USER, "") == expected


def test_assistant_self_memory_rules():
    reply = "Honestly, jazz is overrated. I went to Paris last week. I will remind you. You live in Pune, right"
    ok = item("Riko thinks jazz is overrated", "jazz is overrated", "assistant")
    assert validate_evidence(ok, USER, reply) is None
    # invented experiences and commitments are not opinions
    assert validate_evidence(item("Riko went to Paris", "I went to Paris", "assistant"), USER, reply) \
        == "self_memory_not_opinion"
    assert validate_evidence(item("Riko will remind the user", "I will remind you", "assistant"), USER, reply) \
        in ("self_memory_not_opinion", "meta_request")
    # repeating the user's claim is not independent evidence
    assert validate_evidence(item("Riko thinks user lives in Pune", "You live in Pune", "assistant"), USER, reply) \
        == "self_memory_not_opinion"
    # wrong speaker: the user's words can't support an assistant claim
    assert validate_evidence(item("Riko likes Pune", "I moved to Pune", "assistant"), USER, reply) == "evidence_not_found"


async def test_extraction_validates_corrects_and_anchors(tmp_path):
    m = mgr(tmp_path)
    old, _ = await m.remember("User lives in Delhi", Layer.SEMANTIC, 0.7, "location")

    async def complete(msgs, json_schema=None):
        assert json_schema and "memories" in json_schema["properties"]  # schema-constrained when supported
        return json.dumps({"memories": [
            {"layer": "semantic", "claim": "User lives in Pune", "evidence": "I moved to Pune", "speaker": "user",
             "importance": 8, "confidence": 0.9, "category": "location", "corrects": old.id},
            {"layer": "semantic", "claim": "User lives in Delhi", "evidence": "live in Delhi", "speaker": "user",
             "importance": 8, "confidence": 0.9, "category": "location"},
            {"layer": "semantic", "claim": "User has an exam tomorrow", "evidence": "exam tomorrow", "speaker": "user",
             "importance": 6, "confidence": 0.9, "category": "event"},
            {"layer": "semantic", "claim": "User might like sushi", "evidence": "I moved to Pune", "speaker": "user",
             "importance": 6, "confidence": 0.3, "category": "food"},
            {"layer": "semantic", "claim": "x", "evidence": "y", "speaker": "robot"},
        ]})
    m.complete = complete
    created = await m.extract_from_turn("I don't live in Delhi, I moved to Pune. I have an exam tomorrow.",
                                        "Nice!", source_ts=time.time())
    claims = {c.content for c in created}
    assert "User lives in Pune" in claims and "User lives in Delhi" not in claims
    anchored = next(c for c in created if c.content.startswith("User has an exam tomorrow (said on "))
    assert anchored.layer == Layer.WORKING  # relative time -> short-lived, anchored to the source date
    assert m.store.get(old.id).status == Status.SUPERSEDED
    assert old.id not in {x.id for x in m.store.list("riko")}  # superseded memories are not retrieved
    assert m.rejections["negation_mismatch"] == 1 and m.rejections["low_confidence"] == 1
    assert m.rejections["invalid"] == 1


async def test_deletion_sticks_against_queued_extraction_and_clear(tmp_path):
    m = mgr(tmp_path)
    said_at = time.time() - 10  # the turn happened before the deletion
    mem, _ = await m.remember("User's favourite colour is green")
    m.store.set_summary("riko", "They talked about green.")
    assert m.forget(mem.id) and m.summary() == ""  # derived summary dropped too
    again, _ = await m.remember("User's favourite colour is green", source_ts=said_at)
    assert again is None
    fresh, created = await m.remember("User's favourite colour is green", source_ts=time.time() + 1)
    assert fresh is not None and created  # saying it again later is a new decision
    await m.block("User has a cat")  # forget something before it was ever stored
    assert (await m.remember("User has a cat", source_ts=said_at))[0] is None
    m.clear()
    assert (await m.remember("User likes rain", source_ts=said_at))[0] is None
    assert m.store.list("riko") == []


async def test_user_edit_replaces_claim_and_old_wording_is_blocked(tmp_path):
    m = mgr(tmp_path)
    mem, _ = await m.remember("User works at Acme")
    edited = await m.correct(mem.id, "User works at Initech", pinned=True)
    assert edited.content == "User works at Initech" and edited.pinned and edited.confidence == 1.0
    assert (await m.remember("User works at Acme", source_ts=time.time() - 5))[0] is None


async def test_profiles_and_characters_are_isolated(tmp_path):
    a = mgr(tmp_path, profile="alice")
    b = mgr(tmp_path, profile="bob")
    c = mgr(tmp_path, profile="alice", character="other")
    await a.remember("User likes jazz")
    a.add_message("user", "hello from alice")
    assert b.store.list("riko", profile="bob") == [] and b.context_messages() == []
    assert await b.retrieve("jazz") == [] and await c.retrieve("jazz") == []
    assert len(await a.retrieve("jazz")) == 1


def test_v1_database_is_migrated_with_backup(tmp_path):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript("""
      CREATE TABLE memories (id TEXT PRIMARY KEY, character TEXT NOT NULL, layer TEXT NOT NULL, content TEXT NOT NULL,
        category TEXT NOT NULL DEFAULT 'general', importance REAL NOT NULL, created_at REAL NOT NULL,
        last_accessed REAL NOT NULL, access_count INTEGER NOT NULL DEFAULT 0, pinned INTEGER NOT NULL DEFAULT 0,
        source TEXT NOT NULL DEFAULT 'conversation', meta TEXT NOT NULL DEFAULT '{}', embedding BLOB, embed_model TEXT);
      CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, character TEXT NOT NULL, role TEXT NOT NULL,
        content TEXT NOT NULL, created_at REAL NOT NULL, turn_id INTEGER, interrupted INTEGER NOT NULL DEFAULT 0,
        summarized INTEGER NOT NULL DEFAULT 0);
      CREATE TABLE summaries (character TEXT PRIMARY KEY, content TEXT NOT NULL, updated_at REAL NOT NULL);
      INSERT INTO memories(id, character, layer, content, importance, created_at, last_accessed)
        VALUES ('abc123', 'riko', 'semantic', 'User likes tea', 0.6, 1.0, 1.0);
      INSERT INTO messages(character, role, content, created_at) VALUES ('riko', 'user', 'hi', 1.0);
      INSERT INTO summaries VALUES ('riko', 'old summary', 1.0);""")
    con.commit()
    con.close()
    store = MemoryStore(db)
    m = store.list("riko")[0]
    assert m.content == "User likes tea" and m.profile == "default" and m.status == Status.ACTIVE
    assert store.get_summary("riko") == "old summary" and store.recent_messages("riko", 5)[0]["content"] == "hi"
    assert (tmp_path / "old.db.bak-v1").is_file()
    assert store.unextracted_pairs("riko", 10) == []  # pre-migration turns are not re-extracted
    store.close()
    again = MemoryStore(db)  # idempotent: re-opening doesn't migrate twice
    assert len(again.list("riko")) == 1
    backup = again.backup(tmp_path / "copy.db")
    again.close()
    MemoryStore.restore(backup, tmp_path / "restored.db")
    assert MemoryStore(tmp_path / "restored.db").list("riko")[0].content == "User likes tea"


def test_familiarity_is_session_based_and_tab_safe(tmp_path):
    store = MemoryStore(tmp_path / "f.db")
    from riko.core.config import FamiliarityConfig
    cfg = FamiliarityConfig()
    t0 = 1_000_000.0
    assert store.record_user_turn("riko", "tabA", t0, 1800) is True
    assert store.record_user_turn("riko", "tabB", t0 + 60, 1800) is False  # another tab, same logical session
    assert store.get_relationship("riko")["sessions"] == 1 and store.get_relationship("riko")["user_turns"] == 2
    assert familiarity_stage(store.get_relationship("riko"), cfg, t0) == 0
    for day in range(1, 8):  # one session a day for a week
        store.record_user_turn("riko", f"s{day}", t0 + day * 86400, 1800)
    rel = store.get_relationship("riko")
    assert rel["sessions"] == 8 and familiarity_stage(rel, cfg, t0 + 7 * 86400) == 2
    assert familiarity_stage({"sessions": 3, "first_seen": t0}, cfg, t0 + 86400) == 1
