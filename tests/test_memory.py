import time

import numpy as np

from riko.core.config import MemoryConfig
from riko.memory import (Embedder, EmbeddingError, HashEmbedder, Layer, MemoryManager, MemoryStore,
                         heuristic_extract, parse_json_array)


def mgr(tmp_path, complete=None, embedder=None, **kw):
    return MemoryManager(MemoryStore(tmp_path / "m.db"), embedder or HashEmbedder(256), MemoryConfig(**kw), "riko",
                         complete=complete)


async def test_create_dedup_and_persist(tmp_path):
    m = mgr(tmp_path)
    a, created = await m.remember("User loves spicy ramen", Layer.SEMANTIC, 0.5)
    b, created2 = await m.remember("user loves spicy ramen", Layer.SEMANTIC, 0.9)
    assert created and not created2 and a.id == b.id and b.importance == 0.9
    m2 = mgr(tmp_path)  # reopen: index rebuilt from SQLite
    assert len(m2.store.list("riko")) == 1 and len(m2.index) == 1


async def test_retrieval_ranks_relevant_and_reinforces(tmp_path):
    m = mgr(tmp_path)
    await m.remember("User has a cat named Mochi", Layer.SEMANTIC, 0.6)
    await m.remember("User is learning to play the guitar", Layer.SEMANTIC, 0.6)
    await m.remember("User went hiking in the mountains", Layer.EPISODIC, 0.4)
    res = await m.retrieve("how is my cat Mochi doing?")
    assert res and res[0].memory.content == "User has a cat named Mochi"
    assert m.store.get(res[0].memory.id).access_count == 1


async def test_importance_breaks_ties(tmp_path):
    m = mgr(tmp_path)
    await m.remember("User likes green tea", Layer.SEMANTIC, 0.2)
    await m.remember("User likes green tea a lot every morning", Layer.SEMANTIC, 0.95)
    res = await m.retrieve("green tea")
    assert res[0].memory.importance == 0.95


async def test_decay_cleanup_and_working_expiry(tmp_path):
    m = mgr(tmp_path)
    old, _ = await m.remember("User went to a cafe", Layer.EPISODIC, 0.3)
    pinned, _ = await m.remember("User's birthday is May 3", Layer.SEMANTIC, 0.3, pinned=True)
    goal, _ = await m.remember("User is preparing a talk", Layer.WORKING, 0.6)
    fresh, _ = await m.remember("User adopted a puppy", Layer.EPISODIC, 0.8)
    old.created_at -= 200 * 86400
    m.store.upsert(old)
    assert m.strength(old) < 0.05 < m.strength(fresh)
    assert m.working_memories()
    deleted = m.decay_and_cleanup(now=time.time() + 5 * 86400)  # goal expires after ~4 days
    ids = {x.id for x in m.store.list("riko")}
    assert deleted == 2 and old.id not in ids and goal.id not in ids and pinned.id in ids and fresh.id in ids


async def test_llm_extraction_with_code_fences(tmp_path):
    async def complete(msgs):
        assert "User: my name is Alex" in msgs[0]["content"]
        return '```json\n[{"layer":"semantic","content":"User\'s name is Alex","importance":9,"category":"identity"},' \
               '{"layer":"bogus","content":"x"}]\n```'
    m = mgr(tmp_path, complete)
    created = await m.extract_from_turn("my name is Alex", "[happy] Nice to meet you, Alex.")
    assert [c.content for c in created] == ["User's name is Alex"] and created[0].importance == 0.9


async def test_extraction_falls_back_to_heuristics(tmp_path):
    async def broken(msgs):
        raise RuntimeError("server down")
    m = mgr(tmp_path, broken)
    created = await m.extract_from_turn("Call me Kai. I really love horror movies.", "ok")
    assert {c.content for c in created} == {"User wants to be called Kai", "User loves horror movies"}
    assert heuristic_extract("Remember that my exam is on Friday")[0]["content"] == "User's exam is on Friday"
    assert parse_json_array("nothing here") == []


async def test_short_term_window_and_summary(tmp_path):
    calls = []

    async def complete(msgs):
        calls.append(msgs)
        return "They chatted about games."
    m = mgr(tmp_path, complete, short_term_messages=4, summarize_batch=4)
    for i in range(5):
        m.add_message("user", f"message {i}", i)
        m.add_message("assistant", f"reply {i}", i, interrupted=(i == 4))
    ctx = m.context_messages()
    assert len(ctx) == 4 and ctx[0]["role"] == "user" and "(interrupted" in ctx[-1]["content"]
    assert await m.maybe_summarize() and m.summary() == "They chatted about games."
    assert not await m.maybe_summarize()  # nothing new to fold in


async def test_consolidation(tmp_path):
    async def complete(msgs):
        return "User had a busy week of trips."
    m = mgr(tmp_path, complete)
    for i in range(4):
        mem, _ = await m.remember(f"User visited place number {i} on a trip", Layer.EPISODIC, 0.2)
        mem.created_at -= 30 * 86400
        m.store.upsert(mem)
    assert await m.consolidate() == 4
    left = m.store.list("riko", Layer.EPISODIC)
    assert len(left) == 1 and left[0].category == "consolidated"


class FailingEmbedder(Embedder):
    model_id = "fail"

    async def embed(self, texts):
        raise EmbeddingError("down")


async def test_embedding_failure_degrades_to_keywords(tmp_path):
    m = mgr(tmp_path, embedder=FailingEmbedder())
    await m.remember("User plays the violin", Layer.SEMANTIC, 0.6)
    res = await m.retrieve("do I still play violin?")
    assert res and res[0].memory.content == "User plays the violin"
    assert m.forget(res[0].memory.id) and not m.store.list("riko")
