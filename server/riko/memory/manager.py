"""Layered memory: short-term conversation, episodic, semantic and working memory.

Retrieval score (after Park et al. 2023, "Generative Agents"):

    score = w_rel * relevance + w_imp * strength + w_rec * recency

* relevance - cosine similarity of embeddings (keyword overlap if a memory
  has no compatible embedding);
* strength  - importance decayed by age with a per-layer half-life, boosted by
  how often the memory has been recalled (recall reinforces memory);
* recency   - exponential decay since last access.

Weak, unpinned memories are deleted by ``decay_and_cleanup``; old episodic
memories are merged into summaries by ``consolidate``.
"""
from __future__ import annotations

import json
import math
import re
import time
from typing import Awaitable, Callable, Optional

import numpy as np

from ..core.config import MemoryConfig
from ..core.logging import get_logger
from ..emotion import strip_tags
from .embeddings import Embedder, EmbeddingError
from .models import Layer, Memory, ScoredMemory
from .store import MemoryStore, NumpyVectorIndex, VectorIndex

log = get_logger("memory")

LN2 = math.log(2)
_WORD = re.compile(r"[a-z0-9']+")
_STOP = set("""a an the and or but if to of in on at for with is are was were be been am i you he she it we they
me my your our their this that these those do does did have has had so just really very can could would should
will what when where who how why not no yes okay ok oh um uh like about from as by""".split())

Completer = Callable[[list[dict]], Awaitable[str]]

EXTRACTION_PROMPT = """You maintain the long-term memory of {name}, an AI companion, about the user.
From the latest exchange, extract information worth remembering in future conversations.
Return ONLY a JSON array (it may be empty: []). Each item:
{{"layer": "semantic" | "episodic" | "working", "content": "...", "importance": 1-10, "category": "..."}}
- semantic: stable facts or preferences about the user. Write them as facts, e.g. "User's name is Alex", "User prefers tea over coffee".
- episodic: notable events or experiences the user shared, e.g. "User passed their driving test today".
- working: current goals, tasks or plans that matter for the next hours or days.
Ignore small talk, things {name} said about herself, and anything already known. At most 4 items.

Already known:
{known}

Exchange:
User: {user}
{name}: {assistant}"""

SUMMARY_PROMPT = """Update the running summary of a conversation between the user and {name}.
Keep concrete details (names, plans, decisions, feelings). At most 120 words. Return only the summary text.

Previous summary:
{summary}

New messages:
{messages}"""

CONSOLIDATE_PROMPT = """Merge these related memories about the user into 1-3 sentences that keep every concrete detail.
Return only the merged text.

{items}"""


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


def keyword_relevance(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / math.sqrt(len(ta) * len(tb))


_HEURISTICS: list[tuple[re.Pattern[str], Layer, str, float, str]] = [
    (re.compile(r"\b(?i:my name is|i'm called) ([A-Z][\w'-]+(?: [A-Z][\w'-]+)?)"), Layer.SEMANTIC, "User's name is {0}", 0.9, "identity"),
    (re.compile(r"\b(?i:call me) ([A-Z][\w'-]+)"), Layer.SEMANTIC, "User wants to be called {0}", 0.85, "identity"),
    (re.compile(r"\bmy favou?rite ([\w ]{2,30}?) is ([^.,!?]{2,60})", re.I), Layer.SEMANTIC, "User's favorite {0} is {1}", 0.7, "preference"),
    (re.compile(r"\bi (?:really )?(love|like|enjoy|hate|dislike|prefer) ([^.,!?]{2,60})", re.I), Layer.SEMANTIC, "User {0}s {1}", 0.55, "preference"),
    (re.compile(r"\bi(?: am|'m) (?:a|an) ([^.,!?]{2,50})", re.I), Layer.SEMANTIC, "User is a {0}", 0.6, "identity"),
    (re.compile(r"\bi work (?:as|at) ([^.,!?]{2,60})", re.I), Layer.SEMANTIC, "User works as/at {0}", 0.7, "work"),
    (re.compile(r"\b(?:please )?remember (?:that )?([^.!?]{4,160})", re.I), Layer.SEMANTIC, "{0}", 0.9, "explicit"),
    (re.compile(r"\bi(?:'m| am) (?:going to|planning to|trying to) ([^.!?]{4,100})", re.I), Layer.WORKING, "User is planning to {0}", 0.6, "goal"),
    (re.compile(r"\b(?:today|yesterday|this morning|last night) i ([^.!?]{4,120})", re.I), Layer.EPISODIC, "User {0}", 0.45, "event"),
]


def _third_person(text: str) -> str:
    text = re.sub(r"\bmy\b", "user's", text, flags=re.I)
    text = re.sub(r"\b(?:I am|I'm)\b", "user is", text)
    text = re.sub(r"\bI\b", "user", text)
    text = re.sub(r"\bme\b", "user", text, flags=re.I)
    return text


def heuristic_extract(user_text: str) -> list[dict]:
    out = []
    for pat, layer, template, imp, cat in _HEURISTICS:
        for m in pat.finditer(user_text):
            groups = [_third_person(g.strip()) for g in m.groups()]
            if pat.pattern.startswith("\\bi (?:really )?"):
                groups[0] = groups[0].lower()
            content = template.format(*groups)
            if cat == "explicit":
                content = _third_person(content)
                content = content[0].upper() + content[1:]
            out.append({"layer": layer.value, "content": content, "importance": imp, "category": cat})
    return out[:4]


def parse_json_array(text: str) -> list[dict]:
    text = re.sub(r"```(?:json)?", "", strip_tags(text)).strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [d for d in data if isinstance(d, dict)] if isinstance(data, list) else []


class MemoryManager:
    def __init__(self, store: MemoryStore, embedder: Embedder, cfg: MemoryConfig, character: str,
                 complete: Optional[Completer] = None, character_name: str = "Riko",
                 index: Optional[VectorIndex] = None) -> None:
        self.store = store
        self.embedder = embedder
        self.cfg = cfg
        self.character = character
        self.character_name = character_name
        self.complete = complete
        self.index = index or NumpyVectorIndex()
        self._load_index()

    # ------------------------------------------------------------- setup
    def _load_index(self) -> None:
        for m in self.store.list(self.character):
            if m.embedding is not None and m.embed_model == self.embedder.model_id:
                self.index.upsert(m.id, m.embedding)

    def set_character(self, character: str, name: str) -> None:
        self.character, self.character_name = character, name
        self.index = NumpyVectorIndex()
        self._load_index()

    async def _embed_one(self, text: str) -> Optional[np.ndarray]:
        try:
            return (await self.embedder.embed([text]))[0]
        except (EmbeddingError, Exception) as e:  # noqa: BLE001 - memory must never break a turn
            log.warning("embedding failed; using keyword relevance", extra={"error": str(e)[:200]})
            return None

    # ------------------------------------------------------------- scoring
    def strength(self, m: Memory, now: Optional[float] = None) -> float:
        if m.pinned:
            return max(m.importance, 0.5)
        now = now or time.time()
        half = self.cfg.half_life_days.get(m.layer.value, 30.0)
        age_days = max(0.0, (now - m.created_at) / 86400.0)
        decay = math.exp(-LN2 * age_days / half) if half > 0 else 1.0
        return min(1.0, m.importance * decay * (1.0 + 0.15 * math.log1p(m.access_count)))

    def recency(self, m: Memory, now: Optional[float] = None) -> float:
        hours = max(0.0, ((now or time.time()) - m.last_accessed) / 3600.0)
        return math.exp(-LN2 * hours / max(1e-3, self.cfg.recency_half_life_hours))

    # ------------------------------------------------------------- retrieval
    async def retrieve(self, query: str, k: Optional[int] = None) -> list[ScoredMemory]:
        k = self.cfg.top_k if k is None else k
        if k <= 0 or not query.strip():
            return []
        now = time.time()
        candidates = [m for m in self.store.list(self.character) if m.layer != Layer.WORKING]
        if not candidates:
            return []
        qvec = await self._embed_one(query)
        sims: dict[str, float] = {}
        if qvec is not None:
            sims = dict(self.index.search(qvec, k=max(50, k * 5), allowed={m.id for m in candidates}))
        w = self.cfg.weights
        scored = []
        for m in candidates:
            kw = keyword_relevance(query, m.content)
            vec = sims.get(m.id)
            rel = max(max(0.0, vec), kw * 0.9) if vec is not None else kw
            if rel < self.cfg.min_relevance:
                continue
            st = self.strength(m, now)
            score = w.relevance * rel + w.importance * st + w.recency * self.recency(m, now)
            scored.append(ScoredMemory(m, round(score, 4), round(rel, 4), round(st, 4)))
        scored.sort(key=lambda s: s.score, reverse=True)
        top = scored[:k]
        self.store.touch([s.memory.id for s in top], now)  # recall reinforces memory
        return top

    def working_memories(self, limit: int = 5) -> list[Memory]:
        now = time.time()
        items = [m for m in self.store.list(self.character, Layer.WORKING)
                 if m.meta.get("expires_at", now + 1) > now]
        return items[:limit]

    @staticmethod
    def format_memories(items: list[ScoredMemory]) -> str:
        return "\n".join(f"- {s.memory.content}" for s in items)

    @staticmethod
    def format_working(items: list[Memory]) -> str:
        return "\n".join(f"- {m.content}" for m in items)

    # ------------------------------------------------------------- writing
    async def remember(self, content: str, layer: Layer = Layer.SEMANTIC, importance: float = 0.5,
                       category: str = "general", pinned: bool = False, source: str = "conversation"
                       ) -> tuple[Memory, bool]:
        """Store a memory, merging into a near-duplicate if one exists. Returns (memory, created)."""
        content = " ".join(content.split())[:500]
        importance = max(0.0, min(1.0, importance))
        vec = await self._embed_one(content)
        now = time.time()
        same_layer = self.store.list(self.character, layer)
        dup: Optional[Memory] = None
        low = content.lower()
        for m in same_layer:
            if m.content.lower() == low:
                dup = m
                break
        # Similar-looking *events* are usually distinct ("visited Kyoto" vs "visited Osaka"),
        # so vector dedup only applies to facts and goals.
        if dup is None and vec is not None and layer != Layer.EPISODIC:
            hits = self.index.search(vec, k=1, allowed={m.id for m in same_layer})
            if hits and hits[0][1] >= self.cfg.dedup_similarity:
                dup = next(m for m in same_layer if m.id == hits[0][0])
        if dup is not None:
            dup.importance = max(dup.importance, importance)
            dup.pinned = dup.pinned or pinned
            dup.last_accessed = now
            dup.access_count += 1
            if layer == Layer.SEMANTIC and dup.content.lower() != low:
                dup.content = content  # newer phrasing of the same fact wins
                if vec is not None:
                    dup.embedding, dup.embed_model = vec, self.embedder.model_id
                    self.index.upsert(dup.id, vec)
            self.store.upsert(dup)
            return dup, False
        m = Memory(content=content, layer=layer, character=self.character, importance=importance,
                   category=category, pinned=pinned, source=source)
        if layer == Layer.WORKING:
            m.meta["expires_at"] = now + self.cfg.half_life_days.get("working", 2.0) * 2 * 86400
        if vec is not None:
            m.embedding, m.embed_model = vec, self.embedder.model_id
            self.index.upsert(m.id, vec)
        self.store.upsert(m)
        log.info("memory stored", extra={"layer": layer.value, "category": category, "importance": importance})
        return m, True

    def forget(self, memory_id: str) -> bool:
        self.index.remove(memory_id)
        return self.store.delete([memory_id]) > 0

    async def extract_from_turn(self, user_text: str, assistant_text: str, mode: Optional[str] = None
                                ) -> list[Memory]:
        """Pull durable memories out of one exchange. Runs in the background after a turn."""
        mode = mode or self.cfg.extraction
        if mode == "off" or len(user_text.strip()) < 3:
            return []
        items: list[dict] = []
        if mode == "llm" and self.complete is not None:
            known = "\n".join(f"- {m.content}" for m in self.store.list(self.character, Layer.SEMANTIC, limit=25))
            prompt = EXTRACTION_PROMPT.format(name=self.character_name, known=known or "(nothing yet)",
                                              user=user_text, assistant=strip_tags(assistant_text))
            try:
                raw = await self.complete([{"role": "user", "content": prompt}])
                items = parse_json_array(raw)
            except Exception as e:  # noqa: BLE001
                log.warning("LLM memory extraction failed; using heuristics", extra={"error": str(e)[:200]})
        if not items:
            items = heuristic_extract(user_text)
        created: list[Memory] = []
        for it in items[:4]:
            content = str(it.get("content", "")).strip()
            try:
                layer = Layer(str(it.get("layer", "semantic")).lower())
            except ValueError:
                layer = Layer.SEMANTIC
            imp = float(it.get("importance", 5))
            imp = imp / 10.0 if imp > 1 else imp
            if len(content) < 4:
                continue
            mem, is_new = await self.remember(content, layer, imp, str(it.get("category", "general"))[:32])
            if is_new:
                created.append(mem)
        return created

    # ------------------------------------------------------------- short-term
    def add_message(self, role: str, content: str, turn_id: Optional[int] = None, interrupted: bool = False) -> None:
        self.store.add_message(self.character, role, content, turn_id, interrupted)

    def context_messages(self) -> list[dict]:
        msgs = self.store.recent_messages(self.character, self.cfg.short_term_messages)
        out = []
        for m in msgs:
            content = m["content"] + (" -- (interrupted by the user)" if m["interrupted"] else "")
            out.append({"role": m["role"], "content": content})
        while out and out[0]["role"] == "assistant":  # keep a clean user-first window
            out.pop(0)
        return out

    def summary(self) -> str:
        return self.store.get_summary(self.character)

    async def maybe_summarize(self) -> bool:
        """Fold messages that scrolled out of the short-term window into a running summary."""
        pending = self.store.unsummarized_outside_window(self.character, self.cfg.short_term_messages)
        if len(pending) < self.cfg.summarize_batch or self.complete is None:
            return False
        lines = "\n".join(f"{'User' if p['role'] == 'user' else self.character_name}: {strip_tags(p['content'])}"
                          for p in pending)
        prompt = SUMMARY_PROMPT.format(name=self.character_name, summary=self.summary() or "(none)", messages=lines)
        try:
            text = strip_tags(await self.complete([{"role": "user", "content": prompt}])).strip()
        except Exception as e:  # noqa: BLE001
            log.warning("summarization failed", extra={"error": str(e)[:200]})
            return False
        if text:
            self.store.set_summary(self.character, text[:1500])
            self.store.mark_summarized(p["id"] for p in pending)
            return True
        return False

    # ------------------------------------------------------------- maintenance
    def decay_and_cleanup(self, now: Optional[float] = None) -> int:
        now = now or time.time()
        doomed = []
        for m in self.store.list(self.character):
            if m.pinned:
                continue
            if m.layer == Layer.WORKING and m.meta.get("expires_at", now + 1) <= now:
                doomed.append(m.id)
            elif self.strength(m, now) < self.cfg.cleanup_threshold:
                doomed.append(m.id)
        for i in doomed:
            self.index.remove(i)
        n = self.store.delete(doomed)
        if n:
            log.info("memory cleanup", extra={"deleted": n})
        return n

    async def consolidate(self, now: Optional[float] = None, batch: int = 10) -> int:
        """Merge old, fading episodic memories into summary memories. Returns memories merged."""
        if self.complete is None:
            return 0
        now = now or time.time()
        cutoff = now - self.cfg.consolidate_after_days * 86400
        old = sorted((m for m in self.store.list(self.character, Layer.EPISODIC)
                      if not m.pinned and m.created_at < cutoff and m.category != "consolidated"
                      and self.strength(m, now) < 0.35), key=lambda m: m.created_at)
        merged = 0
        for i in range(0, len(old), batch):
            group = old[i:i + batch]
            if len(group) < 3:
                break
            items = "\n".join(f"- {m.content}" for m in group)
            try:
                text = strip_tags(await self.complete([{"role": "user",
                                                        "content": CONSOLIDATE_PROMPT.format(items=items)}])).strip()
            except Exception as e:  # noqa: BLE001
                log.warning("consolidation failed", extra={"error": str(e)[:200]})
                return merged
            if not text:
                continue
            mem, _ = await self.remember(text, Layer.EPISODIC, min(0.9, max(m.importance for m in group) + 0.05),
                                         "consolidated", source="consolidation")
            for m in group:
                self.index.remove(m.id)
            self.store.delete(m.id for m in group)
            merged += len(group)
        return merged
