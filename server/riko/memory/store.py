"""SQLite persistence + a swappable vector index.

Why SQLite + numpy instead of ChromaDB/Qdrant by default: a single user's
companion accumulates thousands, not millions, of memories. Brute-force
cosine over a few thousand 768-d vectors takes well under a millisecond, and
SQLite is in the standard library, works on every OS, and keeps everything in
one file you can back up. ``VectorIndex`` is the seam for swapping in a
dedicated vector DB later.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .models import Layer, Memory

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY, character TEXT NOT NULL, layer TEXT NOT NULL, content TEXT NOT NULL,
  category TEXT NOT NULL DEFAULT 'general', importance REAL NOT NULL, created_at REAL NOT NULL,
  last_accessed REAL NOT NULL, access_count INTEGER NOT NULL DEFAULT 0, pinned INTEGER NOT NULL DEFAULT 0,
  source TEXT NOT NULL DEFAULT 'conversation', meta TEXT NOT NULL DEFAULT '{}',
  embedding BLOB, embed_model TEXT
);
CREATE INDEX IF NOT EXISTS idx_mem_char_layer ON memories(character, layer);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, character TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
  created_at REAL NOT NULL, turn_id INTEGER, interrupted INTEGER NOT NULL DEFAULT 0,
  summarized INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_msg_char ON messages(character, id);
CREATE TABLE IF NOT EXISTS summaries (character TEXT PRIMARY KEY, content TEXT NOT NULL, updated_at REAL NOT NULL);
"""


class VectorIndex(ABC):
    @abstractmethod
    def upsert(self, id_: str, vec: np.ndarray) -> None: ...
    @abstractmethod
    def remove(self, id_: str) -> None: ...
    @abstractmethod
    def search(self, vec: np.ndarray, k: int, allowed: Optional[set[str]] = None) -> list[tuple[str, float]]: ...


class NumpyVectorIndex(VectorIndex):
    def __init__(self) -> None:
        self._vecs: dict[str, np.ndarray] = {}
        self._ids: list[str] = []
        self._mat: Optional[np.ndarray] = None

    def upsert(self, id_: str, vec: np.ndarray) -> None:
        self._vecs[id_] = vec.astype(np.float32)
        self._mat = None

    def remove(self, id_: str) -> None:
        if self._vecs.pop(id_, None) is not None:
            self._mat = None

    def __len__(self) -> int:
        return len(self._vecs)

    def search(self, vec: np.ndarray, k: int, allowed: Optional[set[str]] = None) -> list[tuple[str, float]]:
        if not self._vecs:
            return []
        if self._mat is None:
            self._ids = list(self._vecs)
            dims = {v.shape[0] for v in self._vecs.values()}
            if len(dims) != 1:  # mixed models; caller filters by model
                self._ids = [i for i in self._ids if self._vecs[i].shape[0] == vec.shape[0]]
            self._mat = np.stack([self._vecs[i] for i in self._ids]) if self._ids else np.zeros((0, vec.shape[0]))
        if self._mat.shape[0] == 0 or self._mat.shape[1] != vec.shape[0]:
            return []
        sims = self._mat @ vec.astype(np.float32)
        order = np.argsort(-sims)
        out = []
        for i in order:
            id_ = self._ids[int(i)]
            if allowed is None or id_ in allowed:
                out.append((id_, float(sims[int(i)])))
                if len(out) >= k:
                    break
        return out


class MemoryStore:
    def __init__(self, path: Path | str) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------- memories
    @staticmethod
    def _row_to_memory(r: sqlite3.Row) -> Memory:
        emb = np.frombuffer(r["embedding"], dtype=np.float32).copy() if r["embedding"] else None
        return Memory(id=r["id"], character=r["character"], layer=Layer(r["layer"]), content=r["content"],
                      category=r["category"], importance=r["importance"], created_at=r["created_at"],
                      last_accessed=r["last_accessed"], access_count=r["access_count"], pinned=bool(r["pinned"]),
                      source=r["source"], meta=json.loads(r["meta"] or "{}"), embedding=emb,
                      embed_model=r["embed_model"])

    def upsert(self, m: Memory) -> None:
        emb = m.embedding.astype(np.float32).tobytes() if m.embedding is not None else None
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (m.id, m.character, m.layer.value, m.content, m.category, float(m.importance), m.created_at,
                 m.last_accessed, m.access_count, int(m.pinned), m.source, json.dumps(m.meta), emb, m.embed_model))
            self._db.commit()

    def get(self, id_: str) -> Optional[Memory]:
        with self._lock:
            r = self._db.execute("SELECT * FROM memories WHERE id=?", (id_,)).fetchone()
        return self._row_to_memory(r) if r else None

    def delete(self, ids: Iterable[str]) -> int:
        ids = list(ids)
        if not ids:
            return 0
        with self._lock:
            cur = self._db.execute(f"DELETE FROM memories WHERE id IN ({','.join('?' * len(ids))})", ids)
            self._db.commit()
            return cur.rowcount

    def list(self, character: str, layer: Optional[Layer] = None, limit: Optional[int] = None) -> list[Memory]:
        q = "SELECT * FROM memories WHERE character=?"
        args: list = [character]
        if layer:
            q += " AND layer=?"
            args.append(layer.value)
        q += " ORDER BY created_at DESC"
        if limit:
            q += " LIMIT ?"
            args.append(int(limit))
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def touch(self, ids: Iterable[str], now: Optional[float] = None) -> None:
        now = now or time.time()
        ids = list(ids)
        if not ids:
            return
        with self._lock:
            self._db.executemany(
                "UPDATE memories SET access_count=access_count+1, last_accessed=? WHERE id=?",
                [(now, i) for i in ids])
            self._db.commit()

    # ------------------------------------------------------------- messages
    def add_message(self, character: str, role: str, content: str, turn_id: Optional[int] = None,
                    interrupted: bool = False) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO messages(character, role, content, created_at, turn_id, interrupted) VALUES (?,?,?,?,?,?)",
                (character, role, content, time.time(), turn_id, int(interrupted)))
            self._db.commit()
            return int(cur.lastrowid)

    def recent_messages(self, character: str, limit: int) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM messages WHERE character=? ORDER BY id DESC LIMIT ?", (character, limit)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def unsummarized_outside_window(self, character: str, window: int) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM messages WHERE character=? AND summarized=0 AND id NOT IN "
                "(SELECT id FROM messages WHERE character=? ORDER BY id DESC LIMIT ?) ORDER BY id",
                (character, character, window)).fetchall()
        return [dict(r) for r in rows]

    def mark_summarized(self, ids: Iterable[int]) -> None:
        with self._lock:
            self._db.executemany("UPDATE messages SET summarized=1 WHERE id=?", [(i,) for i in ids])
            self._db.commit()

    def clear_messages(self, character: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM messages WHERE character=?", (character,))
            self._db.execute("DELETE FROM summaries WHERE character=?", (character,))
            self._db.commit()

    def get_summary(self, character: str) -> str:
        with self._lock:
            r = self._db.execute("SELECT content FROM summaries WHERE character=?", (character,)).fetchone()
        return r["content"] if r else ""

    def set_summary(self, character: str, content: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO summaries VALUES (?,?,?)", (character, content, time.time()))
            self._db.commit()

    def stats(self, character: str) -> dict:
        with self._lock:
            rows = self._db.execute("SELECT layer, COUNT(*) c FROM memories WHERE character=? GROUP BY layer",
                                    (character,)).fetchall()
            msgs = self._db.execute("SELECT COUNT(*) c FROM messages WHERE character=?", (character,)).fetchone()
        return {"memories": {r["layer"]: r["c"] for r in rows}, "messages": msgs["c"]}
