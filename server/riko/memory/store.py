"""SQLite persistence + a swappable vector index.

Why SQLite + numpy instead of ChromaDB/Qdrant: a single user's companion accumulates thousands, not
millions, of memories. Brute-force cosine over a few thousand vectors takes well under a millisecond, and
SQLite is in the standard library and keeps everything in one file you can back up. ``VectorIndex`` is the
seam for a dedicated vector DB later.

Schema versioning: ``PRAGMA user_version``. Migrations are additive and run in one transaction, after a
file backup (``<db>.bak-v<old>``). All state is scoped by (profile, character) - invariant I05.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from ..core.logging import get_logger
from .models import Layer, Memory, Status

log = get_logger("memory")

SCHEMA_VERSION = 2

SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY, character TEXT NOT NULL, layer TEXT NOT NULL, content TEXT NOT NULL,
  category TEXT NOT NULL DEFAULT 'general', importance REAL NOT NULL, created_at REAL NOT NULL,
  last_accessed REAL NOT NULL, access_count INTEGER NOT NULL DEFAULT 0, pinned INTEGER NOT NULL DEFAULT 0,
  source TEXT NOT NULL DEFAULT 'conversation', meta TEXT NOT NULL DEFAULT '{}',
  embedding BLOB, embed_model TEXT
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, character TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
  created_at REAL NOT NULL, turn_id INTEGER, interrupted INTEGER NOT NULL DEFAULT 0,
  summarized INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS summaries (character TEXT PRIMARY KEY, content TEXT NOT NULL, updated_at REAL NOT NULL);
"""

# v1 -> v2: additive columns + new tables. Each statement is idempotent-safe inside the migration.
MIGRATION_V2 = [
    "ALTER TABLE memories ADD COLUMN profile TEXT NOT NULL DEFAULT 'default'",
    "ALTER TABLE memories ADD COLUMN confidence REAL NOT NULL DEFAULT 0.7",
    "ALTER TABLE memories ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
    "ALTER TABLE memories ADD COLUMN evidence TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE memories ADD COLUMN speaker TEXT NOT NULL DEFAULT 'user'",
    "ALTER TABLE memories ADD COLUMN updated_at REAL",
    "ALTER TABLE memories ADD COLUMN source_turn INTEGER",
    "UPDATE memories SET updated_at = created_at WHERE updated_at IS NULL",
    "ALTER TABLE messages ADD COLUMN profile TEXT NOT NULL DEFAULT 'default'",
    "ALTER TABLE messages ADD COLUMN source TEXT NOT NULL DEFAULT 'user'",
    "ALTER TABLE messages ADD COLUMN extracted INTEGER NOT NULL DEFAULT 0",
    # rows that existed before v2 were already processed by the old pipeline
    "UPDATE messages SET extracted = 1",
    """CREATE TABLE IF NOT EXISTS summaries_v2 (profile TEXT NOT NULL, character TEXT NOT NULL, content TEXT NOT NULL,
       updated_at REAL NOT NULL, PRIMARY KEY (profile, character))""",
    "INSERT OR IGNORE INTO summaries_v2 SELECT 'default', character, content, updated_at FROM summaries",
    "DROP TABLE summaries",
]

SCHEMA_V2_EXTRA = """
CREATE INDEX IF NOT EXISTS idx_mem_scope ON memories(profile, character, layer);
CREATE INDEX IF NOT EXISTS idx_msg_scope ON messages(profile, character, id);
CREATE TABLE IF NOT EXISTS tombstones (
  id INTEGER PRIMARY KEY AUTOINCREMENT, profile TEXT NOT NULL, character TEXT NOT NULL, content TEXT NOT NULL,
  embedding BLOB, deleted_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS relationship (
  profile TEXT NOT NULL, character TEXT NOT NULL, first_seen REAL, last_seen REAL,
  sessions INTEGER NOT NULL DEFAULT 0, user_turns INTEGER NOT NULL DEFAULT 0,
  session_key TEXT, session_last_activity REAL, PRIMARY KEY (profile, character)
);
CREATE TABLE IF NOT EXISTS mood (
  profile TEXT NOT NULL, character TEXT NOT NULL, valence REAL NOT NULL, arousal REAL NOT NULL,
  updated_at REAL NOT NULL, PRIMARY KEY (profile, character)
);
CREATE TABLE IF NOT EXISTS notes (
  id TEXT PRIMARY KEY, profile TEXT NOT NULL, character TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY, profile TEXT NOT NULL, character TEXT NOT NULL, title TEXT NOT NULL,
  due_at REAL, status TEXT NOT NULL DEFAULT 'open', created_at REAL NOT NULL, updated_at REAL NOT NULL,
  delivered_at REAL, source_turn INTEGER
);
CREATE TABLE IF NOT EXISTS settings (
  profile TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY (profile, key)
);
"""

_MEM_COLS = ("id, character, layer, content, category, importance, created_at, last_accessed, access_count, "
             "pinned, source, meta, embedding, embed_model, profile, confidence, status, evidence, speaker, "
             "updated_at, source_turn")


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
        self.path = str(path)
        if self.path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._migrate()
        if self.path != ":memory:" and os.name == "posix":
            try:
                os.chmod(self.path, 0o600)  # private conversation data
            except OSError:
                pass

    # ------------------------------------------------------------- schema
    def _migrate(self) -> None:
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        existed = self._db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'").fetchone() is not None
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"database {self.path} has schema v{version}, newer than this Riko "
                               f"(v{SCHEMA_VERSION}); refusing to downgrade it")
        if version == SCHEMA_VERSION:
            return
        if existed and version < 2:
            self.backup(f"{self.path}.bak-v{version or 1}")
        try:
            self._db.execute("BEGIN")
            if not existed:
                for stmt in filter(str.strip, SCHEMA_V1.split(";")):
                    self._db.execute(stmt)
            for stmt in MIGRATION_V2:
                self._db.execute(stmt)
            for stmt in filter(str.strip, SCHEMA_V2_EXTRA.split(";")):
                self._db.execute(stmt)
            self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._db.execute("COMMIT")
        except Exception:
            self._db.execute("ROLLBACK")
            raise
        if existed:
            log.info("memory database migrated", extra={"from": version or 1, "to": SCHEMA_VERSION})

    def backup(self, dest: str | Path) -> Path:
        """Consistent online copy (sqlite backup API). Restore = stop Riko, copy the file back."""
        if self.path == ":memory:":
            raise RuntimeError("cannot back up an in-memory database")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            target = sqlite3.connect(str(dest))
            try:
                self._db.backup(target)
            finally:
                target.close()
        return dest

    @staticmethod
    def restore(backup: str | Path, db_path: str | Path) -> None:
        """Replace the database file with a backup (Riko must not be running)."""
        for suffix in ("-wal", "-shm"):
            Path(f"{db_path}{suffix}").unlink(missing_ok=True)
        shutil.copyfile(backup, db_path)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _exec(self, sql: str, args: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock, self._db:  # one transaction per call
            return self._db.execute(sql, tuple(args))

    def _rows(self, sql: str, args: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, tuple(args)).fetchall()

    # ------------------------------------------------------------- memories
    @staticmethod
    def _row_to_memory(r: sqlite3.Row) -> Memory:
        emb = np.frombuffer(r["embedding"], dtype=np.float32).copy() if r["embedding"] else None
        return Memory(id=r["id"], character=r["character"], layer=Layer(r["layer"]), content=r["content"],
                      category=r["category"], importance=r["importance"], created_at=r["created_at"],
                      last_accessed=r["last_accessed"], access_count=r["access_count"], pinned=bool(r["pinned"]),
                      source=r["source"], meta=json.loads(r["meta"] or "{}"), embedding=emb,
                      embed_model=r["embed_model"], profile=r["profile"], confidence=r["confidence"],
                      status=Status(r["status"]), evidence=r["evidence"], speaker=r["speaker"],
                      updated_at=r["updated_at"] or r["created_at"], source_turn=r["source_turn"])

    def upsert(self, m: Memory) -> None:
        emb = m.embedding.astype(np.float32).tobytes() if m.embedding is not None else None
        self._exec(
            f"INSERT OR REPLACE INTO memories ({_MEM_COLS}) VALUES ({','.join('?' * 21)})",
            (m.id, m.character, m.layer.value, m.content, m.category, float(m.importance), m.created_at,
             m.last_accessed, m.access_count, int(m.pinned), m.source, json.dumps(m.meta), emb, m.embed_model,
             m.profile, float(m.confidence), m.status.value, m.evidence, m.speaker, m.updated_at, m.source_turn))

    def get(self, id_: str, *, profile: Optional[str] = None) -> Optional[Memory]:
        q, args = "SELECT * FROM memories WHERE id=?", [id_]
        if profile is not None:
            q += " AND profile=?"
            args.append(profile)
        rows = self._rows(q, args)
        return self._row_to_memory(rows[0]) if rows else None

    def delete(self, ids: Iterable[str]) -> int:
        ids = list(ids)
        if not ids:
            return 0
        return self._exec(f"DELETE FROM memories WHERE id IN ({','.join('?' * len(ids))})", ids).rowcount

    def list(self, character: str, layer: Optional[Layer] = None, limit: Optional[int] = None, *,
             profile: str = "default", include_inactive: bool = False) -> list[Memory]:
        q = "SELECT * FROM memories WHERE character=? AND profile=?"
        args: list = [character, profile]
        if layer:
            q += " AND layer=?"
            args.append(layer.value)
        if not include_inactive:
            q += " AND status != 'superseded'"
        q += " ORDER BY created_at DESC"
        if limit:
            q += " LIMIT ?"
            args.append(int(limit))
        return [self._row_to_memory(r) for r in self._rows(q, args)]

    def touch(self, ids: Iterable[str], now: Optional[float] = None) -> None:
        now = now or time.time()
        ids = list(ids)
        if not ids:
            return
        with self._lock, self._db:
            self._db.executemany("UPDATE memories SET access_count=access_count+1, last_accessed=? WHERE id=?",
                                 [(now, i) for i in ids])

    # ------------------------------------------------------------- tombstones (deletion must stick, I06)
    def add_tombstone(self, character: str, content: str, embedding: Optional[np.ndarray], *,
                      profile: str = "default", at: Optional[float] = None) -> None:
        emb = embedding.astype(np.float32).tobytes() if embedding is not None else None
        self._exec("INSERT INTO tombstones(profile, character, content, embedding, deleted_at) VALUES (?,?,?,?,?)",
                   (profile, character, content, emb, at or time.time()))

    def tombstones(self, character: str, *, profile: str = "default", after: float = 0.0
                   ) -> list[tuple[str, Optional[np.ndarray], float]]:
        rows = self._rows("SELECT content, embedding, deleted_at FROM tombstones WHERE profile=? AND character=? "
                          "AND deleted_at > ? ORDER BY id DESC LIMIT 500", (profile, character, after))
        return [(r["content"], np.frombuffer(r["embedding"], dtype=np.float32).copy() if r["embedding"] else None,
                 r["deleted_at"]) for r in rows]

    # ------------------------------------------------------------- messages
    def add_message(self, character: str, role: str, content: str, turn_id: Optional[int] = None,
                    interrupted: bool = False, *, profile: str = "default", source: str = "user") -> int:
        cur = self._exec(
            "INSERT INTO messages(character, role, content, created_at, turn_id, interrupted, profile, source) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (character, role, content, time.time(), turn_id, int(interrupted), profile, source))
        return int(cur.lastrowid)

    def recent_messages(self, character: str, limit: int, *, profile: str = "default") -> list[dict]:
        rows = self._rows("SELECT * FROM messages WHERE character=? AND profile=? ORDER BY id DESC LIMIT ?",
                          (character, profile, limit))
        return [dict(r) for r in reversed(rows)]

    def unsummarized_outside_window(self, character: str, window: int, *, profile: str = "default") -> list[dict]:
        rows = self._rows(
            "SELECT * FROM messages WHERE character=? AND profile=? AND summarized=0 AND id NOT IN "
            "(SELECT id FROM messages WHERE character=? AND profile=? ORDER BY id DESC LIMIT ?) ORDER BY id",
            (character, profile, character, profile, window))
        return [dict(r) for r in rows]

    def mark_summarized(self, ids: Iterable[int]) -> None:
        with self._lock, self._db:
            self._db.executemany("UPDATE messages SET summarized=1 WHERE id=?", [(i,) for i in ids])

    def mark_extracted(self, ids: Iterable[int]) -> None:
        with self._lock, self._db:
            self._db.executemany("UPDATE messages SET extracted=1 WHERE id=?", [(i,) for i in ids])

    def unextracted_pairs(self, character: str, limit: int, *, profile: str = "default") -> list[tuple[dict, dict]]:
        """(user, assistant) message pairs not yet processed by memory extraction (restart backfill)."""
        rows = [dict(r) for r in self._rows(
            "SELECT * FROM messages WHERE character=? AND profile=? AND extracted=0 ORDER BY id DESC LIMIT ?",
            (character, profile, limit * 2 + 2))]
        rows.reverse()
        pairs = []
        for a, b in zip(rows, rows[1:]):
            if a["role"] == "user" and b["role"] == "assistant" and a["source"] == "user":
                pairs.append((a, b))
        return pairs[-limit:] if limit else []

    def clear_messages(self, character: str, *, profile: str = "default") -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM messages WHERE character=? AND profile=?", (character, profile))
            self._db.execute("DELETE FROM summaries_v2 WHERE character=? AND profile=?", (character, profile))

    def get_summary(self, character: str, *, profile: str = "default") -> str:
        rows = self._rows("SELECT content FROM summaries_v2 WHERE character=? AND profile=?", (character, profile))
        return rows[0]["content"] if rows else ""

    def set_summary(self, character: str, content: str, *, profile: str = "default") -> None:
        self._exec("INSERT OR REPLACE INTO summaries_v2 VALUES (?,?,?,?)", (profile, character, content, time.time()))

    # ------------------------------------------------------------- relationship / familiarity
    def get_relationship(self, character: str, *, profile: str = "default") -> dict:
        rows = self._rows("SELECT * FROM relationship WHERE profile=? AND character=?", (profile, character))
        return dict(rows[0]) if rows else {"profile": profile, "character": character, "first_seen": None,
                                           "last_seen": None, "sessions": 0, "user_turns": 0,
                                           "session_key": None, "session_last_activity": None}

    def record_user_turn(self, character: str, session_key: str, now: float, gap_s: float, *,
                         profile: str = "default") -> bool:
        """Count one completed user turn atomically. Returns True if it opened a new logical session.

        A logical session is counted once, at its first completed user turn. Activity from another tab
        within ``gap_s`` continues the same logical session, so reconnects and tabs don't inflate counts."""
        with self._lock, self._db:
            r = self._db.execute("SELECT * FROM relationship WHERE profile=? AND character=?",
                                 (profile, character)).fetchone()
            if r is None:
                self._db.execute("INSERT INTO relationship VALUES (?,?,?,?,1,1,?,?)",
                                 (profile, character, now, now, session_key, now))
                return True
            last = r["session_last_activity"] or 0.0
            new_session = (now - last) > gap_s or r["session_key"] is None
            self._db.execute(
                "UPDATE relationship SET last_seen=?, user_turns=user_turns+1, sessions=sessions+?, "
                "session_key=?, session_last_activity=?, first_seen=COALESCE(first_seen, ?) "
                "WHERE profile=? AND character=?",
                (now, 1 if new_session else 0, session_key if new_session else r["session_key"], now, now,
                 profile, character))
            return new_session

    # ------------------------------------------------------------- mood
    def get_mood(self, character: str, *, profile: str = "default") -> Optional[tuple[float, float, float]]:
        rows = self._rows("SELECT valence, arousal, updated_at FROM mood WHERE profile=? AND character=?",
                          (profile, character))
        return (rows[0]["valence"], rows[0]["arousal"], rows[0]["updated_at"]) if rows else None

    def set_mood(self, character: str, valence: float, arousal: float, at: float, *, profile: str = "default") -> None:
        self._exec("INSERT OR REPLACE INTO mood VALUES (?,?,?,?,?)", (profile, character, valence, arousal, at))

    def reset_mood(self, character: str, *, profile: str = "default") -> None:
        self._exec("DELETE FROM mood WHERE profile=? AND character=?", (profile, character))

    # ------------------------------------------------------------- notes
    def add_note(self, character: str, title: str, body: str, *, profile: str = "default") -> str:
        nid = uuid.uuid4().hex[:12]
        self._exec("INSERT INTO notes VALUES (?,?,?,?,?,?)", (nid, profile, character, title, body, time.time()))
        return nid

    def list_notes(self, character: str, query: str = "", limit: int = 20, *, profile: str = "default") -> list[dict]:
        q = "SELECT * FROM notes WHERE profile=? AND character=?"
        args: list = [profile, character]
        for word in query.split()[:6]:
            q += " AND (title LIKE ? OR body LIKE ?)"
            args += [f"%{word}%", f"%{word}%"]
        rows = self._rows(q + " ORDER BY created_at DESC LIMIT ?", [*args, limit])
        return [dict(r) for r in rows]

    def delete_note(self, nid: str, *, profile: str = "default") -> bool:
        return self._exec("DELETE FROM notes WHERE id=? AND profile=?", (nid, profile)).rowcount > 0

    # ------------------------------------------------------------- tasks / reminders
    def add_task(self, character: str, title: str, due_at: Optional[float] = None, *, profile: str = "default",
                 source_turn: Optional[int] = None) -> str:
        tid = uuid.uuid4().hex[:12]
        now = time.time()
        self._exec("INSERT INTO tasks(id, profile, character, title, due_at, status, created_at, updated_at, "
                   "source_turn) VALUES (?,?,?,?,?,'open',?,?,?)",
                   (tid, profile, character, title, due_at, now, now, source_turn))
        return tid

    def list_tasks(self, character: str, status: Optional[str] = "open", limit: int = 50, *,
                   profile: str = "default") -> list[dict]:
        q, args = "SELECT * FROM tasks WHERE profile=? AND character=?", [profile, character]
        if status:
            q += " AND status=?"
            args.append(status)
        return [dict(r) for r in self._rows(q + " ORDER BY COALESCE(due_at, created_at) LIMIT ?", [*args, limit])]

    def update_task(self, tid: str, *, profile: str = "default", **fields: Any) -> bool:
        allowed = {k: v for k, v in fields.items() if k in ("title", "due_at", "status", "delivered_at")}
        if not allowed:
            return False
        sets = ", ".join(f"{k}=?" for k in allowed)
        return self._exec(f"UPDATE tasks SET {sets}, updated_at=? WHERE id=? AND profile=?",
                          [*allowed.values(), time.time(), tid, profile]).rowcount > 0

    def claim_task_delivery(self, tid: str, *, profile: str = "default") -> bool:
        """Atomically mark a due reminder as delivered; False if another tab already claimed it."""
        return self._exec("UPDATE tasks SET delivered_at=? WHERE id=? AND profile=? AND delivered_at IS NULL",
                          (time.time(), tid, profile)).rowcount > 0

    def delete_task(self, tid: str, *, profile: str = "default") -> bool:
        return self._exec("DELETE FROM tasks WHERE id=? AND profile=?", (tid, profile)).rowcount > 0

    def due_tasks(self, character: str, now: float, *, profile: str = "default") -> list[dict]:
        rows = self._rows("SELECT * FROM tasks WHERE profile=? AND character=? AND status='open' AND due_at IS NOT "
                          "NULL AND due_at <= ? AND delivered_at IS NULL ORDER BY due_at", (profile, character, now))
        return [dict(r) for r in rows]

    # ------------------------------------------------------------- settings (per profile)
    def get_setting(self, key: str, default: Optional[str] = None, *, profile: str = "default") -> Optional[str]:
        rows = self._rows("SELECT value FROM settings WHERE profile=? AND key=?", (profile, key))
        return rows[0]["value"] if rows else default

    def set_setting(self, key: str, value: str, *, profile: str = "default") -> None:
        self._exec("INSERT OR REPLACE INTO settings VALUES (?,?,?)", (profile, key, value))

    # ------------------------------------------------------------- bulk
    def clear_memories(self, character: str, *, profile: str = "default") -> int:
        with self._lock, self._db:
            n = self._db.execute("DELETE FROM memories WHERE profile=? AND character=?", (profile, character)).rowcount
            self._db.execute("DELETE FROM summaries_v2 WHERE profile=? AND character=?", (profile, character))
            self._db.execute("INSERT INTO tombstones(profile, character, content, embedding, deleted_at) "
                             "VALUES (?,?,'*',NULL,?)", (profile, character, time.time()))
        return n

    def export(self, character: str, *, profile: str = "default") -> dict:
        mems = [m.public() for m in self.list(character, profile=profile, include_inactive=True)]
        return {"profile": profile, "character": character, "exported_at": time.time(), "schema": SCHEMA_VERSION,
                "memories": mems, "summary": self.get_summary(character, profile=profile),
                "relationship": self.get_relationship(character, profile=profile),
                "notes": self.list_notes(character, limit=1000, profile=profile),
                "tasks": self.list_tasks(character, None, 1000, profile=profile),
                "messages": self.recent_messages(character, 100000, profile=profile)}

    def stats(self, character: str, *, profile: str = "default") -> dict:
        rows = self._rows("SELECT layer, COUNT(*) c FROM memories WHERE character=? AND profile=? "
                          "AND status != 'superseded' GROUP BY layer", (character, profile))
        msgs = self._rows("SELECT COUNT(*) c FROM messages WHERE character=? AND profile=?", (character, profile))
        return {"memories": {r["layer"]: r["c"] for r in rows}, "messages": msgs[0]["c"]}
