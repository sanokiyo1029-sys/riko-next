"""Memory data model."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class Layer(str, Enum):
    EPISODIC = "episodic"          # events and experiences the user shared
    SEMANTIC = "semantic"          # stable facts and preferences
    RELATIONSHIP = "relationship"  # shared history, interaction preferences, assistant opinions about it
    TASK = "task"                  # goals and unfinished tasks (until done or stale)
    WORKING = "working"            # current, short-lived context (expires in days)


class Status(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"   # replaced by an explicit correction; kept for provenance, never retrieved
    DISPUTED = "disputed"       # unresolved contradiction; retrievable only as uncertain


@dataclass
class Memory:
    content: str
    layer: Layer
    character: str
    importance: float = 0.5  # 0..1
    category: str = "general"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    pinned: bool = False
    source: str = "conversation"   # conversation | tool | user_edit | consolidation | heuristic
    meta: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    embed_model: Optional[str] = None
    profile: str = "default"
    confidence: float = 0.7        # 0..1, how sure we are the claim is right
    status: Status = Status.ACTIVE
    evidence: str = ""             # verbatim span from the source text that supports the claim
    speaker: str = "user"          # whose words the evidence came from: user | assistant | system
    updated_at: float = field(default_factory=time.time)
    source_turn: Optional[int] = None

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "content": self.content, "layer": self.layer.value, "category": self.category,
                "importance": round(self.importance, 2), "confidence": round(self.confidence, 2),
                "status": self.status.value, "evidence": self.evidence, "speaker": self.speaker,
                "source": self.source, "created_at": self.created_at, "updated_at": self.updated_at,
                "last_accessed": self.last_accessed, "access_count": self.access_count, "pinned": self.pinned}


@dataclass
class ScoredMemory:
    memory: Memory
    score: float
    relevance: float
    strength: float
