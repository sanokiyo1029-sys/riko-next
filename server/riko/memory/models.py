"""Memory data model."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class Layer(str, Enum):
    EPISODIC = "episodic"   # events, things that happened, conversations worth remembering
    SEMANTIC = "semantic"   # stable facts and preferences
    WORKING = "working"     # current goals / tasks / short-lived context


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
    source: str = "conversation"
    meta: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    embed_model: Optional[str] = None

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "content": self.content, "layer": self.layer.value, "category": self.category,
                "importance": round(self.importance, 2), "created_at": self.created_at,
                "last_accessed": self.last_accessed, "access_count": self.access_count, "pinned": self.pinned}


@dataclass
class ScoredMemory:
    memory: Memory
    score: float
    relevance: float
    strength: float
