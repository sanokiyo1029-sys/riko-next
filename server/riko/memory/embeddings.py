"""Embedding providers. Memories record which model embedded them, so switching
models never mixes vector spaces (old memories fall back to keyword relevance
until re-embedded with ``riko memory reindex``)."""
from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Optional

import httpx
import numpy as np

from ..core.config import EmbeddingConfig, get_secret

_TOKEN = re.compile(r"[\w']+", re.UNICODE)


class EmbeddingError(Exception):
    pass


class Embedder(ABC):
    model_id: str = "base"

    @abstractmethod
    async def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalised vectors."""

    async def aclose(self) -> None:
        return None


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


class HashEmbedder(Embedder):
    """Offline, deterministic feature-hashing embedder (words + word bigrams + char
    trigrams). No model download. Captures lexical overlap, not deep semantics:
    use Ollama ``nomic-embed-text`` or similar for real semantic recall."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.model_id = f"hash:{dim}"

    def _features(self, text: str) -> list[tuple[str, float]]:
        words = [w.lower() for w in _TOKEN.findall(text)]
        feats = [(f"w:{w}", 1.0) for w in words]
        feats += [(f"b:{a}_{b}", 0.7) for a, b in zip(words, words[1:])]
        for w in words:
            padded = f"#{w}#"
            feats += [(f"c:{padded[i:i + 3]}", 0.3) for i in range(len(padded) - 2)]
        return feats

    async def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, w in self._features(text):
                h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if h[4] & 1 else -1.0
                out[row, idx] += sign * w
        return _normalize(out)


class OllamaEmbedder(Embedder):
    def __init__(self, cfg: EmbeddingConfig, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self.model = cfg.model
        self.model_id = f"ollama:{cfg.model}"
        self._client = httpx.AsyncClient(base_url=(cfg.base_url or "http://127.0.0.1:11434").rstrip("/"),
                                         timeout=cfg.timeout_s, transport=transport)

    async def embed(self, texts: list[str]) -> np.ndarray:
        try:
            r = await self._client.post("/api/embed", json={"model": self.model, "input": texts})
        except httpx.HTTPError as e:
            raise EmbeddingError(f"ollama embeddings unreachable: {e.__class__.__name__}") from e
        if r.status_code != 200:
            raise EmbeddingError(f"ollama embeddings HTTP {r.status_code}: {r.text[:200]}")
        return _normalize(np.array(r.json()["embeddings"], dtype=np.float32))

    async def aclose(self) -> None:
        await self._client.aclose()


class OpenAICompatibleEmbedder(Embedder):
    """OpenAI, LM Studio (``/v1/embeddings``), and other compatible servers."""

    def __init__(self, cfg: EmbeddingConfig, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        if not cfg.base_url:
            raise EmbeddingError("memory.embeddings.base_url is required for openai_compatible")
        key = get_secret(cfg.api_key_env)
        self.model = cfg.model
        self.model_id = f"openai:{cfg.model}"
        self._client = httpx.AsyncClient(base_url=cfg.base_url.rstrip("/"), timeout=cfg.timeout_s,
                                         headers={"Authorization": f"Bearer {key}"} if key else {},
                                         transport=transport)

    async def embed(self, texts: list[str]) -> np.ndarray:
        try:
            r = await self._client.post("/embeddings", json={"model": self.model, "input": texts})
        except httpx.HTTPError as e:
            raise EmbeddingError(f"embeddings server unreachable: {e.__class__.__name__}") from e
        if r.status_code != 200:
            raise EmbeddingError(f"embeddings HTTP {r.status_code}: {r.text[:200]}")
        data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
        return _normalize(np.array([d["embedding"] for d in data], dtype=np.float32))

    async def aclose(self) -> None:
        await self._client.aclose()


def create_embedder(cfg: EmbeddingConfig, transport: Optional[httpx.AsyncBaseTransport] = None) -> Embedder:
    if cfg.provider == "ollama":
        return OllamaEmbedder(cfg, transport)
    if cfg.provider == "openai_compatible":
        return OpenAICompatibleEmbedder(cfg, transport)
    return HashEmbedder(cfg.dim)
