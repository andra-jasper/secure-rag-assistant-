"""Embedding backends.

The production backend uses ``sentence-transformers`` locally (no API key,
no network after the first model download). ``HashEmbedder`` is a
deterministic, dependency-free fallback used by the test-suite and offline
demos — it is NOT semantic and must never be mistaken for real embeddings.
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol


class Embedder(Protocol):
    """Anything that maps a list of texts to a list of float vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Local embeddings via sentence-transformers (model loads lazily)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(texts, normalize_embeddings=True)
        return [[float(x) for x in row] for row in vectors]


class HashEmbedder:
    """Deterministic pseudo-random unit vectors from SHA-256 (no network).

    Used in tests so the suite runs offline. Because the vectors carry no
    semantics, retrieval tests that need *meaningful* ranking use their own
    keyword-based embedder instead (see tests/).
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            seed = text.encode("utf-8")
            vec: list[float] = []
            counter = 0
            while len(vec) < self.dim:
                digest = hashlib.sha256(
                    seed + counter.to_bytes(4, "big")
                ).digest()
                vec.extend(b / 255.0 - 0.5 for b in digest)
                counter += 1
            vec = vec[: self.dim]
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out
