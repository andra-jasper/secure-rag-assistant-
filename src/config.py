"""Runtime configuration for the Secure RAG Knowledge Assistant.

All settings come from environment variables (optionally via a ``.env``
file), so no secrets are ever hard-coded or committed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Settings:
    """Tunable settings for ingestion, retrieval, and generation."""

    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "knowledge_base"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    # Max squared-L2 distance (unit-normalized embeddings) for a chunk to
    # count as relevant. Anything farther is treated as off-topic.
    max_distance: float = 1.6

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from the environment (loads ``.env`` if present)."""
        load_dotenv()
        # OPENAI_API_KEY is preferred; LLM_API_KEY is kept for backwards
        # compatibility with earlier versions of the .env template.
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or None
        return cls(
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
            collection_name=os.getenv("CHROMA_COLLECTION", "knowledge_base"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            openai_api_key=api_key,
            max_distance=float(os.getenv("MAX_DISTANCE", "1.6")),
        )
