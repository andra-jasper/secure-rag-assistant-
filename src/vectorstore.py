"""ChromaDB persistent vector-store helpers."""
from __future__ import annotations

import chromadb

# One client per persist dir per process: ChromaDB's local DuckDB backend
# does not like several clients opening the same database files.
_clients: dict[str, chromadb.PersistentClient] = {}


def open_collection(persist_dir: str, collection_name: str = "knowledge_base"):
    """Open (or create) a persistent ChromaDB collection.

    Embeddings are always supplied explicitly by the caller, so ChromaDB
    never tries to download its own default embedding model.
    """
    if persist_dir not in _clients:
        _clients[persist_dir] = chromadb.PersistentClient(path=persist_dir)
    client = _clients[persist_dir]
    return client.get_or_create_collection(name=collection_name)
