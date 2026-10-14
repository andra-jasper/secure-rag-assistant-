"""Document ingestion: read -> chunk -> embed -> upsert into ChromaDB."""
from __future__ import annotations

from pathlib import Path

from src.chunking import chunk_text
from src.config import Settings
from src.embeddings import Embedder, SentenceTransformerEmbedder
from src.vectorstore import open_collection

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}
_EMBED_BATCH_SIZE = 64


def _read_text_file(path: Path) -> str:
    """Read a .txt/.md/.pdf file into plain text."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def ingest_documents(
    docs_dir: str,
    *,
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> int:
    """Chunk and embed documents from *docs_dir* into ChromaDB.

    Files are processed in sorted order; chunk ids are deterministic
    (``<filename>::chunk-<n>``), so re-running ingestion upserts rather
    than duplicating. Returns the number of chunks indexed.

    Raises:
        FileNotFoundError: if *docs_dir* does not exist.
        ValueError: if no supported files are found.
    """
    settings = settings or Settings.from_env()
    docs_path = Path(docs_dir)
    if not docs_path.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    files = sorted(
        p
        for p in docs_path.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise ValueError(f"No .txt/.md/.pdf files found in {docs_dir}")

    embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
    collection = open_collection(settings.chroma_persist_dir, settings.collection_name)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for path in files:
        text = _read_text_file(path)
        for i, chunk in enumerate(
            chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        ):
            ids.append(f"{path.name}::chunk-{i}")
            documents.append(chunk)
            metadatas.append({"source": path.name, "chunk_index": i})

    if not ids:
        return 0

    # Embed in batches to bound memory, then upsert (idempotent on chunk id).
    embeddings: list[list[float]] = []
    for start in range(0, len(documents), _EMBED_BATCH_SIZE):
        embeddings.extend(embedder.embed(documents[start : start + _EMBED_BATCH_SIZE]))
    collection.upsert(
        ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
    )
    return len(ids)
