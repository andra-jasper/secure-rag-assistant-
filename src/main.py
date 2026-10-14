"""Secure RAG Knowledge Assistant.

A FastAPI service exposing document ingestion and guardrailed
question-answering over a ChromaDB vector store.

CLI:
    python -m src.main ingest ./docs   # chunk + embed documents
    python -m src.main                  # run the API on :8000

API:
    GET  /health   liveness probe
    POST /ask      {"question": str, "top_k": int} -> answer + citations
"""
from __future__ import annotations

import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import Settings
from src.embeddings import SentenceTransformerEmbedder
from src.ingest import ingest_documents as _ingest_documents
from src.llm import OpenAIProvider
from src.rag import RAGChain
from src.vectorstore import open_collection

app = FastAPI(title="Secure RAG Knowledge Assistant")


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)


class AskResponse(BaseModel):
    """Response body: the answer plus supporting citations."""

    answer: str
    citations: list[str]


def ingest_documents(docs_dir: str) -> int:
    """Chunk and embed documents from *docs_dir* into ChromaDB.

    Returns the number of chunks indexed.
    """
    return _ingest_documents(docs_dir)


def build_rag_chain(settings: Settings | None = None) -> RAGChain:
    """Build the retrieval -> guardrails -> generation chain."""
    settings = settings or Settings.from_env()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    collection = open_collection(settings.chroma_persist_dir, settings.collection_name)
    # Raises RuntimeError with a helpful message when no API key is set.
    llm = OpenAIProvider(api_key=settings.openai_api_key, model=settings.llm_model)
    return RAGChain(
        collection=collection, embedder=embedder, llm=llm, settings=settings
    )


_chain: RAGChain | None = None


def get_chain() -> RAGChain:
    """Lazily-built chain singleton used by the API.

    Tests monkeypatch this to inject a stub LLM and temp vector store.
    """
    global _chain
    if _chain is None:
        try:
            _chain = build_rag_chain()
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _chain


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Answer *req.question* with citations from indexed documents."""
    result = get_chain().ask(req.question, top_k=req.top_k)
    return AskResponse(answer=result.answer, citations=result.citations)


@app.get("/health")
def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        docs_dir = sys.argv[2] if len(sys.argv) > 2 else "./docs"
        count = ingest_documents(docs_dir)
        print(f"Indexed {count} chunks from {docs_dir}.")
    else:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=8000)
