"""Secure RAG Knowledge Assistant.

Portfolio skeleton: a FastAPI service exposing document ingestion and
guardrailed question-answering over a ChromaDB vector store.

Extend the marked TODO sections with your own embedding model, LLM
provider, and NeMo Guardrails configuration.
"""

import sys

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Secure RAG Knowledge Assistant")


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str
    top_k: int = 4


class AskResponse(BaseModel):
    """Response body: the answer plus supporting citations."""

    answer: str
    citations: list[str]


def ingest_documents(docs_dir: str) -> int:
    """Chunk and embed documents from *docs_dir* into ChromaDB.

    Returns the number of chunks indexed.
    """
    # TODO: implement chunking (e.g. RecursiveCharacterTextSplitter),
    # embedding, and upsert into a persistent ChromaDB collection.
    raise NotImplementedError("Ingestion pipeline not yet implemented")


def build_rag_chain():
    """Build the retrieval -> guardrails -> generation chain."""
    # TODO: wire retriever -> NeMo Guardrails -> LLM.
    raise NotImplementedError("RAG chain not yet implemented")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Answer *req.question* with citations from indexed documents."""
    # TODO: retrieve top_k chunks, apply guardrails, generate answer.
    raise NotImplementedError("Ask endpoint not yet implemented")


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
