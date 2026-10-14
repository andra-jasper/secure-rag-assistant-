# Secure RAG Knowledge Assistant

A retrieval-augmented generation (RAG) service that grounds LLM answers in your own documents — with safety guardrails, PII isolation, and citation-backed responses.

> Portfolio project demonstrating GenAI system design: RAG pipelines, LLM guardrails, and production-style Python services.

## Architecture

```
docs/*.txt ──▶ chunk (overlapping) ──▶ embed (sentence-transformers, local)
                                                        │
                                                        ▼
User question ──▶ input rails ──▶ ChromaDB retrieval ──▶ prompt build ──▶ LLM ──▶ output rails ──▶ answer + citations
                     │                    (top-k chunks)      (PII redacted)                    (PII redacted)
                     ▼
              injection blocked /
              PII redacted
```

**Pipeline stages**

1. **Ingestion** (`src/ingest.py`) — reads `.txt`/`.md`/`.pdf` from a docs
   directory, splits text into overlapping chunks (`src/chunking.py`), embeds
   them locally with `all-MiniLM-L6-v2` (no API key), and upserts them into a
   persistent ChromaDB collection. Chunk ids are deterministic, so re-running
   ingestion never duplicates data.
2. **Retrieval** (`src/rag.py`) — embeds the user question and fetches the
   `top_k` most similar chunks from ChromaDB.
3. **Guardrails** (`src/guardrails.py`) — deterministic input/output rails:
   prompt-injection and jailbreak attempts are blocked, PII (emails, phone
   numbers, SSNs, credit-card numbers) is redacted before anything reaches
   the LLM and again in the final answer. Questions with no relevant context
   get a graceful refusal instead of a hallucination.
4. **Generation** (`src/llm.py`) — pluggable provider interface. The default
   `OpenAIProvider` uses `OPENAI_API_KEY`; tests and offline demos use the
   deterministic `StubLLMProvider` (no network).

**Why custom guardrails instead of NVIDIA NeMo Guardrails?** NeMo's dialogue
rails require a live LLM backend, so they can't run in this project's offline
test suite. These pure-Python rails are fast, dependency-free, and fully
unit-tested — and they stay as the first line of defence even if an
LLM-backed rail layer is added later.

## Project structure

```
secure-rag-assistant/
├── src/
│   ├── main.py        # FastAPI app (/ask, /health) + CLI
│   ├── config.py      # env-based settings
│   ├── chunking.py    # overlapping text splitter
│   ├── embeddings.py  # sentence-transformers + offline hash embedder
│   ├── vectorstore.py # ChromaDB helpers
│   ├── ingest.py      # read -> chunk -> embed -> upsert
│   ├── rag.py         # retrieval -> guardrails -> generation chain
│   ├── llm.py         # OpenAI provider + deterministic stub provider
│   └── guardrails.py  # input/output rails (injection blocking, PII redaction)
├── tests/
│   └── test_rag.py    # offline test suite (chunking, ingestion, retrieval,
│                      #   guardrails, /ask endpoint)
├── docs/              # sample documents for an instant demo
├── Dockerfile
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/andra-jasper/secure-rag-assistant-.git
cd secure-rag-assistant-

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # add your OPENAI_API_KEY
```

## Usage

Ingest documents (the bundled `docs/` samples work out of the box):

```bash
python -m src.main ingest ./docs
# Indexed 6 chunks from ./docs.
```

Run the API:

```bash
uvicorn src.main:app --reload
```

Ask a question:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?", "top_k": 4}'
```

Or with Docker:

```bash
docker build -t secure-rag-assistant .
docker run -p 8000:8000 --env-file .env secure-rag-assistant
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — (required for `/ask`) | API key for answer generation |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `CHROMA_COLLECTION` | `knowledge_base` | Collection name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `CHUNK_SIZE` | `500` | Max characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `MAX_DISTANCE` | `1.6` | Max chunk distance (squared L2 on normalized embeddings) to count as relevant; farther chunks are treated as off-topic |

## Tests

```bash
pytest -v
```

The suite is fully offline — it uses a deterministic keyword embedder and the
stub LLM provider, so no API keys or network access are needed.
