# Secure RAG Knowledge Assistant

A retrieval-augmented generation (RAG) service that grounds LLM answers in your own documents — with safety guardrails, PII isolation, and citation-backed responses.

> Portfolio project demonstrating GenAI system design: RAG pipelines, LLM guardrails, and production-style Python services.

## Features

- **Document ingestion** — chunk and embed documents into a ChromaDB vector store
- **Grounded answers** — hybrid retrieval returns citations with every response
- **Safety guardrails** — NVIDIA NeMo Guardrails for PII isolation, jailbreak blocking, and safe fallback responses
- **API-first** — FastAPI service with structured logging
- **Reproducible** — Dockerized deployment plus a faithfulness evaluation harness

## Tech Stack

`Python` `LangChain` `ChromaDB` `NVIDIA NeMo Guardrails` `FastAPI` `Docker`

## Quickstart

```bash
# 1. Clone and enter the project
git clone https://github.com/andra-jasper/secure-rag-assistant-.git
cd secure-rag-assistant-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env   # add your LLM API key

# 4. Ingest documents
python src/main.py ingest ./docs

# 5. Run the API
uvicorn src.main:app --reload
```

Ask a question:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our refund policy?", "top_k": 4}'
```

Or with Docker:

```bash
docker build -t secure-rag-assistant .
docker run -p 8000:8000 --env-file .env secure-rag-assistant
```

## Project Structure

```
secure-rag-assistant-/
├── src/
│   ├── __init__.py
│   └── main.py          # FastAPI app: ingest + ask endpoints (skeleton)
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Roadmap

- [ ] Streaming responses
- [ ] Hybrid dense + BM25 retrieval
- [ ] Faithfulness eval dashboard

## Note

This is a portfolio skeleton showing architecture and engineering practices. Extend `src/main.py` with your own documents, embedding model, and guardrail configuration.
