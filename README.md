# AI Document Reader (Prototype)

LLM-backed RAG prototype that ingests documents (PDF, DOCX, XLSX, DOC, XLS), builds FAISS indexes, and serves Q&A with citations via a FastAPI backend and a simple web UI.

## Features (Phase 0)
- CLI ingestion with chunking + per-document FAISS indexes
- Global FAISS index builder and retrieval
- Ollama-powered generation (defaults to `llama3`)
- FastAPI endpoint `/api/ask` for question answering with optional upload
- Frontend form posts to `/api/ask` and renders answer + sources

## Project Layout
- `backend/app/ingest_pipeline.py` — ingest + chunk + embed + per-doc FAISS
- `backend/app/rag_local.py` — retrieval + prompt assembly + Ollama call
- `backend/app/server.py` — FastAPI service (upload, index rebuild, Q&A)
- `frontend/app/Index.html` and `styles.css` — static UI posting to backend

## Prerequisites
- Python 3.10+ recommended
- Ollama running locally with a pulled model (e.g., `ollama pull llama3`)
- Build tools for `faiss-cpu`

## Install (backend)
```bash
pip install fastapi uvicorn python-multipart sentence-transformers faiss-cpu sqlalchemy pymupdf python-docx pandas xlrd requests
```

## Run backend (dev)
```bash
cd backend/app
uvicorn server:app --reload --port 8000
```
Environment variables:
- `OLLAMA_URL` (default `http://localhost:11434/api/generate`)
- `LLM_MODEL` (default `llama3`)

## Use the API
`POST /api/ask`
- `form field question` (string, required)
- `form file document` (optional upload; if provided, ingests + rebuilds global index)

Response
```json
{ "answer": "...", "sources": [ { "filename": "...", "page": 1, "char_start": 0, "char_end": 1200, "score": 0.12, "snippet": "..." } ] }
```

## Frontend
Open `frontend/app/Index.html` (or serve statically). The form submits to `/api/ask` and renders answer + sources.

## Typical Workflow
1) Start Ollama with your chosen model.
2) Start backend (`uvicorn ...`).
3) Ingest by uploading via the UI or calling `/api/ask` with a document.
4) Ask follow-up questions (without upload) using the same endpoint; it reuses the global index.

## Notes
- If no documents are indexed, `/api/ask` without a file returns 400.
- Rebuilding the global index happens automatically after an upload inside `/api/ask`.
- Character offsets are included for citations; page is preserved where available.

## Phase 0 Verification (manual)
- Upload a native PDF, ask a factoid, verify citation references the correct page/snippet.
- Check response time against targets (p50 ≈ 3s for ~10-page docs, environment-dependent).
- Validate that unsupported file types are rejected and large files (>200MB) are blocked earlier in ingest.
