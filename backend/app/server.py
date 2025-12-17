import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ingest_pipeline import ingest_file
from rag_local import (
    build_global_index,
    clear_cache,
    get_index_and_mapping,
    answer_question,
)

app = FastAPI(title="AI Document Reader", version="0.1.0")

# Allow local dev from the static frontend or other origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"]
    ,
    allow_headers=["*"],
)


def ensure_global_index():
    """Ensure global FAISS index exists; build if missing."""
    try:
        get_index_and_mapping()
    except FileNotFoundError:
        # Build if no index exists yet
        build_global_index()
        clear_cache()
    except ValueError as e:
        # Likely no per-document indexes yet
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # pragma: no cover - defensive path
        raise HTTPException(status_code=500, detail=f"Index unavailable: {e}")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/api/ask")
async def ask_document(
    question: str = Form(...),
    document: UploadFile | None = File(None),
):
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question is required.")

    temp_path = None
    try:
        # Optional ingestion if a document is provided with the question
        if document:
            suffix = os.path.splitext(document.filename or "")[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(document.file, tmp)
                temp_path = tmp.name

            doc_id = ingest_file(temp_path)
            if not doc_id:
                raise HTTPException(status_code=500, detail="Document ingestion failed.")

            # Rebuild the global index to include the new document
            build_global_index()
            clear_cache()
        else:
            ensure_global_index()

        result = answer_question(question, stream=False)
        if not result.get("answer"):
            detail = result.get("error") or "No answer generated."
            raise HTTPException(status_code=502, detail=detail)

        sources = []
        for r in result.get("sources", []):
            sources.append(
                {
                    "filename": r.get("filename"),
                    "page": r.get("page"),
                    "char_start": r.get("char_start"),
                    "char_end": r.get("char_end"),
                    "score": r.get("score"),
                    "snippet": (r.get("text") or "")[:400],
                }
            )

        return {"answer": result["answer"], "sources": sources}
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
