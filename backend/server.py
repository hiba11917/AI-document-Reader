import json
import os

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ingest_pipeline import (
    clear_chat_messages,
    clear_collection_chat_messages,
    create_collection,
    create_user,
    delete_chat_message,
    delete_collection,
    delete_collection_chat_message,
    delete_document,
    delete_user_account,
    get_collection,
    get_document,
    get_user_by_session_token,
    ingest_file,
    list_chat_messages,
    list_collection_chat_messages,
    list_collections,
    list_documents,
    list_users,
    login_user,
    save_chat_message,
    save_collection_chat_message,
    update_collection_documents,
    update_user,
    user_can_access_collection,
    user_can_access_document,
)
from nextcloud_storage import storage_mode_label
from rag_local import (
    FALLBACK_MODEL,
    LLM_MODEL,
    TOP_K,
    answer_question,
    build_formatted_sources,
    fetch_available_models,
    stream_answer_question,
)


class AskRequest(BaseModel):
    question: str
    document_id: str | None = None
    collection_id: str | None = None
    top_k: int | None = None


class CollectionRequest(BaseModel):
    name: str
    document_ids: list[str]


class CollectionDocumentsRequest(BaseModel):
    document_ids: list[str]


class SignupRequest(BaseModel):
    full_name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    identifier: str
    password: str


class UserCreateRequest(BaseModel):
    full_name: str
    email: str
    password: str = ""
    username: str | None = None
    role: str = "user"
    is_active: bool = True
    auth_source: str = "local"


class UserUpdateRequest(BaseModel):
    full_name: str | None = None
    email: str | None = None
    password: str | None = None
    role: str | None = None
    is_active: bool | None = None
    auth_source: str | None = None


app = FastAPI(title="AI Document Reader API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _public_user_payload(user):
    return {
        "id": user["id"],
        "username": user["username"],
        "full_name": user["full_name"],
        "email": user["email"],
        "role": user["role"],
        "is_active": bool(user.get("is_active", 1)),
        "auth_source": user.get("auth_source", "local"),
        "created_at": user.get("created_at"),
    }


def _extract_token(authorization: str | None):
    # Keep Bearer token parsing in one place so every protected endpoint fails the same way.
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication is required.")
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="Invalid authorization header.")
    token = authorization[len(prefix) :].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Authentication token is missing.")
    return token


def _require_user(authorization: str | None):
    token = _extract_token(authorization)
    user = get_user_by_session_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Your session has expired. Please log in again.")
    return user


def _require_admin(authorization: str | None):
    user = _require_user(authorization)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access is required.")
    return user


def _ensure_document_access(document_id: str, current_user):
    # Centralize document ownership checks so admin/user permissions stay consistent across routes.
    document = get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found.")
    if not user_can_access_document(current_user, document):
        raise HTTPException(status_code=403, detail="You cannot access this document.")
    return document


def _ensure_collection_access(collection_id: str, current_user):
    # Collections follow the same access model as documents because they are built from user-owned files.
    collection = get_collection(collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found.")
    if not user_can_access_collection(current_user, collection):
        raise HTTPException(status_code=403, detail="You cannot access this collection.")
    return collection


@app.get("/health")
async def health():
    return {"status": "ok", "storage": storage_mode_label()}


@app.post("/api/auth/signup")
async def signup(request: SignupRequest):
    try:
        user = create_user(
            full_name=request.full_name,
            email=request.email,
            password=request.password,
            role="user",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"user": _public_user_payload(user)}


@app.post("/api/auth/login")
async def login(request: LoginRequest):
    try:
        session_user = login_user(request.identifier, request.password)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    token = session_user.pop("token")
    return {"token": token, "user": _public_user_payload(session_user)}


@app.get("/api/auth/me")
async def me(authorization: str | None = Header(default=None)):
    current_user = _require_user(authorization)
    return {"user": _public_user_payload(current_user)}


@app.get("/api/users")
async def users(authorization: str | None = Header(default=None)):
    _require_admin(authorization)
    return {"users": [_public_user_payload(user) for user in list_users()]}


@app.post("/api/users")
async def create_user_endpoint(request: UserCreateRequest, authorization: str | None = Header(default=None)):
    _require_admin(authorization)
    try:
        user = create_user(
            full_name=request.full_name,
            email=request.email,
            password=request.password,
            username=request.username,
            role=request.role,
            is_active=request.is_active,
            auth_source=request.auth_source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"user": _public_user_payload(user), "users": [_public_user_payload(item) for item in list_users()]}


@app.put("/api/users/{user_id}")
async def update_user_endpoint(
    user_id: str,
    request: UserUpdateRequest,
    authorization: str | None = Header(default=None),
):
    _require_admin(authorization)
    try:
        user = update_user(
            user_id,
            full_name=request.full_name,
            email=request.email,
            password=request.password,
            role=request.role,
            is_active=request.is_active,
            auth_source=request.auth_source,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"user": _public_user_payload(user), "users": [_public_user_payload(item) for item in list_users()]}


@app.delete("/api/users/{user_id}")
async def delete_user_endpoint(user_id: str, authorization: str | None = Header(default=None)):
    _require_admin(authorization)
    try:
        deleted = delete_user_account(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not deleted:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"deleted": True, "users": [_public_user_payload(item) for item in list_users()]}


@app.get("/api/settings")
async def settings():
    available_models = []
    try:
        available_models = fetch_available_models()
    except Exception:
        available_models = [os.getenv("LLM_MODEL", LLM_MODEL)]

    preferred_model = os.getenv("LLM_MODEL", LLM_MODEL)

    return {
        "default_model": preferred_model,
        "fallback_model": os.getenv("FALLBACK_MODEL", FALLBACK_MODEL),
        "default_top_k": TOP_K,
        "streaming": True,
        "auto_fallback": True,
        "available_models": available_models,
    }


@app.get("/api/models")
async def models():
    try:
        return {"models": fetch_available_models()}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not load models: {exc}") from exc


@app.get("/api/documents")
async def documents(authorization: str | None = Header(default=None)):
    current_user = _require_user(authorization)
    return {"documents": list_documents(current_user)}


@app.get("/api/collections")
async def collections(authorization: str | None = Header(default=None)):
    current_user = _require_user(authorization)
    return {"collections": list_collections(current_user)}


@app.post("/api/collections")
async def create_collection_endpoint(
    request: CollectionRequest,
    authorization: str | None = Header(default=None),
):
    current_user = _require_user(authorization)
    try:
        create_collection(request.name, request.document_ids, current_user)
        return {"collections": list_collections(current_user)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put("/api/collections/{collection_id}/documents")
async def update_collection_documents_endpoint(
    collection_id: str,
    request: CollectionDocumentsRequest,
    authorization: str | None = Header(default=None),
):
    current_user = _require_user(authorization)
    try:
        update_collection_documents(collection_id, request.document_ids, current_user)
        return {"collections": list_collections(current_user)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/collections/{collection_id}")
async def delete_collection_endpoint(collection_id: str, authorization: str | None = Header(default=None)):
    current_user = _require_user(authorization)
    try:
        deleted = delete_collection(collection_id, current_user)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Collection not found.")
    return {"deleted": True, "collection_id": collection_id, "collections": list_collections(current_user)}


@app.get("/api/documents/{document_id}/history")
async def document_history(document_id: str, authorization: str | None = Header(default=None)):
    current_user = _require_user(authorization)
    _ensure_document_access(document_id, current_user)

    messages = []
    for message in list_chat_messages(document_id, current_user):
        citations = []
        if message["citations_json"]:
            try:
                citations = json.loads(message["citations_json"])
            except json.JSONDecodeError:
                citations = []
        messages.append(
            {
                "id": message["id"],
                "role": message["role"],
                "message": message["message"],
                "model": message["model"],
                "citations": citations,
                "created_at": message["created_at"],
            }
        )
    return {"messages": messages}


@app.get("/api/collections/{collection_id}/history")
async def collection_history(collection_id: str, authorization: str | None = Header(default=None)):
    current_user = _require_user(authorization)
    _ensure_collection_access(collection_id, current_user)

    messages = []
    for message in list_collection_chat_messages(collection_id, current_user):
        citations = []
        if message["citations_json"]:
            try:
                citations = json.loads(message["citations_json"])
            except json.JSONDecodeError:
                citations = []
        messages.append(
            {
                "id": message["id"],
                "role": message["role"],
                "message": message["message"],
                "model": message["model"],
                "citations": citations,
                "created_at": message["created_at"],
            }
        )
    return {"messages": messages}


@app.delete("/api/documents/{document_id}/history")
async def delete_document_history(document_id: str, authorization: str | None = Header(default=None)):
    current_user = _require_user(authorization)
    _ensure_document_access(document_id, current_user)
    deleted_count = clear_chat_messages(document_id, current_user)
    return {"deleted": True, "deleted_count": deleted_count, "document_id": document_id}


@app.delete("/api/collections/{collection_id}/history")
async def delete_collection_history(collection_id: str, authorization: str | None = Header(default=None)):
    current_user = _require_user(authorization)
    _ensure_collection_access(collection_id, current_user)
    deleted_count = clear_collection_chat_messages(collection_id, current_user)
    return {"deleted": True, "deleted_count": deleted_count, "collection_id": collection_id}


@app.delete("/api/documents/{document_id}/history/{message_id}")
async def delete_document_history_message(
    document_id: str,
    message_id: str,
    authorization: str | None = Header(default=None),
):
    current_user = _require_user(authorization)
    _ensure_document_access(document_id, current_user)
    deleted_count = delete_chat_message(document_id, message_id, current_user)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Message not found.")
    return {
        "deleted": True,
        "deleted_count": deleted_count,
        "document_id": document_id,
        "message_id": message_id,
    }


@app.delete("/api/collections/{collection_id}/history/{message_id}")
async def delete_collection_history_message(
    collection_id: str,
    message_id: str,
    authorization: str | None = Header(default=None),
):
    current_user = _require_user(authorization)
    _ensure_collection_access(collection_id, current_user)
    deleted_count = delete_collection_chat_message(collection_id, message_id, current_user)
    if deleted_count == 0:
        raise HTTPException(status_code=404, detail="Message not found.")
    return {
        "deleted": True,
        "deleted_count": deleted_count,
        "collection_id": collection_id,
        "message_id": message_id,
    }


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...), authorization: str | None = Header(default=None)):
    current_user = _require_user(authorization)
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file name is required.")

    temp_path = None
    try:
        import shutil
        import tempfile

        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        document = ingest_file(temp_path, original_filename=file.filename, owner_id=current_user["id"])
        return {"document": document}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


@app.delete("/api/documents/{document_id}")
async def remove_document(document_id: str, authorization: str | None = Header(default=None)):
    current_user = _require_user(authorization)
    _ensure_document_access(document_id, current_user)
    deleted = delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Document could not be deleted.")
    return {"deleted": True, "document_id": document_id}


@app.post("/api/ask")
async def ask(request: AskRequest, authorization: str | None = Header(default=None)):
    current_user = _require_user(authorization)
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    if request.document_id and request.collection_id:
        raise HTTPException(status_code=400, detail="Choose either a document or a collection, not both.")
    if not request.document_id and not request.collection_id:
        raise HTTPException(status_code=400, detail="Select a document or collection first.")

    if request.document_id:
        # Reuse the recent thread for the selected scope so follow-up questions keep the right context.
        _ensure_document_access(request.document_id, current_user)
        conversation_history = list_chat_messages(request.document_id, current_user)[-6:]
    else:
        _ensure_collection_access(request.collection_id, current_user)
        conversation_history = list_collection_chat_messages(request.collection_id, current_user)[-6:]

    top_k = request.top_k if request.top_k and request.top_k > 0 else TOP_K
    result = answer_question(
        question=question,
        document_id=request.document_id,
        collection_id=request.collection_id,
        top_k=top_k,
        conversation_history=conversation_history,
    )
    if result["error"]:
        raise HTTPException(status_code=502, detail=result["error"])

    citations = build_formatted_sources(result["sources"])
    if request.document_id:
        save_chat_message(request.document_id, "user", question, current_user["id"])
        save_chat_message(
            request.document_id,
            "assistant",
            result["answer"],
            current_user["id"],
            model=result["model"],
            citations_json=json.dumps(citations),
        )
    else:
        save_collection_chat_message(request.collection_id, "user", question, current_user["id"])
        save_collection_chat_message(
            request.collection_id,
            "assistant",
            result["answer"],
            current_user["id"],
            model=result["model"],
            citations_json=json.dumps(citations),
        )

    return {
        "answer": result["answer"],
        "sources": citations,
        "model": result["model"],
        "fallback_used": result.get("fallback_used", False),
    }


@app.post("/api/ask/stream")
async def ask_stream(request: AskRequest, authorization: str | None = Header(default=None)):
    current_user = _require_user(authorization)
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    if request.document_id and request.collection_id:
        raise HTTPException(status_code=400, detail="Choose either a document or a collection, not both.")
    if not request.document_id and not request.collection_id:
        raise HTTPException(status_code=400, detail="Select a document or collection first.")

    if request.document_id:
        _ensure_document_access(request.document_id, current_user)
        conversation_history = list_chat_messages(request.document_id, current_user)[-6:]
    else:
        _ensure_collection_access(request.collection_id, current_user)
        conversation_history = list_collection_chat_messages(request.collection_id, current_user)[-6:]

    top_k = request.top_k if request.top_k and request.top_k > 0 else TOP_K

    def event_stream():
        yield json.dumps({"type": "status", "stage": "retrieving", "message": "Finding the best document sections."}) + "\n"
        assistant_answer = ""
        final_sources = []
        final_model = os.getenv("LLM_MODEL", LLM_MODEL)

        for event in stream_answer_question(
            question=question,
            document_id=request.document_id,
            collection_id=request.collection_id,
            top_k=top_k,
            conversation_history=conversation_history,
        ):
            if event.get("type") == "meta":
                final_sources = event.get("sources", [])
                final_model = event.get("model", final_model)
            elif event.get("type") == "token":
                assistant_answer += event.get("delta", "")
            elif event.get("type") == "done":
                assistant_answer = event.get("answer", assistant_answer)
                final_sources = event.get("sources", final_sources)
                final_model = event.get("model", final_model)
                if request.document_id:
                    save_chat_message(request.document_id, "user", question, current_user["id"])
                    save_chat_message(
                        request.document_id,
                        "assistant",
                        assistant_answer,
                        current_user["id"],
                        model=final_model,
                        citations_json=json.dumps(final_sources),
                    )
                else:
                    save_collection_chat_message(request.collection_id, "user", question, current_user["id"])
                    save_collection_chat_message(
                        request.collection_id,
                        "assistant",
                        assistant_answer,
                        current_user["id"],
                        model=final_model,
                        citations_json=json.dumps(final_sources),
                    )
            yield json.dumps(event) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
