"""Upload and ingest PDF/DOCX documents into Hybrid RAG pipeline."""
import logging
import json
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.document_processing.document_parser import IMAGE_EXTENSIONS
from app.orchestration.ingestion_flow import InvalidDocumentError, process_and_index_document, upload_document_id
from app.orchestration.study_sessions import SESSION_CORPUS, resolve_scope
from app.retrieval.indexing import scope_mutation_lock
from app.retrieval.vector_store import scope_filter

logger = logging.getLogger("app.api.routes.upload")

router = APIRouter()
UPLOAD_DIR = Path(__file__).resolve().parents[4] / "data" / "runtime" / "uploads"
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB
SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".docx", *IMAGE_EXTENSIONS}



def get_upload_backend(request):
    engine = getattr(request.app.state, "retrieval_engine", None)
    client = getattr(engine, "client", None)
    config = getattr(engine, "config", None) or getattr(request.app.state, "retrieval_config", None)
    if client is None or config is None:
        raise HTTPException(status_code=503, detail="Retrieval is not ready.")
    return client, config


def validate_mutable_scope(corpus_id, workspace_id):
    if not corpus_id.strip() or not workspace_id.strip():
        raise HTTPException(status_code=400, detail="Both scope IDs are required.")
    if (corpus_id, workspace_id) in {("benchmark-v1", "research"), ("dsa-eval-v1", "evaluation")}:
        raise HTTPException(status_code=403, detail="The benchmark scope is read-only.")


def _upload_path(root, corpus_id, workspace_id, document_id, revision, suffix):
    identity = json.dumps([workspace_id, corpus_id, document_id], ensure_ascii=False)
    folder = root / sha256(identity.encode("utf8")).hexdigest()
    return folder / f"{revision}{suffix}"


def _legacy_uuid(payload):
    filename = payload.get("filename", "")
    document_id = payload.get("document_id", "")
    match = re.fullmatch(r"([0-9a-f]{32})_[^/\\]+", filename)
    if match and document_id.startswith(match[1] + "_"):
        return match[1]
    return None


def physical_upload(root, corpus_id, workspace_id, payload):
    filename = payload.get("filename", "")
    metadata = payload.get("metadata") or {}
    revision = metadata.get("upload_revision")
    suffix = Path(filename).suffix.lower()
    if (isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{32}", revision)
            and suffix in SUPPORTED_UPLOAD_EXTENSIONS):
        path = _upload_path(root, corpus_id, workspace_id, payload["document_id"], revision, suffix)
    elif _legacy_uuid(payload):
        path = root / filename
    else:
        return None
    return path if path.resolve().is_relative_to(root.resolve()) else None


def _prior_uploads(client, config, corpus_id, workspace_id, filename, document_id):
    if not client.collection_exists(config.collection):
        return {}
    selector = scope_filter(corpus_id, workspace_id)
    previous = {}
    offset = None
    while True:
        points, offset = client.scroll(
            config.collection, scroll_filter=selector, offset=offset, limit=256,
            with_payload=["document_id", "filename", "metadata.upload_revision"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            old_filename = payload.get("filename", "")
            old_id = payload.get("document_id")
            if old_id == document_id or (_legacy_uuid(payload) and old_filename[33:] == filename):
                previous[old_id] = payload
        if offset is None:
            return previous


def _save_and_index(content, filename, suffix, client, config, corpus_id, workspace_id, registry=None):
    document_id = upload_document_id(filename)
    revision = uuid4().hex
    with scope_mutation_lock(config.collection, corpus_id, workspace_id):
        if registry is not None:
            registry.ensure_active(workspace_id)
        root = registry.directory(workspace_id) / "uploads" if registry is not None else UPLOAD_DIR
        path = _upload_path(root, corpus_id, workspace_id, document_id, revision, suffix)
        previous = _prior_uploads(client, config, corpus_id, workspace_id, filename, document_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as stream:
                stream.write(content)
            result = process_and_index_document(
                path, original_filename=filename, workspace_id=workspace_id,
                corpus_id=corpus_id, config=config, client=client,
                runtime_root=registry.root if registry is not None else None,
                legacy_document_ids=tuple(old_id for old_id in previous if old_id != document_id),
                upload_metadata={
                    "upload_revision": revision,
                    "size_kb": round(len(content) / 1024, 2),
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        except Exception:
            path.unlink(missing_ok=True)
            raise
        for payload in previous.values():
            old_path = physical_upload(root, corpus_id, workspace_id, payload)
            if old_path is not None and old_path != path:
                try:
                    old_path.unlink(missing_ok=True)
                except OSError:
                    logger.exception("Could not remove previous upload %s", old_path.name)
    return {**result, "status": "success", "stored_as": path.name,
            "size_kb": round(len(content) / 1024, 2), "replaced": bool(previous)}


@router.post("/api/upload", tags=["Upload & Ingestion"])
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    workspace_id: str | None = Form(None),
    corpus_id: str | None = Form(None),
):
    workspace_id, corpus_id = resolve_scope(request, workspace_id, corpus_id)
    validate_mutable_scope(corpus_id, workspace_id)
    filename = Path((file.filename or "").replace("\\", "/")).name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and image files are supported.")
    content = await file.read(MAX_FILE_SIZE + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The file is empty.")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="The maximum file size is 15 MB.")
    client, config = get_upload_backend(request)
    registry = request.app.state.study_sessions if corpus_id == SESSION_CORPUS else None
    try:
        result = await run_in_threadpool(
            _save_and_index, content, filename, suffix, client, config, corpus_id, workspace_id, registry,
        )
    except HTTPException:
        raise
    except InvalidDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Could not index upload %s", filename)
        raise HTTPException(status_code=500, detail="Document indexing failed.") from exc
    return {**result, "message": f"Đã lập chỉ mục {result['indexed_chunks']} chunks cho '{filename}'."}
