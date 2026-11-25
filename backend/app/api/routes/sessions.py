"""Create and revoke temporary study-session capabilities."""
import logging

from fastapi import APIRouter, Request, Response

from app.orchestration.study_sessions import resolve_scope
from app.schemas.contracts import SessionResponse

router = APIRouter(prefix="/api/sessions", tags=["Study sessions"])
logger = logging.getLogger(__name__)


@router.post("", status_code=201, response_model=SessionResponse)
def create_session(request: Request, response: Response):
    response.headers["Cache-Control"] = "no-store"
    return request.app.state.study_sessions.create()


@router.delete("/current")
def delete_session(request: Request, workspace_id: str | None = None, corpus_id: str | None = None):
    workspace, _ = resolve_scope(request, workspace_id, corpus_id, require_session=True)
    registry = request.app.state.study_sessions
    registry.mark_deleted(workspace)
    engine = getattr(request.app.state, "retrieval_engine", None)
    if engine is not None:
        try:
            registry.purge(workspace, engine)
            return {"status": "deleted"}
        except Exception as exc:
            logger.warning("Session deletion deferred error=%s", type(exc).__name__)
    return Response(status_code=202)
