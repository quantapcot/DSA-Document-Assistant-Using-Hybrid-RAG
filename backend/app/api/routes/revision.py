"""Scoped revision API."""
from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from app.generation.revision_generator import generate_revision
from app.orchestration.request_flow import retrieve_for_study
from app.schemas.contracts import ChatRequest, RevisionResponse

router = APIRouter()


@router.post("/api/revision", response_model=RevisionResponse)
async def revision_endpoint(body: ChatRequest, request: Request):
    query, evidence, method, empty = await retrieve_for_study(body, request)
    return await run_in_threadpool(
        generate_revision, query, evidence, method_used=method, workspace_empty=empty,
    )
