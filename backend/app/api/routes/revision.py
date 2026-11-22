"""Scoped revision API."""
from fastapi import APIRouter, HTTPException, Request

from app.generation.revision_generator import generate_revision, resolve_exam_date
from app.orchestration.request_flow import retrieve_for_exercise, run_in_threadpool, study_request_budget
from app.schemas.contracts import RevisionRequest, RevisionResponse

router = APIRouter()


@router.post("/api/revision", response_model=RevisionResponse)
async def revision_endpoint(body: RevisionRequest, request: Request):
    async with study_request_budget("revision"):
        try:
            today, exam_date = resolve_exam_date(body)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        query, evidence, empty = await retrieve_for_exercise(body, request, revision=True)
        return await run_in_threadpool(
            generate_revision, query, evidence, today=today, exam_date=exam_date,
            budget=body.minutes_per_day, level=body.level, workspace_empty=empty,
        )
