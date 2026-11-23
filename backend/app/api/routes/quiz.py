"""Scoped quiz API."""
import logging
from time import monotonic

from fastapi import APIRouter, Request

from app.generation.quiz_generator import generate_quiz
from app.orchestration.request_flow import retrieve_for_exercise, run_in_threadpool, study_request_budget
from app.schemas.contracts import QuizRequest, QuizResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/quiz", response_model=QuizResponse)
async def quiz_endpoint(body: QuizRequest, request: Request):
    async with study_request_budget("quiz"):
        started = monotonic()
        try:
            topic, evidence, empty = await retrieve_for_exercise(body, request)
        finally:
            logger.info("quiz stage=retrieval elapsed_ms=%.1f", (monotonic() - started) * 1000)
        return await run_in_threadpool(
            generate_quiz, topic, evidence, num_questions=body.num_questions, workspace_empty=empty,
        )
