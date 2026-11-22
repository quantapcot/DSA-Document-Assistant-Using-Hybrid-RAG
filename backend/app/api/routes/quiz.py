"""Scoped quiz API."""
from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool

from app.generation.quiz_generator import generate_quiz
from app.orchestration.request_flow import retrieve_for_exercise
from app.schemas.contracts import QuizRequest, QuizResponse

router = APIRouter()


@router.post("/api/quiz", response_model=QuizResponse)
async def quiz_endpoint(body: QuizRequest, request: Request):
    query, evidence, empty = await retrieve_for_exercise(body, request)
    return await run_in_threadpool(
        generate_quiz, query, evidence, num_questions=body.num_questions,
        difficulty=body.difficulty, workspace_empty=empty,
    )
