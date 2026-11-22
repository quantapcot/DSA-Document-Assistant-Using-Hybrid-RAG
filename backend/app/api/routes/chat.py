"""Scoped chat API."""
from fastapi import APIRouter, Request

from app.generation.rag_engine import build_chat_response
from app.orchestration.intent_router import route_intent
from app.orchestration.request_flow import retrieve_for_study, run_in_threadpool, study_request_budget
from app.schemas.contracts import ChatRequest, RAGResponse

router = APIRouter()


@router.post("/api/chat", response_model=RAGResponse)
async def chat_endpoint(body: ChatRequest, request: Request):
    async with study_request_budget("chat"):
        query, evidence, method, empty = await retrieve_for_study(body, request)
        return await run_in_threadpool(
            build_chat_response, query, evidence, route_intent(query), method_used=method, workspace_empty=empty,
        )
