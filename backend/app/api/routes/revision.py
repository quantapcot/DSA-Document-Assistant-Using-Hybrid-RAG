"""Study roadmap API for the current scope."""
from fastapi import APIRouter, Request

from app.generation.revision_generator import generate_roadmap
from app.orchestration.request_flow import inventory_for_roadmap, run_in_threadpool, study_request_budget
from app.schemas.contracts import RoadmapRequest, RoadmapResponse

router = APIRouter()


@router.post("/api/roadmap", response_model=RoadmapResponse)
@router.post("/api/revision", response_model=RoadmapResponse, deprecated=True)
async def roadmap_endpoint(request: Request, body: RoadmapRequest):
    async with study_request_budget("revision"):
        evidence, has_documents = await inventory_for_roadmap(body, request)
        return await run_in_threadpool(generate_roadmap, body.topic, evidence, has_documents=has_documents)
