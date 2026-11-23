"""Request metrics shared by the async API and its worker threads."""
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class StudyMetrics:
    retrieval_calls: int = 0
    inventory_reads: int = 0
    retrieval_ms: float = 0
    provider_calls: int = 0
    provider_retries: int = 0
    content_retries: int = 0
    llm_ms: dict[str, float] = field(default_factory=dict)


study_metrics: ContextVar[StudyMetrics | None] = ContextVar("study_metrics", default=None)
