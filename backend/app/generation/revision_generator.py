"""Generate sourced study sections and flashcards."""
from app.schemas.contracts import Flashcard, RevisionResponse, StudySection

from .citations import MAX_EVIDENCES
from .rag_engine import abstention_fields, grounded_fields, generate_study_json


def generate_revision(query, evidences, *, method_used="hybrid_rerank", workspace_empty=False):
    refusal = abstention_fields(query, evidences, method_used, workspace_empty=workspace_empty)
    if refusal:
        return RevisionResponse(**refusal)
    evidence = evidences[:MAX_EVIDENCES]
    payload = generate_study_json(
        query, evidence,
        "Return only a JSON object with sections (array of title, summary, sources objects) "
        "and flashcards (array of question, answer, sources objects). Each sources field "
        "must contain integer source labels. Create at most five of each.",
    )
    sections, flashcards = [], []
    if isinstance(payload, dict):
        try:
            sections = [StudySection.model_validate(item) for item in payload["sections"][:5]]
            flashcards = [Flashcard.model_validate(item) for item in payload["flashcards"][:5]]
            if not all(1 <= label <= len(evidence) for item in [*sections, *flashcards]
                       for label in item.sources):
                sections, flashcards = [], []
        except (KeyError, TypeError, ValueError):
            sections, flashcards = [], []
    mode = "llm" if sections and flashcards else "extractive"
    if mode == "extractive":
        sections, flashcards = [], []
        for label, item in enumerate(evidence, 1):
            source = f"{item.get('filename', 'DSA')}, p. {item.get('page', 1)}"
            excerpt = " ".join(item["text"].split())
            if len(excerpt) > 400:
                excerpt = excerpt[:400].rsplit(" ", 1)[0] + "..."
            title = item.get("section") or source
            sections.append(StudySection(title=title, summary=excerpt, sources=[label]))
            flashcards.append(Flashcard(
                question=f"Ý chính của phần {title} là gì? [{label}]",
                answer=excerpt, sources=[label],
            ))
    labels = [label for item in [*sections, *flashcards] for label in item.sources]
    text = "\n\n".join(section.summary for section in sections)
    fields = grounded_fields(query, text, evidence, method_used, mode, labels)
    return RevisionResponse(**fields, sections=sections, flashcards=flashcards)
