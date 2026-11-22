"""Collect scoped evidence for study workflows without changing ranking."""
from collections import defaultdict
import re
import unicodedata

from app.generation.abstention import SCORE_THRESHOLDS, evidence_score
from app.retrieval.vector_store import scope_filter

QUIZ_FACETS = (
    "khái niệm định nghĩa / definitions concepts",
    "thuật toán ví dụ từng bước / algorithm trace examples",
    "độ phức tạp so sánh ứng dụng / complexity comparison applications",
)
QUIZ_EVIDENCE_LIMIT = 16


def normalized_text(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def topic_key(item):
    return (item.get("document_id") or item["filename"], item.get("section") or "")


def diverse_evidence(result_sets, *, workspace, corpus, limit=QUIZ_EVIDENCE_LIMIT):
    """Interleave ranked lists, preferring evidence from different chapters."""
    groups = defaultdict(list)
    seen_ids, seen_texts = set(), set()
    for rank in range(max(map(len, result_sets), default=0)):
        for results in result_sets:
            if rank >= len(results):
                continue
            item = results[rank]
            if item.get("workspace_id", workspace) != workspace or item.get("corpus_id", corpus) != corpus:
                continue
            if not item.get("filename") or not item.get("text", "").strip():
                continue
            fingerprint = normalized_text(item["text"])
            chunk_id = item.get("chunk_id")
            if not fingerprint or fingerprint in seen_texts or (chunk_id and chunk_id in seen_ids):
                continue
            method = item.get("retrieval_method")
            score = evidence_score(item)
            if score is not None and method in SCORE_THRESHOLDS and score < SCORE_THRESHOLDS[method]:
                continue
            if chunk_id:
                seen_ids.add(chunk_id)
            seen_texts.add(fingerprint)
            groups[topic_key(item)].append(item)
    selected = []
    for rank in range(max(map(len, groups.values()), default=0)):
        for group in groups.values():
            if rank < len(group):
                selected.append(group[rank])
                if len(selected) == limit:
                    return selected
    return selected


def curriculum_inventory(engine, workspace, corpus, relevant=None):
    """Enumerate the scoped index and retain up to two examples per section."""
    client, config = getattr(engine, "client", None), getattr(engine, "config", None)
    if client is None or config is None:
        return relevant or []
    if not client.collection_exists(config.collection):
        return []
    scope = scope_filter(corpus, workspace)
    keep = {topic_key(item) for item in relevant} if relevant is not None else None
    candidates = defaultdict(list)
    offset = None
    while True:
        points, offset = client.scroll(
            config.collection, scroll_filter=scope, limit=256, offset=offset,
            with_payload=["chunk_id", "document_id", "filename", "page", "section", "text",
                          "metadata", "workspace_id", "corpus_id"],
            with_vectors=False,
        )
        for point in points:
            item = point.payload or {}
            if (item.get("workspace_id") != workspace or item.get("corpus_id") != corpus
                    or not item.get("filename") or not item.get("text", "").strip()):
                continue
            key = topic_key(item)
            if keep is not None and key not in keep:
                continue
            group = candidates[key]
            group.append(item)
            group.sort(key=lambda chunk: (chunk["page"], chunk["chunk_id"]))
            del group[2:]
        if offset is None:
            break
    # Prefer the actual ranked matches when focusing on a specific topic.
    if relevant is not None:
        for item in reversed(relevant):
            group = candidates.get(topic_key(item))
            if group:
                group[:] = [item, *(chunk for chunk in group if chunk["chunk_id"] != item.get("chunk_id"))][:2]
    def chapter_order(group):
        filename = re.split(r"(\d+)", group[0]["filename"].casefold())
        return (tuple(int(value) if index % 2 else value for index, value in enumerate(filename)),
                group[0]["page"], group[0].get("section") or "")

    groups = sorted(candidates.values(), key=chapter_order)
    return [item for group in groups for item in group]
