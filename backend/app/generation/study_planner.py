"""Construct a deduplicated, prerequisite-ordered roadmap from scoped chunks."""
from collections import defaultdict
from dataclasses import dataclass
import re

from app.core.config import settings
from app.orchestration.study_evidence import TOPIC_PATTERNS, normalized_text, study_topic

FOUNDATIONS = {
    "BFS": ("Đồ thị", "Hàng đợi"), "DFS": ("Đồ thị", "Ngăn xếp", "Đệ quy"),
    "Dijkstra": ("Đồ thị",), "Cây AVL": ("Cây nhị phân tìm kiếm",),
    "Merge sort": ("Mảng", "Đệ quy"), "Quick sort": ("Mảng", "Đệ quy"),
    "Tìm kiếm nhị phân": ("Mảng",), "Quy hoạch động": ("Đệ quy",),
}
_DEPENDENCY = re.compile(r"(?:prerequisites?|requires?|builds on|tiên quyết|dựa trên|cần nắm)[:\s]+([^.!?\n]{1,180})", re.I)
_HEADING_PREFIX = re.compile(r"^(?:(?:chapter|section|chương|mục)\s+)?\d+(?:\.\d+)*[.)\s:–-]+", re.I)
_PLACEHOLDERS = {"overview", "summary", "introduction", "tổng quan", "giới thiệu", "tóm tắt",
                 "nội dung", "references", "tài liệu tham khảo", "data structures and algorithms"}
_PASSAGE_CHARS = 700


@dataclass(frozen=True)
class Topic:
    id: str
    title: str
    sources: tuple[int, ...]


def _title(item):
    title = _HEADING_PREFIX.sub("", study_topic(item)).strip()
    if normalized_text(title) in _PLACEHOLDERS or not title or re.fullmatch(r"(?:chapter|section|chương|mục)\s+\d+", title, re.I):
        title = study_topic(item, body_only=True)
    if title.startswith("Tr.") or not normalized_text(title):
        return None
    for name, pattern in TOPIC_PATTERNS.items():
        if normalized_text(title) == normalized_text(name) or re.fullmatch(pattern, title, re.I):
            return name
    return title[:150].strip()


def build_curriculum(evidence):
    grouped = defaultdict(list)
    titles, seen = {}, set()
    for item in evidence:
        title = _title(item)
        fingerprint = normalized_text(item["text"])
        if not title or not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        key = normalized_text(title)
        titles.setdefault(key, title)
        if len(grouped[key]) < 2:
            grouped[key].append({**item, "text": item["text"][:_PASSAGE_CHARS]})
    catalog, passages = [], []
    for key, items in grouped.items():
        first = len(passages) + 1
        passages.extend(items)
        catalog.append(Topic(f"T{len(catalog) + 1}", titles[key], tuple(range(first, first + len(items)))))
    return catalog, passages


def _matching_topics(text):
    return {name for name, pattern in TOPIC_PATTERNS.items() if re.search(pattern, text, re.I)}


def curriculum_dependencies(catalog, passages):
    """Infer edges only when a related foundation is present in this material."""
    tags = {topic.id: ({topic.title} if topic.title in TOPIC_PATTERNS else _matching_topics(topic.title))
            for topic in catalog}
    by_tag = defaultdict(list)
    graph = {topic.id: [] for topic in catalog}
    for topic_id, names in tags.items():
        for name in sorted(names):
            by_tag[name].append(topic_id)

    def reaches(start, target):
        pending, seen = [start], set()
        while pending:
            parent = pending.pop()
            if parent == target:
                return True
            if parent not in seen:
                seen.add(parent)
                pending.extend(graph[parent])
        return False

    for topic in catalog:
        text = " ".join(passages[label - 1]["text"] for label in topic.sources)
        cited = _matching_topics(text)
        explicit = set().union(*(_matching_topics(part) for part in _DEPENDENCY.findall(text)))
        related = {parent for tag in tags[topic.id] for parent in FOUNDATIONS.get(tag, ()) if parent in cited}
        for name in sorted(explicit | related):
            parent = next((candidate for candidate in by_tag[name] if not reaches(candidate, topic.id)), None)
            if parent and parent not in graph[topic.id]:
                graph[topic.id].append(parent)
    return graph


def order_topics(catalog, graph):
    pending = list(catalog)
    completed, order = set(), []
    while pending:
        topic = next(candidate for candidate in pending if set(graph[candidate.id]) <= completed)
        pending.remove(topic)
        completed.add(topic.id)
        order.append(topic)
    return order


def roadmap_batches(ordered, passages):
    """Use one structured call when the selected sources fit the input budget."""
    batch, cost = [], 0
    for topic in ordered[:settings.ROADMAP_MAX_TOPICS]:
        size = sum(len(passages[label - 1]["text"].encode("utf-8")) + 200 for label in topic.sources)
        if batch and cost + size > settings.ROADMAP_SOURCE_BUDGET:
            yield batch
            batch, cost = [], 0
        batch.append(topic)
        cost += size
    if batch:
        yield batch
