"""Construct a deduplicated, prerequisite-ordered roadmap from scoped chunks."""
from collections import defaultdict
from dataclasses import dataclass
import re

from app.core.config import settings
from app.orchestration.study_evidence import TOPIC_PATTERNS, normalized_text, roadmap_topic_filter

_CONCEPTS = {
    "Binary Search Tree": r"binary search trees?|bst|cây (?:nhị phân tìm kiếm|tìm kiếm nhị phân)",
    "AVL Tree": r"avl(?: trees?)?|cây avl",
    "Binary Tree": r"binary trees?|cây nhị phân",
    "Tree Traversal": r"tree traversals?|(?:pre|in|post)\s?order traversals?|duyệt cây",
    "Tree Fundamentals": r"trees? fundamentals|(?:tree|trees|cây|cấu trúc cây)",
    "BFS": r"bfs|breadth first search|duyệt theo chiều rộng",
    "DFS": r"dfs|depth first search|duyệt theo chiều sâu",
    "Insertion Sort": r"insertion sort|sắp xếp chèn",
    "Selection Sort": r"selection sort|sắp xếp chọn",
    "Bubble Sort": r"bubble sort|sắp xếp nổi bọt",
    "Heap Sort": r"heap\s?sort|sắp xếp vun đống",
    "Counting Sort": r"counting sort|sắp xếp đếm",
    "Radix Sort": r"radix sort|sắp xếp cơ số",
    **{name: pattern for name, pattern in TOPIC_PATTERNS.items()
       if name not in {"Cây AVL", "Cây nhị phân tìm kiếm", "BFS", "DFS"}},
}

FOUNDATIONS = {
    "BFS": ("Đồ thị", "Hàng đợi"), "DFS": ("Đồ thị", "Ngăn xếp", "Đệ quy"),
    "Dijkstra": ("Đồ thị",), "AVL Tree": ("Binary Search Tree",),
    "Merge sort": ("Mảng", "Đệ quy"), "Quick sort": ("Mảng", "Đệ quy"),
    "Tìm kiếm nhị phân": ("Mảng",), "Quy hoạch động": ("Đệ quy",),
}
_DEPENDENCY = re.compile(
    r"\b(?:prerequisites?|requires?|builds on|extends|(?:is|are) (?:a|an|the)|"
    r"tiên quyết|dựa trên|cần nắm|là một)[:\s]+([^.!?\n]{1,180})", re.I
)
_HEADING_PREFIX = re.compile(r"^(?:(?:chapter|section|chương|mục)\s+)?\d+(?:\.\d+)*[.)\s:–-]+", re.I)
_PLACEHOLDERS = {"overview", "summary", "introduction", "tổng quan", "giới thiệu", "tóm tắt",
                 "nội dung", "references", "tài liệu tham khảo", "data structures and algorithms"}
_SENTENCE = re.compile(
    r"[.!?;]|\b(?:this|we|you|is|are|was|were|can|will|has|have|uses?|represents?|"
    r"works?|requires?|allows?|stores?|finds?|returns?|includes?|supports?|visits?|"
    r"organizes?|contains?|describes?|explains?|là|có thể|được)\b", re.I
)
_BASIC = re.compile(r"\b(?:fundamentals|basics|introduction|cơ bản|nhập môn)\b", re.I)
_ADVANCED = re.compile(r"\b(?:advanced|optimization|nâng cao|tối ưu)\b", re.I)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
_PASSAGE_CHARS = 700


@dataclass(frozen=True)
class Topic:
    id: str
    title: str
    sources: tuple[int, ...]


def _heading(raw):
    title = _HEADING_PREFIX.sub("", raw.strip().lstrip("#").strip()).rstrip(":").strip()
    words = normalized_text(title)
    if (not words or words in _PLACEHOLDERS or len(title) > 90 or len(words.split()) > 10
            or _SENTENCE.search(title) or re.fullmatch(r"(?:chapter|section|chương|mục)\s+\d+", title, re.I)):
        return None
    for name, pattern in _CONCEPTS.items():
        if words == normalized_text(name) or not re.sub(rf"\b(?:{pattern})\b", "", words).strip():
            return name
    return title


def _concept_mentions(text):
    text = normalized_text(text)
    matches = sorted((found.start(), -len(found[0]), name)
                     for name, pattern in _CONCEPTS.items()
                     for found in re.finditer(rf"\b(?:{pattern})\b", text))
    # A generic term inside a more specific alias is not a second concept.
    end = -1
    for start, length, name in matches:
        if start >= end:
            yield name
            end = start - length


def _titles(item, topic, matches):
    titles = []
    first_line, boundary, _ = item["text"].strip().partition("\n")
    for raw in (item.get("section") or "", first_line if boundary else ""):
        title = _heading(raw)
        if title and (title not in _CONCEPTS or matches(title)):
            titles.append(title)
            break
    for sentence in _SENTENCE_BREAK.split(item["text"]):
        concept = next(_concept_mentions(sentence), None)
        if concept and matches(concept):
            titles.append(concept)
    if not titles and (fallback := _heading(topic)):
        titles.append(fallback)
    return list(dict.fromkeys(titles))


def build_curriculum(evidence, topic):
    matches = roadmap_topic_filter(topic)
    grouped = defaultdict(list)
    titles, seen = {}, set()
    for item in evidence:
        if not matches(item["text"]):
            continue
        fingerprint = normalized_text(item["text"])
        if not fingerprint or fingerprint in seen:
            continue
        seen.add(fingerprint)
        for title in _titles(item, topic, matches):
            key = normalized_text(title)
            titles.setdefault(key, title)
            if len(grouped[key]) < 2:
                grouped[key].append((fingerprint, item))
    catalog, passages, labels = [], [], {}
    for key, items in grouped.items():
        sources = []
        for fingerprint, item in items:
            if fingerprint not in labels:
                passages.append({**item, "text": item["text"][:_PASSAGE_CHARS]})
                labels[fingerprint] = len(passages)
            sources.append(labels[fingerprint])
        catalog.append(Topic(f"T{len(catalog) + 1}", titles[key], tuple(sources)))
    return catalog, passages


def _matching_topics(text):
    return set(_concept_mentions(text))


def curriculum_dependencies(catalog, passages):
    """Infer edges only when a related foundation is present in this material."""
    tags = {topic.id: ({topic.title} if topic.title in _CONCEPTS else _matching_topics(topic.title))
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
        relevant_sentences = []
        for sentence in _SENTENCE_BREAK.split(text):
            subject = next(_concept_mentions(sentence), None)
            if subject == topic.title or subject not in by_tag:
                relevant_sentences.append(sentence)
        cited = _matching_topics(" ".join(relevant_sentences))
        references = [part for sentence in relevant_sentences for part in _DEPENDENCY.findall(sentence)]
        explicit = set().union(*(_matching_topics(part) for part in references))
        # Free-form chapter titles can be prerequisites too.
        direct = {other.id for other in catalog if other.id != topic.id and any(
            f" {normalized_text(other.title)} " in f" {normalized_text(part)} " for part in references
        )}
        related = {parent for tag in tags[topic.id] for parent in FOUNDATIONS.get(tag, ()) if parent in cited}
        for name in sorted(explicit | related):
            parent = next((candidate for candidate in by_tag[name] if not reaches(candidate, topic.id)), None)
            if parent and parent not in graph[topic.id]:
                graph[topic.id].append(parent)
        for parent in sorted(direct):
            if parent not in graph[topic.id] and not reaches(parent, topic.id):
                graph[topic.id].append(parent)
    return graph


def order_topics(catalog, graph):
    pending = sorted(catalog, key=lambda topic: bool(_ADVANCED.search(topic.title)) - bool(_BASIC.search(topic.title)))
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
