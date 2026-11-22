"""Derive a grounded curriculum and keep schedule fields under Python control."""
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
import json
import re

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.orchestration.study_evidence import TOPIC_PATTERNS, normalized_text, topic_key
from app.schemas.contracts import SourceLabel, StudyDay, StudyPlan, StudyTask, StudyTopic

LEVEL_MINUTES = {"Beginner": 25, "Intermediate": 20, "Advanced": 15}
FOUNDATIONS = {
    "BFS": ("Đồ thị", "Hàng đợi"), "DFS": ("Đồ thị", "Ngăn xếp", "Đệ quy"),
    "Dijkstra": ("Đồ thị",), "Cây AVL": ("Cây nhị phân tìm kiếm",),
    "Merge sort": ("Mảng", "Đệ quy"), "Quick sort": ("Mảng", "Đệ quy"),
    "Tìm kiếm nhị phân": ("Mảng",), "Quy hoạch động": ("Đệ quy",),
}
_DEPENDENCY = re.compile(r"(?:prerequisites?|requires?|builds on|tiên quyết|dựa trên|cần nắm)[:\s]+([^.!?\n]{1,180})", re.I)
_COMPLEX = re.compile(r"\b(?:invariant|recurrence|amortized|recursion|proof|backtracking|complexity)\b|bất biến|chứng minh|đệ quy|độ phức tạp", re.I)


@dataclass(frozen=True)
class Slot:
    id: int
    date: date
    kind: str
    topic_ids: tuple[str, ...]
    minutes: int
    prerequisites: tuple[str, ...] = ()


class TaskContent(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    slot: int = Field(strict=True, ge=1)
    objective: str = Field(min_length=1, max_length=300)
    knowledge: list[str] = Field(min_length=1, max_length=5)
    activity: str = Field(min_length=1, max_length=400)
    practice: str = Field(min_length=1, max_length=300)
    sources: list[SourceLabel] = Field(min_length=1, max_length=4)
    support_quote: str = Field(min_length=20, max_length=500)


def build_curriculum(evidence):
    grouped = defaultdict(list)
    for item in evidence:
        grouped[topic_key(item)].append(item)
    catalog, passages = [], []
    for index, ((_, title), items) in enumerate(grouped.items(), 1):
        labels = list(range(len(passages) + 1, len(passages) + min(2, len(items)) + 1))
        # Limit prompt excerpts while keeping their real source coordinates.
        passages.extend({**item, "text": item["text"][:1800]} for item in items[:2])
        catalog.append(StudyTopic(id=f"T{index}", title=f"{title} · {items[0]['filename']}", sources=labels))
    return catalog, passages


def _matching_topics(text):
    return {name for name, pattern in TOPIC_PATTERNS.items() if re.search(pattern, text, re.I)}


def curriculum_dependencies(catalog, passages):
    """Infer edges only when a related foundation is present in the material."""
    tags, evidence = {}, {}
    for topic in catalog:
        name = topic.title.split(" · ", 1)[0]
        matched = _matching_topics(name)
        tags[topic.id] = {name} if name in TOPIC_PATTERNS else matched
        evidence[topic.id] = " ".join(passages[label - 1]["text"] for label in topic.sources)
    graph = {topic.id: [] for topic in catalog}
    by_tag = defaultdict(list)
    for identifier, names in tags.items():
        for name in names:
            by_tag[name].append(identifier)

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
        text = evidence[topic.id]
        cited = _matching_topics(text)
        explicit = set().union(*(_matching_topics(part) for part in _DEPENDENCY.findall(text)))
        related = {parent for tag in tags[topic.id] for parent in FOUNDATIONS.get(tag, ()) if parent in cited}
        for name in sorted(explicit | related):
            parent = next((candidate for candidate in by_tag[name] if not reaches(candidate, topic.id)), None)
            if parent and parent not in graph[topic.id]:
                graph[topic.id].append(parent)
    return graph


def schedule_topics(catalog, passages, *, today, exam_date, budget, level):
    graph = curriculum_dependencies(catalog, passages)
    remaining = list(catalog)
    ordered, ordered_ids = [], set()
    while remaining:
        available = [topic for topic in remaining if set(graph[topic.id]) <= ordered_ids]
        if not available:
            break
        topic = available[0]
        ordered.append(topic)
        ordered_ids.add(topic.id)
        remaining.remove(topic)

    days = (exam_date - today).days
    review_days = min(2, days - 1)
    learning_days = days - review_days
    chosen, drafts, completed = [], [], set()
    current_day = spent = source_tokens = 0
    for topic in ordered:
        if len(chosen) >= settings.REVISION_MAX_TOPICS:
            break
        if not set(graph[topic.id]) <= completed:
            continue
        topic_passages = [passages[label - 1] for label in topic.sources]
        # UTF-8 bytes plus overhead is a conservative planning estimate for
        # byte-based tokenizers. Batch prompts send only labels they need.
        cost = sum(len(item["text"].encode("utf-8")) + 120 for item in topic_passages)
        if source_tokens + cost > settings.REVISION_SOURCE_TOKEN_BUDGET:
            continue
        signal = sum(min(3, len(_COMPLEX.findall(item["text"]))) for item in topic_passages)
        minutes = min(budget, LEVEL_MINUTES[level] + min(15, 5 * signal))
        if spent + minutes > budget:
            current_day += 1
            spent = 0
        if current_day >= learning_days:
            break
        chosen.append(topic)
        completed.add(topic.id)
        source_tokens += cost
        spent += minutes
        drafts.append((current_day, topic, minutes))

    if not drafts:
        return [], []
    last_training_day = drafts[-1][0]
    slots = []
    for offset, topic, minutes in drafts:
        when = today + timedelta(days=offset * (learning_days - 1) // max(1, last_training_day))
        slots.append(Slot(len(slots) + 1, when, "learn", (topic.id,), minutes, tuple(graph[topic.id])))

    topic_ids = [topic.id for topic in chosen]
    if review_days:
        for kind, day in (("review", exam_date - timedelta(days=review_days)),
                          ("mock_test", exam_date - timedelta(days=1))):
            available_minutes = budget if review_days == 2 else budget // 2
            # Split a day into sessions of at most four topics, spread the
            # selection across the curriculum when the short budget is full.
            capacity = min(len(topic_ids), 4 * max(1, available_minutes // 10))
            if capacity == 1:
                selected = topic_ids[:1]
            else:
                selected = [topic_ids[i * (len(topic_ids) - 1) // (capacity - 1)] for i in range(capacity)]
            groups = [selected[i:i + 4] for i in range(0, len(selected), 4)]
            duration = min(20, available_minutes // len(groups))
            for group in groups:
                slots.append(Slot(len(slots) + 1, day, kind, tuple(group), duration))
    elif budget - spent >= 10:
        duration = min(15, (budget - spent) // 2)
        group = tuple(topic_ids[:4])
        slots += [
            Slot(len(slots) + 1, today, "review", group, duration),
            Slot(len(slots) + 2, today, "mock_test", group, duration),
        ]
    return chosen, slots


def batch_request(slots, topics, passages, level):
    selected = {topic for slot in slots for topic in (*slot.topic_ids, *slot.prerequisites)}
    catalog = [topic.model_dump() for topic in topics if topic.id in selected]
    identifiers = {
        label: passages[label - 1]
        for topic in topics if topic.id in selected for label in topic.sources
    }
    assignments = [{"slot": s.id, "kind": s.kind, "topic_ids": s.topic_ids,
                    "minutes": s.minutes, "prerequisites": s.prerequisites} for s in slots]
    instructions = (
        f"Write Vietnamese DSA study activities for level {level}. Return only a JSON array with one "
        "object per requested slot. Include slot, objective, knowledge (1-5 points), activity, practice, "
        "sources (at least one provided label per covered topic) and support_quote (a verbatim excerpt "
        "from one cited passage, at least 20 characters). Keep fields concise and activities distinct. "
        "Use the assigned prerequisites for context. Do not add dates, minutes, topics or prerequisites "
        "to your response; they are set by the scheduler. "
        f"Topics: {json.dumps(catalog, ensure_ascii=False)}. Assignments: {json.dumps(assignments)}."
    )
    return identifiers, instructions


def validate_batch(payload, slots, topics, passages):
    if not isinstance(payload, list) or len(payload) != len(slots):
        return None
    try:
        responses = [TaskContent.model_validate(raw) for raw in payload]
    except ValueError:
        return None
    by_slot = {item.slot: item for item in responses}
    if set(by_slot) != {slot.id for slot in slots}:
        return None
    allowed = {topic.id: set(topic.sources) for topic in topics}
    for slot in slots:
        data = by_slot[slot.id]
        sources = set(data.sources)
        groups = [allowed[topic] for topic in slot.topic_ids]
        if len(sources) != len(data.sources) or not sources <= set().union(*groups):
            return None
        if any(not sources & group for group in groups):
            return None
        quote = " ".join(data.support_quote.split())
        if len(quote) < 20 or not any(quote in " ".join(passages[label - 1]["text"].split()) for label in sources):
            return None
        if not all(map(normalized_text, (data.objective, data.activity, data.practice, *data.knowledge))):
            return None
    return by_slot


def assemble_plan(contents, chosen, slots, passages, *, today, exam_date, budget, level):
    if validate_batch([item.model_dump() for item in contents.values()], slots, chosen, passages) is None:
        return None
    days, completed = defaultdict(list), set()
    objectives, activities = set(), set()
    for slot in slots:
        item = contents[slot.id]
        obj, activity = normalized_text(item.objective), normalized_text(item.activity)
        if obj in objectives or activity in activities or not today <= slot.date < exam_date:
            return None
        if not set(slot.prerequisites) <= completed or (slot.kind != "learn" and not set(slot.topic_ids) <= completed):
            return None
        objectives.add(obj)
        activities.add(activity)
        days[slot.date].append(StudyTask(
            **item.model_dump(exclude={"slot"}), kind=slot.kind, topic_id=slot.topic_ids[0],
            covered_topic_ids=list(slot.topic_ids), minutes=slot.minutes, prerequisites=list(slot.prerequisites),
        ))
        if slot.kind == "learn":
            completed.update(slot.topic_ids)
    if list(days) != sorted(days) or any(sum(task.minutes for task in tasks) > budget for tasks in days.values()):
        return None
    if completed != {topic.id for topic in chosen}:
        return None
    return StudyPlan(
        exam_date=exam_date, minutes_per_day=budget, level=level, topics=chosen,
        days=[StudyDay(date=day, tasks=tasks) for day, tasks in days.items()],
    )
