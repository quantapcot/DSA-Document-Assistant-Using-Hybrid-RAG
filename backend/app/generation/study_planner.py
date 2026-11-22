"""Build a bounded curriculum and validate the provider's dated study plan."""
from datetime import timedelta
import json

from app.orchestration.study_evidence import normalized_text, topic_key
from app.schemas.contracts import StudyPlan, StudyTopic

MAX_PLANNED_TOPICS = 10
LESSON_MINUTES = {"Beginner": 30, "Intermediate": 25, "Advanced": 20}


def build_curriculum(evidence):
    groups = {}
    for item in evidence:
        groups.setdefault(topic_key(item), []).append(item)
    catalog, sources = [], []
    for index, items in enumerate(groups.values(), 1):
        first = items[0]
        title = first.get("section") or first["filename"]
        if first.get("section"):
            title = f"{title} · {first['filename']}"
        labels = list(range(len(sources) + 1, len(sources) + min(2, len(items)) + 1))
        sources.extend(items[:2])
        catalog.append(StudyTopic(id=f"T{index}", title=title, sources=labels))
    return catalog, sources


def schedule_topics(catalog, *, today, exam_date, budget, level):
    """Prioritize chapters in document order and reserve review time."""
    days = (exam_date - today).days
    lesson = min(budget, LESSON_MINUTES[level])
    review = min(20, max(5, budget // 3))
    review_days = min(2, days - 1)
    learning_days = days - review_days
    review_possible = review_days > 0 or budget >= lesson + review * 2
    first_day_budget = budget - review * 2 if days == 1 and review_possible else budget
    per_day = (first_day_budget if days == 1 else budget) // lesson
    chosen = catalog[:min(MAX_PLANNED_TOPICS, per_day * learning_days)]
    slots = []
    if not chosen:
        return chosen, slots
    used_learning_days = (len(chosen) + per_day - 1) // per_day
    for index, topic in enumerate(chosen):
        group = index // per_day
        offset = group * (learning_days - 1) // max(1, used_learning_days - 1)
        slots.append((today + timedelta(days=offset), "learn", topic.id, lesson))
    if review_possible:
        slots.append((exam_date - timedelta(days=max(1, review_days)), "review", chosen[0].id, review))
        slots.append((exam_date - timedelta(days=1), "mock_test", chosen[-1].id, review))
    return chosen, slots


def plan_instructions(chosen, slots, *, exam_date, budget, level):
    # Keep source IDs stable between the schedule and the prompt.
    assignments = [
        {"date": day.isoformat(), "kind": kind, "topic_id": topic, "minutes": minutes}
        for day, kind, topic, minutes in slots
    ]
    topics = [topic.model_dump() for topic in chosen]
    return (
        "Create a Vietnamese DSA study plan using the exact dates, task kinds, topic IDs, and minutes below. "
        "Fill each task with objective, knowledge (1-5 concrete points), activity, practice, sources "
        "(integer labels from that topic), and support_quote (a verbatim supporting sentence of at least "
        "20 characters from one of its cited passages). Practice suggests a grounded quiz or recall exercise. "
        "Vary activities by task kind and adapt them to the student's level. Review and mock_test should "
        "connect earlier lessons when relevant. For prerequisites, list only genuine prior topic IDs "
        "already assigned to a learn task earlier in the schedule, or [] when none. "
        "Use only the supplied evidence. Keep each field concise. Return one JSON StudyPlan object with "
        "exam_date, minutes_per_day, level, topics (provided verbatim) and days, each with date and tasks. "
        f"Exam: {exam_date}; minutes_per_day: {budget}; level: {level}. "
        f"Topics: {json.dumps(topics, ensure_ascii=False)}. "
        f"Assignments in order: {json.dumps(assignments)}."
    )


def validate_plan(payload, chosen, slots, evidence, *, today, exam_date, budget, level):
    """Reject invented schedules, labels, forward dependencies and duplicate tasks."""
    try:
        plan = StudyPlan.model_validate(payload)
    except ValueError:
        return None
    if (plan.exam_date, plan.minutes_per_day, plan.level, plan.topics) != (exam_date, budget, level, chosen):
        return None
    dates = [day.date for day in plan.days]
    if dates != sorted(set(dates)) or any(not today <= day < exam_date for day in dates):
        return None
    actual = [(day.date, task.kind, task.topic_id, task.minutes)
              for day in plan.days for task in day.tasks]
    if actual != slots:
        return None
    allowed = {topic.id: set(topic.sources) for topic in chosen}
    completed, objectives, activities = set(), set(), set()
    for day in plan.days:
        if sum(task.minutes for task in day.tasks) > budget:
            return None
        for task in day.tasks:
            if (not set(task.sources) <= allowed[task.topic_id]
                    or len(set(task.sources)) != len(task.sources)
                    or not set(task.prerequisites) <= completed - {task.topic_id}
                    or len(set(task.prerequisites)) != len(task.prerequisites)):
                return None
            objective = normalized_text(task.objective)
            activity = normalized_text(task.activity)
            if not objective or not activity or objective in objectives or activity in activities:
                return None
            if any(not normalized_text(point) for point in task.knowledge):
                return None
            quote = " ".join(task.support_quote.split())
            if len(quote) < 20 or not any(quote in " ".join(evidence[label - 1]["text"].split())
                                          for label in task.sources):
                return None
            objectives.add(objective)
            activities.add(activity)
            if task.kind == "learn":
                completed.add(task.topic_id)
    return plan
