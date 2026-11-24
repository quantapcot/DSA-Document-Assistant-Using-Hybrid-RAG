"""Execute a bounded Chat plan and keep evidence coverage per subject."""
from dataclasses import dataclass
import logging
import re

from app.generation.abstention import SCORE_THRESHOLDS, evidence_score
from app.generation.citations import MAX_EVIDENCES
from .intent_router import ChatQueryPlan, MAX_CHAT_RETRIEVALS, SearchTarget, match_text

logger = logging.getLogger(__name__)
_GRAPH_CONTEXT = re.compile(
    r"\b(?:vertices|edges?|arcs?|adjacency|dinh|canh|nut|duyet|bfs|dfs|ma tran ke|danh sach lan can)\b"
)
_FACT = re.compile(
    r"\b(?:is|are|uses?|consists? of|requires?|works?|visits?|stores?|contains?|divides?|merges?|"
    r"represents?|la|gom|bao gom|su dung|chia|ghep|duyet|chon|thoi gian|do phuc tap)\b"
)
_COMPLEXITY = re.compile(
    r"\b(?:complexity|running time|time complexity|space complexity|do phuc tap|thoi gian tinh)\b|"
    r"\b[OΘΩ]\s*\(", re.I
)
_TOC = re.compile(r"^(?:(?:chapter|chuong)\s*\d+|contents|muc luc)\b")


@dataclass(frozen=True)
class ChatEvidence:
    passages: list[dict]
    covered: dict[str, tuple[int, ...]]
    missing: tuple[str, ...]
    unsearched: tuple[str, ...]
    reason: str | None
    calls: int
    workspace_empty: bool = False


def _contains(text, aliases):
    padded = f" {text} "
    for alias in aliases:
        if not alias:
            continue
        if f" {alias} " in padded:
            return True
        last = alias.rsplit(" ", 1)[-1]
        if last.isascii() and len(last) >= 4 and last.isalpha():
            if last.endswith("s") and f" {alias[:-1]} " in padded:
                return True
            if f" {alias}s " in padded or f" {alias}es " in padded:
                return True
    return False


def _matches_body(text, target):
    if _contains(text, target.aliases):
        return True
    # Some Vietnamese PDFs join "đồ thị" to the following word or vertex name.
    return target.name == "Graph" and bool(re.search(r"\bdothi\w*\b", text))


def _anchors(item: dict, target: SearchTarget, plan: ChatQueryPlan) -> bool:
    raw = item.get("text", "")
    body = match_text(raw)
    if len(body.split()) < 7 or not _matches_body(body, target):
        return False
    if target.name == "Graph" and not _GRAPH_CONTEXT.search(body):
        return False
    if plan.complexity and not (_COMPLEXITY.search(body) or _COMPLEXITY.search(raw)):
        return False
    # A contents page can name both algorithms without teaching either one.
    if plan.intent == "comparison" and _TOC.match(body) and not _FACT.search(body):
        return False
    return True


def _confirmed(hit, target, plan, method):
    raw = hit["text"]
    body = match_text(raw)
    score = evidence_score(hit)
    if score is not None and score >= SCORE_THRESHOLDS[method]:
        return True
    # A low reranker score is not enough to reject a passage that explicitly
    # teaches the requested concept. Require both a focused anchor and substance.
    lead = " ".join(body.split()[:9])
    section = match_text(hit.get("section") or "")
    focused = _matches_body(lead, target) or _matches_body(section, target)
    return focused and bool(_FACT.search(body) or _COMPLEXITY.search(raw))


def _key(item):
    return item.get("chunk_id") or (item["filename"], item.get("page"), match_text(item["text"]))


def _pack(groups):
    selected, seen_ids, seen_text = [], set(), set()
    for rank in range(max(map(len, groups.values()), default=0)):
        for candidates in groups.values():
            if rank >= len(candidates):
                continue
            item = candidates[rank]
            identifier, text = _key(item), match_text(item["text"])
            if identifier in seen_ids or text in seen_text:
                continue
            selected.append(item)
            seen_ids.add(identifier)
            seen_text.add(text)
            if len(selected) == MAX_EVIDENCES:
                return selected
    return selected


async def retrieve_chat_evidence(plan: ChatQueryPlan, retrieve, workspace, corpus) -> ChatEvidence:
    groups, attempted = {}, set()
    had_hits = had_anchors = False
    best_score = None

    async def search(target, query):
        nonlocal had_hits, had_anchors, best_score
        normalized = match_text(query)
        if normalized in attempted or len(attempted) >= MAX_CHAT_RETRIEVALS:
            return []
        attempted.add(normalized)
        hits, method = await retrieve(query)
        accepted = []
        for hit in hits:
            if hit.get("workspace_id", workspace) != workspace or hit.get("corpus_id", corpus) != corpus:
                continue
            had_hits = True
            if (score := evidence_score(hit)) is not None:
                best_score = score if best_score is None else max(best_score, score)
            if not hit.get("filename") or not _anchors(hit, target, plan):
                continue
            had_anchors = True
            if _confirmed(hit, target, plan, method):
                accepted.append({**hit, "retrieval_method": method})
        return accepted

    for target in plan.targets:
        hits = await search(target, target.query)
        if not hits and target.alternate and plan.intent != "comparison":
            hits = await search(target, target.alternate)
        groups[target.name] = hits
    for target in plan.followups:
        pooled = (item for hits in groups.values() for item in hits)
        if not any(_anchors(item, target, plan) for item in pooled):
            groups[target.name] = await search(target, target.query)

    passages = _pack(groups)
    covered = {}
    for name, candidates in groups.items():
        ids = {_key(item) for item in candidates}
        texts = {match_text(item["text"]) for item in candidates}
        labels = tuple(index for index, item in enumerate(passages, 1)
                       if _key(item) in ids or match_text(item["text"]) in texts)
        if labels:
            covered[name] = labels
    missing = tuple(target.name for target in plan.targets if target.name not in covered)
    reason = None
    if not passages:
        reason = "weak_evidence" if had_anchors else "off_topic" if had_hits else "no_results"
    logger.info("chat retrieval intent=%s searches=%d passages=%d covered=%d missing=%d reason=%s best_score=%s",
                plan.intent, len(attempted), len(passages), len(covered), len(missing), reason, best_score)
    return ChatEvidence(passages, covered, missing, plan.unsearched, reason, len(attempted))
