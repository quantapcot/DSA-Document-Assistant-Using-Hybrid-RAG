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
TOPIC_PATTERNS = {
    "Mảng": r"\b(?:arrays?|mảng)\b",
    "Danh sách liên kết": r"linked lists?|danh sách liên kết",
    "Ngăn xếp": r"\b(?:stacks?|ngăn xếp)\b",
    "Hàng đợi": r"\b(?:queues?|hàng đợi)\b",
    "Tìm kiếm nhị phân": r"binary search(?! tree)|tìm kiếm nhị phân",
    "Cây nhị phân tìm kiếm": r"binary search trees?|\bbst\b|cây nhị phân tìm kiếm",
    "Cây AVL": r"\bavl\b",
    "Đồ thị": r"\bgraphs?\b|đồ thị",
    "BFS": r"\bbfs\b|breadth.first|duyệt theo chiều rộng",
    "DFS": r"\bdfs\b|depth.first|duyệt theo chiều sâu",
    "Dijkstra": r"\bdijkstra\b",
    "Merge sort": r"merge[ -]?sort|sắp xếp trộn",
    "Quick sort": r"quick[ -]?sort",
    "Đệ quy": r"recurs(?:ion|ive)|đệ quy",
    "Quy hoạch động": r"dynamic programming|quy hoạch động",
    "Bảng băm": r"hash tables?|bảng băm",
    "Độ phức tạp": r"big[ -]?o|time complexity|độ phức tạp",
}
_HEADING = re.compile(r"^(?:#{1,6}\s+|(?:chapter|chương|section|mục)\s+\d+\b|\d+(?:\.\d+)+\s+)", re.I)


def normalized_text(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def study_topic(item):
    if section := (item.get("section") or "").strip():
        return section
    lines = [line.strip() for line in item["text"].splitlines() if line.strip()]
    if lines and len(lines[0]) <= 100 and _HEADING.match(lines[0]):
        return lines[0].lstrip("# ")
    # Earlier mentions usually identify the topic; incidental supporting terms
    # later in the passage should not determine the grouping.
    text = item["text"][:500].casefold()
    matches = [(match.start(), -len(match[0]), name)
               for name, pattern in TOPIC_PATTERNS.items() if (match := re.search(pattern, text))]
    if matches:
        return min(matches)[2]
    page = max(1, item.get("page", 1))
    section_start = ((page - 1) // 4) * 4 + 1
    chunk = (item.get("metadata") or {}).get("chunk_index")
    if isinstance(chunk, int) and chunk >= 0:
        chunk_start = (chunk // 6) * 6 + 1
        return f"Tr. {page} · Đoạn {chunk_start}–{chunk_start + 5}"
    return f"Tr. {section_start}–{section_start + 3}"


def topic_key(item):
    return (item.get("document_id") or item["filename"], study_topic(item))


def diverse_evidence(result_sets, *, workspace, corpus, limit=QUIZ_EVIDENCE_LIMIT, apply_threshold=True):
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
            if apply_threshold and score is not None and method in SCORE_THRESHOLDS and score < SCORE_THRESHOLDS[method]:
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


_STOPWORDS = {
    "và", "của", "các", "những", "cho", "về", "trong", "là", "một", "thuật", "toán", "bài",
    "the", "and", "of", "for", "in", "on", "a", "an", "to", "algorithm", "algorithms",
}


# Extra aliases used only by the topic gate (they do not affect topic grouping).
GATE_PATTERNS = {
    "Sắp xếp": r"\bsort|sắp xếp",
    "Cây": r"\btrees?\b|\bcây\b",
    "Heap": r"\bheaps?\b|đống",
}


# English names used to expand Vietnamese topic queries, because the course
# slides are mostly in English and cross-lingual retrieval is weak.
ENGLISH_ALIASES = {
    "Mảng": "array", "Danh sách liên kết": "linked list", "Ngăn xếp": "stack",
    "Hàng đợi": "queue", "Tìm kiếm nhị phân": "binary search",
    "Cây nhị phân tìm kiếm": "binary search tree", "Đồ thị": "graph",
    "BFS": "breadth-first search", "DFS": "depth-first search", "Đệ quy": "recursion",
    "Quy hoạch động": "dynamic programming", "Bảng băm": "hash table",
    "Độ phức tạp": "time complexity", "Sắp xếp": "sorting", "Cây": "tree", "Heap": "heap",
}


def expanded_query(query):
    """Append English names of recognised topics to the user's topic query."""
    text = query.casefold()
    patterns = {**GATE_PATTERNS, **TOPIC_PATTERNS}
    extra = [ENGLISH_ALIASES[name] for name, pattern in patterns.items()
             if name in ENGLISH_ALIASES and re.search(pattern, text, re.I)
             and ENGLISH_ALIASES[name] not in text]
    return " ".join([query, *extra])


def topic_terms(query):
    """Return regex patterns that a passage must contain to be about the topic."""
    text = query.casefold()
    patterns = {**GATE_PATTERNS, **TOPIC_PATTERNS}
    known = [pattern for pattern in patterns.values() if re.search(pattern, text, re.I)]
    if known:
        return known
    words = [word for word in normalized_text(query).split() if len(word) >= 3 and word not in _STOPWORDS]
    return [rf"\b{re.escape(word)}" for word in words]


def is_known_topic(terms):
    return bool(terms) and all(term in TOPIC_PATTERNS.values() or term in GATE_PATTERNS.values() for term in terms)


def mentions_topic(item, terms):
    if not terms:
        return True
    text = normalized_text(" ".join((item.get("section") or "", item.get("text", ""))))
    # Known topics match one alias; free-text topics must match every keyword.
    if is_known_topic(terms):
        return any(re.search(term, text, re.I) for term in terms)
    return all(re.search(term, text, re.I) for term in terms)


_BINARY_SEARCH = r"\b(?:binary search(?! trees?\b)|(?<!cay )tim kiem nhi phan)\b"
_BINARY_SEARCH_TREE = r"\b(?:binary search trees?|bst|cay (?:nhi phan tim kiem|tim kiem nhi phan))\b"
_SPECIALIZED_SEARCH = re.compile(
    rf"{_BINARY_SEARCH_TREE}|{_BINARY_SEARCH}|\b(?:(?:breadth|depth) first search|bfs|dfs)\b"
)
_SEARCH = re.compile(r"\b(?:search(?:ing)?|tim kiem)\b")
_SEARCH_OVERVIEW = re.compile(
    r"\b(?:(?:linear|sequential) search|tim kiem (?:tuyen tinh|tuan tu)|"
    r"search(?:ing)? (?:algorithms|methods|techniques|problem)|"
    r"search(?:ing)? is (?:the|a) (?:process|task|operation) of|"
    r"bai toan tim kiem|cac (?:thuat toan|phuong phap) tim kiem)\b"
)


# Exact topic aliases for Quiz. Match the passage body so a broad section
# title or an English substring like "graph" in "paragraph" is insufficient.
_QUIZ_ALIASES = (
    (r"(?:sort|sorting)(?: algorithms?)?|(?:thuat toan )?sap xep",
     "sorting sắp xếp",
     r"\b(?:sort|sorts|sorted order|sorting|(?:merge|quick|heap)sort|sap xep)\b"),
    (r"graphs?|graph theory|(?:ly thuyet )?do thi",
     "graph đồ thị", r"\b(?:graphs?|do thi)\b"),
    (r"binary search(?: algorithm)?|(?:thuat toan )?tim kiem nhi phan",
     "binary search tìm kiếm nhị phân", _BINARY_SEARCH),
    (r"binary search trees?|bst|cay (?:nhi phan tim kiem|tim kiem nhi phan)",
     "binary search tree BST cây nhị phân tìm kiếm", _BINARY_SEARCH_TREE),
    (r"avl(?: trees?)?|cay avl", "AVL cây AVL", r"\bavl\b"),
    (r"bfs(?: graph traversal)?|breadth first search|duyet theo chieu rong",
     "BFS breadth-first search duyệt theo chiều rộng",
     r"\b(?:bfs|breadth first search|duyet theo chieu rong)\b"),
    (r"dfs(?: graph traversal)?|depth first search|duyet theo chieu sau",
     "DFS depth-first search duyệt theo chiều sâu",
     r"\b(?:dfs|depth first search|duyet theo chieu sau)\b"),
)


def _quiz_alias_text(text):
    text = unicodedata.normalize("NFD", normalized_text(text)).replace("đ", "d")
    return "".join(char for char in text if not unicodedata.combining(char))


def _matches_general_search(item):
    text, specialized_mentions = _SPECIALIZED_SEARCH.subn(" ", _quiz_alias_text(item["text"]))
    section = _quiz_alias_text(item.get("section") or "")
    # A specialized passage needs evidence of a broader discussion, such as
    # linear search or a comparison of search methods.
    pattern = _SEARCH_OVERVIEW if specialized_mentions or _SPECIALIZED_SEARCH.search(section) else _SEARCH
    return bool(pattern.search(text))


def quiz_topic_filter(topic):
    """Return a bilingual search query and a Quiz-only evidence predicate."""
    alias_topic = _quiz_alias_text(topic)
    if re.fullmatch(r"search(?:ing)?(?: algorithms?)?|(?:thuat toan )?tim kiem", alias_topic):
        return "searching search algorithms tìm kiếm", _matches_general_search
    for aliases, search, pattern in _QUIZ_ALIASES:
        if re.fullmatch(aliases, alias_topic):
            return search, lambda item: bool(re.search(pattern, _quiz_alias_text(item["text"])))
    normalized = normalized_text(topic)
    for name, pattern in TOPIC_PATTERNS.items():
        if normalized == normalized_text(name) or re.fullmatch(pattern, normalized):
            return expanded_query(topic), lambda item: bool(re.search(pattern, normalized_text(item["text"])))
    words = normalized.split()
    terms = set(words) - _STOPWORDS or set(words)
    return topic, lambda item: bool(terms) and terms.issubset(normalized_text(item["text"]).split())
