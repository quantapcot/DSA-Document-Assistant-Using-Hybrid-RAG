"""Lightweight query planning for scoped Chat retrieval."""
from dataclasses import dataclass
import re
import unicodedata

MAX_CHAT_RETRIEVALS = 4

# Domain aliases expand known terms. Syntax and the free-form fallback also
# work for concepts outside this vocabulary.
_ALIASES = (
    ("Graph traversal", "duyệt đồ thị", "duyệt đồthị", "graph traversals", "traversing a graph", "traversing graphs"),
    ("Graph", "đồ thị", "đồthị", "graphs", "graph theory", "lý thuyết đồ thị"),
    ("BFS", "breadth first search", "duyệt theo chiều rộng", "tìm kiếm theo chiều rộng"),
    ("DFS", "depth first search", "duyệt theo chiều sâu", "tìm kiếm theo chiều sâu"),
    ("Binary Search Tree", "cây nhị phân tìm kiếm", "cây tìm kiếm nhị phân", "BST", "binary search trees"),
    ("Quick Sort", "quicksort", "sắp xếp nhanh"),
    ("Merge Sort", "mergesort", "sắp xếp trộn", "sắp xếp hòa nhập"),
    ("AVL", "cây AVL", "AVL tree"),
    ("Sorting", "sắp xếp", "sorting algorithms", "thuật toán sắp xếp"),
)
_COMPARISON = re.compile(
    r"^(?:please\s+|hãy\s+)?(?:so sánh|phân biệt|compare|khác nhau giữa|"
    r"(?:what is the )?difference between)\s+", re.I,
)
_STRONG_SEPARATORS = re.compile(r"\s+(?:với|with|vs\.?|versus)\s+|\s*[;,]\s*", re.I)
_WEAK_SEPARATORS = re.compile(r"\s+(?:và|and)\s+", re.I)
_VERSUS = re.compile(r"\s+(?:vs\.?|versus)\s+", re.I)
_COMPARISON_FACET = re.compile(r"\s+(?:về|xét theo|in terms of|with respect to)\s+", re.I)
_OVERVIEW = re.compile(
    r"^(?:các\s+)?(?:tổng quan(?: về)?|liệt kê|phân loại|phương pháp|các loại|loại|overview of|list|types of|methods of)\s+", re.I,
)
_EXPLANATION = re.compile(
    r"^(?:hãy\s+)?(?:giải thích|trình bày|nguyên lý(?: của)?|cách hoạt động(?: của)?|"
    r"ví dụ(?: về)?|explain|how does|examples? of|what is)\s+", re.I,
)
_COMPLEXITY = re.compile(r"^(?:độ phức tạp(?: của)?|(?:time |space )?complexity of)\s+", re.I)
_SUBJECT_PREFIX = re.compile(r"^(?:thuật toán|cấu trúc dữ liệu|data structure|algorithm|an?|the)\s+", re.I)
_DEFINITION_END = re.compile(r"\s+(?:là gì|hoạt động như thế nào|works?|work)$", re.I)
_COMPARISON_END = re.compile(r"\s+(?:khác nhau(?: như thế nào)?|are different)$", re.I)


@dataclass(frozen=True)
class SearchTarget:
    name: str
    query: str
    aliases: tuple[str, ...]
    alternate: str | None = None


@dataclass(frozen=True)
class ChatQueryPlan:
    query: str
    intent: str
    targets: tuple[SearchTarget, ...]
    followups: tuple[SearchTarget, ...] = ()
    unsearched: tuple[str, ...] = ()
    complexity: bool = False


def match_text(text: str) -> str:
    # Match accents and layout variants without changing documents or index terms.
    text = unicodedata.normalize("NFKC", text).casefold().replace("đ", "d").replace("\u00ad", "")
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch))
    return " ".join(re.findall(r"[^\W_]+", text))


_ALIAS_BY_TERM = {match_text(alias): aliases for aliases in _ALIASES for alias in aliases}


def _target(topic: str, *, search: str | None = None) -> SearchTarget:
    normalized = match_text(topic)
    equivalents = _ALIAS_BY_TERM.get(normalized)
    if equivalents is None:
        return SearchTarget(topic, search or topic, (normalized,))
    canonical, *variants = equivalents
    # Preserve acronyms and exact keywords on the quick path.
    alternate = " / ".join((canonical, *variants[:2]))
    return SearchTarget(canonical, search or canonical,
                        tuple(dict.fromkeys(map(match_text, equivalents))), alternate)


def _natural_target(subject: str) -> SearchTarget:
    exact = _target(subject)
    if exact.alternate:
        return exact
    # A question can start with a named topic and continue with a qualifier.
    normalized = match_text(subject)
    starts = [alias for alias in _ALIAS_BY_TERM if normalized.startswith(f"{alias} ")]
    if starts:
        name = _ALIAS_BY_TERM[max(starts, key=len)][0]
        return _target(name, search=subject)
    return exact


def _comparison(query: str) -> ChatQueryPlan | None:
    prefix = _COMPARISON.match(query)
    if not prefix and not _VERSUS.search(query) and not _COMPARISON_END.search(query):
        return None
    text = query[prefix.end():] if prefix else query
    text = _COMPARISON_FACET.split(_COMPARISON_END.sub("", text), maxsplit=1)[0]
    parts = _STRONG_SEPARATORS.split(text)
    if len(parts) == 1:
        parts = _WEAK_SEPARATORS.split(text)
    elif re.search(r"[;,]", text):
        # A final conjunction can follow a comma-delimited list.
        parts = [part for entry in parts for part in _WEAK_SEPARATORS.split(entry)]
    entities = {}
    for part in parts:
        part = _SUBJECT_PREFIX.sub("", part.strip(' \t.,?!:;”"'))
        if part and match_text(part):
            target = _target(part)
            entities.setdefault(match_text(target.name), target)
    if len(entities) < 2:
        return None
    targets = tuple(entities.values())
    return ChatQueryPlan(query, "comparison", targets[:MAX_CHAT_RETRIEVALS],
                         unsearched=tuple(target.name for target in targets[MAX_CHAT_RETRIEVALS:]))


def plan_chat_query(query: str) -> ChatQueryPlan:
    query = " ".join(unicodedata.normalize("NFC", query).split())
    bare = query.strip(" ?!.,")
    if plan := _comparison(bare):
        return ChatQueryPlan(query, plan.intent, plan.targets, unsearched=plan.unsearched)

    overview = _OVERVIEW.match(bare)
    explanation = _EXPLANATION.match(bare)
    complexity = _COMPLEXITY.match(bare)
    prefix = overview or explanation or complexity
    subject = _DEFINITION_END.sub("", bare[prefix.end():] if prefix else bare).strip() or bare
    target = _natural_target(_SUBJECT_PREFIX.sub("", subject))
    if complexity:
        search = f"{target.name} time complexity" if match_text(bare).startswith("do phuc tap") else bare
        return ChatQueryPlan(query, "explanation",
                             (SearchTarget(target.name, search, target.aliases, target.query),), complexity=True)

    if target.name == "Graph traversal" and match_text(subject) in target.aliases:
        bfs, dfs = _target("BFS"), _target("DFS")
        traversal = SearchTarget(target.name, "phép duyệt đồthị graph traversal BFS DFS",
                                 target.aliases + bfs.aliases + dfs.aliases)
        return ChatQueryPlan(query, "overview", (traversal,), (bfs, dfs))
    if target.name == "Graph" and match_text(subject) in target.aliases:
        bfs, dfs = _target("BFS"), _target("DFS")
        related = SearchTarget("Graph traversal", "phép duyệt đồthị graph traversal BFS DFS",
                               _target("Graph traversal").aliases + bfs.aliases + dfs.aliases)
        search = "đồthị đồ thị graph đỉnh cạnh định nghĩa"
        if overview and "phuong phap" in match_text(overview[0]):
            search = "biểu diễn đồthị ma trận kề danh sách kề phép duyệt đồthị"
        main = SearchTarget(target.name, search, target.aliases, "định nghĩa đồthị đồthịG đỉnh cạnh")
        definition_only = (subject != bare and not overview and
                           ("la gi" in match_text(bare) or match_text(bare).startswith("what is ")))
        return ChatQueryPlan(query, "overview", (main,), () if definition_only else (related,))

    if overview or target.name == "Sorting":
        primary = SearchTarget(target.name, f"{target.query} definition types methods", target.aliases,
                               target.alternate)
        return ChatQueryPlan(query, "overview", (primary,))
    intent = "explanation" if explanation or subject != bare else "simple"
    if intent == "explanation":
        primary = SearchTarget(target.name, bare, target.aliases, target.alternate or target.query)
    else:
        primary = target
    return ChatQueryPlan(query, intent, (primary,))
