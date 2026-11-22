"""Decode study responses and ground their citations without logging content."""
import json
import logging
import re
import unicodedata

from pydantic import ValidationError

from .llm_client import LLMError

logger = logging.getLogger(__name__)
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.I | re.S)
_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.I | re.S)
_TOKEN = re.compile(r"[\w]+|<=|>=|==|!=|->|[<>≤≥=+\-*/^%()[\]]", re.UNICODE)
_GLYPHS = str.maketrans({
    **dict.fromkeys("‐‑‒–—−", "-"),
    "≤": "<=", "≥": ">=", "≠": "!=", "→": "->",
})


def reject(stage, reason, *, slot=None, source=None):
    logger.warning("study_output stage=%s reason=%s slot=%s source=%s", stage, reason, slot, source)


def decode_json(raw, stage):
    text = _THINK.sub("", raw, count=1).strip()
    fenced = _FENCE.fullmatch(text)
    candidate = fenced[1] if fenced else text
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError as exc:
        # FIT-AI sometimes adds prose or a second fence around the JSON value;
        # decode the first complete array/object instead of rejecting the batch.
        result = _first_json_value(candidate)
        if result is not None:
            logger.info("study_json stage=%s recovered=embedded", stage)
            return result
        # Offsets only: the exception's document contains the original answer.
        logger.warning("study_json stage=%s reason=parse_error offset=%d length=%d",
                       stage, exc.pos, len(text))
        raise LLMError("invalid_json") from None
    logger.info("study_json stage=%s shape=%s", stage, type(result).__name__)
    return result


def _first_json_value(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char in "[{":
            try:
                value, _ = decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                continue
            if isinstance(value, (list, dict)) and value:
                return value
    return None


def rows(payload, stage, *wrappers):
    if isinstance(payload, dict):
        keys = [key for key in ("items", *wrappers) if key in payload]
        lists = [key for key, value in payload.items() if isinstance(value, list) and value
                 and all(isinstance(item, dict) for item in value)]
        if len(keys) == 1:
            payload = payload[keys[0]]
        elif "slot" in payload and not keys:
            payload = [payload]
        elif len(lists) == 1:
            # FIT-AI sometimes wraps the array under another key, e.g. {"answers": [...]}.
            payload = payload[lists[0]]
        elif payload and all(isinstance(value, dict) for value in payload.values()):
            # ...or keys rows by slot, e.g. {"1": {...}, "2": {...}}.
            payload = [{"slot": key, **value} if "slot" not in value else value
                       for key, value in payload.items()]
    if not isinstance(payload, list):
        reject(stage, "unexpected_json_shape")
        return []
    logger.info("study_output stage=%s rows=%d", stage, len(payload))
    return payload


def integer(value, *, label=False):
    if isinstance(value, str):
        pattern = r"(?:\[([0-9]{1,6})\]|([0-9]{1,6}))" if label else r"([0-9]{1,6})"
        match = re.fullmatch(pattern, value.strip())
        if match:
            return int(next(group for group in match.groups() if group is not None))
    return value


def normalize_row(item, *, integers=(), booleans=()):
    if not isinstance(item, dict):
        return item
    result = dict(item)
    for field in integers:
        if field in result:
            result[field] = integer(result[field], label=field == "source")
    for field in booleans:
        value = result.get(field)
        if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
            result[field] = value.strip().casefold() == "true"
    if isinstance(result.get("sources"), list):
        result["sources"] = [integer(value, label=True) for value in result["sources"]]
    return result


def validate_row(model, item, stage):
    try:
        return model.model_validate(item)
    except ValidationError as exc:
        slot = item.get("slot") if isinstance(item, dict) else None
        for error in exc.errors(include_input=False, include_context=False, include_url=False):
            field = error["loc"][0] if error["loc"] else None
            field = field if field in model.model_fields else "(unknown)"
            logger.warning("study_output stage=%s reason=schema slot=%s field=%s type=%s",
                           stage, slot if type(slot) is int else None, field, error["type"])
        return None


def _tokens(text):
    text = unicodedata.normalize("NFC", text).casefold().translate(_GLYPHS)
    # PDF extraction can split a word over a line using a hyphen.
    text = re.sub(r"(?<=[^\W\d_]{2})-\s*\n\s*(?=[^\W\d_]{2})", "", text).replace("\u00ad", "")
    return " ".join(_TOKEN.findall(text))


def supports_quote(quote, passage):
    """Require an exact contiguous token span after layout normalization."""
    normalized = _tokens(quote)
    if len(normalized) < 20 or len(normalized.split()) < 4:
        return False
    return f" {normalized} " in f" {_tokens(passage)} "
