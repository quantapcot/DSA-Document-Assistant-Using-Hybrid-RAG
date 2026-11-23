"""Provider calls with bounded retries and credential-safe logging."""
import logging
import time
from contextvars import ContextVar

from app.core.config import settings
from app.core.telemetry import study_metrics

logger = logging.getLogger(__name__)
_RETRY_DELAYS = (1, 2)
request_deadline: ContextVar[float | None] = ContextVar("llm_request_deadline", default=None)
quiz_provider_calls: ContextVar[int | None] = ContextVar("quiz_provider_calls", default=None)
STUDY_OUTPUT_LIMIT = 8000
ERROR_MESSAGES = {
    "timeout": "Groq đã vượt thời gian xử lý cho phép.",
    "quota_exceeded": "Groq đã hết hạn mức. Hãy thử lại khi hạn mức được làm mới.",
    "rate_limit": "Groq đang giới hạn tần suất. Hãy thử lại sau.",
    "authentication": "Groq từ chối xác thực. Hãy kiểm tra cấu hình.",
    "not_configured": "Chưa cấu hình GROQ_API_KEY.",
    "invalid_json": "Groq trả về dữ liệu JSON không hợp lệ.",
    "truncated_json": "Phản hồi Groq bị cắt trước khi hoàn tất.",
    "invalid_output": "Đầu ra Groq không vượt qua kiểm tra nội dung.",
    "provider": "Không nhận được kết quả từ nhà cung cấp LLM.",
}


class LLMError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(ERROR_MESSAGES[code])


def _call_groq(messages, *, model, api_key, temperature, max_tokens, timeout, response_format=None):
    import openai

    options = {"extra_body": {"reasoning_effort": "low"}} if model.startswith("openai/gpt-oss-") else {}
    if response_format is not None:
        if response_format["type"] == "json_schema" and model not in {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}:
            response_format = {"type": "json_object"}
        options["response_format"] = response_format
    with openai.OpenAI(api_key=api_key, timeout=timeout, max_retries=0,
                       base_url=settings.GROQ_BASE_URL) as client:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_completion_tokens=max_tokens, **options,
        )
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise LLMError("truncated_json")
        return choice.message.content or ""


def classify_error(error):
    if isinstance(error, LLMError):
        return error.code
    code = getattr(error, "status_code", None) or getattr(error, "code", None)
    if callable(code):
        code = code()
    name = getattr(code, "name", None)
    if code in (401, 403) or name in {"UNAUTHENTICATED", "PERMISSION_DENIED"}:
        return "authentication"
    if code == 402:
        return "quota_exceeded"
    if code in (500, 502, 503):
        return "provider_unavailable"
    if code == 429 or name == "RESOURCE_EXHAUSTED":
        # Use provider error codes and quota descriptions for classification only.
        body = f"{getattr(error, 'body', '')} {getattr(error, 'details', '')} {error}".casefold()
        permanent = ("insufficient_quota", "billing_hard_limit", "budget_exceeded",
                     "daily_limit", "daily quota", "daily budget", "per day", "perday", "per_day",
                     "daily", "credits exhausted", "insufficient credits")
        temporary = ("per minute", "per second", "per_minute", "per_second", "perminute",
                     "persecond", "rate_limit_exceeded", "rate limit")
        if any(term in body for term in permanent):
            return "quota_exceeded"
        if any(term in body for term in temporary):
            return "rate_limit"
        return "quota_exceeded" if "quota_exceeded" in body or "quota exceeded" in body else "rate_limit"
    if (code in (408, 504) or name == "DEADLINE_EXCEEDED" or isinstance(error, TimeoutError)
            or type(error).__name__ in {"APITimeoutError", "DeadlineExceeded", "ReadTimeout", "ConnectTimeout"}):
        return "timeout"
    return "provider"


def _remaining(deadline):
    return deadline - time.monotonic() if deadline is not None else float("inf")


def _retry_after(error):
    response = getattr(error, "response", None)
    try:
        return max(0.0, float(response.headers.get("retry-after", 0))) if response is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def generate_answer(system_prompt, user_prompt, *, temperature=0.3, max_tokens=1024,
                    task="chat", raise_errors=False, response_format=None):
    """Generate text, returning an empty string when extraction is needed."""
    key = settings.GROQ_API_KEY
    payload = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    if not key or not key.get_secret_value().strip():
        if raise_errors:
            raise LLMError("not_configured")
        return ""
    deadline = request_deadline.get()
    timeout = getattr(settings, f"LLM_{task.upper()}_TIMEOUT")
    failure = "timeout"
    metrics = study_metrics.get()
    for attempt in range(len(_RETRY_DELAYS) + 1):
        remaining = _remaining(deadline)
        if remaining <= 0:
            failure = "timeout"
            break
        try:
            if task in {"quiz_generation", "quiz_verification"} and (calls := quiz_provider_calls.get()) is not None:
                quiz_provider_calls.set(calls + 1)
            if metrics is not None:
                metrics.provider_calls += 1
                metrics.provider_retries += int(attempt > 0)
            started = time.monotonic()
            try:
                result = _call_groq(
                    payload, model=settings.GROQ_MODEL,
                    api_key=key.get_secret_value(), temperature=temperature, max_tokens=max_tokens,
                    timeout=min(timeout, remaining), response_format=response_format,
                ).strip()
            finally:
                elapsed = (time.monotonic() - started) * 1000
                if metrics is not None:
                    metrics.llm_ms[task] = metrics.llm_ms.get(task, 0.0) + elapsed
                logger.info("groq task=%s attempt=%d elapsed_ms=%.1f", task, attempt + 1, elapsed)
            if _remaining(deadline) <= 0:
                raise LLMError("timeout")
            return result
        except Exception as error:
            failure = classify_error(error)
            if attempt == len(_RETRY_DELAYS) or failure not in {"rate_limit", "timeout", "provider_unavailable"}:
                break
            delay = max(_RETRY_DELAYS[attempt], _retry_after(error))
            if delay > 5:
                break
            if _remaining(deadline) <= delay:
                failure = "timeout"
                break
            logger.info("Retrying LLM task=%s reason=%s", task, failure)
            time.sleep(delay)
    logger.warning("LLM task failed task=%s reason=%s", task, failure)
    if raise_errors:
        raise LLMError("provider" if failure == "provider_unavailable" else failure)
    return ""
