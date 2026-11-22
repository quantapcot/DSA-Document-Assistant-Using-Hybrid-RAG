"""Provider calls with bounded retries and credential-safe logging."""
import logging
import time
from contextvars import ContextVar

from app.core.config import settings

logger = logging.getLogger(__name__)
_RETRY_DELAYS = (1, 2)
request_deadline: ContextVar[float | None] = ContextVar("llm_request_deadline", default=None)
FIT_OUTPUT_LIMIT = 4000
ERROR_MESSAGES = {
    "timeout": "FIT-AI đã vượt thời gian xử lý cho phép.",
    "quota_exceeded": "FIT-AI đã hết hạn mức. Hãy kiểm tra ngân sách hoặc chờ kỳ hạn mức tiếp theo.",
    "rate_limit": "FIT-AI đang giới hạn tần suất. Hãy thử lại sau.",
    "authentication": "FIT-AI từ chối xác thực. Hãy kiểm tra thông tin kết nối.",
    "not_configured": "Chưa cấu hình kết nối LLM.",
    "invalid_json": "FIT-AI trả về dữ liệu JSON không hợp lệ.",
    "truncated_json": "Phản hồi FIT-AI bị cắt trước khi hoàn tất.",
    "invalid_output": "Đầu ra FIT-AI không vượt qua kiểm tra nội dung.",
    "provider": "Không nhận được kết quả từ nhà cung cấp LLM.",
}


class LLMError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(ERROR_MESSAGES[code])


def _call_openai(messages, *, model, api_key, temperature, max_tokens, timeout):
    import openai

    kwargs = {"api_key": api_key, "timeout": timeout, "max_retries": 0}
    if settings.LLM_BASE_URL:
        kwargs["base_url"] = settings.LLM_BASE_URL
    with openai.OpenAI(**kwargs) as client:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
        )
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise LLMError("truncated_json")
        return choice.message.content or ""


def _call_gemini(prompt, *, model, api_key, temperature, max_tokens, timeout):
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    response = genai.GenerativeModel(model).generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=temperature, max_output_tokens=max_tokens,
        ),
        request_options={"timeout": timeout, "retry": None},
    )
    return response.text or ""


def classify_error(error):
    if isinstance(error, LLMError):
        return error.code
    code = getattr(error, "status_code", None) or getattr(error, "code", None)
    if callable(code):
        code = code()
    name = getattr(code, "name", None)
    if code in (401, 403) or name in {"UNAUTHENTICATED", "PERMISSION_DENIED"}:
        return "authentication"
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


def generate_answer(system_prompt, user_prompt, *, provider=None, model=None,
                    temperature=0.3, max_tokens=1024, task="chat", raise_errors=False):
    """Generate text, returning an empty string when extraction is needed."""
    provider = (provider or settings.LLM_PROVIDER).lower()
    if provider == "openai":
        key = settings.OPENAI_API_KEY
        call = _call_openai
        payload = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        default_model = "gpt-4o-mini"
    elif provider == "gemini":
        key = settings.GEMINI_API_KEY
        call = _call_gemini
        payload = f"{system_prompt}\n\n{user_prompt}"
        default_model = "gemini-2.0-flash"
    else:
        logger.warning("Unsupported LLM provider")
        if raise_errors:
            raise LLMError("not_configured")
        return ""

    if not key or not key.get_secret_value().strip():
        if raise_errors:
            raise LLMError("not_configured")
        return ""
    deadline = request_deadline.get()
    timeout = getattr(settings, f"LLM_{task.upper()}_TIMEOUT")
    failure = "timeout"
    for attempt in range(len(_RETRY_DELAYS) + 1):
        remaining = _remaining(deadline)
        if remaining <= 0:
            failure = "timeout"
            break
        try:
            result = call(
                payload, model=model or settings.LLM_MODEL or default_model,
                api_key=key.get_secret_value(), temperature=temperature, max_tokens=max_tokens,
                timeout=min(timeout, remaining),
            ).strip()
            if _remaining(deadline) <= 0:
                raise LLMError("timeout")
            return result
        except Exception as error:
            failure = classify_error(error)
            if attempt == len(_RETRY_DELAYS) or failure not in {"rate_limit", "timeout"}:
                break
            delay = _RETRY_DELAYS[attempt]
            if _remaining(deadline) <= delay:
                failure = "timeout"
                break
            logger.info("Retrying LLM task=%s reason=%s", task, failure)
            time.sleep(delay)
    logger.warning("LLM task failed task=%s reason=%s", task, failure)
    if raise_errors:
        raise LLMError(failure)
    return ""
