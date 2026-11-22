"""Provider calls with bounded retries and credential-safe logging."""
import logging
import time

from app.core.config import settings

logger = logging.getLogger(__name__)
_TIMEOUT = 25
_RETRY_DELAYS = (1, 2)


def _call_openai(messages, *, model, api_key, temperature, max_tokens):
    import openai

    kwargs = {"api_key": api_key, "timeout": _TIMEOUT, "max_retries": 0}
    if settings.LLM_BASE_URL:
        kwargs["base_url"] = settings.LLM_BASE_URL
    with openai.OpenAI(**kwargs) as client:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


def _call_gemini(prompt, *, model, api_key, temperature, max_tokens):
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    response = genai.GenerativeModel(model).generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=temperature, max_output_tokens=max_tokens,
        ),
        request_options={"timeout": _TIMEOUT, "retry": None},
    )
    return response.text or ""


def _is_retryable(error):
    code = getattr(error, "status_code", None) or getattr(error, "code", None)
    if callable(code):
        code = code()
    return (
        code in (429, 408, 504)
        or getattr(code, "name", None) in {"RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED"}
        or isinstance(error, TimeoutError)
        or type(error).__name__ in {"APITimeoutError", "DeadlineExceeded", "ReadTimeout", "ConnectTimeout"}
    )


def generate_answer(system_prompt, user_prompt, *, provider=None, model=None,
                    temperature=0.3, max_tokens=1024):
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
        return ""

    if not key or not key.get_secret_value().strip():
        return ""
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            return call(
                payload, model=model or settings.LLM_MODEL or default_model,
                api_key=key.get_secret_value(), temperature=temperature, max_tokens=max_tokens,
            ).strip()
        except Exception as error:
            # Provider exception bodies can include credentials or request content.
            if attempt == len(_RETRY_DELAYS) or not _is_retryable(error):
                logger.warning("LLM generation failed for %s", provider)
                return ""
            logger.info("Transient LLM failure for %s. Retrying.", provider)
            time.sleep(_RETRY_DELAYS[attempt])
    return ""
