"""C2: model-provider client — gọi LLM để sinh câu trả lời từ context đã truy xuất.

Hỗ trợ 2 provider:
  - ``openai``  : OpenAI-compatible API (GPT-4o, GPT-4o-mini, v.v.)
  - ``gemini``  : Google Gemini API (gemini-2.0-flash, v.v.)

Provider mặc định được đọc từ biến môi trường ``LLM_PROVIDER`` (mặc định "gemini").
Nếu không có API key hợp lệ, client tự động trả về chuỗi rỗng để caller xử lý fallback.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers — OpenAI-compatible
# ---------------------------------------------------------------------------

def _call_openai(messages: list[dict], *, model: str, api_key: str,
                 temperature: float, max_tokens: int) -> str:
    try:
        import openai  # type: ignore
    except ImportError:
        raise ImportError("Cài đặt openai: pip install openai")

    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Helpers — Google Gemini
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str, *, model: str, api_key: str,
                 temperature: float, max_tokens: int) -> str:
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError:
        raise ImportError("Cài đặt google-generativeai: pip install google-generativeai")

    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(model)
    config = genai.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    resp = gen_model.generate_content(prompt, generation_config=config)
    return resp.text or ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_answer(
    system_prompt: str,
    user_prompt: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Gửi prompt đến LLM và trả về chuỗi câu trả lời.

    Parameters
    ----------
    system_prompt : Hướng dẫn role/persona cho LLM.
    user_prompt   : Câu hỏi + context (đã được inject evidence).
    provider      : ``"openai"`` hoặc ``"gemini"``. Mặc định đọc từ ``LLM_PROVIDER``.
    model         : Tên model cụ thể. Mặc định đọc từ ``LLM_MODEL``.
    temperature   : Độ sáng tạo (0 = deterministic, 1 = rất ngẫu nhiên).
    max_tokens    : Giới hạn token đầu ra.

    Returns
    -------
    str — Câu trả lời từ LLM. Trả về chuỗi rỗng nếu không có API key hoặc gặp lỗi.
    """
    effective_provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()

    if effective_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("OPENAI_API_KEY chưa được cấu hình — bỏ qua LLM generation.")
            return ""
        effective_model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return _call_openai(
                messages,
                model=effective_model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.error("OpenAI generation thất bại: %s", exc)
            return ""

    elif effective_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("GEMINI_API_KEY chưa được cấu hình — bỏ qua LLM generation.")
            return ""
        effective_model = model or os.getenv("LLM_MODEL", "gemini-2.0-flash")
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        try:
            return _call_gemini(
                full_prompt,
                model=effective_model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            logger.error("Gemini generation thất bại: %s", exc)
            return ""

    else:
        logger.error("LLM provider '%s' không được hỗ trợ.", effective_provider)
        return ""
