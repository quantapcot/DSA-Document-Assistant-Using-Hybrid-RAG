"""HTTP client: Streamlit → FastAPI backend."""
import os
from typing import Any

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
_STUDY_TIMEOUTS = {"chat": 90, "quiz": 360, "revision": 240}
_HTTP_TIMEOUT_MARGIN = 10
_GENERATION_MESSAGES = {
    "timeout": "LLM vượt thời gian xử lý cho phép.",
    "quota_exceeded": "LLM đã hết hạn mức sử dụng.",
    "rate_limit": "LLM đang giới hạn tần suất. Hãy thử lại sau.",
    "authentication": "LLM từ chối xác thực.",
    "not_configured": "Chưa cấu hình kết nối LLM.",
    "invalid_json": "LLM trả về JSON không hợp lệ.",
    "truncated_json": "Phản hồi LLM bị cắt trước khi hoàn tất.",
    "invalid_output": "Đầu ra LLM không vượt qua kiểm tra nội dung.",
    "provider": "Không nhận được kết quả từ nhà cung cấp LLM.",
}


def generation_error_message(code):
    return _GENERATION_MESSAGES.get(code)


def _error_detail(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail", f"HTTP {response.status_code}")
        return detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def check_health() -> tuple[bool, dict[str, Any]]:
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=10)
        if response.ok:
            data = response.json()
            for task, timeout in data.get("study_request_timeouts", {}).items():
                if task in _STUDY_TIMEOUTS and isinstance(timeout, (int, float)) and 0 < timeout <= 900:
                    _STUDY_TIMEOUTS[task] = timeout
            return True, data
        return False, {"error": _error_detail(response)}
    except requests.RequestException as exc:
        return False, {"error": str(exc)}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def ask_question(
    query: str,
    workspace_id: str | None = None,
    corpus_id: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    return _study_request("chat", query, workspace_id, corpus_id)


def create_quiz(query, num_questions=5, workspace_id=None, corpus_id=None, *, difficulty="Medium"):
    return _study_request(
        "quiz", query, workspace_id, corpus_id,
        num_questions=num_questions, difficulty=difficulty,
    )


def create_revision(query, workspace_id=None, corpus_id=None, *, exam_date=None,
                    days_remaining=None, minutes_per_day=45, level="Intermediate", all_topics=False):
    return _study_request(
        "revision", query, workspace_id, corpus_id,
        exam_date=exam_date, days_remaining=days_remaining, minutes_per_day=minutes_per_day,
        level=level, all_topics=all_topics,
    )


def _study_request(endpoint, query, workspace_id, corpus_id, **fields):
    try:
        payload: dict[str, Any] = {"query": query, **fields}
        if workspace_id:
            payload["workspace_id"] = workspace_id
        if corpus_id:
            payload["corpus_id"] = corpus_id
        response = requests.post(
            f"{BACKEND_URL}/api/{endpoint}",
            json=payload,
            timeout=(10, _STUDY_TIMEOUTS[endpoint] + _HTTP_TIMEOUT_MARGIN),
        )
        if response.ok:
            data = response.json()
            data["process_time_ms"] = response.headers.get("X-Process-Time-MS", "N/A")
            return True, data
        return False, {"error": _error_detail(response)}
    except requests.Timeout:
        return False, {"error": "API đã vượt thời gian chờ.", "error_code": "timeout"}
    except requests.RequestException as exc:
        return False, {"error": str(exc)}


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def upload_pdf(
    filename: str,
    file_bytes: bytes,
    workspace_id: str = "default-workspace",
    corpus_id: str = "user-uploads",
) -> tuple[bool, dict[str, Any]]:
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/upload",
            files={"file": (filename, file_bytes, "application/octet-stream")},
            data={"workspace_id": workspace_id, "corpus_id": corpus_id},
            timeout=300,  # 5 phút — file lớn cần thời gian embedding
        )
        if response.ok:
            data = response.json()
            data["process_time_ms"] = response.headers.get("X-Process-Time-MS", "N/A")
            return True, data
        return False, {"error": _error_detail(response)}
    except requests.RequestException as exc:
        return False, {"error": str(exc)}


# ---------------------------------------------------------------------------
# Document Management
# ---------------------------------------------------------------------------

def list_documents(
    workspace_id: str = "default-workspace",
    corpus_id: str = "user-uploads",
) -> tuple[bool, dict[str, Any]]:
    """Lấy danh sách tài liệu đã nạp trong workspace/corpus chỉ định."""
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/documents",
            params={"workspace_id": workspace_id, "corpus_id": corpus_id},
            timeout=30,
        )
        if response.ok:
            return True, response.json()
        return False, {"error": _error_detail(response)}
    except requests.RequestException as exc:
        return False, {"error": str(exc)}


def delete_document(
    document_id: str,
    workspace_id: str = "default-workspace",
    corpus_id: str = "user-uploads",
) -> tuple[bool, dict[str, Any]]:
    """Xóa tài liệu theo document_id khỏi Qdrant và ổ đĩa."""
    try:
        response = requests.delete(
            f"{BACKEND_URL}/api/documents/{document_id}",
            params={"workspace_id": workspace_id, "corpus_id": corpus_id},
            timeout=30,
        )
        if response.ok:
            return True, response.json()
        return False, {"error": _error_detail(response)}
    except requests.RequestException as exc:
        return False, {"error": str(exc)}
