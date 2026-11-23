"""HTTP client: Streamlit → FastAPI backend."""
import os
from typing import Any

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
_STUDY_TIMEOUTS = {"chat": 90, "quiz": 240, "revision": 240}
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


def _auth(token):
    return {"headers": {"Authorization": f"Bearer {token}"}} if token else {}


def _scope(workspace_id, corpus_id, token):
    if token:
        return {}
    return {key: value for key, value in (("workspace_id", workspace_id), ("corpus_id", corpus_id)) if value}


def create_session():
    try:
        response = requests.post(f"{BACKEND_URL}/api/sessions", timeout=10)
        return (True, response.json()) if response.ok else (False, {"error": _error_detail(response)})
    except requests.RequestException:
        return False, {"error": "Không thể kết nối để tạo phiên học."}


def delete_session(token):
    try:
        response = requests.delete(f"{BACKEND_URL}/api/sessions/current", timeout=30, **_auth(token))
        if response.ok:
            return True, {"pending": response.status_code == 202}
        return False, {"error": _error_detail(response)}
    except requests.RequestException:
        return False, {"error": "Không thể xác nhận đã xóa phiên học."}


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
    *, token=None,
) -> tuple[bool, dict[str, Any]]:
    return _study_request("chat", {"query": query}, workspace_id, corpus_id, token)


def create_quiz(topic, num_questions=5, workspace_id=None, corpus_id=None, *, token=None):
    return _study_request("quiz", {"topic": topic, "num_questions": num_questions}, workspace_id, corpus_id, token)


def create_roadmap(workspace_id=None, corpus_id=None, *, token=None):
    return _study_request("roadmap", {}, workspace_id, corpus_id, token)


def _study_request(endpoint, payload, workspace_id, corpus_id, token):
    try:
        task = "revision" if endpoint == "roadmap" else endpoint
        response = requests.post(
            f"{BACKEND_URL}/api/{endpoint}",
            json={**payload, **_scope(workspace_id, corpus_id, token)},
            timeout=(10, _STUDY_TIMEOUTS[task] + _HTTP_TIMEOUT_MARGIN), **_auth(token),
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
    *, token=None,
) -> tuple[bool, dict[str, Any]]:
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/upload",
            files={"file": (filename, file_bytes, "application/octet-stream")},
            data=_scope(workspace_id, corpus_id, token),
            timeout=300,  # 5 phút — file lớn cần thời gian embedding
            **_auth(token),
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
    *, token=None,
) -> tuple[bool, dict[str, Any]]:
    """Lấy danh sách tài liệu đã nạp trong workspace/corpus chỉ định."""
    try:
        response = requests.get(
            f"{BACKEND_URL}/api/documents",
            params=_scope(workspace_id, corpus_id, token),
            timeout=30, **_auth(token),
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
    *, token=None,
) -> tuple[bool, dict[str, Any]]:
    """Xóa tài liệu theo document_id khỏi Qdrant và ổ đĩa."""
    try:
        response = requests.delete(
            f"{BACKEND_URL}/api/documents/{document_id}",
            params=_scope(workspace_id, corpus_id, token),
            timeout=30, **_auth(token),
        )
        if response.ok:
            return True, response.json()
        return False, {"error": _error_detail(response)}
    except requests.RequestException as exc:
        return False, {"error": str(exc)}
