"""A2: frontend HTTP access to backend. Scaffold only; not implemented."""
"""HTTP client: Streamlit -> FastAPI (PoC only)."""
import os
from typing import Any

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def _error_detail(response: requests.Response) -> str:
    try:
        return str(response.json().get("detail", f"HTTP {response.status_code}"))
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"


def check_health() -> tuple[bool, dict[str, Any]]:
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.ok:
            return True, response.json()
        return False, {"error": _error_detail(response)}
    except requests.RequestException as exc:
        return False, {"error": str(exc)}


def ask_question(query: str) -> tuple[bool, dict[str, Any]]:
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            json={"query": query},
            timeout=15,
        )
        if response.ok:
            data = response.json()
            data["process_time_ms"] = response.headers.get("X-Process-Time-MS", "N/A")
            return True, data
        return False, {"error": _error_detail(response)}
    except requests.RequestException as exc:
        return False, {"error": str(exc)}


def upload_pdf(filename: str, file_bytes: bytes) -> tuple[bool, dict[str, Any]]:
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/upload",
            files={"file": (filename, file_bytes, "application/pdf")},
            timeout=30,
        )
        if response.ok:
            data = response.json()
            data["process_time_ms"] = response.headers.get("X-Process-Time-MS", "N/A")
            return True, data
        return False, {"error": _error_detail(response)}
    except requests.RequestException as exc:
        return False, {"error": str(exc)}