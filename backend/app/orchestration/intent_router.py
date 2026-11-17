"""A1: classify queries into supported learning tasks."""
from __future__ import annotations
import re


def route_intent(query: str) -> str:
    """
    Phân loại ý định của sinh viên DSA:
    - 'quiz': yêu cầu tạo bài kiểm tra, câu hỏi trắc nghiệm
    - 'revision': yêu cầu tóm tắt, lộ trình, đề cương ôn tập
    - 'qa': hỏi đáp lý thuyết, bài tập, giải thuật DSA
    """
    q = query.lower().strip()
    if any(k in q for k in ["trắc nghiệm", "quiz", "làm trắc nghiệm", "bài test", "câu hỏi trắc nghiệm"]):
        return "quiz"
    if any(k in q for k in ["lộ trình", "ôn tập", "kế hoạch", "outline", "đề cương", "tổng hợp kiến thức"]):
        return "revision"
    return "qa"
