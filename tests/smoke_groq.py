"""Opt-in live test using a small synthetic passage and the configured Groq key."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import settings
from app.generation.quiz_generator import generate_quiz
from app.generation.rag_engine import build_chat_response
from app.generation.revision_generator import generate_roadmap


def main():
    if not settings.GROQ_API_KEY or not settings.GROQ_API_KEY.get_secret_value().strip():
        print("SKIP: GROQ_API_KEY is not configured.")
        return 77
    evidence = [{
        "filename": "smoke-fixture.pdf", "page": 1, "section": "BFS", "score": 0.9,
        "text": (
            "Breadth-first search (BFS) sử dụng Queue theo thứ tự FIFO. "
            "BFS duyệt các đỉnh theo khoảng cách tăng dần tính từ đỉnh bắt đầu. "
            "Trên đồ thị không trọng số, BFS tìm đường đi ít cạnh nhất."
        ),
    }]
    chat = build_chat_response("BFS dùng cấu trúc dữ liệu nào?", evidence)
    quiz = generate_quiz("BFS", evidence, num_questions=1)
    roadmap = generate_roadmap("BFS", evidence)
    outcomes = [
        ("Chat", chat.generation_mode == "llm" and bool(chat.citations), chat),
        ("Quiz", quiz.generated_count == 1 and bool(quiz.citations), quiz),
        ("Roadmap", len(roadmap.items) == 1 and bool(roadmap.citations), roadmap),
    ]
    for name, passed, result in outcomes:
        print(f"{name} {'PASS' if passed else 'FAIL'} ({result.generation_mode}, {result.generation_error})")
    return int(not all(passed for _, passed, _ in outcomes))


if __name__ == "__main__":
    raise SystemExit(main())
