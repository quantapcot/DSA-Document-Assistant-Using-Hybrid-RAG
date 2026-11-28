"""Final Evaluation Visualization Consolidation & Scientific Parity Package."""

from __future__ import annotations

from pathlib import Path

EXPECTED_BRANCH = "feat/final-readiness"
EXPECTED_MAIN = "b84f09450f213b03ad57ef8b388f482ee0f0eaa2"
EXPECTED_STAGE05_HEAD = "139957c0d6974df273c86ac2029be1c8fb7d158e"
EXPECTED_AC04_HEAD = "9b09c2e904648cb2eb85acd8986b099b84494f65"

CONSOLIDATED_FIGURES: list[tuple[int, str, str]] = [
    (1, "fig01_danh_gia_chunking", "Hình 01: Đánh giá chiến lược Chunking (Recursive vs Structure-Adaptive)"),
    (2, "fig02_chat_luong_retrieval", "Hình 02: Đánh giá chất lượng Retrieval trên 8 tổ hợp và 9 mốc Top-K"),
    (3, "fig03_chat_luong_evidence", "Hình 03: Đánh giá chất lượng Evidence nguồn xác thực và độ phủ Knowledge Unit"),
    (4, "fig04_hieu_nang_va_tai_nguyen", "Hình 04: Hiệu năng độ trễ Retrieval (tách biệt 4 giao thức) và tài nguyên hệ thống"),
    (5, "fig05_chat_luong_cau_tra_loi", "Hình 05: Chất lượng sinh câu trả lời (10 thước đo Controlled Agent Generation)"),
    (6, "fig06_ablation_va_do_nhay", "Hình 06: Nghiên cứu Ablation và phân tích độ nhạy (Top-K, Candidate Pool, Context Budget)"),
    (7, "fig07_thong_ke_va_tradeoff", "Hình 07: Phân tích thống kê và biên tối ưu cân bằng Chất lượng – Độ trễ – Tài nguyên"),
    (8, "fig08_do_tin_cay_va_phan_tich_loi", "Hình 08: Độ tin cậy (Robustness), tính năng học tập và phân tích lỗi thực nghiệm"),
    (9, "fig09_live_groq_va_http_e2e", "Hình 09: Đánh giá Live Groq API (openai/gpt-oss-120b) và HTTP End-to-End"),
    (10, "fig10_chuc_nang_va_do_on_dinh_api", "Hình 10: Chức năng thực tế (Quiz, Roadmap, Robustness) và độ ổn định API sản phẩm"),
]


def consolidated_png_dir(root: Path) -> Path:
    return root / "evaluation" / "figures" / "consolidated" / "png"


def consolidated_svg_dir(root: Path) -> Path:
    return root / "evaluation" / "figures" / "consolidated" / "svg"


def original_png_dir(root: Path) -> Path:
    return root / "evaluation" / "figures" / "original" / "png"


def original_svg_dir(root: Path) -> Path:
    return root / "evaluation" / "figures" / "original" / "svg"
