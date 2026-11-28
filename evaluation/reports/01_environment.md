# BÁO CÁO GIAI ĐOẠN 01 — CHUẨN BỊ MÔI TRƯỜNG VÀ ỔN ĐỊNH HỆ THỐNG RUNTIME

> **Dự án:** DSA Document Assistant Using Hybrid RAG  
> **Giai đoạn:** Stage 01 (Phase 01 + Round 02 Verification + Model Cache Normalization)  
> **Branch:** `feat/final-readiness` | **Checkpoints:** `8fe5458` (Phase 01), `ff2187e` (Model Cache Fix) | **Immutable `main`:** `b84f09450f213b03ad57ef8b388f482ee0f0eaa2`  
> **Trạng thái:** **ACCEPTED (`20/20` Acceptance Criteria PASS)**

---

## 1. Mục tiêu

Chuẩn bị và ổn định hoàn toàn môi trường thực thi cục bộ (Local Windows) và container hóa (Docker), xử lý các khiếm khuyết về runtime/cấu hình upload/cache mô hình đã phát hiện trong kiểm toán sẵn sàng (`PROJECT_AUDIT_FINAL_READINESS`), tạo nền tảng sạch, tái lập và cô lập trước khi triển khai các giai đoạn đánh giá định lượng (Stage 02–05).

## 2. Phương pháp thực hiện

- Kiểm toán toàn diện hạ tầng phần cứng/phần mềm, phiên bản Python, thư viện PyTorch CUDA, HuggingFace model cache và cấu hình FastAPI/Streamlit/Docker.
- Khắc phục tối thiểu các lỗi vận hành runtime (cấu hình giới hạn upload 15 MiB đồng bộ giữa Streamlit Client, Starlette transport và FastAPI Backend; thêm `.dockerignore`; chuẩn hóa khóa cache mô hình cục bộ để tránh tải lại/lệch đường dẫn khi chạy đánh giá).
- Kiểm chứng trực tiếp bằng trình duyệt thật (Microsoft Edge Chromium Headless qua CDP trên `127.0.0.1`) ở cả môi trường Local và Docker trên các mốc biên chính xác (`51.200 B`, `11 MiB`, `15.728.639 B`, `15.728.640 B`, `15.728.641 B`).

## 3. Dữ liệu đầu vào

- Mã nguồn baseline từ `main` (`b84f09450f213b03ad57ef8b388f482ee0f0eaa2`).
- Tập 5 fixture PDF tổng hợp kiểm thử biên upload (`51.200 B` đến `15.728.641 B`), mỗi tệp 1 trang hợp lệ xác minh bằng PyMuPDF.
- Hồ sơ kiểm toán ban đầu (`PROJECT_AUDIT_FINAL_READINESS`, `EVALUATION_MASTER_AUDIT_REPORT` — đã được lưu trữ bản sao rollback trong `data/runtime/consolidation_rollback_archive/`).

## 4. Nội dung triển khai

1. **Môi trường phần cứng & phần mềm chuẩn hóa:**
   - Hệ điều hành: Windows 10/11; Python `3.11.9` trong `.venv`.
   - GPU: **NVIDIA GeForce RTX 5060 Laptop GPU** (`7,96 GB VRAM`), PyTorch CUDA hỗ trợ trực tiếp `cuda:0`; RAM hệ thống `15,71 GB`.
   - Thư viện cốt lõi: FastAPI, Uvicorn, Streamlit `1.67.0`, Qdrant Client (`QdrantLocal` SQLite), Sentence-Transformers, FlagEmbedding/Transformers, PyMuPDF (`fitz`), `python-docx`, `pytest`, `matplotlib`, `pandas`, `scipy`.
2. **Đồng bộ giới hạn Upload 15 MiB (Round 01 & Round 02 — đóng `DEF-P1-05`):**
   - Phát hiện lỗi upstream của Streamlit `1.67.0` (`starlette_routes.py:869-873`) kiểm tra `content_length > max_size_bytes` mà không cộng overhead `multipart/form-data` (~230 B), gây HTTP `413` giả cho file hợp lệ `15.728.639 B` và `15.728.640 B`.
   - Bản sửa tối thiểu: đặt `maxUploadSize = 16` trong `frontend/.streamlit/config.toml` (dự phòng 1 MiB transport framing) kết hợp `max_upload_size=15` trên `st.file_uploader(...)` trong `frontend/screens/upload_ui.py` và `MAX_FILE_SIZE = 15 * 1024 * 1024` ở `backend/app/api/routes/upload.py`.
3. **Ổn định Docker & Model Caching (`ff2187e`):**
   - Tạo `backend/.dockerignore` và `frontend/.dockerignore` nhằm giảm ngữ cảnh build và loại trừ dữ liệu runtime/secrets.
   - Chuẩn hóa khóa cache mô hình embedding (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, `384d`, `max_seq_length=128`) và reranker (`BAAI/bge-reranker-v2-m3`, `max_length=512`) trong `backend/app/retrieval/`, đảm bảo nạp offline ổn định không phát sinh lệch khóa cache.
   - Xác minh `backend/.env` được `.gitignore` bảo vệ tuyệt đối (`git check-ignore = 0`).

## 5. Kết quả

| Hạng mục kiểm chứng | Local | Docker | Trạng thái |
|---|---|---|---|
| Backend API upload boundary (`50 KB`, `11 MiB`, `15 MiB − 1 B`, `15 MiB`, `15 MiB + 1 B`) | `5/5 PASS` (`200×4`, `413×1`) | `5/5 PASS` (`200×4`, `413×1`) | **PASS** |
| Actual Streamlit UI upload & tab Quản lý kho (Edge Headless CDP) | `5/5 PASS` | `5/5 PASS` | **PASS** |
| GPU CUDA readiness (`cuda:0`, RTX 5060) & Model cache offline | Verified (`0` OOM) | Verified | **PASS** |
| Secrets safety (`backend/.env` gitignored) | `git check-ignore = True` | Verified | **PASS** |

## 6. Kiểm thử và xác minh

- Bổ sung bộ kiểm thử tích hợp `tests/integration/test_phase1_runtime_config.py`.
- Kết quả hồi quy tại checkpoint Stage 01: `276/276 passed` (`0 failed, 0 skipped`), `pip check` trả về `No broken requirements found.` (`exit 0`).
- *(Lưu ý cập nhật về sau ở các giai đoạn kiểm thử Windows: `tests/unit/test_llm_reliability.py` phát sinh 4 lỗi timing `time.monotonic()` đặc thù Windows có sẵn từ nền tảng, được bảo toàn nguyên trạng ngoài phạm vi Evaluation — không tuyên bố toàn bộ full suite pass.)*

## 7. Acceptance Criteria

- **Phase 01 / Stage 01:** **`20 / 20` Acceptance Criteria PASS** (bao gồm `AC-13` nâng lên `PASS` sau kiểm chứng Round 02).

## 8. Vấn đề và giới hạn

- Các lỗi nghiệp vụ phát hiện ở các tầng chat/quiz/roadmap (chỉ lộ diện khi chạy Live Groq & HTTP End-to-End thật ở Stage 04 AC-04) nằm ngoài phạm vi ổn định môi trường Stage 01 và được giữ nguyên trạng để phản ánh trung thực kết quả đánh giá.
- `4` lỗi timing trên Windows trong `tests/unit/test_llm_reliability.py` tồn tại độc lập với các thay đổi Evaluation.

## 9. Kết luận

Môi trường Python `.venv`, GPU CUDA RTX 5060, Docker, cấu hình upload 15 MiB và bộ nhớ đệm mô hình đã được ổn định và kiểm chứng đầy đủ, sẵn sàng làm nền tảng bất biến cho Stage 02–05.

## 10. Checkpoint và tài liệu liên quan

- **Checkpoints Git:** `8fe5458` (`feat: complete phase 01 environment and runtime stabilization`), `ff2187e` (`fix(retrieval): normalize model cache keys for evaluation stability`).
- **Tài liệu liên quan:** [Evaluation README](../README.md) | [Báo cáo Stage 02](02_chunking_indexing.md) | [Relocation Manifest](../manifests/relocation_manifest.json).
