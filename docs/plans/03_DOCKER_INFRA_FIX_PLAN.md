# PLAN 03 — DOCKER & INFRASTRUCTURE FIX

**Branch:** `feat/docker-infra-fixes`  
**Baseline:** `8c48d4d`  
**Owner:** Infrastructure Team (A3)  
**Mục tiêu:** Hoàn thiện môi trường triển khai ổn định, chạy được toàn bộ DSA Document Assistant bằng Docker Compose, không làm thay đổi thuật toán Retrieval hoặc Generation.

**Ưu tiên:** Local end-to-end hoạt động trước, sau đó Docker hóa; GPU là tùy chọn.

## 1. Phạm vi và File Ownership

**Được sửa:**
- `docker-compose.yml`, `docker-compose.gpu.yml` (NEW nếu cần).
- `backend/Dockerfile`, `frontend/Dockerfile`.
- `backend/requirements.txt`, `frontend/requirements.txt`, root `requirements.txt`.
- `.env.example`, `backend/.env.example`, `.dockerignore`.
- `frontend/.streamlit/config.toml`.
- `docs/setup.md`.
- `tests/infrastructure/**` (NEW).

**Không tự sửa:**
- `backend/app/retrieval/**`, `upload.py`, `documents.py`, `ingestion_flow.py` — Retrieval.
- `backend/app/generation/**`, `main.py`, `core/config.py`, `health.py`, `frontend/app.py` — Generation.
- `backend/app/document_processing/**` — Preprocessing.
- `evaluation/**` — Evaluation.

Nếu phát hiện lỗi trong Python source, gửi yêu cầu cho đúng owner thực hiện.

## 2. Implementation Tasks

| ID | Ưu tiên | Công việc | File chính |
|---|---|---|---|
| INFRA-01 | P0 | Sửa Docker Build và Healthchecks | Dockerfiles, Compose |
| INFRA-02 | P0 | Đồng bộ Qdrant Server, Config và Paths | `docker-compose.yml`, env |
| INFRA-03 | P0 | Thiết lập Persistent Storage | `docker-compose.yml` |
| INFRA-04 | P0 | Đồng bộ Environment và Dependencies | requirements, `.env.example`, Compose |
| INFRA-05 | P1 | CPU/GPU Runtime và Model Cache | Dockerfiles, GPU Compose |
| INFRA-06 | P0 | Bảo vệ Website Demo và API | Compose, runtime config |
| INFRA-07 | P0 | Docker Smoke & Integration Tests | `tests/infrastructure/**` (NEW) |
| INFRA-08 | P1 | Viết hướng dẫn Setup & Troubleshooting | `docs/setup.md` |

### INFRA-01 — Docker Build & Healthchecks

**Vấn đề:** Compose hiện sử dụng `curl` để healthcheck, nhưng `python:3.11-slim` không bảo đảm có sẵn `curl`. Backend `/health` cũng chưa kiểm tra Retrieval Engine có sẵn sàng hay không.

**Cần làm:**
- Đảm bảo Backend và Frontend images build thành công.
- Sử dụng healthcheck phù hợp với các công cụ thực sự tồn tại trong image, ví dụ Python `urllib` hoặc cài `curl` khi cần.
- Phân biệt **liveness** (server còn chạy) và **readiness** (Retrieval/Qdrant đã sẵn sàng).
- Cấu hình `depends_on` và startup order phù hợp.
- Chạy container với non-root user nếu tương thích quyền truy cập volume.
- Không dùng thời gian startup cố định như 10 hoặc 45 giây khi chưa đo.

**Đạt khi:** Các container khởi động ổn định; hệ thống không báo ready khi Retrieval chưa hoạt động.

### INFRA-02 — Qdrant Server & Configuration

**Vấn đề hiện tại:**
- `docker-compose.yml` sử dụng Qdrant Server nhưng Retrieval YAML đang đặt `qdrant_mode: local`.
- `main.py` có nguy cơ tìm Retrieval config sai đường dẫn trong container.
- Qdrant Server và Client chưa được kiểm thử tương thích theo phiên bản cụ thể.

**Cần làm:**
- Local development tiếp tục hỗ trợ Qdrant Local.
- Trong Docker, Backend phải sử dụng **Qdrant Server** qua `http://qdrant:6333`.
- Xác định rõ thứ tự ưu tiên cấu hình YAML và environment variables.
- Kiểm tra đường dẫn `retrieval.mvp.yaml` trong container.
- Pin và kiểm thử một cặp phiên bản Qdrant Server/Client tương thích, không mặc định nâng lên `v1.10.0` là đủ.
- Kiểm tra khi khởi động: Qdrant mode, collection, embedding model và BGE reranker thực sự được nạp đúng.

**Phối hợp bắt buộc:**
- Generation sửa `main.py` và cấu hình ứng dụng nếu cần.
- Retrieval sửa `retrieval/config.py` hoặc Retrieval YAML nếu cần.

**Đạt khi:** Docker Backend thật sự kết nối Qdrant Server và chạy Hybrid Retrieval + BGE, không âm thầm fallback về cấu hình khác.

### INFRA-03 — Persistent Storage

**Vấn đề:** Uploaded files, Retrieval metadata, model cache và Qdrant index phải được bảo toàn sau khi container restart.

**Cần làm:**
- Persist `qdrant_data` qua Docker named volume.
- Persist uploaded PDF/DOCX và active-version manifest của Track 1.
- Kiểm tra đường dẫn `UPLOAD_DIR` thực tế với đường dẫn volume mount.
- Persist Hugging Face model cache để tránh tải lại model khi recreate container.
- Không mount đè lên các thư mục chứa source code không cần thiết.
- Có hướng dẫn backup/recovery dữ liệu demo.
- Không chạy `docker compose down -v` trên dữ liệu thật khi chưa được phép.

**Phối hợp:** Retrieval xác nhận nơi lưu file, manifest và cơ chế khôi phục.

**Đạt khi:** Restart/recreate container vẫn giữ đúng documents, active versions và Qdrant index; model cache không phải tải lại khi volume còn nguyên.

### INFRA-04 — Environment & Dependencies

**Cần làm:**
- Đồng bộ các biến môi trường:
  - `QDRANT_MODE`, `QDRANT_URL`.
  - `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`.
  - `OPENAI_API_KEY`, `GEMINI_API_KEY` hoặc credential theo gateway thực tế.
  - Cấu hình embedding/reranker và CPU/GPU.
  - Đường dẫn runtime uploads, configs và model cache.
- Sử dụng placeholder trong `.env.example`; không commit API key thật.
- Kiểm tra việc load `.env` ở local và trong Docker.
- Kiểm tra file `.env` không tồn tại phải có thông báo thiết lập rõ ràng.
- Tổng hợp dependency requests từ Retrieval, Generation và Evaluation.
- Tránh đưa dependencies chỉ dùng cho Evaluation vào Backend production nếu không cần.
- Kiểm tra version conflicts bằng `pip check` và test imports.

**Đạt khi:** Môi trường được cấu hình nhất quán; Backend/Frontend hoạt động mà không cần sửa hardcoded paths hoặc credentials.

### INFRA-05 — CPU/GPU Runtime

**Cần làm:**
- Docker CPU mode phải hoạt động ngay cả khi máy không có NVIDIA GPU.
- Tạo `docker-compose.gpu.yml` nếu cần GPU acceleration.
- Kiểm tra NVIDIA Container Toolkit.
- Kiểm tra PyTorch trong image thật sự có CUDA support.
- Không mặc định GPU passthrough đồng nghĩa CUDA hoạt động.
- Giữ GPU optional; không biến RTX 5060 thành yêu cầu bắt buộc.
- Ghi nhận CPU/GPU startup time và model loading time thực tế.

**Đạt khi:** CPU mode chạy ổn định; GPU mode hoạt động trên máy có cấu hình phù hợp và được kiểm thử riêng.

### INFRA-06 — Controlled Web Demo

**Quyết định sản phẩm:** Website phục vụ nhóm và giảng viên test, sử dụng một workspace demo chung. Chưa triển khai đăng ký/đăng nhập nhiều người dùng.

**Cần làm:**
- Đề xuất cơ chế giới hạn truy cập website demo, ví dụ reverse proxy có authentication hoặc VPN.
- Không expose Qdrant API trực tiếp lên Internet.
- Backend management APIs không được truy cập tự do từ bên ngoài.
- Bảo vệ đồng nhất Upload, Delete, Chat, Quiz và Revision.
- Cấu hình CORS, networking và secrets phù hợp với môi trường demo.
- Nếu truy cập qua Internet, sử dụng HTTPS/TLS.
- Giới hạn tài nguyên/request hợp lý để tránh spam upload hoặc tiêu hao LLM quota.
- Không coi `workspace_id` do client gửi là cơ chế xác thực.

**Đạt khi:** Người được phép có thể test website; Backend và Qdrant không bị truy cập trái phép qua các cổng mở công khai.

### INFRA-07 — Docker Smoke & Integration Tests

**Cần làm:**
- Kiểm thử build và startup toàn bộ services.
- Kiểm tra Backend liveness/readiness.
- Kiểm tra Backend kết nối Qdrant Server.
- Kiểm tra Retrieval config và BGE thực sự được sử dụng.
- Kiểm tra persistence sau restart.
- Kiểm tra CPU mode.
- Kiểm tra GPU mode nếu có GPU phù hợp.
- Kiểm tra Frontend kết nối Backend.
- Sau khi Track 1 và 2 tích hợp, kiểm thử luồng Upload → Retrieval → QA/Quiz/Revision.
- Không sử dụng dữ liệu benchmark thật cho các thao tác phá hủy.

**Đạt khi:** Hệ thống chạy end-to-end trong Docker với dữ liệu kiểm thử riêng; không chỉ dừng ở việc ba container báo healthy.

### INFRA-08 — Setup Documentation

Viết `docs/setup.md` bao gồm:
- Cài đặt Docker và các điều kiện cần thiết.
- Cấu hình `.env`.
- Chạy local và Docker CPU.
- Bật Docker GPU khi phù hợp.
- Kiểm tra health/readiness.
- Vị trí lưu dữ liệu và model cache.
- Backup, restart và troubleshooting.
- Cách bảo vệ website demo khi chia sẻ cho giảng viên.

## 3. Integration Contracts

| Interface | Yêu cầu |
|---|---|
| Qdrant | Docker chạy server mode, Retrieval kết nối đúng URL |
| Retrieval YAML | Track 1 sở hữu nội dung, Track 2 xử lý loader, Docker cung cấp đúng mount/env |
| Document storage | Docker persist uploaded files và manifest theo đường dẫn Track 1 xác định |
| LLM Gateway | Track 2 định nghĩa biến cấu hình, Docker truyền đúng vào Backend |
| Dependencies | Docker là primary owner của requirements; các track gửi requests |
| Health/Readiness | Track 2 cung cấp Backend endpoints, Docker dùng chúng cho probes |
| QA/Quiz/Revision | Không thay đổi API contracts, chỉ bảo đảm môi trường triển khai hoạt động |
| Demo access | Một workspace chung, truy cập có kiểm soát; không expose Backend/Qdrant công khai |

**Các phối hợp bắt buộc:**

- **Retrieval:** Qdrant mode, storage paths, manifest persistence, CPU/GPU configuration.
- **Generation:** `main.py` config path, `.env`, LLM gateway, health/readiness và API routes.
- **Evaluation:** Môi trường tái lập và dependencies; không tác động kết quả benchmark.

**Lưu ý so với proposal:** Proposal có PostgreSQL, nhưng code MVP hiện tại dùng ba services. Không tự bổ sung PostgreSQL trong track này khi chưa có quyết định kiến trúc riêng từ Team Lead.

## 4. Testing & Acceptance Criteria

Tạo `tests/infrastructure/` để kiểm tra Docker, sử dụng dữ liệu test riêng.

| Test | Kết quả mong đợi |
|---|---|
| Docker build | Backend/Frontend build thành công |
| Docker startup | Services hoạt động, không dependency deadlock |
| Backend readiness | Chỉ ready khi Retrieval thực sự sẵn sàng |
| Qdrant connection | Backend kết nối `qdrant:6333` |
| Retrieval config | MiniLM + BGE + server mode được nạp đúng |
| Document persistence | Upload và active version tồn tại sau restart |
| Model cache | Model weights được giữ khi recreate container |
| CPU mode | Retrieval hoạt động không cần GPU |
| GPU mode | CUDA được nhận diện khi cấu hình phù hợp |
| LLM Gateway | Backend nhận đúng provider/model/base URL |
| Frontend connection | Streamlit gọi Backend thành công |
| Web demo security | Backend/Qdrant không bị truy cập trái phép |
| End-to-end | Upload → Retrieve → QA/Quiz/Revision hoạt động |
| Restart recovery | Dữ liệu không mất, không cần reindex ngoài ý muốn |

Các bài test phải ghi rõ đã chạy thật hay chỉ mock, cùng phiên bản image/dependencies sử dụng.

## 5. Definition of Done

- [ ] Docker Compose build và chạy ổn định.
- [ ] Backend hoạt động với Qdrant Server đúng cấu hình.
- [ ] Không còn lỗi Retrieval YAML path hoặc Upload storage path.
- [ ] Healthcheck phản ánh đúng trạng thái hệ thống.
- [ ] Uploaded files, manifest, Qdrant data và model cache được persist.
- [ ] CPU mode hoạt động; GPU mode được kiểm thử nếu có thiết bị hỗ trợ.
- [ ] LLM environment và gateway được cấu hình đúng.
- [ ] Requirements không có dependency conflicts.
- [ ] Website demo có cơ chế kiểm soát truy cập phù hợp.
- [ ] Backend/Qdrant không mở công khai ngoài ý muốn.
- [ ] Docker smoke/integration tests PASS.
- [ ] Không sửa source thuộc Retrieval, Generation hoặc Evaluation.
- [ ] `docs/setup.md` đủ để thành viên khác chạy lại hệ thống.

**Kết quả bàn giao:** Môi trường Docker Compose ổn định, bảo toàn dữ liệu và kết nối đầy đủ Frontend → Backend → Qdrant → LLM, sẵn sàng cho demo nhóm/giảng viên sau khi các track ứng dụng đã được tích hợp.