# PHASE 01 – IMPLEMENTATION REPORT

> **Dự án:** DSA Document Assistant Using Hybrid RAG
> **Phase:** Phase 01 – Environment & Runtime Stabilization (Implementation Round 01)
> **Phạm vi nhiệm vụ:** TASK-01 (Môi trường chuẩn & Baseline Testing) & TASK-04 (Runtime, Docker, Qdrant & Upload Stabilization)
> **Repository:** `D:\DSA-Document-Assistant-Using-Hybrid-RAG`
> **Branch triển khai:** `feat/final-readiness`
> **Ngày thực hiện:** 2026-11-26

---

## 1. Executive Summary

Phase 01 (Round 01) đã hoàn tất toàn bộ mục tiêu **TASK-01** và **TASK-04** trên duy nhất branch `feat/final-readiness`, bảo toàn nguyên vẹn branch `main` tại audited commit `b84f09450f213b03ad57ef8b388f482ee0f0eaa2` và file kiểm toán `PROJECT_AUDIT_FINAL_READINESS.md`.

**Các kết quả chính đã hoàn thành và kiểm chứng thực tế:**
- Khởi tạo môi trường `.venv` dùng **Python 3.11.9**, cài đặt `torch==2.11.0+cu128` tương thích hoàn toàn với **NVIDIA GeForce RTX 5060 Laptop GPU** (`sm_120`, CUDA runtime `12.8`, cả tensor op GPU và CPU fallback đều PASS), cùng toàn bộ dependencies của Backend, Frontend, Evaluation và pytest (`pip check`: `No broken requirements found.`).
- Thiết lập baseline kiểm thử tự động: **271/271 test cases passed** (0 failed, 0 skipped, 0 errors, exit code `0`, 15.32s).
- Khắc phục **BLK-04 (Qdrant Docker Compatibility)**: nâng cấp và pin `docker-compose.yml` từ `qdrant/qdrant:v1.9.2` lên `qdrant/qdrant:v1.20.2` (khớp minor với `qdrant-client==1.20.0`), kiểm chứng thực tế `/readyz` (`200`) và thao tác `client.query_points()` thành công trên Qdrant Server thật.
- Khắc phục **bất nhất đường dẫn runtime/upload**: hợp nhất về một nguồn cấu hình duy nhất `settings.upload_dir` (`Path(self.SESSION_RUNTIME_DIR) / "uploads"`) kết hợp `upload.resolve_upload_dir()` dùng chung cho `upload.py` và `documents.py`, đảm bảo nhất quán ở cả local (`data/runtime/uploads`) và Docker (`/app/data/runtime/uploads`), giữ nguyên tương thích ngược với monkeypatch của test hiện có và không làm suy yếu chống path traversal / session isolation.
- Khắc phục **BLK-06 (Upload Limit 15 MB)**: cập nhật `frontend/.streamlit/config.toml` từ `maxUploadSize = 10` lên `maxUploadSize = 15`, đồng bộ lệnh khởi chạy Streamlit từ thư mục `frontend/` trong `README.md`, xác minh Streamlit nạp đúng `server.maxUploadSize = 15` ở cả local và Docker, và kiểm chứng thực tế file 11 MiB upload thành công (`200`), file 16 MiB bị từ chối (`413`), file `.txt` bị từ chối (`400`).
- Hoàn thiện cấu hình Docker build với `backend/.dockerignore` và `frontend/.dockerignore` (xác minh `/app/.env` không lọt vào image), build và smoke test thành công toàn bộ cụm Docker Compose cách ly (không ảnh hưởng container/port/volume đang chạy của người dùng), bao gồm persistence dữ liệu qua restart Backend.
- Bổ sung 5 regression tests cho Phase 1 tại `tests/integration/test_phase1_runtime_config.py`; toàn bộ suite sau implementation đạt **276/276 passed** (0 regression).

**Giới hạn / vấn đề ngoài phạm vi ghi nhận trung thực:**
- Live Groq API generation chưa chạy vì `GROQ_API_KEY` để trống theo quy định Phase 1 (`UNVERIFIED`, chuyển sang Phase 3).
- Interactive click-through upload trên trình duyệt Chrome bị chặn do Chrome trên host chưa bật remote debugging (`DevToolsActivePort ENOENT`); đã kiểm chứng thay thế bằng Streamlit server config + HTTP health + `AppTest` live kết nối Backend + Backend API live.
- Phát hiện trong local smoke test: `POST /api/chat` trả `500` (`Study retrieval failed`) do `backend/app/retrieval/reranker.py` cache `_get_model` theo key `device="auto"` lúc warmup khác với `device="cuda"` lúc `rerank()` dẫn tới nạp lại lần 2 model `BAAI/bge-reranker-v2-m3` trên 8 GB VRAM — được ghi nhận vào **Deferred Issues** (không sửa thuật toán/retrieval trong Phase 1).

---

## 2. Git and Workspace Verification

| Hạng mục | Giá trị thực tế |
|---|---|
| **Repository path** | `D:/DSA-Document-Assistant-Using-Hybrid-RAG` (xác minh không dùng đường dẫn cũ `d:\MÔN HỌC\...`) |
| **Initial branch** | `main` (`## main...origin/main`) |
| **Final branch** | `feat/final-readiness` |
| **Initial HEAD** | `b84f09450f213b03ad57ef8b388f482ee0f0eaa2` |
| **Final HEAD** | `b84f09450f213b03ad57ef8b388f482ee0f0eaa2` (chưa commit theo quy định) |
| **Initial main commit** | `b84f09450f213b03ad57ef8b388f482ee0f0eaa2` |
| **Final main commit** | `b84f09450f213b03ad57ef8b388f482ee0f0eaa2` |
| **Main integrity** | **INTACT** (`refs/heads/main` không thay đổi, 0 commit/merge vào `main`) |
| **Working tree status** | 6 tracked files modified, 3 new Phase 1 untracked files, 1 preserved audit file (`PROJECT_AUDIT_FINAL_READINESS.md`), 0 staged files, `git diff --check` exit `0` |

Lệnh kiểm chứng đã chạy:
```powershell
git branch --show-current          # feat/final-readiness
git rev-parse HEAD                 # b84f09450f213b03ad57ef8b388f482ee0f0eaa2
git rev-parse refs/heads/main      # b84f09450f213b03ad57ef8b388f482ee0f0eaa2
git diff --check                   # exit code 0
```

---

## 3. Environment Setup

- **Python version:** `Python 3.11.9` (`C:\Users\OS\AppData\Local\Programs\Python\Python311\python.exe`)
- **Virtual environment:** `D:\DSA-Document-Assistant-Using-Hybrid-RAG\.venv` (tạo bằng `py -3.11 -m venv .venv`, nâng `pip` lên `26.3`, xác nhận `.venv/` được `.gitignore` bỏ qua)
- **Dung lượng ổ D:** `164.21 GB` trống
- **Lệnh cài đặt đã thực thi:**
  ```powershell
  .\.venv\Scripts\python.exe -m pip install --upgrade pip
  .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt -r evaluation/requirements.txt pytest
  ```
- **Pip check result:** `.\.venv\Scripts\python.exe -m pip check` → `No broken requirements found.` (exit code `0`)
- **Kiểm tra import & phiên bản 16 package trọng yếu (tất cả `OK`):**

| Package / Module | Phiên bản thực tế trong `.venv` | Trạng thái import |
|---|---|---|
| `fastapi` | `0.165.0` | OK |
| `uvicorn` | `0.54.0` | OK |
| `pydantic` | `2.14.4` | OK |
| `qdrant_client` (`qdrant-client`) | `1.20.0` | OK |
| `sentence_transformers` (`sentence-transformers`) | `5.7.0` | OK |
| `torch` | `2.11.0+cu128` | OK |
| `numpy` | `2.4.6` | OK |
| `fitz` (`PyMuPDF`) | `2.2` *(kèm `FutureWarning` deprecated API `fitz`)* | OK |
| `docx` (`python-docx`) | `1.2.0` | OK |
| `rank_bm25` | `0.2.2` | OK |
| `pyvi` | `0.1.1` | OK |
| `openai` | `3.60.0` | OK |
| `streamlit` | `1.67.0` | OK |
| `matplotlib` | `3.11.2` | OK |
| `openpyxl` | `3.1.5` | OK |
| `pytest` | `9.2.0` | OK |

Không cần chỉnh sửa `backend/requirements.txt`, `frontend/requirements.txt` hay `evaluation/requirements.txt` vì toàn bộ ràng buộc phiên bản giải quyết sạch không xung đột.

---

## 4. Hardware and PyTorch Verification

| Thuộc tính | Kết quả kiểm chứng thực tế |
|---|---|
| **GPU** | `NVIDIA GeForce RTX 5060 Laptop GPU` (8,151 MiB VRAM, kiến trúc Blackwell `sm_120`) |
| **NVIDIA Driver** | `610.74` |
| **PyTorch version (`.venv` host)** | `2.11.0+cu128` (nguồn chính thức `https://download.pytorch.org/whl/cu128`) |
| **PyTorch version (Docker image)** | `2.12.1+cpu` (`CUDA: None`, từ `https://download.pytorch.org/whl/cpu` theo `backend/Dockerfile`) |
| **CUDA runtime version (`.venv`)** | `12.8` |
| **CUDA availability** | `True` (`torch.cuda.is_available() == True`) |
| **Supported CUDA archs** | `['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']` |
| **GPU tensor operation** | `True` (`(torch.randn(16,16,device='cuda') @ torch.randn(16,16,device='cuda')).sum().item()`) |
| **CPU fallback operation** | `True` (`(torch.randn(16,16,device='cpu') @ torch.randn(16,16,device='cpu')).sum().item()`) |
| **Real model warmup on GPU** | `paraphrase-multilingual-MiniLM-L12-v2` và `BAAI/bge-reranker-v2-m3` đều nạp thành công lên `cuda:0` khi chạy Backend local |
| **Limitations** | Docker container giữ kiến trúc CPU-only (`torch==2.12.1+cpu`); trên GPU 8 GB VRAM local cần xử lý cache key mismatch của `reranker.py` ở phase sau để tránh nạp trùng model thứ 2 vào VRAM |

---

## 5. Baseline Test Results

Chạy trước toàn bộ chỉnh sửa mã nguồn/cấu hình (STEP 5):

- **Commands:**
  ```powershell
  .\.venv\Scripts\python.exe -m pytest --collect-only -q
  .\.venv\Scripts\python.exe -m pytest -q
  ```
- **Collected:** `271` test cases (từ 164 hàm test tĩnh + parametrization)
- **Passed:** `271`
- **Failed:** `0`
- **Skipped:** `0`
- **Errors:** `0`
- **Xfailed / Xpassed:** `0 / 0`
- **Warnings:** `1` (`FutureWarning` từ `fitz/__init__.py:17`: API `fitz` deprecated, khuyến nghị `import pymupdf`)
- **Exit code:** `0`
- **Execution time:** `15.32s` (wall `16.55s`)
- **Failure classification:** Không có lỗi baseline (0 lỗi thuộc nhóm A–G).

---

## 6. Implementation Details

### Thay đổi 1 — Đồng bộ phiên bản Qdrant Server trong Docker Compose (BLK-04)
- **Task ID:** TASK-04 (STEP 6)
- **Vấn đề:** `docker-compose.yml` pin `qdrant/qdrant:v1.9.2`, trong khi `backend/app/retrieval/vector_search.py:18` gọi `client.query_points()` (endpoint `/points/query` chỉ có từ Qdrant Server `v1.10+`).
- **Root cause:** Phiên bản Qdrant Server image tụt hậu so với `qdrant-client>=1.10,<2` (thực tế cài `1.20.0`).
- **File path:** `docker-compose.yml` (service `qdrant`, dòng 22)
- **Before:** `image: qdrant/qdrant:v1.9.2`
- **After:** `image: qdrant/qdrant:v1.20.2`
- **Lý do lựa chọn:** Khớp chính xác minor version `1.20` với `qdrant-client==1.20.0` theo khuyến nghị tương thích `±1 minor` của Qdrant, dùng semver tag cố định (không dùng `latest`), đã kiểm tra tồn tại trên Docker Hub (`docker manifest inspect qdrant/qdrant:v1.20.2` exit `0`).
- **Evidence:** `QDRANT_READYZ: 200 all shards are ready VERSION: 1.20.2`; `client.query_points()` trả `[(1, 1.0, 'c1'), (2, 0.0, 'c2')]`.
- **Risk:** Thấp (không có dữ liệu Qdrant cũ của dự án cần migrate; không xóa volume).
- **Verification:** Docker Qdrant smoke test + `test_docker_compose_qdrant_tag_and_runtime_dir` PASS.

### Thay đổi 2 — Thống nhất Runtime Directory và Upload Path
- **Task ID:** TASK-04 (STEP 7)
- **Vấn đề:** `upload.py:21` và `documents.py:25` tính `UPLOAD_DIR = Path(__file__).resolve().parents[4] / "data" / "runtime" / "uploads"`. Trong Docker (`WORKDIR /app`), `parents[4]` trỏ ra `/` (`/data/runtime/uploads`), nằm ngoài volume mount `/app/data/runtime`.
- **Root cause:** Hardcode chiều sâu thư mục tương đối `parents[4]` thay vì đọc `settings.SESSION_RUNTIME_DIR`.
- **File paths:**
  - `backend/app/core/config.py` (`Settings.upload_dir`)
  - `backend/app/api/routes/upload.py` (`_DEFAULT_UPLOAD_DIR`, `UPLOAD_DIR`, `resolve_upload_dir()`, `_save_and_index()`)
  - `backend/app/api/routes/documents.py` (`UPLOAD_DIR`, `delete_document()`)
- **Before:**
  - `upload.py:21` & `documents.py:25`: `UPLOAD_DIR = Path(__file__).resolve().parents[4] / "data" / "runtime" / "uploads"`
  - `upload.py:100`: `root = registry.directory(workspace_id) / "uploads" if registry is not None else UPLOAD_DIR`
  - `documents.py:133-134`: `root = UPLOAD_DIR`, `runtime_root = settings.SESSION_RUNTIME_DIR`
- **After:**
  - `config.py`: thêm property `upload_dir` trả `Path(self.SESSION_RUNTIME_DIR) / "uploads"` (đảm bảo kiểu `Path` kể cả khi bị override bằng `str`).
  - `upload.py`: `_DEFAULT_UPLOAD_DIR = settings.upload_dir`, `UPLOAD_DIR = _DEFAULT_UPLOAD_DIR`, định nghĩa hàm nguồn duy nhất `resolve_upload_dir(override: Path | None = None) -> Path` và gọi trong `_save_and_index()`.
  - `documents.py`: tái sử dụng `resolve_upload_dir(UPLOAD_DIR)` từ `upload.py` và ép kiểu `Path(settings.SESSION_RUNTIME_DIR)` cho `runtime_root`.
- **Lý do lựa chọn:** Một nguồn cấu hình duy nhất từ `SESSION_RUNTIME_DIR`, tránh nhân đôi logic giữa `upload.py` và `documents.py`, tương thích ngược 100% với các test monkeypatch `upload.UPLOAD_DIR` / `documents.UPLOAD_DIR`, giữ nguyên cơ chế `is_relative_to()` chống path traversal và kiểm tra symlink trong `SessionRegistry.directory()`.
- **Evidence:** Trong Docker container `dsa_phase1_smoke_backend`, `settings.SESSION_RUNTIME_DIR`, `settings.upload_dir`, `upload.resolve_upload_dir()`, `documents.UPLOAD_DIR` lần lượt in: `/app/data/runtime | /app/data/runtime/uploads | /app/data/runtime/uploads | /app/data/runtime/uploads`.
- **Risk:** Thấp.
- **Verification:** `test_upload_dir_follows_session_runtime_dir_for_local_and_docker`, `test_upload_and_documents_share_single_runtime_root`, `test_document_lifecycle.py` và Docker smoke test PASS.

### Thay đổi 3 — Đồng bộ giới hạn Upload 15 MB (BLK-06) và lệnh khởi chạy Streamlit
- **Task ID:** TASK-04 (STEP 8)
- **Vấn đề:** `frontend/.streamlit/config.toml:9` đặt `maxUploadSize = 10` MB trong khi Backend (`upload.py:22`), Upload UI (`upload_ui.py:14`) và `README.md` dùng giới hạn 15 MB. Ngoài ra `README.md:26` hướng dẫn chạy `streamlit run frontend/app.py` từ repo root khiến Streamlit không tự nạp `frontend/.streamlit/config.toml`.
- **Root cause:** Lệch cấu hình giữa tầng Streamlit server và tầng ứng dụng; Streamlit chỉ đọc `.streamlit/config.toml` theo current working directory.
- **File paths:** `frontend/.streamlit/config.toml` (dòng 9), `README.md` (dòng 26)
- **Before:** `maxUploadSize = 10`; `README.md`: `.\.venv\Scripts\python.exe -m streamlit run frontend/app.py`
- **After:** `maxUploadSize = 15`; `README.md`: `Set-Location frontend` + `..\.venv\Scripts\python.exe -m streamlit run app.py`
- **Lý do lựa chọn:** Thay đổi tối thiểu, đồng bộ đúng 15 MB và đảm bảo file config thực sự được Streamlit nạp ở cả local lẫn Docker (`WORKDIR /app` chứa `.streamlit/config.toml`).
- **Evidence:** `streamlit config show` tại `frontend/` và `st_config.get_option("server.maxUploadSize")` trong cả local (`15`) và container `dsa_phase1_smoke_frontend` (`15`).
- **Risk:** Thấp.
- **Verification:** `test_upload_limit_is_synchronized_to_15_mb`, `test_upload_size_boundaries_accept_11_mib_and_reject_over_15_mib`, local & Docker upload tests PASS.

### Thay đổi 4 — Bổ sung `.dockerignore` cho cả hai Docker build context
- **Task ID:** TASK-04 (STEP 9)
- **Vấn đề:** Thiếu `backend/.dockerignore` và `frontend/.dockerignore`, làm tăng rủi ro đưa file nhạy cảm (`backend/.env`), `.venv/`, `__pycache__/`, `.pytest_cache/` vào Docker image khi `COPY . .`.
- **Root cause:** Chưa tạo file `.dockerignore` cho hai build context `./backend` và `./frontend`.
- **File paths:** `backend/.dockerignore` (mới), `frontend/.dockerignore` (mới)
- **Before:** Không tồn tại.
- **After:** Loại trừ `.env`, `.env.*` (giữ `!.env.example`), `.venv/`, `venv/`, `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, cache/log/OS files.
- **Evidence:** Trong container `dsa_phase1_smoke_backend` đã build với `backend/.env` tồn tại trên host: `Path('/app/.env').exists()` trả `False`.
- **Risk:** Rất thấp.
- **Verification:** Docker build thành công + secret check PASS + `test_docker_compose_qdrant_tag_and_runtime_dir` PASS.

### Thay đổi 5 — Bổ sung Regression Tests cho Phase 1
- **Task ID:** TASK-01 & TASK-04 (STEP 7, 8, 13)
- **File path:** `tests/integration/test_phase1_runtime_config.py` (mới, 5 test cases)
- **Nội dung:** Kiểm chứng `settings.upload_dir` & `resolve_upload_dir()` cho local/Docker, dùng chung runtime root giữa `upload.py` & `documents.py`, đồng bộ giới hạn 15 MB, kiểm thử biên kích thước upload (`<10 MB`, `11 MiB`, `>15 MiB` → 413, `.txt` → 400), và kiểm chứng `docker-compose.yml` + `.dockerignore`.
- **Verification:** `5 passed` (exit code `0`).

### Cấu hình local `.env` (gitignored, không nằm trong Git diff)
- Sao chép `backend/.env.example` → `backend/.env` (giữ trống `GROQ_API_KEY=`), xác minh bằng `git check-ignore -v backend/.env` → `.gitignore:13:.env backend/.env`.

---

## 7. Qdrant Compatibility Verification

- **Previous server version:** `qdrant/qdrant:v1.9.2` (không hỗ trợ `/points/query`)
- **New server version:** `qdrant/qdrant:v1.20.2`
- **Client version:** `qdrant-client==1.20.0` (trong cả `.venv` local và Docker image `dsa-phase1-smoke-backend`)
- **Compatibility evidence:**
  - `GET http://127.0.0.1:16333/readyz` → HTTP `200` (`all shards are ready`)
  - `GET http://127.0.0.1:16333/` → `version: "1.20.2"`
  - Khớp cùng minor version `1.20` (trong ngưỡng `±1 minor` theo tài liệu Qdrant)
- **Real `query_points()` test:** Tạo collection tạm `phase1_smoke_qdrant_compat` với named vector `dense` (size=4, Cosine), upsert 2 point (`c1`: `[1,0,0,0]`, `c2`: `[0,1,0,0]`), gọi `client.query_points(collection_name="phase1_smoke_qdrant_compat", query=[1,0,0,0], using="dense", limit=2, with_payload=True)` → trả về `[(1, 1.0, 'c1'), (2, 0.0, 'c2')]`, sau đó xóa sạch duy nhất collection tạm `phase1_smoke_qdrant_compat`.
- **Data preservation:** Không tồn tại container/volume `dsa_*` trước Phase 1; thử nghiệm chạy trong project cách ly `-p dsa-phase1-smoke` với volume riêng, không chạy `down -v` hay `prune`, không chạm vào bất kỳ container/volume nào của người dùng.

---

## 8. Runtime Path Verification

- **Local runtime paths:**
  - `settings.SESSION_RUNTIME_DIR` mặc định: `D:\DSA-Document-Assistant-Using-Hybrid-RAG\data\runtime`
  - `settings.upload_dir` / `upload.resolve_upload_dir()`: `D:\DSA-Document-Assistant-Using-Hybrid-RAG\data\runtime\uploads`
  - Session runtime upload: `<SESSION_RUNTIME_DIR>/sessions/<workspace_id>/uploads/<sha256_identity>/<revision>.<ext>` (đã xác minh `SESSION_UPLOAD_DIR_EXISTS: True`, `FILE_COUNT: 2` khi upload 2 PDF)
- **Docker runtime paths:**
  - `SESSION_RUNTIME_DIR`: `/app/data/runtime` (mount tới host volume)
  - `settings.upload_dir` / `upload.resolve_upload_dir()` / `documents.UPLOAD_DIR`: `/app/data/runtime/uploads`
  - Xác minh thực tế file upload trong Docker xuất hiện trực tiếp trên thư mục host được bind mount (`HOST_VOLUME_UPLOAD_FILES: 2`).
- **Volume persistence:** Sau khi thực thi `docker restart dsa_phase1_smoke_backend`, `/ready` trở lại `200` và `GET /api/documents` với cùng Bearer token vẫn trả về `200`, `total = 2` (`['valid_11mib.pdf', 'valid_small.pdf']`), xác nhận cả SQLite session registry và Qdrant server volume đều bền vững qua restart.
- **Session isolation:** Giữ nguyên cơ chế Bearer token + SHA-256 hash trong SQLite, `SessionRegistry.directory()` kiểm tra regex `session-[0-9a-f]{32}` và chống symlink, `physical_upload()` giữ nguyên kiểm tra `path.resolve().is_relative_to(root.resolve())`. Toàn bộ `test_study_sessions.py`, `test_document_lifecycle.py`, `test_p0_production_config.py` đều PASS.

---

## 9. Upload Limit Verification

**Quy ước đơn vị đã xác minh:**
- Backend (`upload.py:22`) và Upload UI (`upload_ui.py:14`): `15 * 1024 * 1024` bytes = **15 MiB** (`15,728,640` bytes).
- Streamlit (`config.toml:9`): `maxUploadSize = 15` (Streamlit cấu hình theo đơn vị MB = `15 * 10^6` bytes hoặc `15 MiB` tùy tầng; trong mọi trường hợp file `11 MiB = 11,534,336` bytes nằm gọn giữa 10 MB và 15 MB, còn `16 MiB = 16,777,216` bytes vượt ngưỡng 15 MiB).

| Test case | File size | Environment | Expected | Actual | Status |
|---|---|---|---|---|---|
| Valid PDF `< 10 MB` (`valid_small.pdf`) | 51,200 B (~50 KB) | Local Backend API (`:18001`) | HTTP 200, indexed 1 chunk | HTTP 200, `indexed_chunks=1` | PASS |
| Valid PDF `> 10 MB & <= 15 MiB` (`valid_11mib.pdf`) | 11,534,336 B (11 MiB) | Local Backend API (`:18001`) | HTTP 200, indexed 1 chunk | HTTP 200, `indexed_chunks=1` | PASS |
| Oversize PDF `> 15 MiB` (`oversize_16mib.pdf`) | 16,777,216 B (16 MiB) | Local Backend API (`:18001`) | HTTP 413 rejected | HTTP 413 (`The maximum file size is 15 MB.`) | PASS |
| Unsupported format (`invalid.txt`) | 17 B | Local Backend API (`:18001`) | HTTP 400 rejected | HTTP 400 (`Only PDF and DOCX files are supported.`) | PASS |
| Valid PDF `< 10 MB` (`valid_small.pdf`) | 51,200 B (~50 KB) | Docker Backend API (`:18002`) | HTTP 200, `status=success` | HTTP 200, `status=success` | PASS |
| Valid PDF `> 10 MB & <= 15 MiB` (`valid_11mib.pdf`) | 11,534,336 B (11 MiB) | Docker Backend API (`:18002`) | HTTP 200, `status=success` | HTTP 200, `status=success` | PASS |
| Oversize PDF `> 15 MiB` (`oversize_16mib.pdf`) | 16,777,216 B (16 MiB) | Docker Backend API (`:18002`) | HTTP 413 rejected | HTTP 413 (`The maximum file size is 15 MB.`) | PASS |
| Unsupported format (`invalid.txt`) | 17 B | Docker Backend API (`:18002`) | HTTP 400 rejected | HTTP 400 (`Only PDF and DOCX files are supported.`) | PASS |
| Streamlit `server.maxUploadSize` config load | N/A | Local (`frontend/`) & Docker (`dsa_phase1_smoke_frontend`) | `15` | `15` (cả local và Docker) | PASS |
| Interactive Chrome UI file-uploader click-through | 11 MiB / 16 MiB | Local Browser (`:18501`) | Accept 11 MiB, reject 16 MiB | Không kết nối được Chrome DevTools (`DevToolsActivePort ENOENT`) | BLOCKED (Host Chrome remote debugging chưa bật) |

---

## 10. Local Runtime Smoke Test

*(Sử dụng port cách ly `18001`/`18501` và `SESSION_RUNTIME_DIR` / `QDRANT_PATH` tạm trong thư mục scratch do host port `8000` đang được container dự án khác của người dùng sử dụng.)*

- **Lệnh khởi động Backend:**
  ```powershell
  .\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 18001 --workers 1 --log-level info
  ```
- **Lệnh khởi động Frontend:**
  ```powershell
  Set-Location 'D:\DSA-Document-Assistant-Using-Hybrid-RAG\frontend'
  $env:BACKEND_URL = 'http://127.0.0.1:18001'
  ..\.venv\Scripts\python.exe -m streamlit run app.py --server.port 18501 --server.address 127.0.0.1 --server.headless true
  ```
- **Startup duration:** `222.52s` (lần đầu tải weight `paraphrase-multilingual-MiniLM-L12-v2` và `BAAI/bge-reranker-v2-m3` từ HuggingFace Hub rồi nạp thành công lên `cuda:0`).
- **Kết quả kiểm chứng:**
  - `GET /health` → `200` (`status: "healthy"`, `environment: "development"`, `study_request_timeouts: {chat: 90.0, quiz: 240.0, revision: 240.0}`)
  - `GET /ready` → `200` (`status: "ready"`, `retrieval: "ready"`, `qdrant: "ready"`, `reranker_configured: true`)
  - `POST /api/sessions` → `201` (`corpus_id: "study-sessions"`, `token_len: 43`, không in token)
  - `GET /api/documents` với token sai → `401`; với Bearer token hợp lệ → `200` (`total: 0`)
  - Upload validation (4 case trong bảng Mục 9) → PASS toàn bộ
  - `DELETE /api/sessions/current` → `200` (`{"status":"deleted"}`)
  - Frontend `GET /_stcore/health` → `200 ok`; `GET /` → `200`
  - Frontend → Backend live handshake qua `streamlit.testing.v1.AppTest` (`app.py` trỏ `http://127.0.0.1:18001`) → 0 exception, tự tạo session `active_study_session` thành công và xóa session sạch (`200`).
  - `POST /api/chat` (khi thiếu `GROQ_API_KEY`) → trả `500` (`ERROR:app.orchestration.request_flow:Study retrieval failed` trước khi tới tầng LLM, do lỗi cache key reranker được ghi vào Mục 13).
- **Cleanup:** Đã dừng cả hai process local test sau khi hoàn tất.

---

## 11. Docker Build and Runtime Smoke Test

*(Sử dụng project cách ly `-p dsa-phase1-smoke` với file override ngoài repo để dùng container name riêng `dsa_phase1_smoke_*`, port riêng `16333`/`18002`/`18502`, và runtime mount riêng — bảo toàn 100% container/volume hiện có của người dùng.)*

- **Config validation:** `docker compose config -q` → exit code `0` (1 cảnh báo `version` obsolete không ảnh hưởng).
- **Build & Up:** `docker compose -p dsa-phase1-smoke -f docker-compose.yml -f <override> up -d --build` → exit code `0`.
  - `backend/Dockerfile`: cài `torch==2.12.1+cpu` (`CUDA: None`) và `qdrant-client-1.20.0` thành công.
  - `frontend/Dockerfile`: build thành công.
  - `.dockerignore`: xác minh `Path('/app/.env').exists() == False` trong container backend.
- **Container status khi kiểm thử:**
  - `dsa_phase1_smoke_qdrant` (`qdrant/qdrant:v1.20.2`): `Up (healthy)`
  - `dsa_phase1_smoke_backend` (`dsa-phase1-smoke-backend`): `Up (healthy)`
  - `dsa_phase1_smoke_frontend` (`dsa-phase1-smoke-frontend`): `Up` (đã xác minh HTTP `/_stcore/health` `200 ok` và `/` `200`)
- **Runtime checks:**
  - Qdrant `/readyz` `200` + `query_points()` thật PASS.
  - Backend `/health` `200` (`environment: "production"`), `/ready` `200`.
  - Upload 50 KB (`200`), 11 MiB (`200`), 16 MiB (`413`), `.txt` (`400`), lưu đúng vào host bind mount (`HOST_VOLUME_UPLOAD_FILES: 2`).
  - Restart `dsa_phase1_smoke_backend` → `/ready` `200`, `GET /api/documents` vẫn còn nguyên 2 file → xóa session `200`.
  - Frontend container nạp `server.maxUploadSize = 15`, gọi `api_client.check_health()`, `create_session()`, `delete_session()` qua Docker network `http://backend:8000` đều trả `True`.
- **Cleanup:** Dừng duy nhất 3 container `dsa_phase1_smoke_*` bằng `docker compose -p dsa-phase1-smoke ... stop` (không xóa volume); xác minh 4 container `escrow_*` của người dùng vẫn `Up (healthy)`.

---

## 12. Regression Test Results

Command: `.\.venv\Scripts\python.exe -m pytest -q`

| Metric | Baseline | After Implementation | Difference |
|---|---|---|---|
| Collected | 271 | 276 | +5 (new Phase 1 tests) |
| Passed | 271 | 276 | +5 |
| Failed | 0 | 0 | 0 |
| Skipped | 0 | 0 | 0 |
| Errors | 0 | 0 | 0 |
| Warnings | 1 | 1 | 0 |
| Exit code | 0 | 0 | 0 |
| Duration | 15.32s | 17.34s | +2.02s |

**Kết luận:** Không phát sinh bất kỳ regression nào (0 new failures).

---

## 13. Deferred Issues

| Issue ID | Description | Evidence | Assigned Phase | Impact |
|---|---|---|---|---|
| DEF-P1-01 | `reranker.py:118` gọi `_get_model(name, resolved_device="cuda", max_length=512)` trong khi `main.py:81` warmup bằng `device="auto"`, gây LRU cache miss và nạp trùng bản `BAAI/bge-reranker-v2-m3` thứ 2 lên GPU 8 GB VRAM, dẫn tới `POST /api/chat` lỗi 500 `Study retrieval failed` khi chạy local GPU. | Local smoke test (`task-101.log:92,119-130`) | Phase 2 / Phase 3 (Retrieval/Runtime Optimization) | Cao khi chạy local CUDA trên GPU 8 GB VRAM; không sửa ở Phase 1 vì thuộc module `reranker.py` (ngoài phạm vi Phase 1). |
| DEF-P1-02 | Live Groq API generation (`Chat`, `Quiz`, `Study Roadmap`) chưa được kiểm chứng vì thiếu `GROQ_API_KEY`. | `backend/.env` giữ trống `GROQ_API_KEY=` theo STEP 10 | Phase 3 | Trung bình (đã lên lịch chính thức ở Phase 3). |
| DEF-P1-03 | PyMuPDF `2.2` phát cảnh báo `FutureWarning: The fitz API is deprecated... Use import pymupdf instead.` | Baseline & post-impl pytest warning summary | Phase 2 (Document Processing) | Thấp (chỉ là warning, chưa lỗi). |
| DEF-P1-04 | `docker-compose.yml` chứa thuộc tính `version: "3.9"` bị Docker Compose v5 báo obsolete warning. | `docker compose config -q` warning output | Phase 6 / Cleanup | Rất thấp (chỉ là warning). |
| DEF-P1-05 | Kiểm thử click-through tương tác bằng Chrome DevTools trên Streamlit UI chưa chạy được do Chrome trên host chưa bật remote debugging. | Browser subagent error `DevToolsActivePort ENOENT` | Phase 6 (Frontend E2E UI) | Thấp (đã kiểm chứng đầy đủ config, HTTP, `AppTest` và Backend API). |

---

## 14. Changed Files

| File | Change Type | Purpose | Risk | Test Evidence |
|---|---|---|---|---|
| `docker-compose.yml` | Modified | Pin `qdrant/qdrant:v1.20.2` tương thích `qdrant-client 1.20.0` và `query_points()` | Thấp | Docker Qdrant `query_points()` + `test_phase1_runtime_config.py` PASS |
| `backend/app/core/config.py` | Modified | Thêm property `Settings.upload_dir` trả `Path(self.SESSION_RUNTIME_DIR) / "uploads"` | Thấp | `test_phase1_runtime_config.py` + Docker path check PASS |
| `backend/app/api/routes/upload.py` | Modified | Thay `parents[4]` bằng `settings.upload_dir` và `resolve_upload_dir()` | Thấp | `test_phase1_runtime_config.py` + `test_document_lifecycle.py` + Docker smoke PASS |
| `backend/app/api/routes/documents.py` | Modified | Tái sử dụng `resolve_upload_dir()` từ `upload.py` và `Path(settings.SESSION_RUNTIME_DIR)` | Thấp | `test_phase1_runtime_config.py` + `test_document_lifecycle.py` PASS |
| `frontend/.streamlit/config.toml` | Modified | Đồng bộ `maxUploadSize = 15` (từ `10`) | Thấp | `streamlit config show` local & Docker = `15` + `test_phase1_runtime_config.py` PASS |
| `README.md` | Modified | Đồng bộ lệnh chạy Streamlit từ thư mục `frontend/` để nạp `.streamlit/config.toml` | Rất thấp | Xác minh `streamlit config show` từ `frontend/` = `15` |
| `backend/.dockerignore` | Added | Loại trừ `.env`, `.venv`, caches, logs khỏi backend Docker build context | Rất thấp | Docker build + `ENV_IN_IMAGE= False` PASS |
| `frontend/.dockerignore` | Added | Loại trừ `.env`, `.venv`, caches, logs khỏi frontend Docker build context | Rất thấp | Docker build + `test_phase1_runtime_config.py` PASS |
| `tests/integration/test_phase1_runtime_config.py` | Added | 5 regression tests cho runtime path, upload limit 15 MB và cấu hình Docker/Qdrant | Rất thấp | `5 passed` trong full pytest suite (`276 passed`) |

*(Lưu ý: `backend/.env` được tạo local từ `backend/.env.example`, đã xác minh bị `.gitignore` loại trừ và không xuất hiện trong `git status`.)*

---

## 15. Acceptance Criteria Matrix

| AC ID | Requirement | Status | Evidence | Notes |
|---|---|---|---|---|
| AC-01 | Workspace đúng repository được chỉ định. | PASS | `git rev-parse --show-toplevel` → `D:/DSA-Document-Assistant-Using-Hybrid-RAG` | Không dùng đường dẫn cũ |
| AC-02 | Chỉ sử dụng branch `feat/final-readiness`. | PASS | `git branch --show-current` → `feat/final-readiness` | Không tạo thêm branch nào khác |
| AC-03 | Main commit không thay đổi. | PASS | `git rev-parse refs/heads/main` = `b84f09450f213b03ad57ef8b388f482ee0f0eaa2` | Trùng khớp trước và sau Phase 1 |
| AC-04 | `.venv` sử dụng Python 3.11 và hoạt động. | PASS | `.\.venv\Scripts\python.exe --version` → `Python 3.11.9` | Được `.gitignore` bỏ qua |
| AC-05 | Các dependencies cần thiết được cài đặt và import thành công. | PASS | 16/16 package trọng yếu import OK | Xem bảng Mục 3 |
| AC-06 | `pip check` không có dependency conflict chưa xử lý. | PASS | `pip check` → `No broken requirements found.` (exit `0`) | |
| AC-07 | PyTorch device và khả năng CUDA/CPU fallback được kiểm chứng trung thực. | PASS | `torch 2.11.0+cu128`, `cuda_available=True`, RTX 5060 `sm_120`, GPU & CPU matmul OK | Docker dùng `2.12.1+cpu` |
| AC-08 | Có baseline pytest thực tế và phân loại lỗi đầy đủ. | PASS | `271 passed, 0 failed, 0 skipped` (exit `0`, 15.32s) | Chạy trước khi sửa code |
| AC-09 | Qdrant Server được pin phiên bản tương thích với client. | PASS | `docker-compose.yml` pin `qdrant/qdrant:v1.20.2` khớp `qdrant-client 1.20.0` | Không dùng `latest` |
| AC-10 | `query_points()` hoạt động trên Qdrant Server thật. | PASS | `QUERY_POINTS_RESULT: [(1, 1.0, 'c1'), (2, 0.0, 'c2')]` trên `v1.20.2` | Đã dọn collection tạm |
| AC-11 | Runtime/upload path nhất quán ở local và Docker. | PASS | Local `data/runtime/uploads`; Docker `/app/data/runtime/uploads` | Một nguồn `settings.upload_dir` |
| AC-12 | Streamlit và Backend sử dụng giới hạn upload nhất quán 15 MB. | PASS | `config.toml=15`, `MAX_FILE_SIZE=15 MiB`, `MAX_UPLOAD_BYTES=15 MiB` | Xác minh `get_option` local & Docker = 15 |
| AC-13 | File hợp lệ >10 MB trong giới hạn 15 MB upload thành công; file vượt giới hạn bị từ chối. | PASS | `11 MiB` → HTTP 200; `16 MiB` → HTTP 413; `.txt` → HTTP 400 (local & Docker API + pytest) | Click-through Chrome UI ghi nhận `BLOCKED` riêng ở Mục 9/13 |
| AC-14 | Backend `/health` và `/ready` hoạt động trong local và Docker. | PASS | Local `:18001` & Docker `:18002` đều `/health` 200 & `/ready` 200 | Kèm persistence sau restart |
| AC-15 | Frontend startup và kết nối Backend thành công trong local và Docker. | PASS | `/_stcore/health` 200 `ok`, `/` 200; live session create/delete PASS ở cả local (`AppTest`) & Docker (`api_client`) | |
| AC-16 | Post-implementation pytest không phát sinh regression mới. | PASS | `276 passed, 0 failed, 0 skipped` (+5 tests mới, exit `0`) | So với baseline 271 passed |
| AC-17 | Các lỗi ngoài phạm vi được ghi nhận và phân bổ đúng phase. | PASS | 5 mục trong Mục 13 (DEF-P1-01 đến DEF-P1-05) | Không tự sửa ngoài phạm vi |
| AC-18 | Không thay đổi dataset, corpus, retrieval algorithm hoặc evaluation logic. | PASS | `git diff --name-status` chỉ có 6 file cấu hình/runtime + 3 file mới Phase 1 | 100% đúng phạm vi |
| AC-19 | Không làm mất dữ liệu hoặc làm suy yếu session isolation. | PASS | Không xóa volume; persistence qua restart PASS; `test_study_sessions.py` & `test_document_lifecycle.py` PASS | 4 container người dùng nguyên vẹn |
| AC-20 | Không commit/push/merge trái phép, không để lộ secret và có báo cáo đầy đủ. | PASS | 0 commit/push/merge; `.env` gitignored & không có trong Docker image; đủ 18 mục báo cáo | |

**Tổng hợp Acceptance Criteria:** **20/20 PASS** (0 FAIL, 0 BLOCKED ở mức tiêu chí AC; các giới hạn phụ trợ như live Groq API và Chrome remote debugging được ghi nhận minh bạch tại Mục 9, 13 và 16).

---

## 16. Remaining Blockers and Limitations

1. **Host Port 8000 đang bận bởi dự án khác:** Container `escrow_marketplace_demo-backend-1` của người dùng đang chiếm port `8000` trên host. Khi chạy `docker compose up` mặc định không override port, cần dừng container kia hoặc đổi port mapping trên host.
2. **Chrome Remote Debugging chưa bật:** Browser subagent không thể kết nối Chrome (`DevToolsActivePort ENOENT`) để bấm trực tiếp widget file uploader trên trình duyệt. Nếu người quản lý muốn kiểm chứng thêm thao tác click-through trên trình duyệt, chỉ cần mở `chrome://inspect/#remote-debugging` hoặc tự mở `http://localhost:8501` để thử nhanh file 11–14 MB.
3. **Thiếu `GROQ_API_KEY`:** Live generation (Chat/Quiz/Roadmap) chưa chạy — nằm trong phạm vi Phase 3.
4. **Lỗi cache key reranker trên GPU 8 GB (`DEF-P1-01`):** Cần xử lý ở Phase 2/3 để tránh nạp trùng `BAAI/bge-reranker-v2-m3` khi `reranker_device: auto`.

---

## 17. Git Integrity and Scope Verification

- `git branch --show-current` → `feat/final-readiness`
- `git rev-parse HEAD` → `b84f09450f213b03ad57ef8b388f482ee0f0eaa2`
- `git rev-parse refs/heads/main` → `b84f09450f213b03ad57ef8b388f482ee0f0eaa2`
- `git diff --cached --stat` → rỗng (chưa stage, chưa commit)
- `git diff --check` → exit code `0` (không lỗi whitespace)
- `git diff --stat`: `6 files changed, 25 insertions(+), 9 deletions(-)`
- Không có thay đổi nào trên `data/corpus/`, `backend/app/retrieval/`, `backend/app/generation/`, `backend/app/safety/`, `evaluation/`.
- `PROJECT_AUDIT_FINAL_READINESS.md` được bảo toàn nguyên vẹn.
- `.venv/` và `backend/.env` được `.gitignore` bỏ qua, không bị track.
- Chưa thực hiện bất kỳ lệnh `git commit`, `git push`, `git merge`, `git rebase`, `git reset --hard`, `git clean` nào.

---

## 18. Agent Self-Assessment

**Trạng thái đề xuất:** **READY FOR REVIEW**

Toàn bộ mục tiêu Phase 01 (TASK-01 & TASK-04) đã được triển khai, kiểm chứng thực tế trên cả môi trường local (Python 3.11 + RTX 5060 CUDA 12.8) và môi trường Docker cách ly (`qdrant/qdrant:v1.20.2` + `backend` + `frontend`), đạt **276/276 pytest passed** (0 regression) và **20/20 Acceptance Criteria PASS** với bằng chứng đầy đủ. Mọi giới hạn và vấn đề ngoài phạm vi đều được báo cáo trung thực. Dừng triển khai tại đây và chờ người quản lý dự án nghiệm thu chính thức trước khi tiến hành commit hoặc sang Phase 2.
