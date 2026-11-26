# PHASE 01 – ROUND 02 VERIFICATION REPORT

> **Dự án:** DSA Document Assistant Using Hybrid RAG
> **Phase:** Phase 01 – Environment & Runtime Stabilization (Implementation Round 02)
> **Phạm vi:** R2-01 (Actual Streamlit UI Upload Verification) & R2-02 (Exact 15 MiB Boundary Verification)
> **Repository:** `D:\DSA-Document-Assistant-Using-Hybrid-RAG`
> **Branch:** `feat/final-readiness`
> **Ngày thực hiện:** 2026-11-27
> **Báo cáo Round 01 (bảo toàn nguyên vẹn):** `PHASE_01_IMPLEMENTATION_REPORT.md`

---

## 1. Executive Summary

Round 02 đã hoàn tất toàn bộ yêu cầu kiểm chứng còn thiếu của **AC-13** (R2-01 & R2-02) và đóng hoàn toàn **DEF-P1-05** trên cả **Local** và **Docker** bằng trình duyệt thật (**Microsoft Edge Chromium Headless** với profile kiểm thử cô lập qua Chrome DevTools Protocol trên loopback `127.0.0.1`, không chạm cấu hình Chrome/Edge cá nhân).

**Phát hiện quan trọng từ kiểm thử biên chính xác (R2-02) và bản sửa tối thiểu:**
- Khi kiểm thử 3 mốc biên chính xác trên giao diện Streamlit `1.67.0` với `maxUploadSize = 15`:
  - Tầng client-side JS của Streamlit dùng đúng đơn vị **15 MiB = 15,728,640 bytes** (chấp nhận `15,728,639 B` và `15,728,640 B`; từ chối `15,728,641 B` với `Error: File must be 15.0MB or smaller.`).
  - **Bất nhất thực tế phát hiện ở tầng Streamlit server (`streamlit/web/server/starlette/starlette_routes.py:869-873`):** Streamlit `1.67.0` kiểm tra nhanh `int(content_length) > max_size_bytes` **quên cộng `_MAX_UPLOAD_MULTIPART_OVERHEAD_BYTES` (~230 byte header multipart/form-data)**, khiến hai file hợp lệ `15,728,639 B` và `15,728,640 B` bị Streamlit server trả HTTP `413` (`Error: AxiosError: Request failed with status code 413`) trước khi tới Backend!
- **Bản sửa tối thiểu trong phạm vi upload configuration (giữ nguyên giới hạn chính xác 15 MiB = 15,728,640 bytes):**
  - `frontend/.streamlit/config.toml`: đặt `maxUploadSize = 16` (chỉ tăng 1 MiB làm khoảng dự phòng framing cho multipart overhead ở tầng Starlette transport).
  - `frontend/screens/upload_ui.py`: thêm `max_upload_size=15` vào `st.file_uploader(...)` để giao diện và tầng client JS vẫn hiển thị và thực thi đúng **`15MB per file`** (`15,728,640 bytes`), đồng bộ tuyệt đối với `upload_ui.MAX_UPLOAD_BYTES = 15 * 1024 * 1024` và Backend `upload.MAX_FILE_SIZE = 15 * 1024 * 1024`.
  - Rebuild duy nhất service `frontend` trong Docker smoke stack và kiểm chứng lại trên cả Local & Docker.
- **Kết quả sau bản sửa:**
  - **10/10 Backend API boundary tests PASS** (5 Local + 5 Docker: `51,200 B`, `11,534,336 B`, `15,728,639 B`, `15,728,640 B` → HTTP `200`, index 1 chunk; `15,728,641 B` → HTTP `413`, 0 chunks).
  - **10/10 Actual Streamlit UI upload tests PASS** (Local & Docker: `51,200 B`, `11,534,336 B`, `15,728,639 B`, `15,728,640 B` → `Thành công: 1 · Chunks đã lập chỉ mục: 1` và xuất hiện đúng trong tab **🗂️ Quản lý kho**; `15,728,641 B` → bị chặn ngay tại UI với `Error: File must be 15.0MB or smaller.` và tab **🗂️ Quản lý kho** trống).
  - **Full pytest regression:** **276/276 passed** (0 failed, 0 skipped, 14.15s, exit code `0`).
  - **AC-13 được nâng cấp chính thức lên `PASS`** → Tổng hợp **20/20 Acceptance Criteria PASS**.

---

## 2. Git Integrity

| Thuộc tính | Giá trị thực tế |
|---|---|
| **Workspace** | `D:\DSA-Document-Assistant-Using-Hybrid-RAG` |
| **Current branch** | `feat/final-readiness` (duy nhất) |
| **HEAD commit** | `b84f09450f213b03ad57ef8b388f482ee0f0eaa2` |
| **Local `main` commit** | `b84f09450f213b03ad57ef8b388f482ee0f0eaa2` (không thay đổi) |
| **Staged changes (`git diff --cached --stat`)** | Trống (0 staged, chưa commit/push/merge) |
| **Whitespace check (`git diff --check`)** | Exit code `0` |
| **Tracked modified files (7)** | `README.md`, `backend/app/api/routes/documents.py`, `backend/app/api/routes/upload.py`, `backend/app/core/config.py`, `docker-compose.yml`, `frontend/.streamlit/config.toml`, `frontend/screens/upload_ui.py` |
| **New Phase 1 files (3)** | `backend/.dockerignore`, `frontend/.dockerignore`, `tests/integration/test_phase1_runtime_config.py` |
| **Preserved reports** | `PROJECT_AUDIT_FINAL_READINESS.md`, `PHASE_01_IMPLEMENTATION_REPORT.md` |
| **Out-of-scope integrity** | 0 thay đổi trên `data/corpus/`, `backend/app/retrieval/`, `backend/app/generation/`, `backend/app/safety/`, `evaluation/` |

---

## 3. Actual Streamlit UI Test Results (R2-01)

**Phương thức tự động hóa trình duyệt:**
- Trình duyệt: `C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe` chế độ `--headless=new` với profile cô lập (`scratch\edge_r2_profile`, `scratch\edge_r2_manage_prof`), remote debugging chỉ lắng nghe loopback `127.0.0.1`, viewport `1440×1080`, điều khiển qua CDP + `websockets==17.4` có sẵn trong `.venv`.
- Mỗi trường hợp chạy trong **phiên học độc lập** (xóa phiên qua nút `Xóa phiên học hiện tại` → `Xác nhận xóa phiên` sau mỗi lượt).
- Tất cả 5 fixture PDF được xác minh parse hợp lệ qua PyMuPDF (`1` trang, `33` từ nội dung DSA hợp lệ mỗi file) trước khi kiểm thử.

### 3.1. Kết quả kiểm thử UI trên Local (`http://127.0.0.1:18501` → Backend `http://127.0.0.1:18001`)

| File Fixture | File Size (bytes) | Browser Action | Expected | Actual Result (UI → Backend → Qdrant → Quản lý kho) | Evidence | Status |
|---|---|---|---|---|---|---|
| `r2_small_50kb.pdf` | `51,200` | `DOM.setFileInputFiles` → click `📤 Nạp vào Hệ thống` | ACCEPT, index 1 chunk | UI hint `15MB per file • PDF, DOCX`; `Thành công: 1 · Thất bại: 0 · Bỏ qua: 0 · Chunks đã lập chỉ mục: 1` | `task-254.log:3` | **PASS** |
| `r2_valid_11mib.pdf` | `11,534,336` (11 MiB) | `DOM.setFileInputFiles` → click `📤 Nạp vào Hệ thống` | ACCEPT, index 1 chunk | `Thành công: 1 · Thất bại: 0 · Bỏ qua: 0 · Chunks đã lập chỉ mục: 1`; `r2_valid_11mib.pdf · 11.00 MB` | `local_ui_r2_valid_11mib_upload.png`, `task-254.log:4` | **PASS** |
| `r2_boundary_15728639.pdf` | `15,728,639` | `DOM.setFileInputFiles` → click `📤 Nạp vào Hệ thống` | ACCEPT, index 1 chunk | `Thành công: 1 · Thất bại: 0 · Bỏ qua: 0 · Chunks đã lập chỉ mục: 1` | `task-254.log:5` | **PASS** |
| `r2_boundary_15728640.pdf` | `15,728,640` (15 MiB) | `DOM.setFileInputFiles` → click `📤 Nạp vào Hệ thống` → click tab `🗂️ Quản lý kho` | ACCEPT, index 1 chunk & hiển thị trong Quản lý kho | `Thành công: 1 · Chunks đã lập chỉ mục: 1`; tab **Quản lý kho** hiển thị `1 tài liệu trong phiên học hiện tại`, `r2_boundary_15728640.pdf`, `Dung lượng: 15360 KB`, `1 chunks · 1 trang · PDF` | `local_ui_r2_boundary_15728640_upload.png`, `local_final_r2_boundary_15728640_manage_ok.png`, `task-281.log:1` | **PASS** |
| `r2_boundary_15728641.pdf` | `15,728,641` (15 MiB + 1 B) | `DOM.setFileInputFiles` → kiểm tra nút nạp & tab `🗂️ Quản lý kho` | REJECT, 0 chunk indexed | Client UI chặn ngay: `Error: File must be 15.0MB or smaller.`, nút `📤 Nạp vào Hệ thống` bị vô hiệu hóa; tab **Quản lý kho** hiển thị `Kho tài liệu đang trống` | `local_ui_r2_boundary_15728641_rejected.png`, `local_final_r2_boundary_15728641_manage_empty.png`, `task-281.log:2` | **PASS** |

### 3.2. Kết quả kiểm thử UI trên Docker (`http://127.0.0.1:18502` → Backend `http://backend:8000` / `:18002`)

| File Fixture | File Size (bytes) | Browser Action | Expected | Actual Result (UI → Backend → Qdrant → Quản lý kho) | Evidence | Status |
|---|---|---|---|---|---|---|
| `r2_small_50kb.pdf` | `51,200` | `DOM.setFileInputFiles` → click `📤 Nạp vào Hệ thống` | ACCEPT, index 1 chunk | UI hint `15MB per file • PDF, DOCX`; `Thành công: 1 · Thất bại: 0 · Bỏ qua: 0 · Chunks đã lập chỉ mục: 1` | `task-254.log:10` | **PASS** |
| `r2_valid_11mib.pdf` | `11,534,336` (11 MiB) | `DOM.setFileInputFiles` → click `📤 Nạp vào Hệ thống` | ACCEPT, index 1 chunk | `Thành công: 1 · Thất bại: 0 · Bỏ qua: 0 · Chunks đã lập chỉ mục: 1` | `docker_ui_r2_valid_11mib_upload.png`, `task-254.log:11` | **PASS** |
| `r2_boundary_15728639.pdf` | `15,728,639` | `DOM.setFileInputFiles` → click `📤 Nạp vào Hệ thống` | ACCEPT, index 1 chunk | `Thành công: 1 · Thất bại: 0 · Bỏ qua: 0 · Chunks đã lập chỉ mục: 1` | `task-254.log:12` | **PASS** |
| `r2_boundary_15728640.pdf` | `15,728,640` (15 MiB) | `DOM.setFileInputFiles` → click `📤 Nạp vào Hệ thống` → click tab `🗂️ Quản lý kho` | ACCEPT, index 1 chunk & hiển thị trong Quản lý kho | `Thành công: 1 · Chunks đã lập chỉ mục: 1`; tab **Quản lý kho** hiển thị `1 tài liệu trong phiên học hiện tại`, `r2_boundary_15728640.pdf`, `Dung lượng: 15360 KB`, `1 chunks · 1 trang · PDF` | `docker_ui_r2_boundary_15728640_upload.png`, `docker_final_r2_boundary_15728640_manage_ok.png`, `task-281.log:3` | **PASS** |
| `r2_boundary_15728641.pdf` | `15,728,641` (15 MiB + 1 B) | `DOM.setFileInputFiles` → kiểm tra nút nạp & tab `🗂️ Quản lý kho` | REJECT, 0 chunk indexed | Client UI chặn ngay: `Error: File must be 15.0MB or smaller.`, nút `📤 Nạp vào Hệ thống` bị vô hiệu hóa; tab **Quản lý kho** hiển thị `Kho tài liệu đang trống` | `docker_final_15728641_rejected.png`, `docker_final_r2_boundary_15728641_manage_empty.png`, `task-281.log:4` | **PASS** |

---

## 4. Exact Upload Boundary Test Results (R2-02)

### 4.1. Phân tích quy ước MB vs MiB theo từng tầng

| Tầng hệ thống | Công thức giới hạn thực tế | Hành vi tại `15,728,639 B` | Hành vi tại `15,728,640 B` | Hành vi tại `15,728,641 B` |
|---|---|---|---|---|
| **Streamlit Client UI (`st.file_uploader(max_upload_size=15)`)** | `15 * 1024 * 1024 = 15,728,640` bytes (15 MiB) | ACCEPT | ACCEPT | REJECT (`Error: File must be 15.0MB or smaller.`) |
| **Streamlit Starlette Server (`config.toml maxUploadSize = 15` — trước khi sửa)** | `int(content_length) > 15 * 1024 * 1024` (lỗi upstream `starlette_routes.py:872` không cộng overhead multipart ~230 B) | **FAIL (HTTP 413)** | **FAIL (HTTP 413)** | Chặn trước ở Client JS |
| **Streamlit Starlette Server (`config.toml maxUploadSize = 16` — sau khi sửa)** | `16 * 1024 * 1024 = 16,777,216` bytes (dự phòng 1 MiB cho multipart framing) | ACCEPT | ACCEPT | Chặn trước ở Client JS (và phòng thủ thêm tại `upload_ui.py` & Backend) |
| **Streamlit App Guard (`upload_ui.MAX_UPLOAD_BYTES`)** | `15 * 1024 * 1024 = 15,728,640` bytes | ACCEPT | ACCEPT | SKIP (`Vượt giới hạn 15 MB.`) |
| **FastAPI Backend (`upload.MAX_FILE_SIZE`)** | `15 * 1024 * 1024 = 15,728,640` bytes | HTTP `200` (1 chunk) | HTTP `200` (1 chunk) | HTTP `413` (0 chunks, `total=0`) |

### 4.2. Kết quả Backend API độc lập trên Local (`:18001`) và Docker (`:18002`)

| Environment | File | Exact Size (bytes) | Expected | HTTP Status | `indexed_chunks` | `GET /api/documents` `total` | Status |
|---|---|---|---|---|---|---|---|
| Local API | `r2_small_50kb.pdf` | `51,200` | ACCEPT | `200` | `1` | `1` (`['r2_small_50kb.pdf']`) | **PASS** |
| Local API | `r2_valid_11mib.pdf` | `11,534,336` | ACCEPT | `200` | `1` | `1` (`['r2_valid_11mib.pdf']`) | **PASS** |
| Local API | `r2_boundary_15728639.pdf` | `15,728,639` | ACCEPT | `200` | `1` | `1` (`['r2_boundary_15728639.pdf']`) | **PASS** |
| Local API | `r2_boundary_15728640.pdf` | `15,728,640` | ACCEPT | `200` | `1` | `1` (`['r2_boundary_15728640.pdf']`) | **PASS** |
| Local API | `r2_boundary_15728641.pdf` | `15,728,641` | REJECT | `413` | `0` | `0` (`[]`) | **PASS** |
| Docker API | `r2_small_50kb.pdf` | `51,200` | ACCEPT | `200` | `1` | `1` (`['r2_small_50kb.pdf']`) | **PASS** |
| Docker API | `r2_valid_11mib.pdf` | `11,534,336` | ACCEPT | `200` | `1` | `1` (`['r2_valid_11mib.pdf']`) | **PASS** |
| Docker API | `r2_boundary_15728639.pdf` | `15,728,639` | ACCEPT | `200` | `1` | `1` (`['r2_boundary_15728639.pdf']`) | **PASS** |
| Docker API | `r2_boundary_15728640.pdf` | `15,728,640` | ACCEPT | `200` | `1` | `1` (`['r2_boundary_15728640.pdf']`) | **PASS** |
| Docker API | `r2_boundary_15728641.pdf` | `15,728,641` | REJECT | `413` | `0` | `0` (`[]`) | **PASS** |

---

## 5. Additional Changes (Round 02)

Chỉ thực hiện bản sửa tối thiểu trong phạm vi cấu hình upload để khắc phục lỗi Starlette `Content-Length` multipart overhead của Streamlit `1.67.0`:

| File | Change | Reason |
|---|---|---|
| `frontend/.streamlit/config.toml` | `maxUploadSize = 15` → `maxUploadSize = 16` (kèm comment giải thích) | Cung cấp 1 MiB transport headroom cho header `multipart/form-data` để tránh lỗi HTTP 413 giả ở `starlette_routes.py:872` tại kích thước `15,728,639` và `15,728,640` bytes. |
| `frontend/screens/upload_ui.py` | Thêm `max_upload_size=15` vào `st.file_uploader(...)` | Giữ nguyên giới hạn thực thi & hiển thị chính xác **15 MiB (`15,728,640` bytes)** tại widget UI và client JS (`Limit 15MB per file`). |
| `tests/integration/test_phase1_runtime_config.py` | Cập nhật `test_upload_limit_is_synchronized_to_15_mb` | Kiểm chứng đồng thời `config.toml` (`maxUploadSize == 16` transport headroom), `upload_ui.py` (`max_upload_size=15` & `MAX_UPLOAD_BYTES == 15 MiB`) và `upload.MAX_FILE_SIZE == 15 MiB`. |

---

## 6. Regression Test Results

Chạy lại toàn bộ suite sau khi áp dụng thay đổi Round 02 và rebuild service `frontend` Docker:

- **Command:** `.\.venv\Scripts\python.exe -m pytest -q`
- **Result:** **`276 passed, 1 warning in 14.15s`** (exit code `0`)
- **So sánh với mốc sau Round 01 (`276 passed`):** `0` regression, `0` failed, `0` skipped, `0` errors.

### Dependency Reproducibility Snapshot (`.venv`, Python `3.11.9`, 110 packages — đã kiểm tra không chứa secret/URL)

```text
altair==6.3.0
annotated-doc==0.0.7
annotated-types==0.8.0
anyio==4.16.0
attrs==26.1.0
certifi==2026.11.20
charset-normalizer==3.5.3
click==8.5.1
cloudpickle==3.1.3
colorama==0.4.6
contourpy==1.3.3
cycler==1.3.3
et_xmlfile==2.0.0
fastapi==0.165.0
filelock==4.2.4
flatbuffers==25.12.19
fonttools==4.69.1
fsspec==2026.9.0
grpcio==1.85.4
h11==0.16.0
h2==4.4.1
hf-xet==1.8.0
hpack==4.2.0
httpcore==1.0.9
httpcore2==2.16.2
httptools==0.9.0
httpx==0.29.0
httpx2==2.16.2
huggingface_hub==2.5.0
hyperframe==6.1.0
idna==3.20
iniconfig==2.3.1
itsdangerous==2.2.0
Jinja2==3.1.6
jiter==0.19.0
joblib==1.6.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kiwisolver==1.5.1
lxml==6.1.3
markdown-it-py==4.3.0
MarkupSafe==3.0.3
matplotlib==3.11.2
mdurl==0.1.2
mpmath==1.3.0
narwhals==2.29.0
networkx==3.6.1
numpy==2.4.6
onnxruntime==1.32.0
openai==3.60.0
openpyxl==3.1.5
opentelemetry-api==1.45.2
packaging==26.4
pandas==3.1.1
pillow==13.0.0
pluggy==1.7.0
portalocker==4.4.2
protobuf==7.36.2
psutil==7.2.2
pyarrow==26.0.1
pydantic==2.14.4
pydantic-settings==2.16.3
pydantic_core==2.50.5
pydeck==0.9.3
Pygments==2.21.0
pymupdf==2.2
pymupdf4llm==2.2
pymupdf_layout==2.2
pyparsing==3.3.3
pytest==9.2.0
python-crfsuite==0.9.12
python-dateutil==4.0.0
python-docx==1.2.0
python-dotenv==1.2.4
python-multipart==0.0.34
pyvi==0.1.1
PyYAML==6.0.4
qdrant-client==1.20.0
rank-bm25==0.2.2
referencing==0.37.0
regex==2026.11.25
requests==2.34.2
rich==15.0.0
rpds-py==2026.9.1
safetensors==0.8.0
scikit-learn==1.9.1
scipy==1.17.1
sentence-transformers==5.7.0
shellingham==1.5.4
sklearn-crfsuite==0.5.0
sniffio==1.3.1
starlette==1.10.0
streamlit==1.67.0
sympy==1.14.0
tabulate==0.10.0
threadpoolctl==3.8.0
tokenizers==0.23.3
torch==2.11.0+cu128
tqdm==4.70.1
transformers==5.24.1
truststore==0.10.5
typer==0.29.3
typing-inspection==0.4.5
typing_extensions==4.16.0
tzdata==2026.6
urllib3==2.8.0
uvicorn==0.54.0
watchdog==7.1.1
watchfiles==1.3.0
websockets==17.4
```

---

## 7. Updated Deferred Issues

| Issue ID | Description | Assigned Phase | Status |
|---|---|---|---|
| **DEF-P1-01** | `reranker.py` cache key mismatch giữa `device="auto"` (warmup) và `device="cuda"` (`rerank()`), gây nạp trùng model `BAAI/bge-reranker-v2-m3` trên GPU 8 GB. | **Bắt buộc đầu Phase 2** (trước Retrieval smoke test & corpus freeze) | Deferred (không sửa trong Round 02) |
| **DEF-P1-02** | Live Groq API generation (`Chat`, `Quiz`, `Roadmap`) chưa kiểm chứng do chưa cung cấp `GROQ_API_KEY`. | **Phase 3** | Deferred |
| **DEF-P1-03** | PyMuPDF `2.2` phát cảnh báo `FutureWarning` deprecated API `fitz`. | **Phase 2** | Deferred |
| **DEF-P1-04** | `docker-compose.yml` chứa thuộc tính `version: "3.9"` bị Compose v5 báo obsolete warning. | **Phase 7** (nếu không gây lỗi) | Deferred |
| **DEF-P1-05** | Kiểm thử upload trực tiếp qua giao diện Streamlit thực tế (Local & Docker). | **Phase 1 – Round 02** | **RESOLVED / CLOSED** |

---

## 8. Updated AC-13 Status

- **AC-13:** *"File hợp lệ >10 MB trong giới hạn 15 MB upload thành công; file vượt giới hạn bị từ chối."*
- **Updated Status:** **PASS**
- **Bằng chứng đầy đủ cho cả Local và Docker:**
  - **Backend API (Local & Docker):** `51,200 B` (`200`), `11,534,336 B` (`200`), `15,728,639 B` (`200`), `15,728,640 B` (`200`, 1 chunk indexed & visible in `/api/documents`), `15,728,641 B` (`413`, 0 chunks, `total=0`).
  - **Streamlit UI thực tế (Headless Edge CDP — Local & Docker):** Toàn bộ luồng `File Uploader → Backend API → Document Processing → Qdrant Indexing → Document Management UI (🗂️ Quản lý kho)` đạt PASS tại `51,200 B`, `11,534,336 B`, `15,728,639 B` và `15,728,640 B` (`15360 KB`, `1 chunks`); file `15,728,641 B` bị chặn ngay trên UI với `Error: File must be 15.0MB or smaller.` và tab `🗂️ Quản lý kho` xác nhận trống (`Kho tài liệu đang trống`).

---

## 9. Updated Acceptance Criteria Summary

| AC ID | Requirement | Status |
|---|---|---|
| AC-01 → AC-12 | (Đã nghiệm thu PASS ở Round 01) | **PASS (12/12)** |
| **AC-13** | File hợp lệ >10 MB trong giới hạn 15 MiB (`<= 15,728,640 B`) upload thành công; file vượt giới hạn (`>= 15,728,641 B`) bị từ chối ở cả Backend API & Streamlit UI (Local và Docker) | **PASS (Updated in Round 02)** |
| AC-14 → AC-20 | (Đã nghiệm thu PASS ở Round 01; AC-16 & AC-20 đã tái xác minh sau Round 02: `276 passed`, Git integrity intact) | **PASS (7/7)** |

**Tổng hợp Acceptance Criteria:** **20 / 20 PASS** | `0` FAIL | `0` BLOCKED | `0` UNVERIFIED.
**Đề xuất trạng thái:** **READY FOR FINAL PHASE 01 ACCEPTANCE** (quyền phê duyệt chính thức thuộc về người quản lý dự án).

---

## 10. Remaining Blockers

- **Không còn blocker nào thuộc phạm vi Phase 01.**
- Lưu ý vận hành: host port `8000` hiện vẫn đang được container dự án khác của người dùng (`escrow_marketplace_demo-backend-1`) sử dụng — toàn bộ container của người dùng đều được bảo toàn nguyên vẹn (`Up (healthy)`), toàn bộ tiến trình/container smoke test của Round 02 đã được dừng sạch (không xóa volume).
