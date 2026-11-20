# KẾ HOẠCH TRIỂN KHAI WORKSTREAM 1 — RETRIEVAL & INGESTION FIX

> **Mã tài liệu:** `PLAN-01-RETRIEVAL`  
> **Nhánh phát triển (Target Branch):** `feat/retrieval-fixes`  
> **Commit gốc chuẩn (Baseline Commit):** `8c48d4daf81af8511c93cd0718fc40dbe40536ad` (HEAD chứa PR #5 Retrieval và PR #6 Week 5 Integration)  
> **Chủ trì (Track Lead):** Senior Retrieval & Data Ingestion Engineer (Phụ trách: C1 + B1)  
> **Trạng thái:** KẾ HOẠCH HOÀN THIỆN ĐÃ PHÊ DUYỆT — NGHIÊM CẤM SỬA CODE TRONG BƯỚC NÀY  

---

## 1. TỔNG QUAN & SNAPSHOT DỰ ÁN (PROJECT SNAPSHOT)

- **Repository:** `DSA-Document-Assistant-Using-Hybrid-RAG`
- **Branch làm việc tài liệu:** `docs/parallel-workstream-plans`
- **Baseline Commit chuẩn:** `8c48d4daf81af8511c93cd0718fc40dbe40536ad`
- **Hiện trạng phân hệ Retrieval sau PR #5 và PR #6:**
  - **BM25 Search:** `rank-bm25` kết hợp PyVi tokenizer tiếng Việt, chuẩn hóa Unicode NFC và xử lý dấu tiếng Việt (`keyword_search.py`).
  - **Dense Search:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` kết hợp Qdrant Cosine Similarity (`vector_search.py`, `vector_store.py`).
  - **Hybrid Fusion:** Reciprocal Rank Fusion (RRF, $k=60$) kết hợp danh sách ứng viên BM25 và Dense (`hybrid_search.py`).
  - **Reranking:** Cross-Encoder `BAAI/bge-reranker-v2-m3` tối ưu chạy trên GPU CUDA (`reranker.py`), đã đo đạc độ trễ warm inference $\sim 53\text{--}67\text{ ms}$ cho Top-10 ứng viên trên RTX 5060 8GB VRAM.
  - **BM25 Cache:** Đã cô lập theo từng instance `RetrievalEngine` (`self._bm25_cache`), kết hợp cơ chế kiểm tra SHA-256 fingerprint (count + ID mẫu + text hash mẫu) và TTL ($300\text{s}$).
  - **Luồng Ingestion & Quản lý tài liệu:** PR #6 đã bổ sung `ingestion_flow.py` kết nối Upload $\to$ Document Processing $\to$ Indexing, và thêm endpoint quản lý `GET/DELETE /api/documents` tại `documents.py`.
  - **Bộ kiểm thử hiện hành:** 52/52 tests PASSED trong pytest (`tests/unit/` và `tests/integration/`).
  - **Dữ liệu Vector DB:** 5.941 chunks gốc thuộc corpus benchmark `benchmark-v1` / workspace `research` được bảo toàn nguyên vẹn.

---

## 2. KIẾN TRÚC & PHẠM VI DỌC CỦA TRACK 1 (VERTICAL SLICE SCOPE)

Track 1 sở hữu trọn vẹn chu trình **Document Ingestion & Retrieval**: Từ giao diện người dùng tải tài liệu (`upload_ui.py`), các API endpoints tiếp nhận và quản lý tài liệu (`upload.py`, `documents.py`), luồng xử lý nạp dữ liệu (`ingestion_flow.py`), indexing vector (`indexing.py`), lưu trữ Qdrant (`vector_store.py`), đến động cơ tìm kiếm lai kết hợp Reranker (`hybrid_search.py`, `keyword_search.py`, `vector_search.py`, `reranker.py`).

```
[Giao diện upload_ui.py]
         │
         ▼ (HTTP Multipart Form)
[POST /api/upload] (upload.py)
         │
         ▼
[orchestration/ingestion_flow.py]
  ├── Document Parser (PyMuPDF / python-docx) - [READ-ONLY]
  ├── Text Cleaner (NFC, whitespace, min_words) - [READ-ONLY]
  ├── Chunker (chunk_size=512, overlap=100) - [READ-ONLY]
  └── Metadata Creator (_make_document_id) - [READ-ONLY]
         │
         ▼ (Safe Staging & Replacement Lifecycle)
[retrieval/indexing.py: build_index()]
  ├── Check Model & Embed Chunks (MiniLM-L12-v2, 384 dims)
  ├── Atomic Swap / Delete Old Points for Document ID
  ├── Upsert New Points to Qdrant (UUID v5)
  └── Invalidate Workspace BM25 Caches
         │
         ▼
[retrieval/hybrid_search.py: RetrievalEngine]
  ├── BM25 Okapi (PyVi tokenized, Cached per engine)
  ├── Dense Cosine (MiniLM embedding)
  ├── RRF Fusion (k=60) -> Top-K Candidates
  └── BGE Reranker v2-m3 (CUDA Cross-Encoder) -> Evidence Chunks
```

### Các Hợp đồng Giao diện Cốt lõi (Core Public Interfaces)
1. `RetrievalConfig`: Dataclass đóng băng (`frozen=True`), nạp từ YAML hoặc biến môi trường.
2. `RetrievalEngine(config=config, client=client)`: Constructor nhận keyword-only arguments.
3. `engine.retrieve(query=..., method=..., top_k=5, corpus_id=..., workspace_id=...)`: Trả về `List[dict]` chuẩn contract (12 trường).
4. `build_index(chunks, *, corpus_id, workspace_id, client=None, config=None, rebuild=False)`: Indexing pipeline.
5. `delete_document_from_index(document_id, *, corpus_id, workspace_id, client=None, config=None)`: Hàm xóa document chuẩn trong Qdrant.

---

## 3. DANH SÁCH VẤN ĐỀ ĐÃ XÁC NHẬN (CONFIRMED ISSUES)

### LỖI ĐÃ XÁC NHẬN (VERIFIED BUGS)

#### ISSUE-RET-01: Nguy cơ Chunk mồ côi (Orphan Chunks) khi tải lại tài liệu chỉnh sửa
- **Vị trí:** `backend/app/document_processing/metadata.py` (dòng 17–27) & `backend/app/retrieval/indexing.py` (dòng 79–90)
- **Mức độ:** **P0 (Lỗi tính toàn vẹn dữ liệu)**
- **Xác minh thực tế:** `_make_document_id(filename)` sinh ID cố định từ tên file (ví dụ `slides_chuong1.pdf` $\to$ `slides_chuong1`). Nếu người dùng chỉnh sửa tài liệu làm số chunk giảm từ 20 xuống 15 rồi tải lên lại:
  `build_index()` chạy với `rebuild=False` chỉ upsert 15 chunk mới (ghi đè chunk 0–14). **5 chunk cũ từ 15 đến 19 vẫn tồn tại trong Qdrant!** Khi tìm kiếm, các chunk lỗi thời này vẫn bị truy xuất và gửi vào LLM, gây hiện tượng ảo giác hoặc sai lệch kiến thức.
- **Giải pháp (Approved Decision 1):** Triển khai chu trình **Safe Staging & Replacement Lifecycle**:
  1. Trích xuất text, làm sạch, chunking và sinh embeddings hoàn tất trong bộ nhớ tạm (in-memory staging).
  2. Chỉ khi toàn bộ embeddings của tài liệu mới đã sinh thành công: thực hiện xóa toàn bộ chunks cũ của `document_id` trong cùng `workspace_id` và upsert chunks mới ngay lập tức.
  3. Nếu quá trình bóc tách hoặc embedding thất bại giữa chừng: Abort/Rollback ngay lập tức, giữ nguyên các chunks cũ hiện tại trong Qdrant, không để mất dữ liệu và không để lại chunks mồ côi.
  4. Invalidate BM25 cache cho workspace tương ứng.

#### ISSUE-RET-02: Xóa file vật lý quá rộng (Overbroad Glob Pattern) trong Document Deletion
- **Vị trí:** `backend/app/api/routes/documents.py` (dòng 198–203)
- **Mức độ:** **P1 (Lỗi xung đột dữ liệu giữa các workspace)**
- **Xác minh thực tế:** Khi gọi `DELETE /api/documents/{document_id}`, code tìm file trên đĩa bằng lệnh:
  ```python
  for f in UPLOAD_DIR.glob(f"*_{filename}"):
      f.unlink(missing_ok=True)
  ```
  Nếu hai người dùng ở hai workspace khác nhau tải lên hai file có cùng tên gốc (ví dụ `baitap.pdf`), việc xóa tài liệu ở Workspace A sẽ **xóa sạch toàn bộ file vật lý của Workspace B** trên đĩa!
- **Giải pháp (Approved Decision 2):**
  1. Quy định chuẩn đặt tên file vật lý khi lưu trên đĩa: `UPLOAD_DIR / f"{workspace_id}_{document_id}_{filename}"` hoặc tổ chức thư mục con theo `UPLOAD_DIR / workspace_id / filename`.
  2. Khi xóa tài liệu: chỉ xóa file vật lý khớp chính xác tiền tố `{workspace_id}_{document_id}_` hoặc trong thư mục của workspace đó. Tuyệt đối không dùng pattern glob mở rộng `*_{filename}`.
  3. Trong Qdrant: Xóa points với bộ lọc chặt chẽ cả `document_id` và `workspace_id`.

---

### NGUY CƠ CẠNH TRANH & VẬN HÀNH CẦN XỬ LÝ (OPERATIONAL RISKS / NEEDS TEST)

#### ISSUE-RET-03: Rủi ro Khóa File SQLite khi chạy Qdrant Local trong Môi trường Đa Tiến trình (High Concurrency Risk)
- **Vị trí:** `backend/app/api/routes/upload.py` (dòng 55–67) & `backend/app/retrieval/vector_store.py` (dòng 25–48)
- **Mức độ:** **P1 (Rủi ro cạnh tranh tài nguyên trong local mode)**
- **Phân tích:** Trong chế độ `qdrant_mode: "local"`, Qdrant sử dụng file lock trên thư mục dữ liệu cục bộ. Nếu `retrieval_engine` chưa khởi tạo xong hoặc bị `None`, `upload.py` gọi `process_and_index_document` với `client=None`. Hàm `indexing.py` tự mở một `QdrantClient(path=...)` thứ hai, dẫn đến lỗi `RuntimeError: Storage folder is already locked by another process`.
- **Giải pháp:**
  1. Trong `upload.py`: Tái sử dụng client từ `retrieval_engine.client`. Nếu engine/client chưa sẵn sàng, trả HTTP 503 ngay lập tức, không cố gắng tạo client mới.
  2. Cung cấp hướng dẫn khuyến nghị chuyển sang `qdrant_mode: "server"` (kết nối Qdrant container qua mạng) trong môi trường Docker và multi-worker.

#### ISSUE-RET-04: Nguy cơ Fallback ngầm sang Benchmark Corpus làm sai lệch Provenance (Approved Decision 3)
- **Vị trí:** `backend/app/retrieval/hybrid_search.py` & `backend/app/orchestration/request_flow.py`
- **Mức độ:** **P1 (Nguyên tắc bảo vệ dữ liệu)**
- **Phân tích:** Nếu người dùng đang học trong workspace `demo`, nếu workspace đó chưa có tài liệu hoặc tìm kiếm không khớp, hệ thống TUYỆT ĐỐI KHÔNG được tự ý tìm kiếm ngầm trên corpus `benchmark-v1` / `research`. Điều này sẽ vi phạm ranh giới cách ly dữ liệu và khiến sinh viên bị nhầm lẫn về nguồn gốc tài liệu môn học của mình.
- **Giải pháp:** Khi `workspace_id` được truyền vào, `retrieve()` chỉ được phép truy vấn trong đúng `workspace_id` đó. Nếu rỗng, trả về danh sách rỗng để tầng sinh trả lời (Track 2) đưa ra câu thông báo từ chối phù hợp.

#### ISSUE-RET-05: Nghẽn Event Loop khi Ingestion file PDF/DOCX dung lượng lớn
- **Vị trí:** `backend/app/api/routes/upload.py` (dòng 60–67)
- **Mức độ:** **P2 (Cần kiểm thử tải)**
- **Phân tích:** Mặc dù `upload.py` đã dùng `run_in_threadpool`, việc nạp tài liệu PDF dài ($> 50$ trang, $> 500$ chunks) trên CPU mất nhiều thời gian sinh embedding MiniLM.
- **Giải pháp:** Hỗ trợ cấu hình `indexing_batch_size: 64` khi có GPU CUDA; ghi log tiến độ indexing theo từng batch rõ ràng.

---

## 4. CÁC NHIỆM VỤ TRIỂN KHAI CHI TIẾT (IMPLEMENTATION TASKS)

| Mã Task | Tên nhiệm vụ cụ thể | File tác động | Phụ thuộc | Tiêu chí chấp nhận (Acceptance Criteria) |
|---|---|---|---|---|
| **RET-01** | Bổ sung hàm xóa document chuẩn vào `indexing.py` (`delete_document_from_index`) | `backend/app/retrieval/indexing.py` | Không | Nhận `document_id, corpus_id, workspace_id`. Dùng `Filter` xóa triệt để các point trong Qdrant. Gọi `invalidate_all_bm25_caches` và trả về số point đã xóa. |
| **RET-02** | Cài đặt Safe Staging & Replacement Lifecycle trong `ingestion_flow.py` | `backend/app/orchestration/ingestion_flow.py` | RET-01 | Thực hiện bóc tách và sinh vector trước trong bộ nhớ. Chỉ khi thành công mới xóa chunks cũ và upsert chunks mới. Nếu lỗi giữa chừng, abort và bảo toàn chunks cũ. Không còn chunk mồ côi. |
| **RET-03** | Khắc phục triệt để lỗi xóa file vật lý đa workspace trong `documents.py` | `backend/app/api/routes/documents.py` | RET-01 | Lưu và xóa file vật lý với định danh duy nhất `{workspace_id}_{document_id}_{filename}`. Xóa tài liệu ở Workspace A không làm mất file cùng tên ở Workspace B. |
| **RET-04** | Chuẩn hóa `upload.py` và xử lý an toàn Qdrant Client Lock | `backend/app/api/routes/upload.py` | RET-02 | Tái sử dụng client tập trung. Nếu engine chưa sẵn sàng, trả về HTTP 503 tường minh. Bắt lỗi định dạng file không hỗ trợ và file rỗng. |
| **RET-05** | Tối ưu hóa UI tải tài liệu trong `upload_ui.py` | `frontend/screens/upload_ui.py` | RET-03, RET-04 | Hiển thị danh sách tài liệu đã tải lên kèm dung lượng, số chunks, ngày tải; hỗ trợ nút Xóa tài liệu gọi `DELETE /api/documents/{id}`; hiển thị cảnh báo khi tài liệu bị trùng tên. |
| **RET-06** | Đảm bảo tính cô lập tuyệt đối của Workspace và nghiêm cấm Fallback ngầm | `backend/app/retrieval/hybrid_search.py` | Không | Truy vấn đúng scope `workspace_id`. Không tự động chuyển hướng tìm kiếm sang `benchmark-v1`/`research` khi workspace không có dữ liệu. |
| **RET-07** | Tối ưu hóa hiệu năng Ingestion & Batch Embedding | `backend/app/retrieval/indexing.py` | Không | Hỗ trợ cấu hình `batch_size: 64` trên CUDA; hiển thị progress log chi tiết. |
| **RET-08** | Xây dựng bộ Integration Tests toàn diện cho Document Lifecycle | `tests/integration/test_document_lifecycle.py` (TẠO MỚI) | RET-01–06 | Kiểm thử tự động: Upload PDF/DOCX $\to$ Tìm kiếm thấy $\to$ Re-upload file sửa đổi ít chunk hơn $\to$ Xác minh không còn chunk mồ côi $\to$ Kiểm tra abort khi lỗi $\to$ Xóa document $\to$ Xác minh cô lập giữa 2 workspace. |

---

## 5. PHÂN QUYỀN FILE & RANH GIỚI TRÁCH NHIỆM (FILE OWNERSHIP)

Tuân thủ nghiêm ngặt **Approved Decision 4**, Track 1 sở hữu độc quyền toàn bộ lát cắt Ingestion & Retrieval:

### File ĐƯỢC QUYỀN SỬA (Owned Files — Toàn quyền sửa và chịu trách nhiệm)
- `backend/app/retrieval/*` (`config.py`, `embeddings.py`, `hybrid_search.py`, `indexing.py`, `keyword_search.py`, `reranker.py`, `vector_search.py`, `vector_store.py`, `README.md`)
- `backend/app/orchestration/ingestion_flow.py` (Chủ trì luồng nạp)
- `backend/app/api/routes/upload.py` (Chủ trì API upload)
- `backend/app/api/routes/documents.py` (Chủ trì API quản lý tài liệu)
- `frontend/screens/upload_ui.py` (Chủ trì màn hình upload và quản lý tài liệu)
- `configs/retrieval.example.yaml`, `configs/retrieval.mvp.yaml`
- `tests/unit/test_embeddings.py`
- `tests/unit/test_hybrid_search.py`
- `tests/unit/test_keyword_search.py`
- `tests/unit/test_optimizations.py`
- `tests/unit/test_reranker.py`
- `tests/integration/test_retrieval_pipeline.py`
- `tests/integration/test_document_lifecycle.py` (TẠO MỚI)
- `tests/smoke_backend_integration.py`
- `tests/smoke_reranker.py`

### File CHỈ ĐƯỢC ĐỌC (Read-Only Files — Nghiêm cấm sửa)
- `backend/app/document_processing/*` (`chunker.py`, `document_parser.py`, `metadata.py`, `text_cleaner.py` - Module B1 đã đóng băng)
- `backend/app/generation/*` (Thuộc Track 2)
- `backend/app/orchestration/request_flow.py` (Thuộc Track 2)
- `backend/app/orchestration/intent_router.py` (Thuộc Track 2)
- `backend/app/core/config.py` (Thuộc Track 2)
- `backend/app/schemas/contracts.py` (Thuộc Track 2)
- `backend/app/main.py` (Thuộc Track 2)
- `frontend/app.py`, `frontend/api_client.py` (Thuộc Track 2)
- `frontend/screens/chat_ui.py`, `quiz_ui.py`, `revision_ui.py` (Thuộc Track 2)
- `docker-compose.yml`, `backend/Dockerfile`, `frontend/Dockerfile` (Thuộc Track 3)
- `evaluation/*` (Thuộc Track 4)
- `data/corpus/dataset_clean/chunks_all.jsonl` (Dữ liệu chuẩn - Cấm sửa)

### GIAO THỨC YÊU CẦU THAY ĐỔI LIÊN TRACK (Cross-Track Change Request Protocol - Approved Decision 8)
- Nếu Track 1 cần bổ sung trường mới trong `DocumentUploadResponse` hoặc cấu hình chung: Gửi yêu cầu RFC đến Track 2 (người sở hữu `contracts.py` và `config.py`) để bổ sung các trường tùy chọn (`Optional` có giá trị mặc định). Tuyệt đối không tự ý sửa file của Track 2.

### VÙNG NGHIÊM CẤM CHẠM VÀO (Forbidden Scope)
- Tuyệt đối không xóa hay rebuild bộ dữ liệu 5.941 chunks gốc của Qdrant trong thư mục `data/runtime/qdrant_local/`.
- Tuyệt đối không thay đổi model embedding hoặc reranker đã được phê duyệt.
- Tuyệt đối không sửa Docker hay Evaluation files.

---

## 6. THỨ TỰ THỰC HIỆN (EXECUTION ORDER)

```
Bước 1 (Độc lập — Hạ tầng lưu trữ & Indexing):
  RET-01 (delete_document_from_index trong indexing.py)
    └── RET-04 (Chuẩn hóa upload.py và kiểm soát client lock)
    └── RET-06 (Bảo đảm cô lập workspace trong hybrid_search.py)

Bước 2 (Tích hợp luồng nạp & Quản lý file):
  RET-02 (Safe Staging & Replacement Lifecycle trong ingestion_flow.py)
    └── RET-03 (Sửa xóa file vật lý theo workspace trong documents.py)
    └── RET-05 (Cập nhật giao diện upload_ui.py kết nối API)
    └── RET-07 (Tối ưu batch embedding)

Bước 3 (Kiểm chứng & Kiểm thử hồi quy):
  RET-08 (Viết và chạy test_document_lifecycle.py)
    └── Chạy lại toàn bộ 52 tests hiện có để bảo đảm Zero Regression.
```

---

## 7. KẾ HOẠCH KIỂM THỬ (TESTING PLAN)

### Lệnh chạy kiểm thử tự động
```powershell
# 1. Chạy regression test toàn bộ phân hệ Retrieval hiện có (bắt buộc 52/52 PASS)
$env:PYTHONPATH = "backend"
& C:\Users\USER\miniconda3\envs\dsa-rag-gpu\python.exe -m pytest `
  tests/unit/test_embeddings.py tests/unit/test_hybrid_search.py `
  tests/unit/test_keyword_search.py tests/unit/test_optimizations.py `
  tests/unit/test_reranker.py tests/integration/test_retrieval_pipeline.py -v `
  --basetemp "$env:USERPROFILE\pytest-dsa-temp-gpu" -p no:cacheprovider

# 2. Chạy test mới về vòng đời tài liệu (Document Lifecycle & Orphan Chunk Prevention)
& C:\Users\USER\miniconda3\envs\dsa-rag-gpu\python.exe -m pytest `
  tests/integration/test_document_lifecycle.py -v `
  --basetemp "$env:USERPROFILE\pytest-dsa-temp-gpu" -p no:cacheprovider

# 3. Kiểm thử tích hợp Backend Smoke Test
& C:\Users\USER\miniconda3\envs\dsa-rag-gpu\python.exe tests/smoke_backend_integration.py
```

### Tiêu chí kiểm thử bắt buộc trong `test_document_lifecycle.py`
1. **Test Upload:** Nạp file PDF và DOCX mẫu thành công, kiểm tra các points xuất hiện trong Qdrant với đúng `workspace_id`.
2. **Test Retrieval sau Upload:** Thực hiện truy vấn tìm kiếm, đảm bảo tìm thấy nội dung vừa nạp.
3. **Test Re-upload không còn chunk mồ côi:** Nạp file có 10 chunks $\to$ Re-upload cùng tên file nhưng nội dung chỉ còn 6 chunks $\to$ Xác minh tổng số chunks trong Qdrant giảm chính xác từ 10 xuống 6, không còn 4 chunk mồ côi.
4. **Test Rollback khi lỗi Ingestion:** Giả lập lỗi sinh embedding $\to$ Xác minh các chunks cũ của tài liệu vẫn được bảo toàn nguyên vẹn trong Qdrant.
5. **Test Xóa Document:** Gọi `DELETE /api/documents/{id}` $\to$ Xác minh points trong Qdrant bị xóa sạch và file vật lý trên đĩa bị xóa chính xác.
6. **Test Cô lập Workspace:** Hai workspace cùng upload file trùng tên `test.pdf` $\to$ Xóa ở Workspace 1 $\to$ Xác minh Workspace 2 vẫn còn nguyên cả chunks trong Qdrant lẫn file trên đĩa.

---

## 8. TIÊU CHUẨN HOÀN THÀNH (DEFINITION OF DONE - DoD)

- [ ] Hàm `delete_document_from_index()` hoạt động ổn định và có unit test riêng.
- [ ] Safe Staging & Replacement Lifecycle hoạt động chuẩn xác: Re-upload dọn sạch chunk mồ côi; lỗi giữa chừng tự động rollback an toàn.
- [ ] Endpoint `DELETE /api/documents/{id}` chỉ xóa đúng file vật lý thuộc sở hữu của workspace đó.
- [ ] `upload_ui.py` hiển thị trực quan danh sách tài liệu, hỗ trợ upload và xóa tài liệu mượt mà.
- [ ] Không còn lỗi crash Qdrant file lock khi client local được gọi.
- [ ] Không có hiện tượng fallback ngầm sang benchmark corpus khi user workspace rỗng.
- [ ] Toàn bộ 5.941 chunks dữ liệu gốc của môn DSA được giữ nguyên vẹn.
- [ ] Tất cả các bài test (52 test cũ + các test mới) đều đạt 100% PASS trên môi trường `dsa-rag-gpu`.
- [ ] `python -m pip check` báo cáo *No broken requirements found*.
- [ ] Không có merge conflict với `main`.

---

## 9. QUY TRÌNH GIT (GIT WORKFLOW)

- **Tên nhánh:** `feat/retrieval-fixes` (tạo từ nhánh `docs/parallel-workstream-plans` hoặc `main` sau khi merge).
- **Quy ước Commit:** Tuân thủ Conventional Commits: `feat(retrieval): ...`, `fix(retrieval): ...`, `test(retrieval): ...`.
- **Yêu cầu PR:** Mô tả rõ các hàm lifecycle đã thêm, đính kèm kết quả chạy `pytest` và log kiểm thử orphan chunks.
