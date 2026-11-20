# KẾ HOẠCH TRIỂN KHAI WORKSTREAM 2 — GENERATION, PROMPTS & RAG ENGINE FIX

> **Mã tài liệu:** `PLAN-02-GENERATION`  
> **Nhánh phát triển (Target Branch):** `feat/generation-fixes`  
> **Commit gốc chuẩn (Baseline Commit):** `8c48d4daf81af8511c93cd0718fc40dbe40536ad` (HEAD chứa PR #5 Retrieval và PR #6 Week 5 Integration)  
> **Chủ trì (Track Lead):** Senior LLM & RAG Generation Engineer (Phụ trách: C2 + A1)  
> **Trạng thái:** KẾ HOẠCH HOÀN THIỆN ĐÃ PHÊ DUYỆT — NGHIÊM CẤM SỬA CODE TRONG BƯỚC NÀY  

---

## 1. TỔNG QUAN & SNAPSHOT DỰ ÁN (PROJECT SNAPSHOT)

- **Repository:** `DSA-Document-Assistant-Using-Hybrid-RAG`
- **Branch làm việc tài liệu:** `docs/parallel-workstream-plans`
- **Baseline Commit chuẩn:** `8c48d4daf81af8511c93cd0718fc40dbe40536ad`
- **Hiện trạng phân hệ Generation sau PR #6:**
  - **LLM Client:** `backend/app/generation/llm_client.py` đã hỗ trợ 2 provider: `gemini` (qua `google.generativeai`) và `openai` (qua `openai`).
  - **RAG Engine:** `backend/app/generation/rag_engine.py` đã có hàm `build_answer()` liên kết kiểm tra từ chối (abstention), định dạng prompt và gọi LLM, có chế độ fallback Extractive khi không có API key.
  - **Prompting:** `backend/app/generation/prompts/dsa_prompts.py` định nghĩa system prompt cho 3 ý định: Hỏi đáp (`qa`), Trắc nghiệm (`quiz`), Ôn tập (`revision`).
  - **Trích dẫn (Citations):** `backend/app/generation/citations.py` tạo `Citation` objects từ danh sách chunks.
  - **Từ chối (Abstention):** `backend/app/generation/abstention.py` kiểm tra ngưỡng điểm tương quan trước khi gọi LLM.
  - **Orchestration & Router:** `backend/app/orchestration/request_flow.py` và `intent_router.py` đã nối vào `RetrievalEngine` trong FastAPI lifespan.
  - **Giao diện sinh viên (Student Frontend):** Streamlit app đã có khung sườn điều hướng nhưng các màn hình `quiz_ui.py` và `revision_ui.py` chưa được kết nối đầy đủ với backend API.

---

## 2. KIẾN TRÚC & PHẠM VI DỌC CỦA TRACK 2 (VERTICAL SLICE SCOPE)

Track 2 sở hữu trọn vẹn **Trải nghiệm Học tập của Sinh viên & Phân hệ Sinh câu trả lời**: Từ luồng điều phối yêu cầu (`request_flow.py`), phân loại ý định (`intent_router.py`), toàn bộ lõi LLM/Prompt/Generation (`generation/*`), các API endpoints phục vụ học tập (`chat.py`, `quiz.py`, `revision.py`), cấu hình ứng dụng (`config.py`), hợp đồng dữ liệu (`contracts.py`), ứng dụng FastAPI (`main.py`), đến toàn bộ giao diện tương tác học tập của sinh viên trên Streamlit (`app.py`, `api_client.py`, `chat_ui.py`, `quiz_ui.py`, `revision_ui.py`).

```
[Student Browser]
       │
       ▼
[frontend/app.py & api_client.py]
  ├── chat_ui.py: Hỏi đáp kiến thức DSA theo tài liệu
  ├── quiz_ui.py: Tạo & làm bài trắc nghiệm kiến thức
  └── revision_ui.py: Tạo dàn bài ôn tập & Flashcards
       │
       ▼ (REST API: /api/chat, /api/quiz, /api/revision)
[backend/app/main.py] (FastAPI Routers)
       │
       ▼
[orchestration/request_flow.py] ──► [orchestration/intent_router.py]
       │
       ├── Gọi RetrievalEngine.retrieve() (Interface của Track 1)
       │     └── Kiểm soát scope workspace: TUYỆT ĐỐI KHÔNG fallback ngầm
       │
       ▼ (Evidence Chunks)
[generation/rag_engine.py: build_answer()]
  ├── 1. should_abstain(evidences, method) -> Nhận diện đúng thang điểm RRF/Rerank
  ├── 2. format_evidence_for_prompt() -> [1], [2] Citation tagging
  ├── 3. llm_client.py: generate_answer() (Hỗ trợ LLM_BASE_URL cho Gateway)
  ├── 4. filter_actual_citations() -> Chỉ giữ trích dẫn được LLM sử dụng
  ├── 5. calculate_answer_confidence() -> Tính toán điểm tin cậy grounded thực tế
  └── 6. Extractive Fallback -> Hoạt động trơn tru khi không có API key
```

---

## 3. DANH SÁCH VẤN ĐỀ ĐÃ XÁC NHẬN (CONFIRMED ISSUES)

### LỖI ĐÃ XÁC NHẬN (VERIFIED BUGS)

#### ISSUE-GEN-01: Lỗi nhầm lẫn thang điểm (Scale Conflation) trong `abstention.py` — Chế độ `hybrid_rrf` bị từ chối 100%
- **Vị trí:** `backend/app/generation/abstention.py` (dòng 11 và dòng 38–42)
- **Mức độ:** **P0 (Lỗi nghiêm trọng nhất của hệ thống sinh)**
- **Xác minh thực tế (Empirical Proof):**
  Trong `abstention.py`:
  ```python
  _SCORE_THRESHOLD = 0.25
  scored = [ev for ev in evidences if ev.get("score") is not None]
  if scored:
      best_score = max(ev["score"] for ev in scored)
      if best_score < _SCORE_THRESHOLD:
          return True  # Từ chối trả lời!
  ```
  Khi hệ thống chạy với phương thức `hybrid_rrf` (hoặc khi tự động fallback từ Reranker do CUDA OOM):
  Điểm RRF tính theo $\sum \frac{1}{60 + r_i}$. Một chunk đứng top 1 ở cả BM25 và Dense có điểm tối đa:
  $$\text{Score}_{\max} = \frac{1}{61} + \frac{1}{61} \approx 0.0328$$
  Vì $0.0328 < 0.25$, **MỌI TRUY VẤN CHẠY BẰNG RRF ĐỀU BỊ TỪ CHỐI TRẢ LỜI 100%!** Kể cả khi tài liệu chứa lời giải hoàn hảo, hệ thống vẫn trả về thông báo từ chối.
- **Giải pháp:** Cập nhật `should_abstain(evidences, method="hybrid_rerank")` nhận diện phương thức retrieval:
  - Nếu `method == "hybrid_rerank"`: Ngưỡng `0.25` (Cross-Encoder score).
  - Nếu `method == "hybrid_rrf"`: Ngưỡng `0.015` (RRF score tương đương Top-5 ở ít nhất một bảng xếp hạng).
  - Nếu `method == "dense"`: Ngưỡng `0.40` (Cosine similarity).
  - Nếu `method == "bm25"`: Ngưỡng `5.0` (BM25 raw score).

#### ISSUE-GEN-02: Điểm tin cậy bị gán cứng giả lập (Hardcoded Fake Confidence Score)
- **Vị trí:** `backend/app/generation/rag_engine.py` (dòng 68 và dòng 71)
- **Mức độ:** **P0 (Vi phạm tính trung thực khoa học)**
- **Xác minh thực tế:**
  ```python
  if not llm_answer:
      answer = _extractive_synthesis(...)
      confidence = 0.70  # Gán cứng!
  else:
      answer = llm_answer
      confidence = 0.92  # Gán cứng!
  ```
  Điểm tin cậy hoàn toàn là con số ảo, không phản ánh chất lượng câu trả lời hay độ tin cậy của tài liệu.
- **Giải pháp:** Xây dựng hàm `calculate_answer_confidence()` tính toán điểm thực tế dựa trên:
  1. Tỷ lệ trích dẫn được viện dẫn (`cited_evidences / total_evidences`).
  2. Điểm tương quan trung bình đã chuẩn hóa của các đoạn trích được dùng.
  3. Mức phạt nếu văn bản câu trả lời chứa các cụm từ thể hiện sự mơ hồ/không chắc chắn ("tài liệu không nêu", "có thể là").
  4. Trả về giá trị thực trong khoảng $[0.1, 0.98]$.

#### ISSUE-GEN-03: Thiếu cấu hình `LLM_BASE_URL` cho API Gateway trường cấp
- **Vị trí:** `backend/app/core/config.py` (dòng 23–26) & `backend/app/generation/llm_client.py` (dòng 30)
- **Mức độ:** **P1 (Chặn tích hợp môi trường trường học)**
- **Xác minh thực tế:** Trong `llm_client.py`, `OpenAI(api_key=api_key)` không nhận `base_url`. Trường học hoặc lab thường cung cấp API proxy/gateway (OpenRouter, LiteLLM, vLLM nội bộ). Do thiếu cấu hình này, toàn bộ request bị gửi thẳng ra máy chủ OpenAI quốc tế và thất bại khi mạng bị chặn.
- **Giải pháp:** Bổ sung `LLM_BASE_URL: Optional[str] = None` vào `Settings` (`config.py`) và truyền `base_url=settings.LLM_BASE_URL` khi khởi tạo `OpenAI()`.

#### ISSUE-GEN-04: Trích dẫn mù (Blind Citations) — Trả về toàn bộ chunks thay vì chunks được dùng
- **Vị trí:** `backend/app/generation/citations.py` (dòng 10–26) & `rag_engine.py` (dòng 51)
- **Mức độ:** **P1 (Trích dẫn sai lệch nguồn gốc)**
- **Xác minh thực tế:** Hệ thống biến toàn bộ danh sách 5 chunk đầu vào thành `citations` trả về giao diện, bất kể LLM có sử dụng chúng hay không.
- **Giải pháp:** Bổ sung bước lọc bằng regex tìm các ký hiệu `[1]`, `[2]` trong nội dung trả lời của LLM; danh sách `citations` cuối cùng chỉ giữ lại những đoạn trích thực sự được viện dẫn.

---

### NGUY CƠ CẦN KIỂM THỬ & VẬN HÀNH (LIKELY RISKS / NEEDS TEST)

#### ISSUE-GEN-05: Tuyệt đối Không Fallback Ngầm sang Benchmark Corpus (Approved Decision 3)
- **Vị trí:** `backend/app/orchestration/request_flow.py`
- **Mức độ:** **P1 (Ranh giới dữ liệu người dùng)**
- **Phân tích:** Khi sinh viên học trong workspace `demo`, nếu tài liệu của sinh viên không có câu trả lời, hệ thống TUYỆT ĐỐI KHÔNG được tự động chuyển sang tìm kiếm trong corpus `benchmark-v1`/`research`. Việc fallback ngầm sẽ tạo ra ảo giác rằng tài liệu của sinh viên có chứa thông tin đó.
- **Giải pháp:** Khi workspace không có tài liệu hoặc điểm số dưới ngưỡng: Trả về câu thông báo rõ ràng:
  - Nếu workspace rỗng: *"Workspace chưa có tài liệu nào. Vui lòng tải tài liệu môn học (PDF/DOCX) để bắt đầu."*
  - Nếu điểm dưới ngưỡng: *"⚠️ Không tìm thấy nội dung phù hợp trong tài liệu của bạn cho câu hỏi này."*

#### ISSUE-GEN-06: Nguy cơ Lộ API Keys qua Logging / Pydantic Models
- **Vị trí:** `backend/app/core/config.py` & `backend/app/generation/llm_client.py`
- **Mức độ:** **P2 (An toàn thông tin)**
- **Giải pháp:** Sử dụng `SecretStr` cho các trường API Key trong `Settings`. Tuyệt đối không log chuỗi key ra màn hình console hay file log.

#### ISSUE-GEN-07: Xử lý Lỗi Rate Limit (HTTP 429) và Timeout của Nhà cung cấp LLM
- **Mức độ:** **P2 (Độ tin cậy vận hành)**
- **Giải pháp:** Cài đặt cơ chế retry 2 lần với exponential backoff khi gặp lỗi 429 hoặc timeout; nếu vẫn thất bại, tự động chuyển sang Extractive Fallback một cách êm thuận kèm cảnh báo trên UI.

---

## 4. CÁC NHIỆM VỤ TRIỂN KHAI CHI TIẾT (IMPLEMENTATION TASKS)

| Mã Task | Tên nhiệm vụ cụ thể | File tác động | Phụ thuộc | Tiêu chí chấp nhận (Acceptance Criteria) |
|---|---|---|---|---|
| **GEN-01** | Khắc phục lỗi scale trong `abstention.py` theo từng retrieval method | `backend/app/generation/abstention.py` | Không | `should_abstain()` nhận `method: str`. Đối với `hybrid_rrf`, ngưỡng là `0.015`. Query với điểm RRF hợp lệ ($0.02\text{--}0.03$) KHÔNG bị từ chối sai. |
| **GEN-02** | Xây dựng công thức tính `confidence_score` grounded thực tế | `backend/app/generation/rag_engine.py` | GEN-01 | Loại bỏ hoàn toàn số gán cứng `0.70` và `0.92`. Điểm tin cậy được tính toán từ tỷ lệ citation và điểm relevance chuẩn hóa, nằm trong $[0.1, 0.98]$. |
| **GEN-03** | Hỗ trợ cấu hình `LLM_BASE_URL` cho API Gateway trường cấp | `backend/app/core/config.py`, `backend/app/generation/llm_client.py` | Không | Nạp `LLM_BASE_URL` từ file `.env`. Khởi tạo `OpenAI(api_key=..., base_url=...)` chuẩn xác khi có biến cấu hình gateway. |
| **GEN-04** | Lọc trích dẫn thực tế theo nhãn `[i]` trong câu trả lời | `backend/app/generation/citations.py` | Không | Trích xuất các nhãn `[i]` trong câu trả lời; danh sách `citations` trả về chỉ chứa các chunk được viện dẫn thực tế. |
| **GEN-05** | Triển khai hoàn chỉnh Endpoint `POST /api/quiz` và UI Trắc nghiệm | `backend/app/api/routes/quiz.py`, `backend/app/generation/quiz_generator.py`, `frontend/screens/quiz_ui.py` | GEN-02 | Sinh bài trắc nghiệm DSA (4 lựa chọn, đáp án, giải thích) từ tài liệu; UI Streamlit hiển thị trắc nghiệm tương tác, chấm điểm trực tiếp. Hỗ trợ extractive fallback. |
| **GEN-06** | Triển khai hoàn chỉnh Endpoint `POST /api/revision` và UI Ôn tập | `backend/app/api/routes/revision.py`, `backend/app/generation/revision_generator.py`, `frontend/screens/revision_ui.py` | GEN-02 | Sinh dàn bài tóm tắt và câu hỏi ôn tập theo tài liệu; UI Streamlit hiển thị flashcards/bản tóm tắt trực quan kèm trích dẫn. |
| **GEN-07** | Chuẩn hóa `main.py`, `request_flow.py` và cấm fallback ngầm sang benchmark | `backend/app/main.py`, `backend/app/orchestration/request_flow.py` | GEN-05, GEN-06 | Đăng ký đầy đủ router `chat`, `quiz`, `revision`. Không fallback ngầm sang `research` khi workspace người dùng rỗng. |
| **GEN-08** | Cập nhật Client API và Giao diện Chat của Sinh viên | `frontend/api_client.py`, `frontend/screens/chat_ui.py`, `frontend/app.py` | GEN-01–07 | Streamlit hiển thị câu trả lời kèm `confidence_score` thực tế, trích dẫn rõ ràng, cảnh báo khi không tìm thấy tài liệu hoặc chạy extractive fallback. |
| **GEN-09** | Xây dựng bộ Unit & Integration Tests cho toàn bộ tầng Generation | `tests/unit/test_generation_fixes.py` (TẠO MỚI) | GEN-01–08 | 100% tests PASS sử dụng Mock LLM: Kiểm chứng scale RRF, tính confidence, trích dẫn, Quiz, Revision, và Extractive Fallback. |

---

## 5. PHÂN QUYỀN FILE & RANH GIỚI TRÁCH NHIỆM (FILE OWNERSHIP)

Tuân thủ nghiêm ngặt **Approved Decision 4**, Track 2 sở hữu độc quyền toàn bộ lát cắt Nghiệp vụ Sinh câu trả lời, Điều phối Request, và Giao diện Sinh viên:

### File ĐƯỢC QUYỀN SỬA (Owned Files — Toàn quyền sửa và chịu trách nhiệm)
- `backend/app/generation/*` (`abstention.py`, `citations.py`, `llm_client.py`, `quiz_generator.py`, `rag_engine.py`, `revision_generator.py`, `prompts/*`, `README.md`)
- `backend/app/orchestration/request_flow.py`
- `backend/app/orchestration/intent_router.py`
- `backend/app/api/routes/chat.py`
- `backend/app/api/routes/quiz.py`
- `backend/app/api/routes/revision.py`
- `backend/app/main.py`
- `backend/app/core/config.py`
- `backend/app/schemas/contracts.py`
- `configs/generation.example.yaml`
- `frontend/app.py`
- `frontend/api_client.py`
- `frontend/screens/chat_ui.py`
- `frontend/screens/quiz_ui.py`
- `frontend/screens/revision_ui.py`
- `tests/unit/test_generation_fixes.py` (TẠO MỚI)

### File CHỈ ĐƯỢC ĐỌC (Read-Only Files — Nghiêm cấm sửa)
- `backend/app/retrieval/*` (Thuộc Track 1 — Tuyệt đối không can thiệp thuật toán tìm kiếm)
- `backend/app/orchestration/ingestion_flow.py` (Thuộc Track 1)
- `backend/app/api/routes/upload.py` (Thuộc Track 1)
- `backend/app/api/routes/documents.py` (Thuộc Track 1)
- `frontend/screens/upload_ui.py` (Thuộc Track 1)
- `backend/app/document_processing/*` (Module B1 đã đóng băng)
- `configs/retrieval.*`
- `data/corpus/*`
- `docker-compose.yml`, `backend/Dockerfile`, `frontend/Dockerfile` (Thuộc Track 3)
- `evaluation/*` (Thuộc Track 4)

### GIAO THỨC YÊU CẦU THAY ĐỔI LIÊN TRACK (Cross-Track Change Request Protocol - Approved Decision 8)
- Khi Track 1 hoặc Track 3 cần thêm cấu hình trong `config.py` hoặc thêm trường vào `contracts.py`: Họ sẽ tạo Change Request / Issue. Track 2 sẽ thực hiện cập nhật với nguyên tắc giữ tương thích ngược (`Optional` với default value).

### VÙNG NGHIÊM CẤM CHẠM VÀO (Forbidden Scope)
- Tuyệt đối không can thiệp vào các phương thức tìm kiếm trong `retrieval/`.
- Tuyệt đối không sửa mã nguồn quản lý upload tài liệu (`upload.py`, `documents.py`, `upload_ui.py`).
- Tuyệt đối không chỉnh sửa Dockerfile hoặc Docker Compose.

---

## 6. THỨ TỰ THỰC HIỆN (EXECUTION ORDER)

```
Bước 1 (Sửa lỗi cốt lõi & Cấu hình):
  GEN-01 (Sửa abstention scale cho RRF trong abstention.py)
    └── GEN-02 (Tính confidence thực tế trong rag_engine.py)
    └── GEN-03 (Bổ sung LLM_BASE_URL trong config.py và llm_client.py)
    └── GEN-04 (Lọc trích dẫn theo nhãn trong citations.py)

Bước 2 (Mở rộng tính năng Quiz, Revision & Orchestration):
  GEN-05 (Hoàn thiện endpoint quiz.py và quiz_ui.py)
    └── GEN-06 (Hoàn thiện endpoint revision.py và revision_ui.py)
    └── GEN-07 (Đăng ký router trong main.py và chặn fallback ngầm trong request_flow.py)
    └── GEN-08 (Hoàn thiện giao diện Streamlit: chat_ui.py, app.py, api_client.py)

Bước 3 (Kiểm thử & Đóng gói):
  GEN-09 (Viết và chạy test_generation_fixes.py với Mock LLM)
    └── Kiểm thử giao diện Streamlit nội bộ qua Mock API.
```

---

## 7. KẾ HOẠCH KIỂM THỬ (TESTING PLAN)

### Lệnh chạy kiểm thử tự động
```powershell
# 1. Chạy unit tests cho các bản sửa lỗi của Generation (Dùng Mock LLM, không tốn tiền API)
$env:PYTHONPATH = "backend"
& C:\Users\USER\miniconda3\envs\dsa-rag-gpu\python.exe -m pytest `
  tests/unit/test_generation_fixes.py -v `
  --basetemp "$env:USERPROFILE\pytest-dsa-temp-gpu" -p no:cacheprovider

# 2. Kiểm thử nhanh scale RRF không bị từ chối sai
& C:\Users\USER\miniconda3\envs\dsa-rag-gpu\python.exe -c "
import sys; sys.path.insert(0, 'backend')
from app.generation.abstention import should_abstain
# Kiểm tra case RRF điểm 0.025 không bị từ chối
assert not should_abstain([{'chunk_id': 'c1', 'text': 'Binary Search Tree definition', 'score': 0.025}], method='hybrid_rrf')
# Kiểm tra case RRF điểm cực thấp (< 0.015) bị từ chối chính xác
assert should_abstain([{'chunk_id': 'c2', 'text': 'irrelevant text', 'score': 0.005}], method='hybrid_rrf')
print('Abstention Scale Verification: PASS')
"
```

### Tiêu chí kiểm thử bắt buộc trong `test_generation_fixes.py`
1. **Test Abstention Scale:** Đảm bảo `hybrid_rrf` điểm $0.025$ không bị từ chối; `hybrid_rerank` điểm $0.15$ bị từ chối; tài liệu rỗng bị từ chối.
2. **Test Confidence Calculation:** `confidence_score` thay đổi phụ thuộc vào mức độ tương quan và tỷ lệ trích dẫn thực tế; không còn xuất hiện giá trị cố định `0.70` hay `0.92`.
3. **Test Citation Filtering:** Câu trả lời chỉ viện dẫn `[1]` $\to$ Chỉ trả về 1 citation tương ứng, loại bỏ các chunks không được dùng.
4. **Test Quiz Generation:** Mock LLM trả về cấu trúc câu hỏi trắc nghiệm hợp lệ; Extractive fallback tạo câu hỏi tóm tắt khi không có API key.
5. **Test Revision Generation:** Sinh nội dung ôn tập theo tài liệu hợp lệ; Extractive fallback hoạt động ổn định.
6. **Test LLM Base URL:** Khởi tạo `OpenAI` client với `base_url` tùy chỉnh chính xác khi có cấu hình.
7. **Test Chặn Fallback Ngầm:** Khi workspace của người dùng không có tài liệu, hệ thống trả về thông báo yêu cầu tải tài liệu, không tự động query sang `research`.

---

## 8. TIÊU CHUẨN HOÀN THÀNH (DEFINITION OF DONE - DoD)

- [ ] Lỗi từ chối sai của `hybrid_rrf` được khắc phục 100%, tỷ lệ false refusal của RRF bằng 0%.
- [ ] Điểm tin cậy `confidence_score` phản ánh đúng mức độ grounding khoa học của câu trả lời.
- [ ] Hỗ trợ đầy đủ biến môi trường `LLM_BASE_URL` cho proxy/gateway trường cấp.
- [ ] Danh sách `citations` trả về phản ánh chính xác các nguồn được viện dẫn thực tế.
- [ ] Các tính năng Chat, Quiz, Revision hoạt động đầy đủ cả ở API và giao diện Streamlit.
- [ ] Cơ chế Extractive Fallback hoạt động mượt mà khi không có API key hoặc khi mạng gặp sự cố.
- [ ] Không có hiện tượng fallback ngầm sang corpus benchmark khi workspace của người dùng rỗng.
- [ ] Tuyệt đối không log hoặc làm rò rỉ API key ra màn hình console hay file log.
- [ ] Toàn bộ unit tests chạy 100% PASS mà không cần mạng Internet hay API key thật.
- [ ] Không có merge conflict với `main`.

---

## 9. QUY TRÌNH GIT (GIT WORKFLOW)

- **Tên nhánh:** `feat/generation-fixes` (tạo từ nhánh `docs/parallel-workstream-plans` hoặc `main`).
- **Quy ước Commit:** Tuân thủ Conventional Commits: `feat(gen): ...`, `fix(gen): ...`, `test(gen): ...`.
- **Yêu cầu PR:** Đính kèm log kiểm thử unit test và ảnh chụp màn hình tương tác Chat/Quiz/Revision trên Streamlit.
