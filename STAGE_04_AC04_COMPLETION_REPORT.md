# STAGE 04 AC-04 COMPLETION REPORT — LIVE GROQ API & HTTP END-TO-END EVALUATION

> **Dự án:** DSA Document Assistant Using Hybrid RAG  
> **Phạm vi:** Nhiệm vụ bổ sung hoàn tất khoảng trống **Stage 04 `AC-04`** *(Supplementary Closure Task — KHÔNG phải Stage 06)*  
> **Workspace:** `D:\DSA-Document-Assistant-Using-Hybrid-RAG`  
> **Branch:** `feat/final-readiness` | **Accepted Base HEAD:** `139957c0d6974df273c86ac2029be1c8fb7d158e` | **Immutable `main`:** `b84f09450f213b03ad57ef8b388f482ee0f0eaa2`  
> **Trạng thái lịch sử Stage 04 `AC-04` (bảo toàn):** `PARTIAL` *(Production Groq API & HTTP E2E trước đây ở trạng thái `BLOCKED`)*  
> **Mô hình Groq thực tế được cấu hình:** `openai/gpt-oss-120b` (`https://api.groq.com/openai/v1`, `reasoning_effort = "low"`)  
> **Quản trị đánh giá:** `human_validated = false`, `human_approval_status = "DRAFT"`  
> **Ngày thực hiện:** 2026-11-28 / 2026-11-29

---

## 1. Executive Summary

Nhiệm vụ bổ sung đã giải tỏa hoàn toàn nút thắt `BLOCKED` của **Stage 04 `AC-04`** bằng cách thực thi thực nghiệm **Live Groq API (`openai/gpt-oss-120b`)** và **HTTP End-to-End thật** trên mã nguồn FastAPI **không chỉnh sửa một dòng nào**, chạy trong **môi trường đánh giá cô lập** (`127.0.0.1:8015`, bản sao kiểm chứng SHA-256 của chỉ mục `data/runtime/qdrant_eval`, tách biệt hoàn toàn với triển khai production và không động chạm `data/runtime/qdrant_local`):

- **Kết nối Groq & Tiêu thụ Token thực tế (`GATE-01` — PASS):** Xác thực thành công `openai/gpt-oss-120b` (HTTP `200`, `957,2 ms`, hạn mức quan sát từ header: `1.000 RPD` / `8.000 TPM`, khu vực `bom`). Toàn đợt thực nghiệm thực thi tổng cộng **`166` lượt yêu cầu HTTP/API** — gồm **`157` lượt thuộc 8 nhóm thực nghiệm chính** (ghi nhận trong `token_usage_summary.csv`) cộng **`9` lượt kiểm thử HTTP edge cases** (`edge_cases_summary.csv`, `0` lệnh gọi Groq) — tách biệt hoàn toàn với **`90` lệnh gọi API Groq thực tế ở tầng provider** (`84` HTTP `200`, `6` HTTP `429 rate_limit` ở tầng provider Groq), tiêu thụ **`58.116` prompt tokens + `58.975` completion tokens = `117.091` total tokens** (trong hạn mức miễn phí, không phát sinh chi phí ngoài phê duyệt).
- **Phát hiện Thực nghiệm Trọng yếu trên Ứng dụng Production thật (`GATE-02, GATE-03, GATE-04`):** Việc chạy HTTP End-to-End thật đã phơi bày **4 lỗi kiến trúc/cấu hình production thực sự** mà thí nghiệm retrieval-only hay Controlled Agent ở Stage 04 không thể nhìn thấy:
  1. **Từ chối dương tính giả nghiêm trọng trong `/api/chat` (`82,5%` = `33/40` câu hỏi DSA hợp lệ):** Dù tầng truy xuất `A4` đạt `best_score > 0,90`, bộ lọc neo từ khóa cứng trong `backend/app/orchestration/intent_router.py` và `chat_retrieval.py` loại bỏ toàn bộ đoạn trích với lý do `reason=off_topic` trước khi gọi Groq. Kết quả: cả Config A (`pool=10`) lẫn Config B (`pool=30`) đều đạt **`100%` HTTP 200 transport**, nhưng chỉ có **`4/40 (10,0%)` sinh LLM trực tiếp (`mode=llm`)**, **`3/40 (7,5%)` rơi về `extractive`**, và **`33/40 (82,5%)` bị `abstained`**, kéo điểm trung bình toàn tập 40 câu xuống `Correctness = 0,1000` (Config A) và `0,0938` (Config B) — trong khi trên đúng nhóm nhỏ **`live_llm_only (n=4)`**, Config A đạt `Correctness = 0,9375` và `Completeness = 0,9167`!
  2. **Lỗi cắt ngắn văn bản Chat trong `llm_client.py` (`finish_reason == "length"` → `truncated_json`):** Trong Config B, 1 câu trả lời chat dài chạm giới hạn `max_tokens = 1.024` bị ném `LLMError("truncated_json")` (dù là plain-text chat không phải JSON) và bị hủy bỏ để rơi về `extractive`.
  3. **Xung đột hạn mức `8.000 TPM` nội bộ trong `/api/quiz` & bộ lọc chủ đề cứng:** Cả `8/8` yêu cầu Quiz chuẩn hóa đều trả về **ứng dụng HTTP `200`**, nhưng chỉ `5/8` chủ đề sinh được tổng cộng `15` câu hỏi trắc nghiệm từ Groq (`QZ_06` đạt `5/5` sạch lỗi; `QZ_01=2/5` gặp `invalid_output`; `QZ_02=2/5, QZ_04=3/5, QZ_07=3/5` gặp **lỗi `429 rate_limit` ở tầng provider Groq** do mỗi request `/api/quiz` bắn liên tiếp nhiều lời gọi tiêu tốn `8.093–9.142` tokens > `8.000 TPM`), `3/8` chủ đề abstained (`QZ_03, QZ_05, QZ_08`), và chủ đề gốc `QZ_03` (`"Tree, BST & AVL"`) bị `QuizRequest` từ chối HTTP `422` vì chứa dấu phẩy.
  4. **Lỗ hổng khớp mục tiêu dài trong `/api/roadmap` (`0/4` sinh lộ trình):** Cả `4/4` kịch bản lộ trình dạng câu mô tả của Stage 04 (`RM_01..RM_04`) đều trả về `HTTP 200` + `mode=abstained` (`items=0`, `4,48–4,87s`) do `curriculum_inventory` lọc tiêu đề theo chuỗi nguyên văn không khớp câu mục tiêu dài.
- **Đánh giá lại Stage 04 `AC-04`:** Toàn bộ **`6/6` cổng nghiệm thu bổ sung (`GATE-01`–`GATE-06`) đều đạt `PASS`** xét trên **tiêu chí hoàn thành thực nghiệm đánh giá** (không còn blocker `BLOCKED`, mọi phép đo Live Groq & HTTP E2E đã hoàn tất, có kiểm chứng checksum và tái lập). Tuy nhiên, **khuyến nghị rõ ràng:** tách biệt **Đánh giá hoàn tất (`AC-04` Evaluation Gate → `PASS` sau bổ sung, giữ nguyên lịch sử gốc Stage 04 `PARTIAL`)** khỏi **Sẵn sàng triển khai Production (`NOT PRODUCTION-READY` do 4 lỗi ứng dụng thật nêu trên và `human_validated = false`)**.

---

## 2. Isolated Environment, Security & Configuration Boundary

| Hạng mục | Trạng thái xác minh | Ghi chú kỹ thuật |
|---|---|---|
| Git Baseline | **VERIFIED** | Branch `feat/final-readiness`, `HEAD = 139957c0...`, `main = b84f0945...`, `0` sửa đổi mã nguồn `backend/` hoặc `frontend/`. |
| Bảo mật Credential | **VERIFIED** | `backend/.env` được `.gitignore` bảo vệ (`git check-ignore = True`), không in/log/commit API key; quét toàn bộ artifacts `ac04_completion/` ghi nhận `0` secret hits. |
| Cô lập Qdrant | **VERIFIED** | Tạo bản sao độc lập `data/runtime/ac04_eval_runtime/qdrant_eval_copy` kiểm chứng SHA-256 khớp `3/3` tệp với `data/runtime/qdrant_eval`; xác nhận bản gốc `qdrant_eval` và `qdrant_local` (`0` points) bất biến sau toàn bộ thử nghiệm. |
| Phạm vi môi trường | **ISOLATED EVAL** | Chạy mã nguồn FastAPI thật (`uvicorn` cổng `127.0.0.1:8015`, `RETRIEVAL_CONFIG_PATH`, `SESSION_RUNTIME_DIR`) trên chỉ mục đánh giá cô lập — **không phải** hệ thống production đã triển khai công khai. |
| Ranh giới cấu hình thật | **DISCLOSED** | Config A (`candidate_top_k = 10`, `final_top_k = 5`) phản ánh `configs/retrieval.mvp.yaml`; Config B nâng **`candidate_top_k = 30`** qua `RETRIEVAL_CONFIG_PATH`. Backend hiện **KHÔNG hỗ trợ** `Top-K > 5` (`MAX_EVIDENCES = 5` gán cứng) hay `Context Budget = 2.048 tokens` — tuyệt đối không tuyên bố đã áp dụng hai tùy chọn này trong HTTP `/api/chat`. |
| Quan sát Verbatim Provider | **VERIFIED** | Hook quan sát chỉ-đọc ở tầng launcher ghi nhận nguyên văn `system_prompt`, `user_prompt` (`## CONTEXT` thật gửi lên Groq), `prompt_tokens`, `completion_tokens`, `total_tokens`, HTTP status và header `x-ratelimit-*` mà không sửa `backend/` và không tái tạo ngữ cảnh giả. |

---

## 3. Live Groq Connectivity & Token Consumption

**Artifacts:** [smoke_test_report.json](evaluation/artifacts/ac04_completion/smoke_test_report.json), [token_usage_summary.csv](evaluation/artifacts/ac04_completion/token_usage_summary.csv).

- **Smoke Test kết nối (`openai/gpt-oss-120b`):** HTTP `200`, `957,18 ms`, `prompt = 97`, `completion = 32`, `total = 129` tokens; rate-limit header thực đo: `x-ratelimit-limit-requests = 1000`, `x-ratelimit-limit-tokens = 8000` (TPM).

| Nhóm thực nghiệm | Yêu cầu HTTP / Tác vụ | Lệnh gọi Groq thực tế | Thành công (`200`) | Lỗi / `429` | Prompt Tokens | Completion Tokens | Total Tokens | TB Token / Call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Connectivity_Smoke_1q` | `1` | `1` | `1` | `0 / 0` | `97` | `32` | `129` | `129,0` |
| `HTTP_Smoke_3q` | `3` | `1` | `1` | `0 / 0` | `1.494` | `59` | `1.553` | `1.553,0` |
| `HTTP_Config_A_pool10_40q` | `40` | `7` | `7` | `0 / 0` | `7.752` | `2.784` | `10.536` | `1.505,1` |
| `HTTP_Config_B_pool30_40q` | `40` | `7` | `7` | `0 / 0` | `8.707` | `3.398` | `12.105` | `1.729,3` |
| `Direct_Live_Groq_NO_RAG_40q` | `40` | `40` | `40` | `0 / 0` | `8.774` | `38.567` | `47.341` | `1.183,5` |
| `HTTP_Robustness_20` | `20` | `1` | `1` | `0 / 0` | `1.326` | `1.024` | `2.350` | `2.350,0` |
| `HTTP_Quiz_8_Topics` (+1 raw 422) | `9` | `33` | `27` | `6 / 6` | `29.966` | `13.111` | `43.077` | `1.595,4` |
| `HTTP_Roadmap_4_Scenarios` | `4` | `0` | `0` | `0 / 0` | `0` | `0` | `0` | `0,0` |
| **TỔNG CỘNG (8 NHÓM LIVE GROQ)** | **`157`** | **`90`** | **`84`** | **`6 / 6`** | **`58.116`** | **`58.975`** | **`117.091`** | **`1.393,9`** |
| *Đối chiếu: `HTTP_Edge_Cases_9` (`edge_cases_summary.csv`)* | *`9`* | *`0`* | *`0`* | *`0 / 0`* | *`0`* | *`0`* | *`0`* | *`0,0`* |
| **TỔNG CỘNG TOÀN BỘ YÊU CẦU HTTP/API ĐÃ THỬ** | **`166`** | **`90`** | **`84`** | **`6 / 6`** | **`58.116`** | **`58.975`** | **`117.091`** | **`1.393,9`** |

> **Làm rõ mẫu số & tầng phát sinh lỗi `429`:** (1) `157` là tổng số yêu cầu thuộc 8 nhóm thực nghiệm chính ghi nhận trong `token_usage_summary.csv` (`1 + 3 + 40 + 40 + 40 + 20 + 9 + 4`); khi cộng thêm `9` yêu cầu kiểm thử HTTP edge cases (`edge_cases_summary.csv`, `0` lệnh gọi Groq), tổng số yêu cầu HTTP/API đã thực thi là **`166`** — tách biệt hoàn toàn với **`90` lệnh gọi API Groq thực tế ở tầng provider**. (2) Toàn bộ `6` lỗi `429 rate_limit` trong `HTTP_Quiz_8_Topics` đều phát sinh **ở tầng provider Groq** (`api.groq.com`), trong khi endpoint ứng dụng `/api/quiz` vẫn trả về **HTTP `200`** cho cả `8/8` chủ đề chuẩn hóa (chỉ đầu dò gốc `QZ_03` trả về HTTP `422` do validation dấu phẩy).

---

## 4. Real Groq Answer Generation Quality (Protocol-Separated)

**Artifacts:** [answer_quality_summary.csv](evaluation/artifacts/ac04_completion/answer_quality_summary.csv), [answer_metrics_per_query_live.csv](evaluation/artifacts/ac04_completion/answer_metrics_per_query_live.csv), [answer_quality_audit_live.json](evaluation/artifacts/ac04_completion/answer_quality_audit_live.json).  
**Kiểm toán chấm mù:** `120` bản chấm mù bởi `antigravity_agent_pro` + `14` mẫu kiểm tra chéo tầng `flash` đạt **`100,0%` đồng thuận trong ngưỡng ±1 điểm về Correctness** và **`87,5%` về Faithfulness** (bảo toàn `3/3 PASS` synthetic controls từ Stage 04 và `human_validated = false`). Giá trị `N/A` được bảo toàn tuyệt đối ở mọi ca không có ngữ cảnh verbatim gửi lên Groq (`NO_RAG` và `33` ca `abstained`).

| Giao thức & Nhóm tách biệt | `n` (`llm / ext / abs`) | Correct. | Relev. | Faith. *(ctx)* | Complet. | Cit. Acc. *(ctx)* | Cit. Val. *(ctx)* | Unsup.↓ *(ctx)* | Contra.↓ | Complex. | Code |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **HTTP Config A (`pool=10`) — All 40** | `40` (`4/3/33`) | `0,1000` | `0,0938` | `0,7143` *(n=7)* | `0,0917` | `0,8571` *(n=7)* | `1,0000` *(n=7)* | `0,2586` *(n=7)* | `0,0073` | `0,2031` | `0,1875` |
| **HTTP Config A — Live LLM Only** | `4` (`4/0/0`) | **`0,9375`** | `0,8750` | `0,5000` | **`0,9167`** | `0,7500` | `1,0000` | `0,4525` | `0,0725` | `1,0000` | `0,7500` |
| **HTTP Config B (`pool=30`) — All 40** | `40` (`4/3/33`) | `0,0938` | `0,0938` | `0,7143` *(n=7)* | `0,0958` | `0,8214` *(n=7)* | `1,0000` *(n=7)* | `0,2371` *(n=7)* | `0,0103` | `0,1974` | `0,0500` |
| **HTTP Config B — Live LLM Only** | `4` (`4/0/0`) | `0,7500` | `0,7500` | `0,5000` | `0,8750` | `0,6875` | `1,0000` | `0,4150` | `0,0675` | `0,8333` | `N/A` |
| **Direct Live Groq `NO_RAG` (All 40)** | `40` (`40/0/0`) | **`0,7312`** | **`0,9125`** | `N/A` | `0,6333` | `N/A` | `N/A` | `N/A` | `0,0575` | `0,8750` | `0,8158` |
| *Đối chiếu Stage 04 Controlled Agent A4 (tách biệt)* | *`40` (`flash`, `2048t`)* | *`0,9250`* | *`0,9563`* | *`0,9187`* | *`0,9167`* | *`0,8187`* | *`1,0000`* | *`0,0638`* | *`0,0000`* | *`1,0000`* | *`0,9500`* |

- **Hình bổ sung 1 ([PNG](evaluation/artifacts/ac04_completion/figures/png/sup_fig01_live_groq_answer_quality.png) | [SVG](evaluation/artifacts/ac04_completion/figures/svg/sup_fig01_live_groq_answer_quality.svg)):** *Chất lượng 10 thước đo trên Live Groq (`openai/gpt-oss-120b`) và phân bổ chế độ sinh (`4` LLM / `3` Extractive / `33` Abstained ở cả Config A và B).*
- **Hình bổ sung 2 ([PNG](evaluation/artifacts/ac04_completion/figures/png/sup_fig02_controlled_agent_vs_live_groq.png) | [SVG](evaluation/artifacts/ac04_completion/figures/svg/sup_fig02_controlled_agent_vs_live_groq.svg)):** *Đối chiếu tách biệt giao thức giữa Stage 04 Controlled Agent (`flash`, `budget=2048`) và Live Groq HTTP E2E / `NO_RAG` (`openai/gpt-oss-120b`).*

**Diễn giải khoa học quan trọng:**
1. Khi câu hỏi **vượt qua** bộ lọc `chat_retrieval.py` để đi tới Groq (`live_llm_only`, `n=4`), `openai/gpt-oss-120b` trên Config A đạt `Correctness = 0,9375` và `Completeness = 0,9167`, nhưng có `Faithfulness = 0,5000` và `Unsupported Claim Rate = 0,4525` vì mô hình bổ sung kiến thức DSA nền từ tham số nội tại ngoài 5 đoạn trích ngắn.
2. Điểm trung bình toàn bộ 40 câu hỏi HTTP (`/api/chat`) tụt xuống `0,0938–0,1000` **hoàn toàn do tỷ lệ từ chối sai `82,5%` của tầng điều phối `intent_router.py` / `chat_retrieval.py`**, không phải do suy giảm chất lượng truy xuất `A4` (`nDCG@5 = 0,7053`).
3. Trên **Live Groq `NO_RAG` (`n=40`)**, `openai/gpt-oss-120b` đạt `Correctness = 0,7312`, `Relevancy = 0,9125`, `Completeness = 0,6333` (thấp hơn Controlled Agent `NO_RAG = 0,9688` một phần do trần `max_completion_tokens = 1.024` làm cắt ngắn một số câu trả lời chi tiết bằng tiếng Việt và có `Contradiction = 0,0575`).

---

## 5. Real HTTP End-to-End Latency & Reliability

**Artifacts:** [http_latency_summary.csv](evaluation/artifacts/ac04_completion/http_latency_summary.csv), [edge_cases_summary.csv](evaluation/artifacts/ac04_completion/edge_cases_summary.csv).

| Nhóm đo HTTP / API | Requests (`Provider Calls`) | HTTP 200 | Live LLM / Ext / Abs | Err / 429 / Retry | Mean (`ms`) | P50 (`ms`) | P95 (`ms`) | Min / Max (`ms`) | Std (`ms`) | Retrieval Mean (`ms`) | LLM Mean (`ms`) | Overhead (`ms`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `HTTP_Config_A_pool10` (`/api/chat`) | `40` (`7`) | `100%` | `4 / 3 / 33` | `0 / 0 / 0` | `2.099,5` | `1.287,6` | `4.671,9` | `889 / 11.682` | `1.994,0` | `1.814,0` | `281,3` | `5,3` |
| `HTTP_Config_B_pool30` (`/api/chat`) | `40` (`7`) | `100%` | `4 / 3 / 33` | `1 / 0 / 0` | `3.022,1` | `1.741,4` | `6.644,9` | `1.393 / 12.209` | `2.206,0` | `2.689,5` | `328,1` | `5,8` |
| `Direct_Live_Groq_NO_RAG` | `40` (`40`) | `100%` | `40 / 0 / 0` | `0 / 0 / 0` | `2.596,1` | `2.617,1` | `2.874,3` | `1.337 / 3.903` | `355,1` | `0,0` | `2.596,1` | `0,0` |
| `HTTP_Robustness_20` (`/api/chat`) | `20` (`1`) | `100%` | `0 / 0 / 20` | `0 / 0 / 0` | `3.637,7` | `2.010,6` | `5.398,0` | `1.170 / 29.729` | `6.206,5` | `3.628,9` | `0,0` | `10,1` |
| `HTTP_Quiz_8_Topics` (`/api/quiz`) | `8` (`33`) | `100%` | `1 full (5 any) / 0 / 3` | `4 / 6 / 3` | `9.833,7` | `10.477,2` | `18.854,8` | `1.412 / 18.868` | `7.647,5` | `2.578,3` | `6.179,4` | `1.076,0` |
| `HTTP_Roadmap_4_Scenarios` (`/api/roadmap`) | `4` (`0`) | `100%` | `0 / 0 / 4` | `0 / 0 / 0` | `4.641,9` | `4.609,9` | `4.840,3` | `4.477 / 4.871` | `171,3` | `4.640,8` | `0,0` | `4,7` |

- **Kiểm thử Edge Cases & Lỗi HTTP (`9 / 9 PASSED`):** `/health` (`200`, `29,4 ms`), `/ready` (`200`, `1,3 ms`), Empty Workspace cho `/api/chat`, `/api/quiz`, `/api/roadmap` (`200` + `abstained`), Empty query (`400`, `3,2 ms`), Extra field trên `/api/roadmap` (`422`, `4,6 ms`), Client timeout (`0,001s` → timeout sạch), và Server `CHAT_REQUEST_TIMEOUT=0,05s` (`504`, `125,0 ms`).
- **Hình bổ sung 3 ([PNG](evaluation/artifacts/ac04_completion/figures/png/sup_fig03_http_e2e_latency.png) | [SVG](evaluation/artifacts/ac04_completion/figures/svg/sup_fig03_http_e2e_latency.svg)):** *Độ trễ HTTP End-to-End thực tế đo từ phía client (`P50, Mean, P95`) trên các endpoint Chat, Robustness, Quiz và Roadmap.*
- **Hình bổ sung 4 ([PNG](evaluation/artifacts/ac04_completion/figures/png/sup_fig04_retrieval_vs_generation_vs_total_time.png) | [SVG](evaluation/artifacts/ac04_completion/figures/svg/sup_fig04_retrieval_vs_generation_vs_total_time.svg)):** *Phân rã thời gian Retrieval vs Live Groq Generation vs HTTP Overhead (tách rõ bằng đường nét đứt khỏi benchmark chỉ-truy-xuất Stage 04).* Lưu ý: trên nhóm câu thực sự gọi Groq (`n=7`), thời gian sinh LLM trung bình của `/api/chat` là `~1,6–1,9s` (nhưng trung bình trên cả 40 câu chỉ là `281–328 ms` vì `33/40` câu không gọi LLM); còn `Max = 11,7–12,2s` ở `Q001` phản ánh chi phí khởi tạo cache BM25 lần đầu (`~10s`).

---

## 6. Live HTTP Quiz & Study Roadmap Evaluation

**Artifacts:** [study_features_summary_live.csv](evaluation/artifacts/ac04_completion/study_features_summary_live.csv), [http_quiz.jsonl](evaluation/artifacts/ac04_completion/http_quiz.jsonl), [http_roadmap.jsonl](evaluation/artifacts/ac04_completion/http_roadmap.jsonl).

- **Bảo toàn nguyên trạng Stage 04 Backend Contracts (`11 / 11 PASSED`):** Giữ tách biệt với đánh giá HTTP Live Groq dưới đây.
- **Live HTTP `/api/quiz` (`8` chủ đề chuẩn hóa + `1` đầu dò lỗi gốc `QZ_03`):**
  - *Bảo toàn lỗi validation gốc:* Gửi nguyên văn `"Tree, BST & AVL"` (`QZ_03_RAW_VALIDATION`) nhận **`HTTP 422`** (`3,71 ms`) đúng theo validator cấm dấu phẩy của `QuizRequest.single_topic`.
  - *Kết quả 8 chủ đề chuẩn hóa:* `8/8` endpoint `/api/quiz` trả về **HTTP `200`**, `100%` hợp lệ schema (`duplicate_rate = 0,0`), **`5/8 (62,5%)` sinh được câu hỏi từ Groq** (`15` câu hỏi tổng cộng: `QZ_06 = 5/5` sạch lỗi; `QZ_01 = 2/5` gặp `invalid_output`; `QZ_02 = 2/5`, `QZ_04 = 3/5`, `QZ_07 = 3/5` gặp **`rate_limit` `429` ở tầng provider Groq** trong khi `/api/quiz` vẫn trả về ứng dụng HTTP `200`, do mỗi request vượt `8.000 TPM`), **`3/8` (`QZ_03, QZ_05, QZ_08`) abstained (`0/5`)**. Điểm trung bình trên đủ 8 chủ đề (tính `0` cho ca abstained): `Correctness = 0,2812`, `Source Grounding = 0,3438`, `Pedagogical Quality = 0,2500`, `Mean HTTP = 9.833,7 ms` (`P95 = 18.854,8 ms`).
- **Live HTTP `/api/roadmap` (`4` kịch bản Stage 04 `RM_01..RM_04`):**
  - `4/4` `HTTP 200`, `100%` hợp lệ schema, **`0/4` sinh được mục lộ trình (`mode=abstained`, `items=0`)**, `Mean HTTP = 4.641,9 ms` (`4.640,8 ms` cho `curriculum_inventory`, `0` lệnh gọi Groq) do bộ lọc catalog không khớp câu mô tả mục tiêu học tập dài.
- **Hình bổ sung 5 ([PNG](evaluation/artifacts/ac04_completion/figures/png/sup_fig05_live_quiz_and_roadmap.png) | [SVG](evaluation/artifacts/ac04_completion/figures/svg/sup_fig05_live_quiz_and_roadmap.svg)):** *Chất lượng Live HTTP Quiz & Roadmap đối chiếu tách biệt với Stage 04 Controlled Agent, cùng biểu đồ phân tách HTTP `200` transport, sinh một phần (`62,5%`) và sinh đủ sạch lỗi (`12,5%`).*

---

## 7. Live HTTP Robustness Evaluation (`n = 20`)

**Artifacts:** [robustness_summary_live.csv](evaluation/artifacts/ac04_completion/robustness_summary_live.csv), [http_robustness.jsonl](evaluation/artifacts/ac04_completion/http_robustness.jsonl).

- **Khác biệt giao thức Prompt Injection (Condition #9):** Ở Stage 04, prompt injection được kiểm thử ở mức **context-level** (tiêm lệnh vào ngữ cảnh giả lập). Ở đợt HTTP End-to-End thật này, prompt injection được kiểm thử ở mức **query-level HTTP input** (gắn chuỗi tiêm lệnh trực tiếp vào `ChatRequest.query` gửi lên `/api/chat`).
- **Kết quả thực đo:** `20/20` trả về `HTTP 200` + `mode = abstained` (chỉ có 1 ca `ROB_OOD_04` gọi Groq `2.350` tokens rồi model từ chối đúng).
  - `out_of_domain` (`n=4`): **`100%` xử lý đúng** (`Quality = 1,00`).
  - `unanswerable_in_corpus` (`n=4`): **`100%` xử lý đúng** (`Quality = 1,00`).
  - `prompt_injection` (`n=3`, query-level): `100%` vắng `forbidden_token`, **nhưng `0%` trả lời được câu hỏi DSA hợp lệ đi kèm** (`Quality = 0,25`, từ chối dương tính giả do `chat_retrieval` loại bỏ câu hỏi; `n=3` không chứng minh an toàn bảo mật tổng thể).
  - `distractor_context` (`n=3`), `semantic_paraphrase` (`n=3`), `typo_noisy` (`n=3`): đều **`0%` xử lý đúng** (`Quality = 0,00`) do bộ lọc từ khóa cứng abstained toàn bộ.
- **Hình bổ sung 6 ([PNG](evaluation/artifacts/ac04_completion/figures/png/sup_fig06_live_api_robustness_and_reliability.png) | [SVG](evaluation/artifacts/ac04_completion/figures/svg/sup_fig06_live_api_robustness_and_reliability.svg)):** *Kết quả Live HTTP Robustness (`n=20`) và ma trận độ tin cậy phân tách HTTP `200`, Live LLM sạch lỗi và tỷ lệ lỗi.*

---

## 8. Acceptance Gates (`GATE-01` – `GATE-06`) & Updated Stage 04 `AC-04` Assessment

**Artifact:** [ac04_gates_assessment.json](evaluation/artifacts/ac04_completion/ac04_gates_assessment.json), [manifest.json](evaluation/artifacts/ac04_completion/manifest.json).

| Gate | Tiêu chí kiểm định | Trạng thái | Bằng chứng thực đo |
|---|---|---|---|
| **GATE-01** | Live Groq connectivity verified | **PASS** | `openai/gpt-oss-120b` HTTP `200` (`957 ms`), `1.000 RPD / 8.000 TPM`, `90` lệnh gọi Groq thực tế (`84` HTTP `200`), `117.091` tokens, `0` lộ secret. |
| **GATE-02** | Actual application Answer Generation evaluated | **PASS** | Hoàn tất `40` Config A + `40` Config B + `40` Direct Live Groq `NO_RAG` = `120` câu chấm mù 10 chỉ số (dùng ngữ cảnh verbatim) + kiểm toán `14` mẫu (`100%` Correctness / `87,5%` Faithfulness ±1pt). |
| **GATE-03** | Actual HTTP End-to-End evaluated | **PASS** | Hoàn tất `80` Chat E2E + `9/9` Edge Cases (kể cả `400/422/504`/empty workspace/client timeout) + `20` HTTP Robustness + phân rã đầy đủ `retrieval_ms / llm_ms / total_ms`. |
| **GATE-04** | Real Quiz & Roadmap API evaluated | **PASS** | Hoàn tất `8` Quiz chuẩn hóa + `1` raw `QZ_03` `422` probe (`5/8` sinh `15` câu hỏi) và `4` Roadmap scenarios (`4/4` HTTP `200`, ghi nhận thực chất lỗi abstention). |
| **GATE-05** | Robustness, answer quality, latency & token tracking valid & reproducible | **PASS** | Đủ CSV/JSONL checkpoint, bảo toàn `N/A`, không chế ngữ cảnh, `original_qdrant_eval_untouched = True`. |
| **GATE-06** | Supplementary artifacts, 6 figures, reports, regression tests & integrity checks | **PASS** | Đủ `6` hình `300 DPI PNG + SVG` kiểm tra trực quan sạch lỗi, báo cáo + phụ lục Stage 05, bộ test hồi quy `PASS`, Stage 04/05 gốc bất biến `100%`. |

### Đánh giá cập nhật trạng thái Stage 04 `AC-04`
- **Trạng thái lịch sử Stage 04 (giữ nguyên):** **`PARTIAL`** (tại checkpoint Stage 04 `966ecfa4...`).
- **Khuyến nghị trạng thái cập nhật cho cổng đánh giá Stage 04 `AC-04`:** **`PASS` (ở phương diện Hoàn tất Thực nghiệm Đánh giá — Evaluation Gate)** vì toàn bộ `6/6` cổng `GATE-01`–`GATE-06` đã hoàn thành thực đo thật trên Groq API và HTTP End-to-End mà không còn blocker `BLOCKED`.
- **Cảnh báo Chất lượng Sản phẩm bắt buộc đi kèm:** Hệ thống backend hiện tại **CHƯA SẴN SÀNG PRODUCTION** cho đến khi khắc phục 4 lỗi đã đo được trong một PR kỹ thuật riêng: (1) nới lỏng/sửa bộ lọc `intent_router.py` & `chat_retrieval.py` gây `82,5%` abstention sai; (2) sửa `llm_client.py` không ném `truncated_json` khi `finish_reason == "length"` ở plain-text chat; (3) bổ sung pacing/giảm token prompt trong `quiz_generator.py` để không vượt `8.000 TPM` nội bộ; và (4) hỗ trợ khớp mục tiêu học tập tự nhiên trong `study_planner.py` / `/api/roadmap`.
