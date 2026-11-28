# BÁO CÁO GIAI ĐOẠN 04 — ĐÁNH GIÁ TOÀN DIỆN HỆ THỐNG (STAGE 04 & BỔ SUNG AC-04 LIVE GROQ / HTTP E2E)

> **Dự án:** DSA Document Assistant Using Hybrid RAG  
> **Giai đoạn:** Stage 04 Comprehensive System Evaluation + Supplementary AC-04 Completion *(Live Groq `openai/gpt-oss-120b` & HTTP End-to-End)*  
> **Branch:** `feat/final-readiness` | **Checkpoints:** `966ecfa` (Stage 04), `9b09c2e` (AC-04 Completion) | **Immutable `main`:** `b84f09450f213b03ad57ef8b388f482ee0f0eaa2`  
> **Quản trị đánh giá:** `human_validated = false`, `human_approval_status = "DRAFT"`  
> **Trạng thái AC-04:** Lịch sử Stage 04 gốc = **`PARTIAL`** *(bảo toàn)* | Sau bổ sung (Evaluation Completion Gate) = **`PASS`** (`6/6` gates)  
> **Trạng thái sản phẩm:** **`NOT PRODUCTION-READY`** *(do 4 lỗi backend thực đo chưa sửa trong phạm vi đánh giá)*

---

## 1. Mục tiêu

Đánh giá định lượng toàn diện hệ thống Hybrid RAG trên cả 8 tổ hợp thực nghiệm (`A1–A4` Recursive, `B1–B4` Adaptive) qua 9 mốc `Top-K ∈ {1, 3, 5, 10, 15, 20, 30, 45, 60}`, bao phủ: (1) chẩn đoán chunking, (2) chất lượng truy xuất & bằng chứng nguồn, (3) hiệu năng độ trễ & tài nguyên tách biệt theo giao thức, (4) chất lượng sinh câu trả lời (Controlled Agent & Live Groq), (5) độ tin cậy (Robustness) & tính năng học tập (Quiz/Roadmap), (6) Ablation, độ nhạy, thống kê Bootstrap, phân tích lỗi & biên Pareto, và (7) đánh giá **Live Groq API & HTTP End-to-End thực tế** để hoàn tất khoảng trống `AC-04`.

## 2. Phương pháp thực hiện

- **Tách biệt giao thức nghiêm ngặt (không trộn lẫn phép đo):**
  - *Retrieval chính thức:* `320` lượt truy xuất (`40` câu × `8` tổ hợp, `top_k=60`, `candidate_top_k=120`, `0` ranking drift so với Stage 03).
  - *Hiệu năng truy xuất (4 giao thức retrieval-only độc lập — KHÔNG phải HTTP E2E):* (1) Main Warm Latency **`14.400` lượt** (`5` repeats xen kẽ, `pool=120`, `ms`); (2) Stage Breakdown **`640` lượt** (`K=5/60`, `pool=120`, `ms`); (3) Production-Parameter Retrieval **`1.600` lượt** (`K=5`, `pool=10`, `ms`); (4) Cold-Start **`24` thử nghiệm** độc lập (`giây`).
  - *Controlled Agent Generation (Stage 04 gốc):* **`400` câu trả lời** (`360` câu chính = `40` câu × `9` cấu hình gồm `A1–A4, B1–B4, NO_RAG`, ngân sách `2.048` tokens + `40` câu độ nhạy Context Budget `1.024 / 4.096` trên `n=10`), sinh bởi `antigravity_agent_flash` và chấm mù 10 chỉ số bởi `antigravity_agent_pro`.
  - *Thống kê:* Paired Bootstrap **`10.000` lượt** (`seed=42`), hiệu chỉnh đa kiểm định **Holm-Bonferroni**; không diễn giải kết quả không có ý nghĩa thống kê (`p ≥ 0,05`) là tương đương đã chứng minh.
  - *Bổ sung AC-04 Live Groq & HTTP End-to-End:* Chạy mã nguồn FastAPI thật (`uvicorn 127.0.0.1:8015`, **0 sửa đổi `backend/`**) trên bản sao kiểm chứng SHA-256 của `data/runtime/qdrant_eval` (`data/runtime/ac04_eval_runtime/qdrant_eval_copy`, tách biệt hoàn toàn với `qdrant_local`) kết nối trực tiếp Groq **`openai/gpt-oss-120b`**. Giữ tách biệt 4 giao thức: HTTP Config A (`pool=10`, `top_k=5`), HTTP Config B (`pool=30`, `top_k=5`), Direct Live Groq `NO_RAG` (`n=40`) và Controlled Agent trước đó. Backend hiện **không hỗ trợ** `2.048`-token context budget hay `Top-K > 5` (`MAX_EVIDENCES = 5`).

## 3. Dữ liệu đầu vào

- Hai frozen benchmark corpora Stage 02 (`6.366` Recursive chunks, `13.390` Adaptive chunks) và chỉ mục Qdrant cô lập (`19.756` points).
- Bộ dữ liệu đóng băng Stage 03 (`40` câu hỏi, `91` KUs, `107` bằng chứng nguồn, `9.068` nhãn Ground Truth; `human_validated = false`).
- Tập kiểm thử độc lập: `20` ca Robustness (`6` nhóm), `8` chủ đề Quiz (+ đầu dò lỗi gốc `QZ_03` `"Tree, BST & AVL"`), `4` kịch bản Roadmap, và `9` ca HTTP Edge Cases.

## 4. Nội dung triển khai

- Mã nguồn Stage 04: `evaluation/stage04/{chunking_eval,retrieval_benchmark,evidence_metrics,latency_benchmark,generation_eval,metrics,analysis}.py` và CLI `evaluation/scripts/run_comprehensive_evaluation.py`.
- Mã nguồn bổ sung AC-04: `evaluation/ac04_completion/{groq_smoke,asgi_harness,server_runner,http_experiments,judging,visualizations}.py` và CLI `evaluation/scripts/run_ac04_completion.py`.
- Toàn bộ kết quả thí nghiệm đã được tổ chức theo nội dung nghiên cứu dưới `evaluation/results/` (xem bản đồ di chuyển đầy đủ tại [relocation_manifest.json](../manifests/relocation_manifest.json)).

## 5. Kết quả

### 5.1. Chunking, Retrieval và Evidence (Stage 04 — Chi tiết tại `evaluation/results/{chunking,retrieval,evidence}/`)
- **Chunking:** Profile A chịu **`43,20%`** cắt cụt MiniLM (`114.961` tokens bị bỏ) nhưng mang ngữ cảnh gấp `~2,1–2,3×` mỗi K và đạt độ phủ hiệu dụng `91,36%`; Profile B đạt **`0,00%`** cắt cụt embedding và giảm overlap còn `1,06%` nhưng phân mảnh thành `13.390` chunks ngắn (trung vị `112` ký tự).
- **Retrieval (`320` lượt, `Judged@60 = 1,000`, `0` drift):** **A4 (Rec+Rerank)** dẫn đầu **`nDCG@5 = 0,7053`**, `Precision@5 = 0,7000`, `MRR@5 = 0,9875`, `nDCG@15 = 0,7335`, `Pooled Recall@60 = 0,9447`; tiếp theo là **B4** (`nDCG@5 = 0,5591`, `Pooled Recall@5 = 0,3509`) và **A1** (`nDCG@5 = 0,5584`). BM25 vượt mạnh Dense trên cả hai profile (`A1 − A2 = +0,1662`, `B1 − B2 = +0,1286`).
- **Evidence:** **A4** đạt **`Evidence Recall@15 = 0,9083`** và `Complete Evidence Rate@15 = 80,0%` (so với `B4 = 47,5%`). Bất biến `Complete ≤ EvRecall ≤ Partial ≤ KU_Any` được bảo toàn trên mọi tổ hợp.

### 5.2. Hiệu năng Độ trễ Retrieval và Tài nguyên (`evaluation/results/performance/`)
- **Main Warm (`14.400` lượt, `pool=120`, `ms`):** A1 nhanh nhất (`P95@5 = 607,5 ms`); A4 `P95@5 = 3.693,1 ms`; B4 `P95@5 = 3.214,3 ms` (nhanh hơn A4 `13,0%` ở tổng P95 warm và `23,6%` ở P50).
- **Stage Breakdown (`640` lượt, `K=5`):** Ở riêng bước **BGE Reranking cô lập**, B4 nhanh hơn A4 **`41,2%`** (`1.364,8 ms` vs `2.320,4 ms`) nhờ chunk ngắn hơn — cải thiện này áp dụng cho bước reranking, **không phải** toàn bộ độ trễ truy xuất.
- **Production-Parameter Retrieval (`1.600` lượt, `K=5/pool=10`):** A4 `P95` giảm từ `3.693 ms` xuống `1.231 ms`.
- **Cold-Start (`24` thử nghiệm, `giây`):** `9,6–34,3s` (dựng BM25 Profile B mất `15,2–16,0s` vs Profile A `9,6–10,9s`).
- **Tài nguyên:** BM25 dùng `0 MB` VRAM; Reranker đỉnh `~2,69 GB` VRAM (`~34%` RTX 5060, `0` CUDA OOM); RAM tiến trình `965–993 MB`.

### 5.3. Controlled Agent Generation, Ablation, Sensitivity, Statistics & Error Analysis
- **Controlled Generation (`400` câu; bảo toàn `N/A` của `NO_RAG`):** **`NO_RAG` đạt `Correctness = 0,9688` và `Relevancy = 1,0000` cao hơn A4 (`Correctness = 0,9250`, `Relevancy = 0,9563`)**; **A1 nhỉnh hơn A4 về `Faithfulness` (`0,9250` vs `0,9187`) và `Citation Accuracy` (`0,8500` vs `0,8187`)**. A4 không thắng mọi thước đo, nhưng giảm `Contradiction` xuống `0,0000` (so với `0,0175` của `NO_RAG`).
- **Ablation & Sensitivity:** Reranker tạo bước nhảy lớn nhất (`A4 − A3 = +0,2016`, `Holm p = 0,001`). Độ nhạy Candidate Pool (`640` single-pass): **A4 tại `pool=30`** giữ `99,8%` `nDCG@5` (`0,7039` vs `0,7053`) trong khi giảm độ trễ trung bình từ `3.486 ms` xuống `1.628 ms`. Context Budget thăm dò (`n=10`): `2.048` tokens tối ưu cho A4; `4.096` tokens làm giảm Faithfulness (`0,825`) và Citation Accuracy (`0,700`).
- **Controlled Robustness (`n=20`) & Study Features:** OOD (`4/4`), Unanswerable (`4/4`), Context-level Prompt Injection (`3/3`, cỡ mẫu nhỏ) đạt `100%`, nhưng **tiếng Việt không dấu (`n=3`) thất bại `0%`** (`EvRecall@5` giảm `-0,67` do `accent_insensitive=false`). Kiểm thử hợp đồng module backend thật đạt `11/11 PASSED`.
- **Phân tích lỗi:** `7` dạng lỗi, phổ biến nhất là bằng chứng có trong Top-60 nhưng xếp ngoài Top-5 (`29` lượt), không truy xuất trong Top-60 (`17`) và thiếu KU (`17`).

### 5.4. Kết quả Bổ sung AC-04: Live Groq API (`openai/gpt-oss-120b`) & HTTP End-to-End (`evaluation/results/{generation/live_groq,http_e2e,robustness/live_http,study_features/live_http}/`)
- **Kế toán Yêu cầu vs Lệnh gọi Provider Groq (Mẫu số phân tách rõ ràng):**
  - Tổng số yêu cầu HTTP/API đã thực thi: **`166` yêu cầu** = **`157` yêu cầu** thuộc 8 nhóm thực nghiệm chính trong [token_usage_summary.csv](../results/http_e2e/token_usage_summary.csv) (`1` Connectivity Smoke + `3` HTTP Smoke + `40` Config A + `40` Config B + `40` Direct Live Groq `NO_RAG` + `20` Live Robustness + `9` Quiz [`8` chuẩn hóa + `1` raw `QZ_03` `422`] + `4` Roadmap) **`+ 9` yêu cầu HTTP Edge Cases** trong [edge_cases_summary.csv](../results/http_e2e/edge_cases_summary.csv) (`0` lệnh gọi Groq; `9/9 PASSED`).
  - Tách biệt hoàn toàn với **`90` lệnh gọi API Groq thực tế ở tầng provider** (`84` HTTP `200`, `6` HTTP `429 rate_limit` ở tầng provider Groq), tiêu thụ **`117.091` tokens** (`58.116` prompt + `58.975` completion; miễn phí trong hạn mức quan sát `1.000 RPD / 8.000 TPM`).
- **HTTP `/api/chat` Config A (`pool=10`) & Config B (`pool=30`) vs Direct Live Groq `NO_RAG`:**
  - Cả Config A và B đạt **`100%` ứng dụng HTTP `200`** (`Mean = 2.099,5 ms` / `P95 = 4.671,9 ms` ở Config A; `Mean = 3.022,1 ms` / `P95 = 6.644,9 ms` ở Config B), **nhưng chỉ `4/40 (10,0%)` sinh Live LLM**, `3/40 (7,5%)` rơi về `extractive`, và **`33/40 (82,5%)` bị từ chối dương tính giả (`abstained`)** bởi bộ lọc cứng trong `intent_router.py` / `chat_retrieval.py` dù `best_score > 0,90`.
  - Điểm `Correctness` toàn tập 40 câu tụt xuống `0,1000` (Config A) và `0,0938` (Config B), trong khi riêng nhóm **`live_llm_only (n=4)`** Config A đạt `Correctness = 0,9375`, `Completeness = 0,9167` (nhưng `Faithfulness = 0,5000` do model bổ sung kiến thức nội tại). Direct Live Groq `NO_RAG` (`n=40`) đạt `Correctness = 0,7312`, `Relevancy = 0,9125`, `Completeness = 0,6333`, `Mean = 2.596,1 ms` (bảo toàn `N/A` cho các thước đo cần ngữ cảnh).
- **Live `/api/quiz`, `/api/roadmap` và Live Robustness (`n=20`):**
  - **Quiz:** Cả **`8/8` yêu cầu chuẩn hóa trả về ứng dụng HTTP `200`** (chỉ raw `QZ_03` `"Tree, BST & AVL"` trả đúng HTTP `422`), `5/8 (62,5%)` sinh được tổng cộng `15` câu hỏi từ Groq, chỉ **`1/8 (12,5%)` sạch lỗi `5/5`** (`QZ_06`), `1/8` gặp `invalid_output` (`QZ_01`), `3/8` gặp **lỗi `429 rate_limit` ở tầng provider Groq** (`6` lời gọi `429`) vì mỗi request `/api/quiz` tiêu thụ `8.093–9.142` tokens > `8.000 TPM`, và `3/8` abstained (`Mean = 9.833,7 ms`, `P95 = 18.854,8 ms`).
  - **Roadmap:** `4/4` trả về ứng dụng HTTP `200`, **`0/4` sinh được lộ trình (`abstained`, `items=0`)**, `Mean = 4.641,9 ms`.
  - **Live Robustness (`n=20`, query-level prompt injection):** `20/20` trả HTTP `200` + `abstained` → **`8/8` đúng** trên OOD & Unanswerable, nhưng **`0/12` thành công** trên các câu DSA hợp lệ.

- **Biểu đồ tổng hợp liên quan:** [Hình 01](../figures/consolidated/png/fig01_danh_gia_chunking.png) – [Hình 10](../figures/consolidated/png/fig10_chuc_nang_va_do_on_dinh_api.png).
- **Biểu đồ gốc bổ sung AC-04 (lưu trữ đối chiếu):** [sup_fig01_live_groq_answer_quality](../figures/original/png/sup_fig01_live_groq_answer_quality.png) | [sup_fig02_controlled_agent_vs_live_groq](../figures/original/png/sup_fig02_controlled_agent_vs_live_groq.png) | [sup_fig03_http_e2e_latency](../figures/original/png/sup_fig03_http_e2e_latency.png) | [sup_fig04_retrieval_vs_generation_vs_total_time](../figures/original/png/sup_fig04_retrieval_vs_generation_vs_total_time.png) | [sup_fig05_live_quiz_and_roadmap](../figures/original/png/sup_fig05_live_quiz_and_roadmap.png) | [sup_fig06_live_api_robustness_and_reliability](../figures/original/png/sup_fig06_live_api_robustness_and_reliability.png).

## 6. Kiểm thử và xác minh

- Kiểm định tự động Stage 04 (`run_comprehensive_evaluation.py --mode validate`): `VALIDATE_MISSING: []` (`59/59` artifacts).
- Kiểm định tự động AC-04 (`run_ac04_completion.py --mode validate`): `6/6` PNG @ `300 DPI`, `6/6` SVG, `0` secret hits, bản gốc `qdrant_eval` và `qdrant_local` bất biến `100%`.
- Kiểm toán chấm mù bổ sung `14` mẫu tầng `flash`: `100,0%` đồng thuận ±1 điểm về Correctness, `87,5%` về Faithfulness.
- Bộ kiểm thử đơn vị `test_comprehensive_evaluation.py`, `test_ac04_completion.py`, `test_retrieval_metrics.py`, `test_ground_truth.py` đều **PASS** (`4` lỗi timing trên Windows trong `test_llm_reliability.py` là có sẵn từ trước và nằm ngoài phạm vi).

## 7. Acceptance Criteria

- **Stage 04 gốc (bảo toàn lịch sử tại `966ecfa`):** `5 PASS` (`AC-01, AC-02, AC-03, AC-05, AC-06`) + **`1 PARTIAL` (`AC-04`)**.
- **Bổ sung AC-04 (`9b09c2e`):** **`6/6` cổng thực nghiệm (`GATE-01`–`GATE-06`) PASS** (chi tiết tại [ac04_gates_assessment.json](../manifests/ac04_gates_assessment.json)) → Cập nhật cổng hoàn thành thực nghiệm **`AC-04` → `PASS`** *(tách biệt hoàn toàn khỏi trạng thái sẵn sàng sản phẩm)*.

## 8. Vấn đề và giới hạn

1. **4 Lỗi Backend Production thực đo (CHƯA SỬA — khiến hệ thống `NOT PRODUCTION-READY`):**
   - *Lỗi #1 (`/api/chat`):* `82,5%` (`33/40`) từ chối dương tính giả (`reason=off_topic`) trong `intent_router.py` / `chat_retrieval.py`.
   - *Lỗi #2 (`llm_client.py`):* Ném `LLMError("truncated_json")` khi plain-text chat chạm `finish_reason == "length"` (`1` ca Config B → fallback).
   - *Lỗi #3 (`/api/quiz`):* Mỗi request bắn nhiều lời gọi vượt `8.000 TPM` gây `6` lỗi `429` ở tầng provider Groq (`3/8` chủ đề) dù `/api/quiz` trả HTTP `200`; `QuizRequest` trả `422` khi chủ đề chứa dấu phẩy (`QZ_03`).
   - *Lỗi #4 (`/api/roadmap`):* `0/4` sinh mục (`abstained`) do bộ lọc catalog không khớp câu mục tiêu dài.
2. **Giới hạn phương pháp:** `human_validated = false`, `human_approval_status = "DRAFT"`, chưa có tập kiểm chứng held-out độc lập, `n=10` cho Context Budget mang tính thăm dò, `n=3` cho Prompt Injection không chứng minh an toàn bảo mật toàn hệ thống.

## 9. Kết luận

Toàn bộ thí nghiệm định lượng Stage 04 và bổ sung Live Groq / HTTP End-to-End đã hoàn tất đầy đủ, tái lập và minh bạch. Tầng truy xuất lõi **A4 (Recursive + Hybrid Rerank)** thể hiện năng lực vượt trội (`nDCG@5 = 0,7053`, `EvRecall@15 = 0,9083`), nhưng ứng dụng FastAPI tổng thể **CHƯA SẴN SÀNG PRODUCTION** cho đến khi 4 lỗi backend nêu trên được khắc phục trong một đợt phát triển riêng.

## 10. Checkpoint và tài liệu liên quan

- **Checkpoints Git:** `966ecfa` (Stage 04), `9b09c2e` (AC-04 Completion).
- **Manifests & Tables:** [stage04_manifest.json](../manifests/stage04_manifest.json) | [ac04_manifest.json](../manifests/ac04_manifest.json) | [ac04_gates_assessment.json](../manifests/ac04_gates_assessment.json) | [table_10_stage04_ac_reconciliation.csv](../tables/table_10_stage04_ac_reconciliation.csv).
- **Tài liệu liên quan:** [Báo cáo Stage 03](03_dataset_ground_truth.md) | [Báo cáo Tổng kết Stage 05](05_final_evaluation.md) | [Evaluation README](../README.md).
