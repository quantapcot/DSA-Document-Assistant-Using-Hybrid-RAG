# BÁO CÁO GIAI ĐOẠN 05 — TRỰC QUAN HÓA, PHÂN TÍCH SO SÁNH VÀ KẾT LUẬN ĐÁNH GIÁ CUỐI CÙNG

> **Dự án:** DSA Document Assistant Using Hybrid RAG  
> **Giai đoạn:** Stage 05 Final Evaluation & Consolidation *(Hợp nhất 34 biểu đồ gốc thành 10 biểu đồ đa khung tiếng Việt chuẩn công bố)*  
> **Branch:** `feat/final-readiness` | **Checkpoints:** `139957c` (Stage 05), `9b09c2e` (Supplementary AC-04) | **Immutable `main`:** `b84f09450f213b03ad57ef8b388f482ee0f0eaa2`  
> **Quản trị:** `human_validated = false` | `human_approval_status = "DRAFT"` | Không có tập held-out độc lập  
> **Lịch sử Stage 04 AC-04:** `PARTIAL` *(Production Groq/HTTP ban đầu `BLOCKED`)* → **Sau bổ sung:** `PASS` *(Evaluation Completion Gate)* | **Sản phẩm:** **`NOT PRODUCTION-READY`**

---

## 1. Mục tiêu

Tổng hợp toàn bộ kết quả thực nghiệm của 5 giai đoạn đánh giá và đợt bổ sung Live Groq / HTTP End-to-End thành **10 biểu đồ đa khung tiếng Việt chuẩn học thuật** (tái sinh trực tiếp từ dữ liệu có cấu trúc, bảo toàn nguyên vẹn 34 biểu đồ gốc), trả lời toàn bộ **8 câu hỏi nghiên cứu (`RQ1`–`RQ8`)**, đưa ra kết luận thống kê, khuyến nghị cấu hình thăm dò theo kịch bản vận hành, đối chiếu thí nghiệm có kiểm soát với API thực tế, và bàn giao kết luận đánh giá cuối cùng.

## 2. Phương pháp thực hiện

- Tái sinh trực tiếp 10 biểu đồ đa khung (`300 DPI PNG` + `SVG` vector, font Arial Unicode tiếng Việt, bảng màu Okabe-Ito nhất quán, phân biệt nét liền Recursive vs nét đứt Adaptive) từ dữ liệu nguồn bất biến trong `evaluation/results/` mà không ghép ảnh PNG.
- Kiểm định đối chiếu dữ liệu khoa học 100% qua [figure_mapping.json](../figures/figure_mapping.json) và [figure_data_parity.csv](../tables/figure_data_parity.csv) (`34/34` biểu đồ gốc, `0` sai lệch số liệu, bảo toàn `N/A` và tách biệt mọi giao thức).
- Tham chiếu số liệu chi tiết từ [Báo cáo Stage 04](04_system_evaluation.md) và 10 bảng tóm tắt trong `evaluation/tables/` nhằm tránh trùng lặp bảng dữ liệu thô.

## 3. Dữ liệu đầu vào

- Toàn bộ 59 artifacts Stage 04, 39 artifacts bổ sung AC-04, 10 bảng tổng hợp (`evaluation/tables/table_01..table_10`), 34 biểu đồ gốc (`evaluation/figures/original/{png,svg}/`), và 9 checksums đóng băng Stage 02/03 (`54` tài liệu, `6.366` Recursive chunks, `13.390` Adaptive chunks, `40` câu hỏi, `91` KUs, `9.068` nhãn Ground Truth).

## 4. Nội dung triển khai — 10 Biểu đồ Hợp nhất Tiếng Việt

1. **Hình 01 — Đánh giá chiến lược Chunking** *(từ Stage 05 Hình 01–05)*: [PNG](../figures/consolidated/png/fig01_danh_gia_chunking.png) | [SVG](../figures/consolidated/svg/fig01_danh_gia_chunking.svg) | Bảng [table_01_chunking_summary.csv](../tables/table_01_chunking_summary.csv).
2. **Hình 02 — Đánh giá chất lượng Retrieval** *(từ Stage 05 Hình 06–12)*: [PNG](../figures/consolidated/png/fig02_chat_luong_retrieval.png) | [SVG](../figures/consolidated/svg/fig02_chat_luong_retrieval.svg) | Bảng [table_02_retrieval_k5_k15_k60.csv](../tables/table_02_retrieval_k5_k15_k60.csv).
3. **Hình 03 — Đánh giá chất lượng Evidence** *(từ Stage 05 Hình 13–15)*: [PNG](../figures/consolidated/png/fig03_chat_luong_evidence.png) | [SVG](../figures/consolidated/svg/fig03_chat_luong_evidence.svg) | Bảng [table_03_evidence_summary.csv](../tables/table_03_evidence_summary.csv).
4. **Hình 04 — Hiệu năng và Tài nguyên Retrieval** *(từ Stage 05 Hình 16–20)*: [PNG](../figures/consolidated/png/fig04_hieu_nang_va_tai_nguyen.png) | [SVG](../figures/consolidated/svg/fig04_hieu_nang_va_tai_nguyen.svg) | Bảng [table_04_latency_protocols.csv](../tables/table_04_latency_protocols.csv).
5. **Hình 05 — Chất lượng sinh câu trả lời (Controlled Agent)** *(từ Stage 05 Hình 21)*: [PNG](../figures/consolidated/png/fig05_chat_luong_cau_tra_loi.png) | [SVG](../figures/consolidated/svg/fig05_chat_luong_cau_tra_loi.svg) | Bảng [table_05_generation_summary.csv](../tables/table_05_generation_summary.csv).
6. **Hình 06 — Ablation và Phân tích độ nhạy** *(từ Stage 05 Hình 22–24)*: [PNG](../figures/consolidated/png/fig06_ablation_va_do_nhay.png) | [SVG](../figures/consolidated/svg/fig06_ablation_va_do_nhay.svg) | Bảng [table_06_ablation_and_stats.csv](../tables/table_06_ablation_and_stats.csv), [table_07_sensitivity_summary.csv](../tables/table_07_sensitivity_summary.csv).
7. **Hình 07 — Thống kê và Cân bằng Hiệu năng (Pareto)** *(từ Stage 05 Hình 25–26)*: [PNG](../figures/consolidated/png/fig07_thong_ke_va_tradeoff.png) | [SVG](../figures/consolidated/svg/fig07_thong_ke_va_tradeoff.svg) | Bảng [table_09_pareto_and_recommendations.csv](../tables/table_09_pareto_and_recommendations.csv).
8. **Hình 08 — Độ tin cậy và Phân tích lỗi thực nghiệm** *(từ Stage 05 Hình 27–28)*: [PNG](../figures/consolidated/png/fig08_do_tin_cay_va_phan_tich_loi.png) | [SVG](../figures/consolidated/svg/fig08_do_tin_cay_va_phan_tich_loi.svg) | Bảng [table_08_robustness_and_study.csv](../tables/table_08_robustness_and_study.csv).
9. **Hình 09 — Đánh giá Live Groq và HTTP End-to-End** *(từ Supplementary Hình 01–04)*: [PNG](../figures/consolidated/png/fig09_live_groq_va_http_e2e.png) | [SVG](../figures/consolidated/svg/fig09_live_groq_va_http_e2e.svg).
10. **Hình 10 — Chức năng thực tế và Độ ổn định API** *(từ Supplementary Hình 05–06)*: [PNG](../figures/consolidated/png/fig10_chuc_nang_va_do_on_dinh_api.png) | [SVG](../figures/consolidated/svg/fig10_chuc_nang_va_do_on_dinh_api.svg).

## 5. Kết quả — Trả lời 8 Câu hỏi Nghiên cứu (`RQ1`–`RQ8`) và Khuyến nghị Cấu hình

### 5.1. Giải đáp Câu hỏi Nghiên cứu (`RQ1`–`RQ8`)
- **`RQ1` (Recursive vs Structure-Adaptive Chunking — Hình 01, 02, 03, 06):** Dù Profile B loại bỏ hoàn toàn cắt cụt MiniLM (`0,00%` vs `43,20%`) và giảm overlap còn `1,06%`, việc chia thành `13.390` chunk ngắn (trung vị `112` ký tự) làm suy giảm có ý nghĩa `nDCG@5` (`B4 − A4 = −0,1462`, `Holm p = 0,001`; `B1 − A1 = −0,1170`, `Holm p = 0,031`) và `Complete Evidence@15` (`47,5%` vs `80,0%`). Ngược lại, chunk dài hơn của Profile A bảo toàn ngữ cảnh đa câu tốt hơn trong `Top-K` nhỏ.
- **`RQ2` (BM25 vs Dense vs Hybrid RRF vs Rerank — Hình 02, 06):** `BM25` vượt mạnh `Dense` trên cả hai profile (`A1 − A2 = +0,1662`; `B1 − B2 = +0,1286`) nhờ khớp chính xác thuật ngữ DSA; `RRF` tăng độ phủ ở `K=60` nhưng không thắng `BM25` ở `K=5`; **BGE Reranker (`A4`, `B4`) là nguồn cải thiện lớn nhất** (`A4 − A3 = +0,2016`, `Holm p = 0,001`, thắng `36/40` câu).
- **`RQ3` (Chất lượng Evidence nguồn — Hình 03):** `A4` đạt `Evidence Recall@15 = 0,9083` và `Complete Evidence@15 = 80,0%`; độ phủ ngữ nghĩa KU (`Semantic KU`) luôn cao hơn `Source-Backed Evidence` vì corpus chứa nhiều đoạn giải thích trùng kiến thức ngoài khoảng chuẩn đã đăng ký.
- **`RQ4` (Hiệu năng Độ trễ & Tài nguyên — Hình 04):** Trên Main Warm (`14.400` lượt), `A1` nhanh nhất (`P95 = 607,5 ms`, `0 MB VRAM`). `B4` nhanh hơn `A4` **`41,2%` ở riêng bước reranking cô lập** (`1.364,8` vs `2.320,4 ms`), nhưng chỉ giảm `13,0%` tổng `P95` warm (`3.214` vs `3.693 ms`) do chi phí QdrantLocal SQLite. Mọi cấu hình dùng tối đa `2,69 GB VRAM` (`0` OOM).
- **`RQ5` (Đóng góp RAG vs `NO_RAG` & Controlled Generation — Hình 05, 06):** Trong thí nghiệm Controlled Agent (`400` câu), **`NO_RAG` vượt `A4` về `Correctness` (`0,9688` vs `0,9250`, `p = 0,238`, không chứng minh tương đương)** và `Relevancy` (`1,000` vs `0,956`), còn **`A1` vượt `A4` về `Faithfulness` (`0,9250` vs `0,9187`) và `Citation Accuracy` (`0,8500` vs `0,8187`)**. Giá trị của RAG nằm ở trích dẫn nguồn xác thực và triệt tiêu mâu thuẫn (`Contradiction = 0,0000` ở A4 vs `0,0175` ở `NO_RAG`).
- **`RQ6` (Độ nhạy `Top-K`, `Candidate Pool`, `Context Budget` — Hình 06):** `A4` với **`candidate_top_k = 30`** (thăm dò) giữ `99,8%` `nDCG@5` (`0,7039` vs `0,7053`) và giảm độ trễ single-pass hơn một nửa (`3.486 → 1.628 ms`). `Context Budget = 2.048` tokens tối ưu trên tập thăm dò `n=10`, nhưng lưu ý backend HTTP hiện tại **chưa hỗ trợ** `2.048` budget hay `Top-K > 5`.
- **`RQ7` (Biên Pareto & Khuyến nghị Cấu hình — Hình 07):** Không dùng trọng số chủ quan, 3 tổ hợp không bị trội là **`A1`** (độ trễ thấp nhất, `0 MB VRAM`), **`B4`** (trung gian, ít trùng lặp bằng chứng, rerank nhanh hơn `41,2%`), và **`A4`** (chất lượng cao nhất). Khuyến nghị thăm dò cân bằng: **`A4` với `pool=30`** (chi tiết tại [stage05_configuration_recommendations.json](../manifests/stage05_configuration_recommendations.json)).
- **`RQ8` (Controlled vs Live Groq API / HTTP End-to-End — Hình 08, 09, 10):** Trong khi Controlled Agent cho thấy tiềm năng cao của pipeline `A4`, thực nghiệm HTTP End-to-End thật (`166` yêu cầu HTTP/API, `90` lệnh gọi Groq, `117.091` tokens) phơi bày **khoảng cách lớn giữa tiềm năng thuật toán và chất lượng ứng dụng thực tế**: `82,5%` từ chối dương tính giả trong `/api/chat`, lỗi `truncated_json` trong `llm_client.py`, `3/8` chủ đề `/api/quiz` chạm `429` ở tầng provider Groq dù endpoint trả HTTP `200`, và `0/4` `/api/roadmap`.

### 5.2. Điểm rút ra sẵn sàng trình bày (Presentation-Ready Takeaways)
1. **Retrieval lõi mạnh (`A4 nDCG@5 = 0,7053`, `EvRecall@15 = 0,9083`):** Reranker là đòn bẩy số 1 (`+40%` `nDCG@5`); Recursive vượt Adaptive về độ hoàn chỉnh bằng chứng dù bị `43,2%` cắt cụt MiniLM.
2. **Minh bạch khoa học:** `A4` không thắng mọi thước đo (`NO_RAG` cao hơn ở Correctness, `A1` cao hơn ở Faithfulness & Citation Accuracy); không có ý nghĩa thống kê không đồng nghĩa tương đương.
3. **Đánh giá HTTP End-to-End thật là bắt buộc:** Chỉ chạy retrieval-only hoặc controlled agent sẽ bỏ sót `82,5%` false-positive abstention, `0/4` roadmap và xung đột `8.000 TPM` của `/api/quiz`.

## 6. Kiểm thử và xác minh

- Kiểm định hợp nhất (`generate_consolidated_visualizations.py --mode validate`): `10/10` PNG (`≥300 DPI`), `10/10` SVG, `34/34` biểu đồ gốc khớp SHA-256 byte-for-byte, `34/34` parity `PASS` (`0` sai lệch), `168/168` tệp di chuyển khớp SHA-256, `5/5` báo cáo chính thức & `evaluation/README.md` có `0` liên kết gãy.
- Toàn bộ 3 validator lịch sử (`Stage 04`, `Stage 05`, `AC-04`) đều tiếp tục đạt **PASS** ở chế độ chỉ-đọc qua `evaluation/path_resolver.py`.
- Toàn bộ targeted unit tests (`test_consolidation.py`, `test_ac04_completion.py`, `test_stage05_visualizations.py`, `test_comprehensive_evaluation.py`, `test_retrieval_metrics.py`, `test_ground_truth.py`) đều **PASS** và `pip check` sạch (`4` lỗi timing có sẵn trên Windows trong `test_llm_reliability.py` được giữ nguyên ngoài phạm vi).

## 7. Acceptance Criteria

- **Stage 05 gốc (`139957c`):** `6/6` Acceptance Criteria (`AC-01`–`AC-06`) đạt **PASS**.
- **Hợp nhất & Dọn dẹp cuối cùng:** Đạt đầy đủ `AC-01`–`AC-06` (dữ liệu bất biến, đủ 10 biểu đồ tiếng Việt + 34 biểu đồ gốc, kiểm tra trực quan sạch lỗi, đúng 5 báo cáo chính thức, tái lập & hồi quy PASS, README & Git integrity sạch).

## 8. Vấn đề, Đe dọa Tính hợp lệ (Threats to Validity) và Hướng phát triển

- **Đe dọa tính hợp lệ:** `human_validated = false` (`DRAFT`), không có tập held-out độc lập, cỡ mẫu nhỏ ở Context Budget (`n=10`) & Prompt Injection (`n=3`), và giới hạn free-tier `8.000 TPM` của Groq ảnh hưởng trực tiếp tới `/api/quiz`.
- **Hướng phát triển ưu tiên (trong PR kỹ thuật riêng — KHÔNG thuộc phạm vi đánh giá này):**
  1. Sửa 4 lỗi backend đã định lượng (`intent_router.py`/`chat_retrieval.py`, `llm_client.py`, pacing/prompt `/api/quiz`, bộ lọc catalog `/api/roadmap`).
  2. Bật chuẩn hóa tiếng Việt không dấu (`accent_insensitive=true`) để khắc phục `0%` typo VI.
  3. Bổ sung thẩm định chuyên gia con người trên tập mẫu đại diện và xây dựng tập kiểm chứng held-out độc lập.

## 9. Kết luận

Chuỗi đánh giá 5 giai đoạn cùng đợt bổ sung Live Groq / HTTP End-to-End đã hoàn tất trọn vẹn với bằng chứng thực nghiệm tái lập cao. Về mặt nghiên cứu truy xuất, **A4 (`recursive_512_100` + `Hybrid Rerank`, khuyến nghị thăm dò `pool=30`)** là cấu hình cân bằng hiệu quả nhất; về mặt kỹ thuật sản phẩm, ứng dụng hiện ở trạng thái **`NOT PRODUCTION-READY`** cho đến khi 4 lỗi backend đo được được khắc phục và xác minh lại.

## 10. Checkpoint, Tái lập và Phụ lục 34 Biểu đồ Gốc

- **Checkpoints Git:** `8fe5458` (Stage 01) → `ff2187e` (Cache) → `0012a0c` (Stage 02) → `c21700a` (Stage 03) → `966ecfa` (Stage 04) → `139957c` (Stage 05) → `9b09c2e` (AC-04 Completion).
- **Lệnh tái lập & kiểm định chỉ-đọc:**
  ```powershell
  .\.venv\Scripts\python.exe evaluation/scripts/generate_consolidated_visualizations.py --mode validate
  .\.venv\Scripts\python.exe evaluation/scripts/run_comprehensive_evaluation.py --mode validate
  .\.venv\Scripts\python.exe evaluation/scripts/generate_final_visualizations.py --mode validate --require-report
  .\.venv\Scripts\python.exe evaluation/scripts/run_ac04_completion.py --mode validate
  ```
- **Phụ lục đối chiếu 34 biểu đồ gốc (bảo toàn nguyên vẹn byte-for-byte dưới `evaluation/figures/original/`):**
  - *Stage 05 (28 biểu đồ):* [fig01_chunk_count_and_storage](../figures/original/png/fig01_chunk_count_and_storage.png), [fig02_chunk_length_distribution](../figures/original/png/fig02_chunk_length_distribution.png), [fig03_token_distribution_and_context_volume](../figures/original/png/fig03_token_distribution_and_context_volume.png), [fig04_truncation_rate_and_discarded_tokens](../figures/original/png/fig04_truncation_rate_and_discarded_tokens.png), [fig05_source_coverage_ku_and_redundancy](../figures/original/png/fig05_source_coverage_ku_and_redundancy.png), [fig06_precision_at_k](../figures/original/png/fig06_precision_at_k.png), [fig07_pooled_recall_at_k](../figures/original/png/fig07_pooled_recall_at_k.png), [fig08_mrr_at_k](../figures/original/png/fig08_mrr_at_k.png), [fig09_ndcg_at_k](../figures/original/png/fig09_ndcg_at_k.png), [fig10_map_at_k](../figures/original/png/fig10_map_at_k.png), [fig11_f1_at_k](../figures/original/png/fig11_f1_at_k.png), [fig12_hit_rate_and_judged_at_k](../figures/original/png/fig12_hit_rate_and_judged_at_k.png), [fig13_evidence_recall_and_semantic_ku_at_k](../figures/original/png/fig13_evidence_recall_and_semantic_ku_at_k.png), [fig14_evidence_precision_and_redundancy_at_k](../figures/original/png/fig14_evidence_precision_and_redundancy_at_k.png), [fig15_complete_vs_partial_evidence_coverage](../figures/original/png/fig15_complete_vs_partial_evidence_coverage.png), [fig16_mean_latency_at_k](../figures/original/png/fig16_mean_latency_at_k.png), [fig17_p95_latency_at_k](../figures/original/png/fig17_p95_latency_at_k.png), [fig18_retrieval_latency_breakdown](../figures/original/png/fig18_retrieval_latency_breakdown.png), [fig19_cold_vs_warm_and_prod_profile_latency](../figures/original/png/fig19_cold_vs_warm_and_prod_profile_latency.png), [fig20_resource_usage](../figures/original/png/fig20_resource_usage.png), [fig21_answer_quality_comparison](../figures/original/png/fig21_answer_quality_comparison.png), [fig22_rag_vs_norag_and_ablation_study](../figures/original/png/fig22_rag_vs_norag_and_ablation_study.png), [fig23_candidate_pool_and_topk_sensitivity](../figures/original/png/fig23_candidate_pool_and_topk_sensitivity.png), [fig24_context_budget_sensitivity](../figures/original/png/fig24_context_budget_sensitivity.png), [fig25_quality_latency_pareto_tradeoff](../figures/original/png/fig25_quality_latency_pareto_tradeoff.png), [fig26_statistical_ci_subgroups_and_audit](../figures/original/png/fig26_statistical_ci_subgroups_and_audit.png), [fig27_robustness_and_study_features](../figures/original/png/fig27_robustness_and_study_features.png), [fig28_error_distribution_and_taxonomy](../figures/original/png/fig28_error_distribution_and_taxonomy.png).
  - *Bổ sung AC-04 (6 biểu đồ):* [sup_fig01_live_groq_answer_quality](../figures/original/png/sup_fig01_live_groq_answer_quality.png), [sup_fig02_controlled_agent_vs_live_groq](../figures/original/png/sup_fig02_controlled_agent_vs_live_groq.png), [sup_fig03_http_e2e_latency](../figures/original/png/sup_fig03_http_e2e_latency.png), [sup_fig04_retrieval_vs_generation_vs_total_time](../figures/original/png/sup_fig04_retrieval_vs_generation_vs_total_time.png), [sup_fig05_live_quiz_and_roadmap](../figures/original/png/sup_fig05_live_quiz_and_roadmap.png), [sup_fig06_live_api_robustness_and_reliability](../figures/original/png/sup_fig06_live_api_robustness_and_reliability.png).
