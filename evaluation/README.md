# ĐÁNH GIÁ HỆ THỐNG — DSA DOCUMENT ASSISTANT USING HYBRID RAG

> **Trạng thái 5 giai đoạn:** Stage 01–05 + Bổ sung AC-04 (Live Groq & HTTP E2E) **HOÀN TẤT** trên nhánh `feat/final-readiness`  
> **Quản trị đánh giá:** `human_validated = false` | `human_approval_status = "DRAFT"` | Chưa có tập held-out độc lập  
> **Trạng thái sản phẩm:** **`NOT PRODUCTION-READY`** *(do 4 lỗi backend thực đo được ghi nhận bên dưới)*

---

## 1. Năm Báo cáo Chính thức

| Giai đoạn | Báo cáo chính thức | Trạng thái | Checkpoint |
|---|---|---|---|
| **Stage 01** | [01_environment.md](reports/01_environment.md) — Môi trường & Ổn định Runtime | ACCEPTED (`20/20 PASS`) | `8fe5458`, `ff2187e` |
| **Stage 02** | [02_chunking_indexing.md](reports/02_chunking_indexing.md) — Phân đoạn kép & Chỉ mục Qdrant cô lập | ACCEPTED (`0` lỗi provenance) | `0012a0c` |
| **Stage 03** | [03_dataset_ground_truth.md](reports/03_dataset_ground_truth.md) — Bộ 40 câu hỏi & Ground Truth | ACCEPTED (`human_validated=false`) | `c21700a` |
| **Stage 04 + AC-04** | [04_system_evaluation.md](reports/04_system_evaluation.md) — Đánh giá Toàn diện & Live Groq/HTTP E2E | Lịch sử `PARTIAL` → Bổ sung `PASS` (Eval Gate) | `966ecfa`, `9b09c2e` |
| **Stage 05** | [05_final_evaluation.md](reports/05_final_evaluation.md) — Trực quan hóa, Phân tích RQ1–RQ8 & Kết luận | ACCEPTED (`6/6 PASS`) | `139957c` |

---

## 2. Mười Biểu đồ Hợp nhất Tiếng Việt (`300 DPI PNG` + `SVG`)

Tất cả 10 biểu đồ được tái sinh trực tiếp từ dữ liệu thực nghiệm gốc, bảo toàn nguyên vẹn **34 biểu đồ gốc** trong `figures/original/` và ánh xạ đầy đủ trong [figure_mapping.json](figures/figure_mapping.json) & [figure_data_parity.csv](tables/figure_data_parity.csv).

1. **Hình 01 — Đánh giá chiến lược Chunking:** [PNG](figures/consolidated/png/fig01_danh_gia_chunking.png) | [SVG](figures/consolidated/svg/fig01_danh_gia_chunking.svg)
2. **Hình 02 — Đánh giá chất lượng Retrieval:** [PNG](figures/consolidated/png/fig02_chat_luong_retrieval.png) | [SVG](figures/consolidated/svg/fig02_chat_luong_retrieval.svg)
3. **Hình 03 — Đánh giá chất lượng Evidence:** [PNG](figures/consolidated/png/fig03_chat_luong_evidence.png) | [SVG](figures/consolidated/svg/fig03_chat_luong_evidence.svg)
4. **Hình 04 — Hiệu năng và Tài nguyên Retrieval:** [PNG](figures/consolidated/png/fig04_hieu_nang_va_tai_nguyen.png) | [SVG](figures/consolidated/svg/fig04_hieu_nang_va_tai_nguyen.svg)
5. **Hình 05 — Chất lượng sinh câu trả lời (Controlled Agent):** [PNG](figures/consolidated/png/fig05_chat_luong_cau_tra_loi.png) | [SVG](figures/consolidated/svg/fig05_chat_luong_cau_tra_loi.svg)
6. **Hình 06 — Ablation và Phân tích độ nhạy:** [PNG](figures/consolidated/png/fig06_ablation_va_do_nhay.png) | [SVG](figures/consolidated/svg/fig06_ablation_va_do_nhay.svg)
7. **Hình 07 — Thống kê và Cân bằng Hiệu năng (Pareto):** [PNG](figures/consolidated/png/fig07_thong_ke_va_tradeoff.png) | [SVG](figures/consolidated/svg/fig07_thong_ke_va_tradeoff.svg)
8. **Hình 08 — Độ tin cậy và Phân tích lỗi thực nghiệm:** [PNG](figures/consolidated/png/fig08_do_tin_cay_va_phan_tich_loi.png) | [SVG](figures/consolidated/svg/fig08_do_tin_cay_va_phan_tich_loi.svg)
9. **Hình 09 — Đánh giá Live Groq và HTTP End-to-End:** [PNG](figures/consolidated/png/fig09_live_groq_va_http_e2e.png) | [SVG](figures/consolidated/svg/fig09_live_groq_va_http_e2e.svg)
10. **Hình 10 — Chức năng thực tế và Độ ổn định API:** [PNG](figures/consolidated/png/fig10_chuc_nang_va_do_on_dinh_api.png) | [SVG](figures/consolidated/svg/fig10_chuc_nang_va_do_on_dinh_api.svg)

---

## 3. Bộ dữ liệu, 8 Tổ hợp & Phát hiện Trọng tâm

- **Dữ liệu đóng băng:** `54` tài liệu DSA (`1.733` trang), `6.366` Recursive chunks (Profile A) vs `13.390` Adaptive chunks (Profile B), `40` câu hỏi dùng chung (`24` VI + `16` EN), `91` Knowledge Units, `107` bằng chứng nguồn xác thực, `9.068` nhãn Ground Truth (`Judged@60 = 1,000`).
- **8 Tổ hợp thực nghiệm:** Profile A (`A1` BM25, `A2` Dense, `A3` Hybrid RRF, `A4` Hybrid Rerank) và Profile B (`B1` BM25, `B2` Dense, `B3` Hybrid RRF, `B4` Hybrid Rerank), đánh giá trên 9 mốc `K ∈ {1, 3, 5, 10, 15, 20, 30, 45, 60}`.
- **Phát hiện chính:**
  - `A4` dẫn đầu truy xuất (`nDCG@5 = 0,7053`, `Evidence Recall@15 = 0,9083`, `Controlled Correctness = 0,9250`).
  - `NO_RAG` đạt `Controlled Correctness = 0,9688` (cao hơn `A4`); `A1` vượt `A4` ở `Faithfulness` (`0,9250`) và `Citation Accuracy` (`0,8500`).
  - `Profile B` loại bỏ cắt cụt MiniLM (`0%` vs `43,2%` ở Recursive) và giúp `B4` tăng tốc **`~41,2%` ở riêng bước reranking cô lập**, nhưng làm giảm `nDCG@5` và `Complete Evidence` do chunk ngắn.
  - **Khuyến nghị cấu hình thăm dò:** `A4` với `candidate_top_k = 30` (giữ `99,8%` `nDCG@5`, giảm mạnh độ trễ truy xuất; mang tính thăm dò vì chưa có held-out và `human_validated = false`).

---

## 4. Cảnh báo Sản phẩm (`NOT PRODUCTION-READY`) & 4 Lỗi Backend Thực đo

Đợt đánh giá **Live Groq (`openai/gpt-oss-120b`) & HTTP End-to-End** (`166` yêu cầu HTTP/API gồm `157` nhóm chính + `9` edge cases; **`90` lệnh gọi Groq thực tế**; `117.091` tokens) phơi bày 4 lỗi backend thật (chưa sửa trong phạm vi đánh giá):
1. **`/api/chat`:** `82,5%` (`33/40`) từ chối dương tính giả (`reason=off_topic`) trong `intent_router.py` / `chat_retrieval.py`.
2. **`llm_client.py`:** Ném `LLMError("truncated_json")` khi plain-text chat chạm `max_tokens=1.024`.
3. **`/api/quiz`:** Ứng dụng trả HTTP `200` nhưng `3/8` chủ đề gặp `6` lỗi `429` ở tầng provider Groq do vượt `8.000 TPM`; chủ đề gốc `QZ_03` trả HTTP `422`.
4. **`/api/roadmap`:** `0/4` kịch bản sinh lộ trình (`abstained`).

---

## 5. Cấu trúc Thư mục Kết quả & Bản sao Cục bộ

- `reports/` — Đúng 5 báo cáo chính thức (`01`–`05`).
- `results/` — Kết quả thực nghiệm phân loại theo nội dung nghiên cứu: `chunking/`, `retrieval/`, `evidence/`, `performance/`, `generation/{controlled_agent,live_groq}/`, `http_e2e/`, `robustness/{controlled,live_http}/`, `study_features/{controlled,live_http}/`, `ablation/`, `sensitivity/`, `statistics/`, `error_analysis/`, `tradeoff/`.
- `figures/` — `consolidated/{png,svg}/` (10 hình), `original/{png,svg}/` (34 hình gốc), `figure_mapping.json`.
- `tables/` — 10 bảng tổng hợp CSV + `figure_data_parity.csv`.
- `manifests/` — Toàn bộ manifests lịch sử (bất biến checksum) + `relocation_manifest.json`, `consolidated_figure_manifest.json`, `consolidated_chart_validation.json`.
- `datasets/` & `artifacts/{recursive_512_100,structure_adaptive,stage03}/` — Dữ liệu đầu vào đóng băng Stage 02/03 (giữ nguyên đường dẫn gốc để bảo toàn SHA-256).
- **Bản sao cục bộ bắt buộc (Git-ignored):** `data/runtime/eval_stage04_backups/` (5 bản sao thô Stage 04), `data/runtime/qdrant_eval/` (chỉ mục Qdrant cô lập), `data/runtime/consolidation_rollback_archive/` (bản sao rollback trước dọn dẹp).

---

## 6. Lệnh Tái lập và Kiểm định Toàn vẹn (Read-Only)

```powershell
# 1. Kiểm định 10 biểu đồ hợp nhất, 34 biểu đồ gốc, parity, 5 báo cáo & README
.\.venv\Scripts\python.exe evaluation/scripts/generate_consolidated_visualizations.py --mode validate

# 2. Kiểm định lịch sử Stage 04, Stage 05 và AC-04 (qua evaluation/path_resolver.py)
.\.venv\Scripts\python.exe evaluation/scripts/run_comprehensive_evaluation.py --mode validate
.\.venv\Scripts\python.exe evaluation/scripts/generate_final_visualizations.py --mode validate --require-report
.\.venv\Scripts\python.exe evaluation/scripts/run_ac04_completion.py --mode validate

# 3. Bộ test hồi quy mục tiêu Evaluation (bảo toàn 4 lỗi timing Windows có sẵn ở test_llm_reliability.py)
.\.venv\Scripts\python.exe -m pytest tests/unit/test_consolidation.py tests/unit/test_ac04_completion.py tests/unit/test_stage05_visualizations.py tests/unit/test_comprehensive_evaluation.py tests/unit/test_retrieval_metrics.py tests/unit/test_ground_truth.py -q
.\.venv\Scripts\python.exe -m pip check
```
