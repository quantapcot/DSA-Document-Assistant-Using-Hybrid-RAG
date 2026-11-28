# BÁO CÁO GIAI ĐOẠN 03 — BỘ DỮ LIỆU CÂU HỎI VÀ GROUND TRUTH ĐÁNH GIÁ

> **Dự án:** DSA Document Assistant Using Hybrid RAG  
> **Giai đoạn:** Stage 03 (40 Shared Questions, Top-60 Pooling & Automated LLM-as-a-Judge Ground Truth)  
> **Branch:** `feat/final-readiness` | **Checkpoint:** `c21700a` | **Dataset Version:** `stage03_dsa_40q_v1.1_ai_reviewed`  
> **Quản trị gán nhãn:** **`human_validated = false`**, **`human_approval_status = "DRAFT"`**  
> **Trạng thái:** **ACCEPTED (WITH AI-JUDGED GOVERNANCE DISCLOSURE)**

---

## 1. Mục tiêu

Xây dựng bộ câu hỏi chuẩn dùng chung cho cả hai chiến lược phân đoạn, chuẩn hóa đáp án tham chiếu, đơn vị kiến thức (Knowledge Units) và bằng chứng nguồn xác thực đến từng ký tự (`source_evidence`), thực hiện pooling Top-60 / Pool-120 trên cả 8 tổ hợp thực nghiệm, và tạo Ground Truth gán nhãn liên quan phân bậc (`0/1/2`) có kiểm toán chéo tự động.

## 2. Phương pháp thực hiện

- **Bộ 40 câu hỏi dùng chung (`Q001–Q040`):** Thiết kế cân bằng trên **8 chủ đề DSA** và **10 dạng câu hỏi** (4 câu/dạng), gồm **24 câu tiếng Việt + 16 câu tiếng Anh** (độc lập, không dịch máy nguyên văn), phân bố độ khó **12 Easy + 18 Medium + 10 Hard**.
- **Xác thực bằng chứng nguồn (`Source Evidence`):** Định vị **91 Knowledge Units (KUs)** và **107 khoảng văn bản chuẩn hóa** (`single`, `alternative`, `composite`) khớp nguyên văn 100% trên `canonical_fingerprint_sha256 = 712f3847...`, đảm bảo cả Profile A và Profile B đều bảo toàn **`91/91` KUs (`100%`)**.
- **Pooling Top-60 / Candidate-120 (Blind-Method):** Chạy **320 lượt truy xuất chính thức** (`40` câu × `8` tổ hợp) trên GPU RTX 5060, thu `19.200` ứng viên thô → loại trùng `53,07%` còn `9.011` ứng viên duy nhất → bổ sung `57` đoạn chứa bằng chứng nguồn (`11` Profile A + `46` Profile B) → tổng cộng **`9.068` ứng viên gán nhãn** (`4.526` Profile A + `4.542` Profile B).
- **Quy trình gán nhãn tự động LLM-as-a-Judge có phân xử (Adjudication):**
  - Chạy đánh giá câu hỏi tự động (`v1 → v1.1_ai_reviewed`, 5 chỉnh sửa siêu nhỏ, 0 sửa văn bản câu hỏi).
  - Gán nhãn chính (Primary AI Judge) trên toàn bộ `9.068` cặp câu hỏi–chunk theo thang bậc `0` (không liên quan), `1` (hỗ trợ/một phần), `2` (trực tiếp trả lời).
  - Rà soát bậc hai (Secondary Audit) và phân xử độc lập (Adjudication) cho các ca mâu thuẫn, tạo tập Ground Truth cuối cùng cùng workbook Excel kiểm chứng (`ground_truth_automated.xlsx`).

## 3. Dữ liệu đầu vào

- Hai frozen benchmark corpora Stage 02 (`6.366` Recursive chunks và `13.390` Adaptive chunks) và chỉ mục Qdrant cô lập `data/runtime/qdrant_eval/` (`19.756` points).
- Ngân hàng 40 câu hỏi & kịch bản định vị bằng chứng trong `evaluation/stage03/question_bank_40.py`.

## 4. Nội dung triển khai

- `evaluation/stage03/dataset_schema.py`, `question_bank_40.py`, `pooling.py`, `workbook.py`, `automated_workbook.py`, `judge.py`, `importer.py`.
- CLI scripts: `evaluation/scripts/prepare_evaluation_dataset.py`, `generate_annotation_pools.py`, `import_ground_truth.py`.
- Xuất bộ dữ liệu đóng băng dưới `evaluation/datasets/`:
  - [questions_40.jsonl](../datasets/questions_40.jsonl) (`40` câu hỏi, `91` KUs)
  - [source_evidence.jsonl](../datasets/source_evidence.jsonl) (`107` khoảng nguồn xác thực)
  - [ground_truth_recursive.jsonl](../datasets/ground_truth_recursive.jsonl) (`4.526` nhãn Profile A)
  - [ground_truth_adaptive.jsonl](../datasets/ground_truth_adaptive.jsonl) (`4.542` nhãn Profile B)
  - [dataset_manifest.json](../datasets/dataset_manifest.json) (SHA-256 đóng băng toàn bộ bộ dữ liệu)

## 5. Kết quả

| Hạng mục Stage 03 | Giá trị thực đo | Ghi chú |
|---|---:|---|
| Số câu hỏi dùng chung (`Q001–Q040`) | **`40`** (`24` VI + `16` EN) | `12` Easy / `18` Medium / `10` Hard; `8` chủ đề; `10` dạng câu |
| Knowledge Units / Canonical Evidence Spans | **`91` KUs / `107` spans** | `91/91` KUs bảo toàn trên cả Profile A và Profile B |
| Official Pooling Calls (`top_k=60`, `pool=120`) | **`320 / 320`** | `0` lỗi, `0` OOM, `0` lượt `< 60` kết quả |
| Raw / Unique / Evidence-Added / Final Candidates | `19.200` / `9.011` / `57` / **`9.068`** | Profile A: **`4.526`** \| Profile B: **`4.542`**; dedup `53,07%` |
| Độ phủ gán nhãn trên Top-60 (`Judged@60`) | **`1,000` (`100%`)** | `0` vị trí chưa gán nhãn trên toàn bộ `2.880` ô (`40×8×9`) |
| Độ nhạy phân xử AI (`|Δ nDCG@5|`, Stage 04 audit) | **`≤ 0,011`** | Không làm thay đổi thứ hạng của 8 tổ hợp |

## 6. Kiểm thử và xác minh

- Kiểm chứng 100% tọa độ `[char_start, char_end)` của `107/107` evidence spans khớp văn bản chuẩn hóa gốc.
- Bộ unit test `tests/unit/test_ground_truth.py` kiểm định schema, blind-method pooling, importer, phân bổ nhãn và kiểm tra toàn vẹn checksum đều đạt **PASS** (`308 passed` tại checkpoint chuẩn bị, toàn bộ targeted ground truth tests pass liên tục ở các giai đoạn sau).

## 7. Acceptance Criteria

- Toàn bộ cổng kỹ thuật Stage 03 (câu hỏi, bằng chứng, pooling, deduplication, gán nhãn tự động, kiểm toán bậc hai, import Ground Truth và checksums) đạt **PASS** trong phạm vi tự động hóa được phê duyệt, kèm gắn cờ quản trị bắt buộc **`human_validated = false`** và **`human_approval_status = "DRAFT"`**.

## 8. Vấn đề và giới hạn

- **Không có xác nhận thủ công của chuyên gia con người (`human_validated = false`):** Nhãn liên quan `0/1/2` được sinh và phân xử tự động bằng LLM-as-a-Judge; dù phân tích độ nhạy ở Stage 04 cho thấy chênh lệch giữa Primary AI và Final Adjudicated rất nhỏ (`|Δ nDCG@5| ≤ 0,011`), điều này **không thay thế** thẩm định con người.
- **Không có tập kiểm chứng độc lập (no independent held-out validation set):** Mọi kết luận cấu hình đều chỉ mang tính thăm dò trên tập 40 câu hỏi này.
- **Pooled Recall@K:** Mẫu số Recall dựa trên tập ứng viên đã pooling (Top-60 từ 8 tổ hợp + evidence-added), không phải toàn bộ 19.756 chunks trong corpus.

## 9. Kết luận

Bộ dữ liệu đánh giá `stage03_dsa_40q_v1.1_ai_reviewed` (`40` câu hỏi, `91` KUs, `107` bằng chứng nguồn, `9.068` nhãn Ground Truth) hoàn chỉnh, tái lập và bảo đảm độ phủ `Judged@60 = 1,000` cho toàn bộ Stage 04 và Stage 05.

## 10. Checkpoint và tài liệu liên quan

- **Checkpoint Git:** `c21700a` (`feat(evaluation): complete automated pooled ground truth benchmark`).
- **Frozen Artifacts (giữ nguyên đường dẫn gốc):** `evaluation/datasets/*` và `evaluation/artifacts/stage03/*` (liệt kê đầy đủ trong [relocation_manifest.json](../manifests/relocation_manifest.json)).
- **Tài liệu liên quan:** [Báo cáo Stage 02](02_chunking_indexing.md) | [Báo cáo Stage 04](04_system_evaluation.md) | [Evaluation README](../README.md).
