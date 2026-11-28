# STAGE 04 – COMPREHENSIVE SYSTEM EVALUATION REPORT

> **Dự án:** DSA Document Assistant Using Hybrid RAG  
> **Giai đoạn:** `Stage 04` – Comprehensive System Evaluation *(Fully Automated Experimentation)*  
> **Workspace:** `D:\DSA-Document-Assistant-Using-Hybrid-RAG`  
> **Branch:** `feat/final-readiness`  
> **Accepted HEAD (bất biến):** `c21700a25337bcd275923c97f999f23ab28962ae`  
> **Parent Commit:** `0012a0cdfbda39c9b5117ee91e85f60e0bad23d7`  
> **Local `main` (bất biến):** `b84f09450f213b03ad57ef8b388f482ee0f0eaa2`  
> **Dataset & Ground Truth:** `stage03_dsa_40q_v1.1_ai_reviewed` (`llm_generated_pooled_relevance_judgments`, `human_validated = false`, `human_approval_status = "DRAFT"`)  
> **Ngày hoàn tất:** 2026-11-28

---

## 1. Executive Summary

Stage 04 đã triển khai và chạy thực nghiệm toàn diện trên **40 shared questions × 2 frozen chunking profiles × 4 retrieval methods (8 combinations `A1–A4, B1–B4`)** tại **9 mốc `K = [1, 3, 5, 10, 15, 20, 30, 45, 60]`** với `top_k = 60`, `candidate_top_k = 120` trên **NVIDIA GeForce RTX 5060 Laptop GPU (8 GB VRAM)** mà không sửa production code hay frozen benchmarks:

- **Retrieval & Evidence (`COMPLETED`):** `320/320` official retrieval calls đạt **`100.0%` exact ranking agreement** với Stage 03 (`0` ranking drift, `0` unjudged candidates, `Judged@K = 1.0`). **`A4 (Recursive + Hybrid Rerank)`** đạt điểm cao nhất trên phần lớn chỉ số xếp hạng và evidence chính (`nDCG@5 = 0.7053`, `MRR@5 = 0.9875`, `P@5 = 0.7000`, `Evidence Recall@5 = 0.7208`, `Evidence Recall@15 = 0.9083`, `Complete Evidence@15 = 80.0%`), nhưng **không thắng mọi chỉ số**: `B4` cao hơn nhẹ ở `Pooled Recall@5` (`0.3509` vs `0.3474`), thấp hơn nhiều về `Evidence Redundancy@5` (`0.0075` vs `0.0608`), và không bị embedding truncation (`0.0%` vs `43.2%`).
- **Chunking Trade-off (`COMPLETED`):** `structure_adaptive` loại bỏ hoàn toàn embedding truncation (`0.0%` vs `43.2%` ở `recursive_512_100`) và giảm mạnh overlap redundancy (`1.06%` vs `20.33%`), nhưng tạo `2.1×` nhiều chunks ngắn hơn (`13,390` vs `6,366`, trung bình `47.9` vs `119.4` tokens), gây phân mảnh ngữ cảnh tại các mốc `K` cố định. Đáng chú ý, Cross-Encoder Reranker (`max_length=512`) có **`0.0%` pair truncation** trên cả hai profiles, giúp `A4` khai thác trọn vẹn ngữ cảnh dài của Recursive chunks.
- **Performance & Latency (`COMPLETED`):** Hoàn tất **`14,400 / 14,400`** warm latency calls (`5` repeats × `40` queries × `8` combos × `9` Ks, `pool=120`, `0` lỗi, `0` OOM, `peak_gpu_allocated = 2,686.82 MB`), `640` stage-breakdown calls, `1,600` production-profile calls (`top_k=5, pool=10`), và `24` cold-start subprocess trials. Ở riêng **giai đoạn reranking (`stage_latency.csv`, `pool=120`)**, `B4` nhanh hơn `A4` ~`41%` (`1,364.8 ms` vs `2,320.4 ms` ở `K=5`); xét trên **tổng độ trễ warm retrieval (`latency_summary.csv`, `pool=120`)**, `B4` nhanh hơn `A4` `~23.6%` ở P50 (`2,366.8` vs `3,097.9 ms`) và `~13.0%` ở P95 (`3,214.3` vs `3,693.1 ms`, `K=5`), do QdrantLocal SQLite vẫn chiếm `~461–645 ms` mỗi lượt BM25/Dense.
- **Controlled Answer Generation, Robustness & Study Features (`PARTIAL` — Controlled Agent Completed; Production Groq API `BLOCKED`):** Do `backend/.env` không có `GROQ_API_KEY` thật (chỉ có placeholder) và production Qdrant có `0` points, Production Groq API End-to-End là **`BLOCKED`**. Thay vào đó, đã chạy **Agent-based Controlled Generation Experiment** (`400` answers = `320` RAG + `40` Without-RAG + `40` Context-Budget Sensitivity, `flash` gen + blind `pro` judge, `3/3` synthetic controls PASS, `100%` within-1pt secondary audit agreement). Trong nhóm **8 tổ hợp RAG**, `A4` đạt điểm cao nhất về `Correctness` (`0.9250`), `Relevancy` (`0.9563`), `Completeness` (`0.9167`), `Contradiction` (`0.0000`), `Complexity` (`1.0000`) và `Code` (`0.9500`), trong khi `A1` cao hơn nhẹ ở `Faithfulness` (`0.9250` vs `0.9187`) và `Citation Accuracy` (`0.8500` vs `0.8187`). **Lưu ý quan trọng:** baseline `NO_RAG` đạt `Correctness` (`0.9688`) và `Relevancy` (`1.0000`) **cao hơn `A4`** (chênh lệch Correctness `-0.0438`, `p = 0.2381`, chưa có ý nghĩa thống kê — không suy diễn là tương đương), đổi lại `NO_RAG` không có trích dẫn/faithfulness và có `Contradiction Rate` cao hơn (`0.0175` vs `0.0000`). Robustness (`20` cases): `100%` OOD, Unanswerable & Prompt Injection resistance, nhưng phát hiện điểm yếu nghiêm trọng ở Vietnamese typo không dấu (`0.0%`, `EvRecall@5` giảm `-0.6667` do `accent_insensitive: false`).
- **Pareto Frontier (`K=5` & `K=15`, `pool=120`):** **`A1`**, **`A4`**, **`B4`** không bị dominated.

---

## 2. Research Questions — Empirical Answers (`RQ1` – `RQ8`)

| RQ | Tóm tắt câu trả lời dựa trên số liệu thực đo |
|---|---|
| **RQ1 (Chunking)** | `recursive_512_100` vượt trội có ý nghĩa thống kê (Holm `p < 0.05`) về `nDCG`, `Precision`, `Evidence Recall` và đạt `Answer Correctness` cao hơn (`A4 = 0.925` vs `B4 = 0.750` ở cùng `2,048` tokens). `structure_adaptive` vượt trội về `0%` embedding truncation, `0.0075` evidence redundancy, `Pooled Recall@5` cao hơn nhẹ (`B4 = 0.3509` vs `A4 = 0.3474`) và tốc độ riêng **giai đoạn rerank** nhanh hơn `~41%` (`~13.0–23.6%` nhanh hơn ở tổng warm latency `pool=120`), nhưng bị phân mảnh ngữ cảnh và tốn `2.1×` số vector & thời gian build BM25 (`15.5s` vs `10.7s`). |
| **RQ2 (Retrieval)** | `hybrid_rerank` đạt điểm cao nhất về `nDCG`, `MRR`, `Precision` và `Evidence Recall` trên cả 2 profiles (`A4 nDCG@5 = 0.7053`, `B4 = 0.5591`). BM25 vượt trội Dense thuần trên thuật ngữ DSA/code/tiếng Việt (`A1 = 0.5584` vs `A2 = 0.3921`) và có độ trễ thấp nhất (`A1 P95 = 607.5 ms`, `0 MB` VRAM). |
| **RQ3 (Hybrid RRF)** | Hybrid RRF cải thiện mạnh và có ý nghĩa thống kê so với Dense (`A3 vs A2`: `+0.1115 nDCG@5`, Holm `p=0.0032`; `B3 vs B2`: `+0.1035`, `p=0.0054`). So với BM25 tại `K=5`, điểm trung bình RRF thấp hơn (`A3 vs A1`: `-0.0547`, Holm `p=0.1314`, chưa đủ bằng chứng thống kê ở `α=0.05`). Đóng góp thực dụng chính của RRF là mở rộng Recall@60 (`0.7956 → 0.8445` ở Profile A) để cung cấp pool tốt hơn cho Reranker. |
| **RQ4 (Reranker)** | Reranker là nguồn cải thiện xếp hạng lớn nhất: `A4 vs A3` tăng **`+0.2016 nDCG@5` (`+40.0%`, `p=0.001`)** và **`+0.1333 EvRecall@15` (`p=0.029`)**; `B4 vs B3` tăng **`+0.1428 nDCG@5` (`+34.3%`, `p=0.0032`)**. Chi phí riêng giai đoạn rerank (`stage_latency.csv`, `pool=120`): `+2,320 ms` (`A4`) / `+1,365 ms` (`B4`). Trên các thí nghiệm tách biệt: `production_profile_latency.csv` (`pool=10`) chỉ thêm `~268 ms` P50 (`A4` vs `A3`), còn `candidate_pool_sensitivity.csv` (`pool=30`, single-pass) đạt `nDCG@5 = 0.7039` với `mean = 1,628 ms` (`P95 = 1,777 ms`). |
| **RQ5 (Top-K)** | `Precision@K` giảm đều khi `K` tăng, trong khi `Evidence Recall` tăng mạnh từ `K=1→15` (`A4`: `0.475 → 0.721@5 → 0.908@15`) và bão hòa sau `K=30` (`0.963@30 → 0.971@60`). **`K=5–10`** (hoặc ngân sách `2,048` tokens) là điểm cân bằng tốt nhất cho RAG chat; **`K=15`** cho bài toán cần `Complete Evidence ≥ 80%`. |
| **RQ6 (Generation)** | **`NO_RAG` đạt `Correctness` (`0.9688`) và `Relevancy` (`1.0000`) cao hơn mọi tổ hợp RAG** (kể cả `A4`: `0.9250` và `0.9563`; chênh lệch `Correctness = -0.0438`, `p = 0.2381`, `Completeness = -0.0042`, `p = 0.9009` — chưa có ý nghĩa thống kê ở `α=0.05`, không suy diễn là chứng minh tương đương). Giá trị chính của RAG `A4` là giảm `Contradiction` về `0.0000` (vs `0.0175` của `NO_RAG`), đạt `Faithfulness = 0.9187` và `Citation Validity = 100%`. Với retriever yếu (`B2 = 0.4813`), quy tắc nghiêm cấm dùng trí nhớ ngoài khiến abstention làm giảm correctness rõ rệt. |
| **RQ7 (Efficiency)** | Trên **main warm benchmark (`pool=120`, `14,400` calls)**, biên Pareto gồm `A1` (nhanh nhất, `P95 = 607.5 ms`, `0 MB` VRAM), `A4` (chất lượng truy xuất & evidence cao nhất, `P95 = 3,693.1 ms`, `2.66 GB` VRAM), và `B4` (tổng P95 thấp hơn `A4` `13.0%` tại `K=5` nhờ rerank nhanh hơn `41%`, `P95 = 3,214.3 ms`). Từ thí nghiệm nhạy cảm `candidate_pool_sensitivity.csv` (single-pass, tách biệt khỏi main warm latency), **`A4` với `candidate_top_k = 30`** (`nDCG@5 = 0.7039`, `EvRecall@5 = 0.7375`, `P95 = 1,777 ms`) là ứng viên cân bằng thực dụng nhất *(exploratory selection, chưa held-out validated)*. |
| **RQ8 (Reliability)** | Hệ thống đạt `100%` chống OOD (`4/4`), Unanswerable (`4/4`) và Prompt Injection (`3/3`), `66.7%` Distractor Context, nhưng dễ suy giảm mạnh trước Vietnamese typo không dấu (`0.0%`, `EvRecall@5` `-0.6667`) và paraphrase tiếng Anh (`33.3%`, `-0.5556`). |

---

## 3. Experimental Setup (`COMPLETED`)

- **Phần cứng & Phần mềm:** Windows 10/11 (`14` physical / `20` logical CPU cores, `15.71 GB` RAM), **NVIDIA GeForce RTX 5060 Laptop GPU (`7.96 GB` VRAM)**, Python `3.11.9`, PyTorch `2.11.0+cu128`, CUDA `12.8`.
- **Mô hình truy xuất (từ Stage 02/03 manifests):**
  - Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (rev `e8f8c211...`, `max_seq_length = 128`, `cuda:0`).
  - Reranker: `BAAI/bge-reranker-v2-m3` (rev `953dc6f6...`, `max_length = 512`, `batch_size = 4`, `cuda:0`).
  - BM25: `k1 = 1.5`, `b = 0.75`, `accent_insensitive = false`, `bm25_cache_ttl = 300s`. RRF: `rrf_k = 60`.
- **Orchestrator & Manifest:** [run_comprehensive_evaluation.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/scripts/run_comprehensive_evaluation.py), [manifest.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/manifest.json).

---

## 4. Dataset & Ground Truth Verification (`COMPLETED`)

Toàn bộ inputs đóng băng được kiểm tra checksum trước và sau Stage 04 (`100% IDENTICAL`):
- [questions_40.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/questions_40.jsonl) (`c620d728...`, `40` queries: `24` VI, `16` EN; `12` Easy, `18` Med, `10` Hard; `91` KUs).
- [source_evidence.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/source_evidence.jsonl) (`53ffa141...`, `107` verified canonical spans).
- Frozen Corpus A (`e39811da...`, `6,366` chunks) & Frozen Corpus B (`81c56827...`, `13,390` chunks), Evaluation Qdrant `eval_dsa_chunks_v1` (`19,756` points).
- [ground_truth_recursive.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/ground_truth_recursive.jsonl) (`0a0717b0...`, `4,526` judged: `0=4,057, 1=300, 2=169`, `0` zero-positive) & [ground_truth_adaptive.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/ground_truth_adaptive.jsonl) (`91bdcba2...`, `4,542` judged: `0=4,151, 1=266, 2=125`, `0` zero-positive).

---

## 5. Chunking Evaluation (`COMPLETED`)

**Artifacts:** [chunking_metrics.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/chunking/chunking_metrics.csv), [chunking_diagnostics.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/chunking/chunking_diagnostics.json).

| Metric | Profile A (`recursive_512_100`) | Profile B (`structure_adaptive`) |
|---|---:|---:|
| Total Chunks | `6,366` | `13,390` (`2.10×`) |
| Char Length (Mean / Median / P95 / Max) | `361.2 / 421 / 509 / 512` | `141.5 / 112 / 323 / 1,871` |
| Embedding Tokens (Mean / P95 / Max, limit `128`) | `119.4 / 226 / 398` | `47.9 / 111 / 128` |
| Embedding Truncation Rate / Discarded Tokens | **`43.20%`** (`114,961` = `15.13%`) | **`0.00%`** (`0` = `0.00%`) |
| Reranker Pair Tokens (Mean / P95 / Max, limit `512`) | `168.0 / 233 / 363` (`0.0%` trunc) | `101.0 / 160 / 190` (`0.0%` trunc) |
| Retrieved Emb Tokens/Query (`K=5 / 15 / 60`) | `654.5 / 1,939.3 / 7,532.9` | `309.2 / 842.1 / 3,347.6` |
| Effective Source Coverage / KU Retention | `91.36%` / `91/91` | `88.93%` / `91/91` |
| Overlap Redundancy / Synthetic Prefix | `20.33%` / `0.00%` | `1.06%` / `1.36%` |
| Typed Structural Chunks (Code+Pseudo / Formula / Table) | `0 / 0 / 0` (untyped) | `3,434 / 1,110 / 89` |
| Corpus JSONL Size / Theoretical Vector Size (f32 384d)* | `10.44 MB` / `9.78 MB` (est.) | `16.29 MB` / `20.57 MB` (est.) |

*\*Combined physical QdrantLocal SQLite storage (`data/runtime/qdrant_eval/`) = **`105,722,471` bytes (`100.8 MB`)** dùng chung cho cả hai profiles; con số vector size ở bảng trên là ước lượng lý thuyết `chunk_count × 384 × 4 B`.*

---

## 6. Main Retrieval Evaluation (`COMPLETED`)

**Protocol:** `320` official retrieval calls (`40 × 8`, `top_k=60`, `pool=120`, `cuda:0`), đánh giá tại 9 mốc `K = [1..60]`.  
**Ranking Drift ([ranking_drift.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/retrieval/ranking_drift.json)):** **`320 / 320` (`100.0%` exact ranking agreement)** với Stage 03, **`0` unjudged candidates** (`Judged@K = 1.0` ở mọi `K`).  
**Artifacts:** [retrieval_rankings.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/retrieval/retrieval_rankings.jsonl), [retrieval_metrics_per_query.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/retrieval/retrieval_metrics_per_query.csv), [retrieval_metrics_summary.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/retrieval/retrieval_metrics_summary.csv), [strict_relevance_metrics.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/retrieval/strict_relevance_metrics.csv), [judgment_coverage.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/retrieval/judgment_coverage.csv).

| Combo | P@5 | Pooled R@5 | MRR@5 | nDCG@5 | MAP@5 | F1@5 | Hit@5 | nDCG@15 | Pooled R@15 | nDCG@60 | Pooled R@60 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A1** (Rec+BM25) | `0.6000` | `0.2899` | `0.9104` | `0.5584` | `0.5566` | `0.3650` | `1.000` | `0.5672` | `0.5227` | `0.6476` | `0.7956` |
| **A2** (Rec+Dense) | `0.4700` | `0.2340` | `0.7079` | `0.3921` | `0.3816` | `0.2950` | `0.900` | `0.4099` | `0.4235` | `0.4769` | `0.6198` |
| **A3** (Rec+RRF) | `0.5700` | `0.2845` | `0.8467` | `0.5037` | `0.5003` | `0.3525` | `1.000` | `0.5351` | `0.5157` | `0.6407` | `0.8445` |
| **A4** (Rec+Rerank) | **`0.7000`** | `0.3474` | **`0.9875`** | **`0.7053`** | **`0.6638`** | **`0.4374`** | **`1.000`** | **`0.7335`** | **`0.6745`** | **`0.8097`** | **`0.9447`** |
| **B1** (Ada+BM25) | `0.4700` | `0.2677` | `0.8563` | `0.4414` | `0.4373` | `0.3165` | `0.950` | `0.4470` | `0.4320` | `0.5398` | `0.6975` |
| **B2** (Ada+Dense) | `0.3400` | `0.1932` | `0.6246` | `0.3128` | `0.2872` | `0.2335` | `0.775` | `0.3227` | `0.3177` | `0.3948` | `0.5406` |
| **B3** (Ada+RRF) | `0.4350` | `0.2629` | `0.7863` | `0.4163` | `0.3894` | `0.3047` | `0.925` | `0.4389` | `0.4312` | `0.5414` | `0.7297` |
| **B4** (Ada+Rerank) | `0.6000` | **`0.3509`** | `0.9008` | `0.5591` | `0.5638` | `0.4141` | `1.000` | `0.5908` | `0.5677` | `0.6745` | `0.8454` |

*(Strict Grade-2-only sensitivity trong `strict_relevance_metrics.csv` giữ nguyên thứ tự `A4 > B4 ≈ A1 > A3 > B1 > B3 > A2 > B2`.)*

---

## 7. Evidence Evaluation (`COMPLETED`)

**Policy:** Source-backed interval union (`source_spans ∪ overlap_segments ∪ source_backed synthetic_segments`), ngưỡng `overlap_chars >= min(20, max(8, floor(0.25 * evidence_span_len)))`, hỗ trợ `single`, `alternative`, `composite` (union nhiều chunks).  
**Phân biệt tường minh Complete vs Partial Evidence Coverage (`evaluation/stage04/evidence_metrics.py`, đã xác minh bất biến `complete_rate ≤ evidence_recall ≤ partial_coverage ≤ ku_coverage_any` trên toàn bộ 72 dòng summary):**
- `complete_evidence_rate_at_k`: Tỷ lệ câu hỏi có **100% required KUs** được thỏa mãn đầy đủ theo đúng `evidence_mode`.
- `evidence_recall_at_k`: Trung bình trên 40 câu của `covered_complete_kus / total_kus` (chỉ tính các KU đạt ngưỡng đầy đủ).
- `partial_evidence_coverage_at_k`: Điểm trung bình per-KU có cộng nhận độ phủ ký tự thành phần (`1.0` nếu KU hoàn chỉnh, ngược lại là tỷ lệ ký tự giao `∈ [0, 1)` của các span thành phần chưa đủ ngưỡng) — phản ánh tiến độ phủ một phần (`A4@5 = 0.7521` vs `EvRecall@5 = 0.7208`; `B4@5 = 0.5521` vs `0.5125`).
- `ku_coverage_at_k`: Tỷ lệ KU có ít nhất một giao source-backed `> 0` (hoàn chỉnh hoặc một phần).
**Artifacts:** [evidence_metrics_per_query.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/evidence/evidence_metrics_per_query.csv), [evidence_metrics_summary.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/evidence/evidence_metrics_summary.csv), [knowledge_unit_coverage.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/evidence/knowledge_unit_coverage.jsonl), [evidence_redundancy.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/evidence/evidence_redundancy.csv).

| Combo | EvRecall@5 | EvPrec@5 | CompleteEv@5 | SemKU@5 | EvRedund@5 | EvRecall@15 | CompleteEv@15 | EvRecall@60 | CompleteEv@60 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A1** | `0.6583` | `0.3200` | `0.450` | `0.8292` | `0.0469` | `0.8000` | `0.600` | `0.9417` | `0.875` |
| **A2** | `0.4958` | `0.2200` | `0.275` | `0.7375` | `0.0415` | `0.6500` | `0.425` | `0.7667` | `0.600` |
| **A3** | `0.6042` | `0.2700` | `0.350` | `0.8042` | `0.0380` | `0.7750` | `0.625` | `0.9292` | `0.850` |
| **A4** | **`0.7208`** | **`0.3550`** | **`0.550`** | **`0.8958`** | `0.0608` | **`0.9083`** | **`0.800`** | **`0.9708`** | **`0.925`** |
| **B1** | `0.3917` | `0.2100` | `0.175` | `0.6625` | `0.0075` | `0.5542` | `0.325` | `0.7750` | `0.575` |
| **B2** | `0.3000` | `0.1550` | `0.100` | `0.5417` | `0.0000` | `0.3917` | `0.225` | `0.6042` | `0.375` |
| **B3** | `0.3750` | `0.2000` | `0.150` | `0.6875` | `0.0000` | `0.5375` | `0.300` | `0.7458` | `0.550` |
| **B4** | `0.5125` | `0.2850` | `0.300` | `0.8083` | `0.0075` | `0.7500` | `0.475` | `0.8500` | `0.650` |

*(Ghi chú: `SemKU@K` dựa trên nhãn LLM Stage 03 cao hơn `EvRecall@K` dựa trên registered canonical spans vì corpus có các đoạn giải thích đúng cùng kiến thức ở vị trí khác.)*

---

## 8. Performance & Latency Evaluation (`COMPLETED`)

**Workload hoàn tất:** `14,400` warm calls (`5` repeats × `40` × `8` × `9`, `seed=42`, `16` warmup calls, `18,308.4s`, `0` lỗi, `0` OOM, `115` BM25 TTL rebuilds được ghi nhận) + `640` stage-breakdown calls + `1,600` production-profile calls (`top_k=5, pool=10`) + `24` cold-start subprocess trials.  
**Artifacts:** [latency_raw.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/performance/latency_raw.jsonl), [latency_summary.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/performance/latency_summary.csv), [stage_latency.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/performance/stage_latency.csv), [cold_start_latency.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/performance/cold_start_latency.csv), [resource_usage.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/performance/resource_usage.csv), [production_profile_latency.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/performance/production_profile_latency.csv), [performance_run_manifest.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/performance/performance_run_manifest.json).

| Combo | Warm K=5 P50 / P95 (ms, pool=120) | Warm K=60 P50 / P95 (ms, pool=120) | Prod Profile K=5, pool=10 P50 / P95 (ms) | Stage Rerank Mean (ms, pool=120) | Cold-Start Mean (s) | Peak GPU Alloc (MB) |
|---|---:|---:|---:|---:|---:|---:|
| **A1** | `360.4 / 607.5` | `365.6 / 610.4` | `352.0 / 447.0` | `0.0` | `15.42` | `0.0` |
| **A2** | `389.4 / 626.5` | `397.5 / 677.7` | `371.9 / 479.6` | `0.0` | `10.44` | `458.5` |
| **A3** | `757.6 / 1,321.8` | `765.0 / 1,303.2` | `720.3 / 943.6` | `0.0` | `21.03` | `458.5` |
| **A4** | `3,097.9 / 3,693.1` | `3,087.8 / 3,692.1` | `988.7 / 1,230.9` | `2,320.4` | `32.13` | `2,662.0` |
| **B1** | `452.4 / 756.7` | `454.4 / 700.1` | `429.5 / 553.8` | `0.0` | `20.49` | `0.0` |
| **B2** | `457.6 / 754.0` | `456.1 / 710.7` | `433.4 / 586.1` | `0.0` | `9.62` | `458.5` |
| **B3** | `896.1 / 1,308.4` | `904.8 / 1,415.8` | `868.3 / 1,151.1` | `0.0` | `24.92` | `458.5` |
| **B4** | `2,366.8 / 3,214.3` | `2,317.4 / 2,997.0` | `1,045.7 / 1,368.1` | `1,364.8` | `34.32` | `2,648.3` |

---

## 9. Answer Generation Evaluation (`PARTIAL` — Controlled Agent Completed; Production Groq API Unavailable)

**Methodology:** Agent-based controlled generation experiment (`antigravity_agent_flash` generator + blind `antigravity_agent_pro` 10-metric judge; matched `token_budget = 2,048`, production `QA_SYSTEM_PROMPT` rules). `human_validated = false`; có rủi ro family-level self-preference bias do cùng hệ sinh thái Agent (đã kiểm chứng bằng `3/3` synthetic controls PASS và 10% secondary audit `100.0%` correctness / `96.77%` faithfulness within-1pt agreement).  
**Artifacts:** [generated_answers.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/generation/generated_answers.jsonl), [no_rag_baseline_answers.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/generation/no_rag_baseline_answers.jsonl), [answer_judgments.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/generation/answer_judgments.jsonl), [answer_metrics_per_query.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/generation/answer_metrics_per_query.csv), [answer_metrics_summary.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/generation/answer_metrics_summary.csv), [answer_quality_audit.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/generation/answer_quality_audit.json), [citation_validation.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/generation/citation_validation.csv), [generation_run_manifest.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/generation/generation_run_manifest.json).

| Combo | Correctness | Relevancy | Faithfulness | Completeness | Cit. Acc. | Cit. Valid. | Unsup. Rate | Contra. Rate | Complexity | Code |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A1** | `0.8125` | `0.8500` | `0.9250` | `0.7917` | `0.8500` | `1.0000` | `0.0590` | `0.0027` | `0.9231` | `0.7778` |
| **A2** | `0.7063` | `0.7688` | `0.8562` | `0.6667` | `0.8187` | `1.0000` | `0.1125` | `0.0000` | `0.8462` | `0.8889` |
| **A3** | `0.8187` | `0.8500` | `0.8500` | `0.8083` | `0.8187` | `1.0000` | `0.1225` | `0.0027` | `0.9062` | `0.8333` |
| **A4** | **`0.9250`** | **`0.9563`** | **`0.9187`** | **`0.9167`** | `0.8187` | **`1.0000`** | `0.0638` | **`0.0000`** | **`1.0000`** | **`0.9500`** |
| **B1** | `0.6625` | `0.7438` | `0.8562` | `0.6000` | `0.8000` | `1.0000` | `0.1263` | `0.0035` | `0.9091` | `0.5625` |
| **B2** | `0.4813` | `0.5938` | `0.8750` | `0.3500` | `0.8438` | `1.0000` | `0.1205` | `0.0043` | `0.6154` | `0.3333` |
| **B3** | `0.6562` | `0.7312` | `0.8313` | `0.5917` | `0.8063` | `1.0000` | `0.1439` | `0.0020` | `0.8462` | `0.7778` |
| **B4** | `0.7500` | `0.8000` | `0.8500` | `0.7333` | `0.8125` | `1.0000` | `0.1131` | `0.0136` | `0.9583` | `0.7000` |
| **NO_RAG** | `0.9688` | `1.0000` | `N/A` | `0.9208` | `N/A` | `N/A` | `N/A` | `0.0175` | `1.0000` | `0.9500` |

---

## 10. End-to-End Production API Evaluation (`BLOCKED`)

- **Trạng thái:** **`BLOCKED`** — Không có `GROQ_API_KEY` thật trong `backend/.env` (chỉ có placeholder) và production Qdrant (`data/runtime/qdrant_local`) có `0` points (cấm populate production storage cho mục đích Evaluation). Không tạo số liệu giả cho end-to-end HTTP API latency.

---

## 11. Robustness Evaluation (`COMPLETED` — Controlled Agent + Real RTX 5060 Retrieval)

**Artifacts:** [robustness_questions.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/robustness/robustness_questions.jsonl), [robustness_results.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/robustness/robustness_results.jsonl), [robustness_metrics.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/robustness/robustness_metrics.csv).

| Category | Cases | Correct Handling | Abstention Rate | Unsup. Rate | Injection Resist. | EvRecall@5 Degradation | Quality (0–1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `out_of_domain` | `4` | **`1.0000`** | `1.0000` | `0.0` | `—` | `—` | `1.0000` |
| `unanswerable_in_corpus` | `4` | **`1.0000`** | `1.0000` | `0.0` | `—` | `—` | `1.0000` |
| `prompt_injection` | `3` | **`1.0000`** | `0.0000` | `0.0` | **`1.0000`** | `0.0000` | `0.9167` |
| `distractor_context` | `3` | `0.6667` | `0.0000` | `0.0` | `—` | `0.0000` | `0.7500` |
| `semantic_paraphrase` | `3` | `0.3333` | `0.3333` | `0.0` | `—` | **`-0.5556`** | `0.5000` |
| `typo_noisy` (VI không dấu) | `3` | **`0.0000`** | `0.6667` | `0.0` | `—` | **`-0.6667`** | `0.1667` |

---

## 12. Study Features & Functional Evaluation (`PARTIAL` — Real Contract Checks + Controlled Agent)

- **Real Module Contract Checks ([functional_test_results.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/study_features/functional_test_results.json)):** **`11 / 11 PASSED`** (`plan_chat_query` intents, empty-workspace abstention cho Chat/Quiz/Roadmap, low-score abstention, `study_planner` DAG topological sort trên 4 chủ đề DSA, citation builder). Phát hiện kiến trúc: `backend/app/safety/guardrail.py` (`0` B) và `policies.py` là scaffolds chưa triển khai.
- **Controlled Agent Study Evaluation ([quiz_evaluation.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/study_features/quiz_evaluation.jsonl), [roadmap_evaluation.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/study_features/roadmap_evaluation.jsonl), [study_features_summary.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/study_features/study_features_summary.csv)):**
  - Quiz (`8` DSA topics): `schema_valid = 1.00`, `correctness = 0.6875`, `grounding = 0.6250`, `pedagogical = 0.6250`, `dup = 0.0`.
  - Study Roadmap (`4` scenarios): `schema_valid = 1.00`, `correctness = 0.6250`, `grounding = 0.4375`, `pedagogical = 0.5000`, `dup = 0.0`.

---

## 13. Ablation Study (`COMPLETED`)

**Artifacts:** [ablation_results.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/ablation/ablation_results.csv), [paired_method_comparisons.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/ablation/paired_method_comparisons.csv), [rag_contribution.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/ablation/rag_contribution.csv).

| Comparison | Category | nDCG@5 Abs (Rel) | Holm p | EvRecall@15 Abs (Rel) | Holm p |
|---|---|---:|---:|---:|---:|
| `A3 vs A1` | Hybrid vs BM25 (Rec) | `-0.0547` (`-9.8%`) | `0.1314` (ns) | `-0.0250` (`-3.1%`) | `1.0000` (ns) |
| `A3 vs A2` | Hybrid vs Dense (Rec) | **`+0.1115`** (`+28.4%`) | **`0.0032`** (*) | **`+0.1250`** (`+19.2%`) | **`0.0060`** (*) |
| `B3 vs B2` | Hybrid vs Dense (Ada) | **`+0.1035`** (`+33.1%`) | **`0.0054`** (*) | **`+0.1458`** (`+37.2%`) | **`0.0060`** (*) |
| `A4 vs A3` | Reranker (Rec) | **`+0.2016`** (`+40.0%`) | **`0.0010`** (*) | **`+0.1333`** (`+17.2%`) | **`0.0291`** (*) |
| `B4 vs B3` | Reranker (Ada) | **`+0.1428`** (`+34.3%`) | **`0.0032`** (*) | **`+0.2125`** (`+39.5%`) | **`0.0036`** (*) |
| `B4 vs A4` | Adaptive vs Recursive (Rerank) | **`-0.1461`** (`-20.7%`) | **`0.0010`** (*) | **`-0.1583`** (`-17.4%`) | **`0.0035`** (*) |

- **RAG vs Without-RAG (`A4 vs NO_RAG`):** `NO_RAG` đạt `Correctness` (`0.9688` vs `0.9250`, diff `-0.0438`, `p = 0.2381`), `Relevancy` (`1.0000` vs `0.9563`, diff `-0.0438`, `p = 0.0644`) và `Completeness` (`0.9208` vs `0.9167`, diff `-0.0042`, `p = 0.9009`) cao hơn về giá trị mẫu; các chênh lệch này chưa đạt ý nghĩa thống kê ở `α = 0.05` (không suy diễn là chứng minh tương đương với `n = 40`). Đổi lại, `A4` giảm `Contradiction` về `0.0000` (vs `0.0175` ở `NO_RAG`), đạt `Faithfulness = 0.9187` và `100%` valid citations.

---

## 14. Sensitivity Analysis (`COMPLETED`)

- **Top-K Sensitivity ([top_k_sensitivity.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/sensitivity/top_k_sensitivity.csv)):** Lưu đủ 72 hàng (`8` combos × `9` Ks) ghép Retrieval, Evidence và **Main Warm Latency (`14,400` calls, `5` repeats, `pool=120`)**.
- **Candidate Pool Sensitivity (`top_k=5`, `pool=[10,30,60,120]`, [candidate_pool_sensitivity.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/sensitivity/candidate_pool_sensitivity.csv)):** Đây là thí nghiệm đo single-pass tách biệt (`40` queries × `4` combos × `4` pools = `640` calls, không trộn với Main Warm Latency 5-repeat ở Mục 8). `Judged@5 = 1.0` (`0` unjudged). `A4` đạt điểm uốn tại **`pool = 30`** (`nDCG@5 = 0.7039`, `EvRecall@5 = 0.7375`, single-pass `mean = 1,628 ms` / `P95 = 1,777 ms`) gần sát `pool = 120` trong cùng thí nghiệm single-pass (`nDCG@5 = 0.7053`, `mean = 3,486 ms` / `P95 = 3,980 ms`; đối chiếu Main Warm Latency 5-repeat tại `K=5, pool=120`: `mean = 3,210 ms` / `P95 = 3,693 ms`).
- **Context Budget Sensitivity (`1024 / 2048 / 4096` tokens, 10 queries × `A4, B4`, [context_budget_sensitivity.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/sensitivity/context_budget_sensitivity.csv)):** `A4` cân bằng tốt nhất ở **`2,048` tokens** (`Correctness = 0.950`, `Faithfulness = 0.925`); lên `4,096` làm giảm `Faithfulness` (`0.825`) và `Citation Accuracy` (`0.700`).

---

## 15. Statistical Analysis (`COMPLETED`)

**Artifacts:** [statistical_summary.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/statistics/statistical_summary.csv), [confidence_intervals.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/statistics/confidence_intervals.csv) (`10,000` resamples, `seed=42`), [paired_bootstrap_results.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/statistics/paired_bootstrap_results.csv), [win_tie_loss.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/statistics/win_tie_loss.csv), [subgroup_metrics.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/statistics/subgroup_metrics.csv), [annotation_sensitivity.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/statistics/annotation_sensitivity.csv).
- **Annotation Uncertainty & Sensitivity:** So sánh `ai_annotations_primary.jsonl` vs `ai_annotations_final.jsonl` cho thấy mức dịch chuyển tối đa `|Δ nDCG| ≤ 0.011` trên toàn bộ 8 combos tại `K=5, 15`, **không làm thay đổi bất kỳ thứ hạng hay kết luận chính nào**.

---

## 16. Error Analysis (`COMPLETED`)

**Artifacts:** [error_analysis.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/error_analysis/error_analysis.jsonl), [failure_taxonomy_summary.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/error_analysis/failure_taxonomy_summary.csv), [representative_case_studies.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/error_analysis/representative_case_studies.json).
- Top failure modes trên `A4/B4`: `2_relevant_evidence_ranked_too_low` (`29` occurrences), `1_relevant_evidence_not_retrieved` (`17`), `9_missing_knowledge_unit` (`17`), `10_irrelevant_context_overload` (`7`), `7_fragmented_evidence` (`6`), `4_semantic_mismatch` (`5`), `8_duplicate_context` (`4`).

---

## 17. Quality-Latency-Resource Trade-off (`COMPLETED`)

**Artifacts:** [quality_latency_tradeoff.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/tradeoff/quality_latency_tradeoff.csv), [pareto_frontier.csv](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/tradeoff/pareto_frontier.csv), [configuration_recommendations.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage04/tradeoff/configuration_recommendations.json).
- **Pareto-Optimal Set (`K=5` & `K=15`, `pool=120`, không dùng trọng số chủ quan):** **`A1`** (độ trễ thấp nhất, `0 MB` VRAM, `Faithfulness = 0.9250` & `Citation Accuracy = 0.8500` cao nhất), **`A4`** (dẫn đầu `nDCG`, `MRR`, `Evidence Recall` và `Answer Correctness` trong 8 tổ hợp RAG), **`B4`** (riêng giai đoạn rerank nhanh hơn `A4` ~`41%`, tổng warm P95 thấp hơn `13.0%` ở `K=5` và `16.8%` ở `K=15`, `0%` embedding truncation, `0.0075` evidence redundancy).
- **Exploratory Product Recommendation (chưa held-out validated):** Triển khai **`A4 (Recursive + Hybrid Rerank)` với `candidate_top_k = 30`, `token_budget = 2,048`**, bật `accent_insensitive: true` cho BM25 tiếng Việt, và chuyển từ `QdrantLocal` SQLite sang Qdrant Server (HNSW) để cắt `~900 ms` overhead.

---

## 18. Known Limitations

1. **Ground Truth & Answer Judge:** Hoàn toàn tự động (`human_validated = false`), dùng hai model tiers trong cùng hệ sinh thái Antigravity (`flash` / `pro`); tồn tại nguy cơ family-level bias đã được kiểm soát một phần qua synthetic controls & secondary cross-check.
2. **Production LLM API:** `GROQ_API_KEY` chưa được cấu hình và production Qdrant rỗng (`0` points) nên End-to-End HTTP API & Production Groq Generation ở trạng thái **`BLOCKED`**.
3. **Latency Environment:** Không xóa OS disk cache giữa các cold-start trials; không khóa nhiệt độ/tiến trình nền Windows; `115` lần BM25 TTL (`300s`) rebuild xảy ra tự nhiên trong phiên đo 5 giờ và đã được ghi nhận rõ.
4. **No Held-out Validation:** Đề xuất cấu hình (`A4`, `pool=30`, `budget=2048`) là *exploratory selection* trên cùng tập 40 câu hỏi.

---

## 19. Reproducibility Instructions

```powershell
Set-Location 'D:\DSA-Document-Assistant-Using-Hybrid-RAG'
.\.venv\Scripts\python.exe evaluation/scripts/run_comprehensive_evaluation.py --mode all
.\.venv\Scripts\python.exe evaluation/scripts/run_comprehensive_evaluation.py --mode validate
.\.venv\Scripts\python.exe -m pytest tests/unit/test_comprehensive_evaluation.py tests/unit/test_retrieval_metrics.py tests/unit/test_ground_truth.py -q
```
*(Các thí nghiệm Controlled Generation, Answer Judge, Robustness & Study Features phụ thuộc Antigravity `run_workflow` được tách riêng khỏi script thuần Python trên.)*

---

## 20. Tests & Integrity (`COMPLETED`)

- **Targeted Tests (`test_comprehensive_evaluation.py`, `test_retrieval_metrics.py`, `test_ground_truth.py`):** **`53 / 53 PASSED`** (`7.02s`).
- **Full `pytest -q`:** **`319 passed, 4 failed, 1 warning (fitz)`** (`21.56s`, `+4` tests mới so với baseline `315`, **`0` regression mới**). Bốn failures giữ nguyên đúng nguyên nhân pre-existing trên Windows `time.monotonic()` (`0.8000000000029104 <= 0.8`) tại `tests/unit/test_llm_reliability.py::test_timeout_per_task_and_remaining_budget[*]`; không sửa production/test ngoài phạm vi và không tuyên bố full suite PASS.
- **`pip check`:** `No broken requirements found.` (`exit 0`).
- **Frozen & Production Checksums:** Toàn bộ 9 checksums (`questions_40`, `source_evidence`, `corpus_a`, `corpus_b`, `gt_recursive`, `gt_adaptive`, `prod_chunks`, `prod_qdrant`, `orig_wb`) đều **`IDENTICAL`**.
- **Git Integrity:** Branch `feat/final-readiness`, `HEAD = c21700a2...`, `main = b84f0945...`, `0` staged files, chỉ `evaluation/metrics/retrieval_metrics.py` được nâng cấp tương thích ngược trong số các tracked files.

---

## 21. Acceptance Criteria (`AC-01` – `AC-06`)

| AC ID | Tiêu chí | Trạng thái | Bằng chứng & Ghi chú |
|---|---|---|---|
| **AC-01** | Benchmark Infrastructure & Integrity | **PASS** | Frozen inputs verified, `retrieval_metrics.py` upgraded backward-compatibly, Top-60/Pool-120 & zero-positive supported, `0` prod edits |
| **AC-02** | Retrieval & Evidence Evaluation | **PASS** | `8` combos × `40` queries × `9` Ks × `8` retrieval metrics + `6` evidence metrics, `0` ranking drift, `Judged@K = 1.0` |
| **AC-03** | Performance Evaluation | **PASS** | `14,400` warm calls (`5` repeats, interleaved `seed=42`), `640` stage breakdown, `24` cold-start trials, `1,600` prod-profile calls, full GPU/RAM usage |
| **AC-04** | Generation, Robustness & Study Features | **PARTIAL** | Controlled Agent Generation (`400` answers + blind `pro` judge + `10%` audit), `20` Robustness cases, `11/11` real contract checks & `8` Quizzes / `4` Roadmaps **COMPLETED**; Production Groq API & End-to-End HTTP API **BLOCKED** (no `GROQ_API_KEY`, empty prod Qdrant) |
| **AC-05** | Comprehensive Experimental Analysis | **PASS** | Ablation, Top-K / Candidate-Pool / Context-Budget sensitivity, `10,000`-resample paired bootstrap + Holm, Error Analysis, Pareto frontier |
| **AC-06** | Regression, Reproducibility & Handover | **PASS** | `53/53` targeted PASS, `0` new regressions (`319 passed, 4 pre-existing Windows timing flakes`), `pip check = 0`, `VALIDATE_MISSING = []`, report complete |

---

## 22. Stage 05 Visualization Data Handover

Toàn bộ dữ liệu CSV/JSONL chuẩn hóa cho 28 biểu đồ trong Stage 05 Data Contract đã sẵn sàng trong `evaluation/artifacts/stage04/` (`chunking/`, `retrieval/`, `evidence/`, `performance/`, `generation/`, `robustness/`, `study_features/`, `ablation/`, `sensitivity/`, `statistics/`, `error_analysis/`, `tradeoff/`). Đã dừng chính xác tại Stage 04 Final Handover — chưa commit, chưa push, chưa bắt đầu Stage 05.
