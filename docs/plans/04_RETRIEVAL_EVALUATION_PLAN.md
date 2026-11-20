# PLAN 04 — RETRIEVAL EVALUATION

**Branch:** `feat/retrieval-evaluation`  
**Baseline:** `8c48d4d`  
**Owner:** Evaluation Team (D1)  
**Mục tiêu:** Xây dựng Ground Truth có nhãn và đánh giá khách quan bốn phương pháp Retrieval, bảo đảm kết quả chính xác, có thể tái lập và sử dụng được trong báo cáo đồ án.

## 1. Phạm vi và File Ownership

**Được sửa:**
- `evaluation/datasets/**`
- `evaluation/metrics/**`
- `evaluation/scripts/**`
- `evaluation/results/**`
- `evaluation/reports/**`
- `evaluation/README.md`
- `tests/unit/test_retrieval_metrics.py`
- `evaluation/requirements.txt` (NEW nếu cần).

**Không tự sửa:**
- `backend/app/retrieval/**` — Retrieval.
- `backend/app/generation/**`, `backend/app/orchestration/**`, API routes — Generation/Retrieval.
- `backend/app/document_processing/**` — Preprocessing.
- Dockerfiles, Compose, production requirements — Docker.
- `configs/retrieval.*` — Retrieval.
- `data/corpus/dataset_clean/chunks_all.jsonl` — Corpus gốc, read-only.

**Nguyên tắc:** Evaluation chỉ sử dụng public interface của Retrieval, không sửa thuật toán hoặc dữ liệu gốc để cải thiện kết quả.

## 2. Implementation Tasks

| ID | Ưu tiên | Công việc | File chính |
|---|---|---|---|
| EVAL-01 | P0 | Xây dựng Ground Truth DSA có nhãn | `evaluation/datasets/` |
| EVAL-02 | P0 | Hoàn thiện Retrieval Metrics | `retrieval_metrics.py` |
| EVAL-03 | P0 | Hoàn thiện Benchmark Runner | `run_benchmark.py` |
| EVAL-04 | P1 | Ablation Study cho RRF và Reranker | `run_ablation.py` |
| EVAL-05 | P1 | Phân tích lỗi Retrieval | `error_analysis.py` (NEW) |
| EVAL-06 | P0 | Tạo báo cáo so sánh 4 phương pháp | `generate_report.py` (NEW), `reports/` |
| EVAL-07 | P0 | Unit Tests và Reproducibility | `test_retrieval_metrics.py`, Evaluation scripts |

### EVAL-01 — Ground Truth Dataset

**Hiện trạng:** `evaluation/datasets/` chưa có bộ Ground Truth hoàn chỉnh để chạy benchmark.

**Cần làm:**
- Xây dựng tối thiểu 35 câu hỏi DSA, dự kiến 20 tiếng Việt và 15 tiếng Anh.
- Bao phủ các chủ đề thực sự có trong corpus: BST, AVL, Sorting, BFS/DFS, Graph, Hashing, v.v.
- Gán `relevant_ids` bằng `chunk_id` có thật từ corpus đã index.
- Có thể gán graded relevance:
  - `2`: Chứa bằng chứng trực tiếp để trả lời.
  - `1`: Có thông tin bổ trợ liên quan.
  - `0`: Không liên quan.
- Người gán nhãn phải kiểm tra nội dung chunk, không chỉ dựa vào tên file hoặc kết quả do model trả về.
- Chia tập `development` và `held-out test`, tránh tuning trực tiếp trên tập test.
- Kiểm tra mức độ đầy đủ của nhãn relevant để Recall@K có ý nghĩa.

**Schema đề xuất:**

```json
{
  "query_id": "q001",
  "query": "Cây AVL là gì?",
  "language": "vi",
  "topic": "AVL Tree",
  "split": "development",
  "relevant_ids": ["real_chunk_id_1", "real_chunk_id_2"],
  "relevance_grades": {
    "real_chunk_id_1": 2,
    "real_chunk_id_2": 1
  }
}
```

*Các ID trên chỉ minh họa schema, không được dùng làm nhãn thực.*

**Đạt khi:** Dataset có nhãn được kiểm chứng; mọi relevant ID tồn tại trong corpus, không chứa nhãn giả hoặc ID bịa đặt.

### EVAL-02 — Retrieval Metrics

**Hiện trạng:** `retrieval_metrics.py` đã có Precision, Recall và Reciprocal Rank, chưa có nDCG.

**Cần làm:**
- Giữ và kiểm tra:
  - Precision@K.
  - Recall@K.
  - MRR@K.
- Bổ sung nDCG@K sử dụng graded relevance.
- Hỗ trợ K = 1, 3, 5.
- Xử lý đúng empty results, duplicate IDs, missing relevance và IDCG = 0.
- Phân biệt Recall dựa trên tập nhãn đầy đủ và Recall dựa trên tập nhãn chưa được đánh giá hết.

**Đạt khi:** Các metrics có unit tests đối chiếu với kết quả tính toán thủ công, không sai lệch công thức.

### EVAL-03 — Benchmark Runner

**Hiện trạng:** `run_benchmark.py` đã hỗ trợ bốn Retrieval methods và xuất CSV/JSON nhưng còn hạn chế về latency và metrics.

**Bốn phương pháp bắt buộc:**

| Method | Cấu hình |
|---|---|
| BM25 | Sparse Retrieval |
| Dense | Multilingual MiniLM + Qdrant |
| Hybrid RRF | BM25 + Dense + RRF |
| Hybrid Rerank | Hybrid RRF + BGE Reranker v2-m3 |

**Cần làm:**
- Dùng cùng query set, corpus, config và K cho tất cả phương pháp.
- Sử dụng `RetrievalEngine.retrieve()` thay vì tự triển khai Retrieval riêng.
- Tính Precision@K, Recall@K, MRR@K, nDCG@K.
- Warm-up các thành phần cần thiết trước khi đo warm latency.
- Đo latency bằng `time.perf_counter()`, báo cáo mean, median/p50 và p95.
- Tách cold-start measurements khỏi warm-query measurements; không chỉ gọi một truy vấn warm-up rồi suy ra toàn bộ cold-start.
- Xuất per-query results và summary ra CSV/JSON.
- Ghi metadata: commit SHA, corpus/query SHA-256, config, model IDs, device, K và môi trường.
- Bảo đảm CLI hỗ trợ các tham số thực tế, đặc biệt `--split` và giá trị K.

**Đạt khi:** Chạy đầy đủ bốn methods trên cùng dataset, xuất kết quả thực và có metadata để tái lập.

### EVAL-04 — Ablation Study

**Hiện trạng:** `run_ablation.py` chủ yếu gọi lại benchmark runner, chưa có parameter sweep thực sự.

**Cần làm:**
- Thử RRF `k = 20, 60, 100`.
- Thử `candidate_top_k = 5, 10, 20`.
- Đánh giá thay đổi về Recall, MRR, nDCG và latency.
- Thay đổi tham số trong cấu hình thực nghiệm riêng, không sửa `configs/retrieval.mvp.yaml`.
- Chỉ tune trên development set.
- Chọn cấu hình trên development, sau đó đánh giá một lần trên held-out test.

**Đạt khi:** Có bảng so sánh tác động của RRF và candidate pool, không chỉ so sánh tên bốn methods.

### EVAL-05 — Error Analysis

**Cần làm:**
- Tìm các query mà Retrieval không lấy được relevant chunks trong Top-K.
- So sánh query mà BM25 tốt hơn Dense hoặc ngược lại.
- Phân tích các trường hợp Hybrid Rerank kém hơn Hybrid RRF.
- Phân loại lỗi dựa trên bằng chứng: tokenization, từ viết tắt, ngôn ngữ, thiếu context, chunking hoặc reranking.
- Không tự kết luận nguyên nhân nếu chưa kiểm tra nội dung chunk thực tế.

**Đạt khi:** Báo cáo có ít nhất 3 trường hợp lỗi tiêu biểu, kèm query, relevant chunks, retrieved chunks và phân tích.

### EVAL-06 — Benchmark Report

Tạo `evaluation/reports/BENCHMARK_REPORT.md` và script `generate_report.py`.

**Nội dung bắt buộc:**
- Thông tin corpus và Ground Truth.
- Cấu hình bốn methods.
- Bảng so sánh Precision@K, Recall@K, MRR@K, nDCG@K.
- Bảng latency.
- Kết quả ablation.
- Phân tích các trường hợp Retrieval thất bại.
- Giới hạn dataset và phương pháp đánh giá.
- Kết luận dựa trên số liệu thực tế.

**Không giả định Hybrid Rerank chắc chắn tốt nhất.** Nếu kết quả không như kỳ vọng, báo cáo trung thực và giải thích dựa trên bằng chứng.

### EVAL-07 — Tests & Reproducibility

**Cần làm:**
- Unit test các công thức metrics.
- Validate schema và IDs của Ground Truth.
- Test benchmark runner bằng Mock RetrievalEngine.
- Test missing/empty/invalid dataset.
- Kiểm tra CLI của benchmark và ablation.
- Kiểm tra metadata lưu đúng phiên bản dữ liệu/model.
- Tạo output directory khi cần.
- Không chạy nhiều Qdrant Local processes trên cùng storage path.
- Không chỉnh sửa dataset hoặc tham số sau khi nhìn kết quả held-out test để làm đẹp số liệu.

**Đạt khi:** Toàn bộ Evaluation tests PASS; một thành viên khác có thể tái lập thực nghiệm trên cùng dữ liệu và cấu hình.

## 3. Integration Contracts

| Interface | Yêu cầu |
|---|---|
| `RetrievalEngine.retrieve()` | Dùng public API của Track 1 |
| Retrieval methods | `bm25`, `dense`, `hybrid_rrf`, `hybrid_rerank` |
| Evidence | Sử dụng `chunk_id`, `score`, `rank`, `retrieval_method` và metadata hiện có |
| Ground Truth | `query_id`, `query`, `relevant_ids`, `relevance_grades`, `split` |
| Benchmark scope | `workspace_id=research`, `corpus_id=benchmark-v1` |
| Corpus | Dùng snapshot cố định của `chunks_all.jsonl` |
| Reproducibility | Ghi commit, model, config, dataset checksums, môi trường và K |

**Phối hợp với track khác:**

- **Retrieval:** Giữ public Retrieval methods và evidence schema ổn định. Nếu Track 1 thay đổi indexing/chunk IDs, cần cập nhật và xác minh lại Ground Truth.
- **Docker:** Thống nhất dependencies và cách khởi tạo Qdrant trong môi trường thực nghiệm.
- **Generation:** Evaluation không đo LLM trong đợt này; không phụ thuộc trực tiếp vào QA/Quiz/Revision.

**Quan trọng:** Có thể xây dựng dataset, metrics và runner song song với Track 1, nhưng benchmark cuối phải chạy trên một commit Retrieval đã ổn định và được ghi rõ trong báo cáo.

## 4. Testing & Acceptance Criteria

| Test | Kết quả mong đợi |
|---|---|
| Ground Truth validation | IDs tồn tại và schema hợp lệ |
| Precision@K | Đúng công thức |
| Recall@K | Đúng công thức |
| MRR@K | Đúng thứ hạng relevant đầu tiên |
| nDCG@K | Đúng graded relevance và xử lý IDCG = 0 |
| Benchmark 4 methods | Chạy và xuất kết quả thành công |
| Equal evaluation conditions | Cùng corpus, query set, K và config |
| Warm/cold latency | Được đo và báo cáo riêng |
| Ablation | Thay đổi tham số và lưu kết quả thật |
| Development/Test split | Không dùng test để tune |
| Error Analysis | Có các trường hợp lỗi được kiểm chứng |
| Reproducibility | Đủ metadata và checksum |
| Isolation | Không sửa hoặc xóa corpus benchmark |

**Lệnh benchmark tham khảo:**

```powershell
python evaluation/scripts/run_benchmark.py `
  --queries evaluation/datasets/dsa_benchmark_v1.jsonl `
  --corpus data/corpus/dataset_clean/chunks_all.jsonl `
  --corpus-id benchmark-v1 `
  --workspace-id research `
  --config configs/retrieval.mvp.yaml `
  --split development `
  --output evaluation/results/benchmark_dev.csv
```

Chạy sau khi dataset đã được tạo và scope benchmark đã index đúng. Các tham số bổ sung như `--k` phải được triển khai trong runner trước khi sử dụng.

## 5. Definition of Done

- [ ] Có tối thiểu 35 câu hỏi Ground Truth được gán nhãn thực.
- [ ] Ground Truth khớp corpus và chunk IDs hiện tại.
- [ ] Precision, Recall, MRR, nDCG có unit tests.
- [ ] Benchmark chạy thành công cả bốn Retrieval methods.
- [ ] Có so sánh chất lượng và latency khách quan.
- [ ] Ablation Study chạy thực sự.
- [ ] Có phân tích lỗi Retrieval.
- [ ] Kết quả được xuất CSV/JSON và báo cáo Markdown.
- [ ] Commit, config, model và dataset provenance được lưu đầy đủ.
- [ ] Held-out test không bị sử dụng để tune.
- [ ] Không sửa Retrieval Core hoặc benchmark corpus gốc.
- [ ] Toàn bộ Evaluation tests PASS.

**Kết quả bàn giao:** Bộ Ground Truth có nhãn, Evaluation pipeline có thể tái lập và báo cáo định lượng so sánh BM25, Dense, Hybrid RRF, Hybrid + BGE Reranker, đủ làm cơ sở thực nghiệm cho đồ án.