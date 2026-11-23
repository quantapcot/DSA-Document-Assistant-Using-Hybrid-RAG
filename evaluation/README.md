# Đánh giá — D1 và D2

`datasets/` chứa dữ liệu đầu vào có kiểm soát và `results/` chứa kết quả
từng lần chạy. D1 phụ trách Retrieval benchmark và ablation. D2 phụ trách
red-team và safety.

## Retrieval benchmark

Chạy trên một corpus đã được index, với file JSONL ground truth đã có.
Mỗi dòng cần `query_id` duy nhất, `query`, `relevant_ids`, `relevance_grades`
và `labels_complete: true`. `relevance_grades` ánh xạ chunk ID sang 0
(không liên quan), 1 (hỗ trợ), hoặc 2 (liên quan trực tiếp).
`relevant_ids` phải trùng tập ID có điểm lớn hơn 0. Runner từ chối nhãn
thiếu, mâu thuẫn hoặc chưa hoàn chỉnh.

Hiện `datasets/` chỉ có tài liệu hướng dẫn. Khi chưa có ground truth đã gán
nhãn, chưa thể báo cáo metrics thực nghiệm. Fixture trong test chỉ dùng để
kiểm tra phần mềm.

Từ thư mục gốc repository, trong môi trường Python của dự án:

```powershell
python -m pip install -r evaluation/requirements.txt
python evaluation/scripts/run_benchmark.py `
  --queries "DUONG_DAN_GROUND_TRUTH.jsonl" `
  --corpus data/corpus/dataset_clean/chunks_all.jsonl `
  --corpus-id benchmark-v1 --workspace-id research `
  --config configs/retrieval.mvp.yaml --split held-out `
  --output evaluation/results/k15/run.json
python evaluation/scripts/plot_retrieval_metrics.py `
  --input evaluation/results/k15/run.json `
  --output-dir evaluation/results/k15/plots
```

Thay placeholder `--queries` bằng đường dẫn file thật và chọn scope thực
tế của index. `--corpus` ghi SHA-256 của corpus tham chiếu, nên người chạy
cần xác nhận index được dựng từ chính phiên bản đó.

Mặc định runner chạy BM25, Dense, Hybrid RRF và Hybrid Rerank với pool 30.
`--candidate-pool` cho phép đổi pool cho riêng lần benchmark (tối thiểu 15).
Lượt chấm chất lượng gọi Retrieval đúng một lần cho mỗi cặp query/phương
thức để lấy Top-15. Cùng ranking đó được cắt ở K = 1, 3, 5, 10, 15 để
tính Precision, Recall, MRR và
nDCG. `--ks` giới hạn K được báo cáo và `--methods` chọn phương thức.
Để chạy Hybrid Rerank, YAML cần cấu hình reranker.

### Đo latency

Runner warmup toàn bộ tập query/phương thức trước khi đo. Warmup mặc định
một vòng (`--warmup-rounds 1`), không ghi latency. Sau đó runner đo tuần tự
theo `--repeats` vòng (mặc định 5). Mỗi vòng xáo thứ tự query và xoay thứ
tự phương thức. `--seed` (mặc định 42) giúp tái lập thứ tự. Sau các lượt
đo, profile Evaluation mới lấy ranking dùng để tính chất lượng.

Thời gian đo bao gồm đúng lời gọi Retrieval bằng đồng hồ đơn điệu, không
bao gồm warmup, tính metrics hoặc ghi file. Mean, p50 và p95 tổng hợp từ
tất cả mẫu thực đo của mỗi phương thức, với p95 nội suy tuyến tính.
Mỗi mẫu giữ `query_id`, `method`, `repeat` và `latency_ms` theo thứ tự chạy.
Summary lưu số mẫu cho từng phương thức. Đây là latency khi đã warmup
trên các query lặp lại. Cache vẫn có thể ảnh hưởng số đo. Dùng cùng máy,
cấu hình, workload, số lượt và điều kiện tải khi so sánh.

Để đo riêng cấu hình Production **Top-5/pool 10**, dùng
`--latency-profile production`. Profile này chỉ đo latency và chấp nhận
file query JSONL đã có với `query_id` và `query` mà không cần gán nhãn.
Giữ kết quả trong file khác với profile Evaluation:

```powershell
python evaluation/scripts/run_benchmark.py `
  --latency-profile production --repeats 5 `
  --queries "DUONG_DAN_QUERY_JSONL_DA_CO.jsonl" `
  --corpus data/corpus/dataset_clean/chunks_all.jsonl `
  --corpus-id benchmark-v1 --workspace-id research `
  --config configs/retrieval.mvp.yaml --split held-out `
  --output evaluation/results/k15/production-latency.json
python evaluation/scripts/plot_retrieval_metrics.py `
  --input evaluation/results/k15/production-latency.json `
  --output-dir evaluation/results/k15/plots
```

Plotter xuất `latency_summary_production.csv` và
`latency_comparison_production.png` riêng, không ghi bảng hoặc biểu đồ
chất lượng.

Runner lưu ranking, nhãn, mẫu latency, summary và provenance trong JSON.
Với CSV, giữ file `*.metadata.json` bên cạnh. Ở profile Evaluation, CSV
chứa ranking và sidecar chứa mẫu latency. Ở profile Production, CSV chứa
các mẫu latency. Plotter đọc
trực tiếp cả hai định dạng và ghi `metrics_summary.csv`,
`latency_summary.csv`, `metrics_by_k.png`, `latency_comparison.png` và
`metric_comparison_k15.png`. Biểu đồ cột mặc định dùng K lớn nhất có trong
kết quả Evaluation. Dùng `--comparison-k` để chọn K khác đã chạy. Kết quả
schema cũ được ghi chú là latency chưa warmup trên biểu đồ.

```powershell
python -m pytest tests/unit/test_retrieval_metrics.py tests/unit/test_retrieval_benchmark.py tests/integration/test_retrieval_evaluation.py -q
```
