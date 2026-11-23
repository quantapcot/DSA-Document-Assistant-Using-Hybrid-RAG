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
Mỗi cặp query/phương thức gọi Retrieval đúng một lần để lấy Top-15. Cùng
ranking đó được cắt ở K = 1, 3, 5, 10, 15 để tính Precision, Recall, MRR và
nDCG. `--ks` giới hạn K được báo cáo và `--methods` chọn phương thức.
Để chạy Hybrid Rerank, YAML cần cấu hình reranker.

Latency đo thời gian gọi Retrieval bằng đồng hồ đơn điệu, tính theo
milliseconds. Mean, p50 và p95 được tính trên một mẫu mỗi query/phương thức,
với p95 nội suy tuyến tính. Runner chạy tuần tự theo thứ tự phương thức đã
chọn và không warmup. Lần gọi đầu có thể bao gồm nạp model hoặc tạo cache.
So sánh các lần chạy với cùng cấu hình, máy và trạng thái cache.

Runner lưu ranking, nhãn, latency, summary và provenance trong JSON. Khi
`--output` có đuôi CSV, giữ file `*.metadata.json` bên cạnh. Plotter đọc
trực tiếp cả hai định dạng và ghi `metrics_summary.csv`,
`latency_summary.csv`, `metrics_by_k.png`, `latency_comparison.png` và
`metric_comparison_k15.png`. Biểu đồ cột mặc định dùng K lớn nhất có trong
kết quả. Dùng `--comparison-k` để chọn K khác đã chạy.

```powershell
python -m pytest tests/unit/test_retrieval_metrics.py tests/unit/test_retrieval_benchmark.py tests/integration/test_retrieval_evaluation.py -q
```
