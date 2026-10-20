# Chỉ số đánh giá

`retrieval_metrics.py`: D1, Recall@K và MRR.
`answer_metrics.py`: D1, correctness và faithfulness.
`safety_metrics.py`: D2, mức chặn đúng/chặn nhầm/bỏ sót theo bộ testcase.

Chưa triển khai phép tính. Mỗi chỉ số cần định nghĩa mẫu số, ground truth và
cách xử lý trường hợp không có đáp án phù hợp.
