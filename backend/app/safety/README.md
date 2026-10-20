# Kiểm tra an toàn — D2

`guardrail.py` được chuyển từ service rỗng; `policies.py` dành cho chính sách.
D2 viết logic kiểm tra, A1 gọi từ orchestration. Dùng GuardrailResult hiện có.

Phân biệt từ chối theo chính sách với thiếu bằng chứng do C2 xử lý. Bộ red-team
thuộc `evaluation/datasets/`, phép đo an toàn thuộc `evaluation/metrics/`.
Hiện chưa có logic kiểm tra thực tế.
