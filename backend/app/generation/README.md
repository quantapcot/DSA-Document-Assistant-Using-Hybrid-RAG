# Sinh nội dung — C2

`rag_engine.py`, `quiz_generator.py`, `revision_generator.py` được chuyển từ các
service rỗng. `llm_client.py` dành cho giao tiếp mô hình; `citations.py` liên kết
bằng chứng; `abstention.py` xử lý thiếu bằng chứng; `prompts/` chứa mẫu hướng dẫn.

Sử dụng kết quả retrieval của C1 và schema hiện có. Quy tắc an toàn thuộc D2.
Không đưa API key vào prompt hoặc code. Tất cả hiện chưa triển khai.
