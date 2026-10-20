# Xử lý tài liệu — B1

`document_parser.py` là điểm vào PDF → DocumentChunk, chuyển từ file service rỗng.
`text_cleaner.py` dành cho làm sạch văn bản; `chunker.py` chia theo cấu trúc;
`metadata.py` bảo toàn filename/page/section và định danh nguồn.

Dùng DocumentChunk tại `app.schemas.contracts`. Không gọi LLM, không xây index
và không triển khai HTTP upload ở đây. Tất cả hiện là khung chưa triển khai.
