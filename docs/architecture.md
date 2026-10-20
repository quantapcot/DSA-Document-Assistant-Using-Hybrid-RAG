# Kiến trúc và luồng dữ liệu

## Hiện trạng

FastAPI hiện đăng ký health và chat giả lập. Chat trả dữ liệu AVL cố định,
confidence và is_safe gán sẵn. Chưa nối các module mới vào `main.py` hoặc route.
Middleware hiện có thêm header thời gian cho `/api/`, chưa ghi log đầy đủ.

## Luồng mục tiêu

```text
Nạp tài liệu:
UI upload → API upload → orchestration/ingestion_flow
          → document_processing (B1)
          → DocumentChunk
          → retrieval/indexing (C1)

Yêu cầu học tập:
UI → API → orchestration/request_flow
         → safety/guardrail (D2)
         → orchestration/intent_router (A1)
         → retrieval/hybrid_search → reranker (C1), khi cần bằng chứng
         → generation: RAG / quiz / revision (C2)
         → trích dẫn hoặc lý do từ chối → API → UI
```

Các bước trên là thiết kế, chưa phải chuỗi gọi đã triển khai.

## Vai trò các tầng

- API quản lý giao tiếp HTTP; không chứa thuật toán tìm kiếm hoặc prompt dài.
- Orchestration điều phối thứ tự các module và chuyển trạng thái/lỗi.
- Document processing giữ metadata nguồn qua làm sạch và chunking.
- Retrieval lập chỉ mục và trả các chunk kèm điểm, thứ hạng theo contract sẽ chốt.
- Generation dùng bằng chứng, không tự tạo trích dẫn không có nguồn.
- Safety cung cấp chính sách kiểm tra do D2 sở hữu; A1 tích hợp.
- Frontend chỉ gọi API và trình bày kết quả, không import backend trực tiếp.

## Cấu hình và dữ liệu

`backend/app/core/config.py` tiếp tục đọc cấu hình môi trường; không di chuyển
vì code hiện có đang import đường dẫn này. Các YAML trong `configs/` chỉ là ví dụ
để nhóm thống nhất, chưa được ứng dụng đọc.

`data/corpus/` giữ bộ tài liệu đánh giá; `data/runtime/` dành cho dữ liệu phát sinh.
VectorDB có thể dùng volume riêng. Không lưu thêm một bản index trùng chỉ để lấp
đầy `data/runtime/indexes/`.

## Nguyên tắc mở rộng

Không tạo một cây `src/` thứ hai song song với `backend/app/`. Mỗi chức năng
có một nơi sở hữu chính. Không tạo lại các class đã có trong `schemas/contracts.py`.
