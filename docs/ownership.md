# Phân công và ranh giới trách nhiệm A1–D2

Đường dẫn dưới đây tính từ thư mục gốc repo.

| Người | Phần sở hữu chính | Đầu ra / phối hợp |
| --- | --- | --- |
| A1 | `backend/app/api/`, `backend/app/orchestration/` | API, Intent Router, nối luồng upload/chat/quiz/revision; gọi guardrail của D2 |
| A2 | `frontend/` | Bốn màn hình, API client, hiển thị trích dẫn và trạng thái từ chối |
| A3 | Dockerfile, Compose, cấu hình môi trường, tích hợp | Môi trường chạy tái lập, hướng dẫn chạy, phối hợp kiểm thử tích hợp |
| B1 | `backend/app/document_processing/` | PDF → danh sách DocumentChunk có nguồn gốc |
| C1 | `backend/app/retrieval/` | Embedding, index, BM25 + vector search, fusion, reranker |
| C2 | `backend/app/generation/` | Sinh câu trả lời/quiz/revision, trích dẫn, từ chối khi thiếu bằng chứng |
| D1 | `data/corpus/`, benchmark, chỉ số chất lượng, ablation | Corpus, ground truth, Recall@K/MRR, correctness/faithfulness, báo cáo |
| D2 | `backend/app/safety/`, red-team, chỉ số an toàn | Guardrail, testcase tấn công/lạc đề, phân loại lỗi và phân tích rủi ro |

## Ranh giới cần giữ

- A1 tích hợp guardrail; D2 viết quy tắc và logic kiểm tra. Không có hai bộ lọc
  độc lập cùng làm một việc trong route và safety.
- A1 định tuyến ý định; C2 tạo nội dung. Route HTTP và Intent Router là hai tầng
  khác nhau.
- A1 nhận upload và điều phối lưu file; B1 đọc/làm sạch/chia chunk; C1 lập chỉ mục.
- C2 từ chối do thiếu bằng chứng; D2 từ chối do quy tắc an toàn/phạm vi. Không gộp
  hai lý do thành một cờ khó đánh giá.
- C1 phụ trách code kết nối và truy xuất VectorDB; A3 phụ trách chạy dịch vụ/volume.
- A1 điều phối schema chung; B1, C1, C2, D2 xác nhận contract thuộc phần mình.
- Mỗi owner viết unit test cho module mình. A3 phối hợp integration test với A1
  và các owner liên quan. D1/D2 phụ trách evaluation, không thay thế unit test.

## Những phần chưa hoàn thành

Tên thư mục chỉ xác định nơi làm việc. Parser, retrieval, generation, guardrail,
frontend và evaluation vẫn chưa được triển khai. Dockerfile/Compose còn rỗng.
