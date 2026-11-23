# Hợp đồng dữ liệu MVP V2

`backend/app/schemas/contracts.py` định nghĩa DocumentChunk, Citation, ChatRequest, QuizRequest, RoadmapRequest và các phản hồi. OpenAPI được tạo ở FastAPI `/docs`.

## Phiên tạm

| Endpoint | Input | Kết quả |
| --- | --- | --- |
| `POST /api/sessions` | Không cần body | 201 với `token`, `workspace_id`, `corpus_id`, `expires_at` (Unix UTC). |
| `DELETE /api/sessions/current` | `Authorization: Bearer <token>` | 200 khi đã xóa, 202 khi đã thu hồi token và đang chờ dọn. |
| `POST /api/upload` | Bearer token và multipart `file` | Kết quả lập chỉ mục PDF/DOCX trong phiên. |
| `GET /api/documents` | Bearer token | Danh sách tài liệu của phiên. |
| `DELETE /api/documents/{document_id}` | Bearer token | Xóa tài liệu và chunks của tài liệu trong phiên. |

Token nằm trong HTTP Authorization. Backend tự xác định workspace/corpus, từ chối token sai, hết hạn hoặc bị thu hồi với 401. Nếu yêu cầu kèm ID không khớp, backend trả 403. API scope legacy vẫn dùng `workspace_id` và `corpus_id` cho dữ liệu cũ. Scope phiên tạm luôn yêu cầu bearer token.

## Tính năng học

Các endpoint dùng bearer token của phiên. Request trong phiên chỉ cần các trường sau:

| Endpoint | Body | Trường chính của phản hồi |
| --- | --- | --- |
| `POST /api/chat` | `{"query":"..."}` | `answer`, `citations`, `generation_mode`, `generation_error`. |
| `POST /api/quiz` | `{"topic":"BFS","num_questions":3}` | `requested_count`, `generated_count`, `questions`. Mỗi câu giữ đáp án, giải thích, nhãn nguồn và `evidence` với tên file, trang, trích đoạn. |
| `POST /api/roadmap` | `{}` hoặc body rỗng | `items` có `id`, `title`, `objective`, `knowledge`, `activity`, `prerequisites`, `sources`, `support_quote`. Kèm `citations`, `uncovered_topics`, `warnings`. |

`/api/revision` là alias có đánh dấu deprecated của Roadmap. Các trường lịch ngày, thời lượng, trình độ của Revision cũ đã được gỡ. `label` trong bằng chứng là nhãn 1-based. `correct_index` của Quiz là 0-based. `X-Process-Time-MS` ghi thời gian xử lý request.
