# Thay đổi cấu trúc

## Cấu trúc cuối cùng

`backend/`, `frontend/`, `configs/`, `data/`, `evaluation/`, `tests/` và
`docker-compose.yml` đặt trực tiếp ở gốc repo. Lớp `dsa_ai_agent/` đã được bỏ theo
yêu cầu. README giới thiệu ở gốc; bản đồ chi tiết ở `docs/folder_structure.md`.

Nội dung mọi file code được giữ nguyên khi di chuyển. Các import `app.*` vẫn
giữ nguyên cấu trúc bên trong backend. Chạy từ `backend/` bằng
`python -m uvicorn app.main:app` theo hướng dẫn `setup.md`.

## Phần đã tổ chức

- Bổ sung nhóm orchestration, document_processing, retrieval, generation, safety.
- Tách màn hình UI khỏi components dùng lại.
- Bổ sung thư mục dữ liệu, cấu hình mẫu, evaluation và kiểm thử.
- Viết hướng dẫn phân công, kiến trúc, hợp đồng và cách chạy.
- Cập nhật đường dẫn trong tài liệu và .gitignore theo gốc repo mới.

Đây là việc tổ chức khung, chưa triển khai thuật toán hoặc nối module mới vào API.
Không thêm dependencies, không commit/push.

## Nội dung code được bảo toàn

- Giữ `backend/app/core/config.py`, không đổi thành `settings.py`.
- Giữ `backend/app/api/middleware/observability.py`, không đổi tên.
- Giữ `backend/app/schemas/contracts.py`, không tách hoặc sao chép schema.
- Không sửa nội dung Python, requirements, Dockerfile hoặc Compose khi bỏ lớp ngoài.
- Các chú thích đầu file code cũ có thể còn đường dẫn trước khi chuyển; chúng được
  giữ nguyên để tuân thủ yêu cầu không sửa code và không ảnh hưởng import.
- YAML có hậu tố `.example.yaml` vẫn là mẫu, chưa được app nạp.

## Các file rỗng đã chuyển trong lần tổ chức module

Đường dẫn trong bảng được biểu diễn theo gốc repo mới. Mọi file trong bảng đều
rỗng trước khi chuyển và vẫn giữ nguyên nội dung.

| Vị trí module cũ | Vị trí module mới |
| --- | --- |
| backend/app/services/__init__.py | backend/app/orchestration/__init__.py |
| backend/app/services/document_service.py | backend/app/document_processing/document_parser.py |
| backend/app/services/rag_service.py | backend/app/generation/rag_engine.py |
| backend/app/services/quiz_service.py | backend/app/generation/quiz_generator.py |
| backend/app/services/revision_service.py | backend/app/generation/revision_generator.py |
| backend/app/services/guardrail_service.py | backend/app/safety/guardrail.py |
| frontend/components/chat_ui.py | frontend/screens/chat_ui.py |
| frontend/components/upload_ui.py | frontend/screens/upload_ui.py |
| frontend/components/quiz_ui.py | frontend/screens/quiz_ui.py |
| frontend/components/revision_ui.py | frontend/screens/revision_ui.py |

Thư mục `services/` cũ đã bỏ khi trống. `frontend/components/` dành cho thành phần
dùng lại. Hai báo cáo JSON/PDF 0 byte cũ đã bỏ; báo cáo thật sẽ vào
`evaluation/reports/`, dữ liệu chạy vào `evaluation/results/`.

## Kiểm tra

Đối chiếu SHA-256 trước/sau cho tất cả file được chuyển, ngoại trừ README bản đồ
được cập nhật đường dẫn và chuyển vào docs. Kiểm tra liên kết tài liệu, quy tắc
bỏ qua dữ liệu runtime/kết quả và Git diff. Không chạy lại ứng dụng hoặc cam kết
các chức năng khung đã hoạt động chỉ dựa vào việc sắp xếp thư mục.
