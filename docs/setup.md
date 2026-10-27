# Chạy backend hiện tại

## Những gì có thể chạy

Backend FastAPI có `/`, `/health`, `/api/chat` và trang `/docs`.
Chat là mock, không cần gọi LLM/VectorDB. Frontend, upload, quiz, revision và
Docker Compose chưa có triển khai đầy đủ.

## Chạy từ thư mục gốc repo bằng PowerShell

Cần Python 3.11.x phù hợp với dependencies của dự án (khuyến nghị tạo môi trường
Conda `dsa-rag` hoặc virtualenv với Python 3.11.x). Các lệnh cài đặt dưới đây do
người phát triển chạy khi muốn chuẩn bị môi trường:

```powershell
# Cách 1: Sử dụng Conda environment dsa-rag (Python 3.11.x)
conda activate dsa-rag
cd backend
pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Cách 2: Sử dụng virtualenv với Python 3.11.x
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Chỉ sao chép `.env.example` nếu chưa có `.env`; không ghi đè cấu hình cá nhân.
API docs: http://127.0.0.1:8000/docs

Dùng `app.main:app` từ thư mục `backend` để khớp các import `app.*` hiện có.
Không dựa vào block chạy trực tiếp trong `main.py` cho hướng dẫn này.

## Cấu hình

`app/core/config.py` đọc `.env` theo thư mục chạy. Vì vậy ví dụ môi trường nằm
ở `backend/.env.example`; đặt `.env` bên cạnh khi chạy theo hướng dẫn trên.
Không đưa API key thật vào Git. Các biến LLM/Qdrant (`QDRANT_URL`) hiện được khai báo để dùng
về sau, chưa có retrieval/generation thực tế.

Mặc định Qdrant chạy tại cổng 6333 (`QDRANT_URL=http://localhost:6333` trên local, hoặc
`http://qdrant:6333` khi chạy trong Docker Compose sau này), tách biệt rõ ràng với cổng 8000
của backend FastAPI. Lưu ý: kết nối cơ sở dữ liệu Qdrant và luồng Hybrid RAG chưa được kích
hoạt trong mã nguồn hiện tại (hiện chỉ có API mock).

## Chưa thể chạy toàn hệ thống bằng Docker

Hai Dockerfile và `docker-compose.yml` hiện vẫn rỗng. A3 cần triển khai và kiểm
tra frontend + backend + VectorDB trước khi hướng dẫn `docker compose up --build`.
Không xem sự tồn tại của các file này là bằng chứng hệ thống đã đóng gói xong.
