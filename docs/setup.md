# Chạy backend hiện tại

## Những gì có thể chạy

Backend FastAPI có `/`, `/health`, `/api/chat` và trang `/docs`.
Chat là mock, không cần gọi LLM/VectorDB. Frontend, upload, quiz, revision và
Docker Compose chưa có triển khai đầy đủ.

## Chạy từ thư mục gốc repo bằng PowerShell

Cần Python phù hợp với dependencies của nhóm. Các lệnh cài đặt dưới đây do
người phát triển chạy khi muốn chuẩn bị môi trường; lần sắp xếp thư mục không
cài đặt hoặc thay đổi dependencies.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Chỉ sao chép `.env.example` nếu chưa có `.env`; không ghi đè cấu hình cá nhân.
API docs: http://127.0.0.1:8000/docs

Dùng `app.main:app` từ thư mục `backend` để khớp các import `app.*` hiện có.
Không dựa vào block chạy trực tiếp trong `main.py` cho hướng dẫn này.

## Cấu hình

`app/core/config.py` đọc `.env` theo thư mục chạy. Vì vậy ví dụ môi trường nằm
ở `backend/.env.example`; đặt `.env` bên cạnh khi chạy theo hướng dẫn trên.
Không đưa API key thật vào Git. Các biến LLM/ChromaDB hiện được khai báo để dùng
về sau, chưa có retrieval/generation thực tế.

Khi triển khai VectorDB, C1/A3 cần thống nhất địa chỉ và cổng; tránh chạy backend
và VectorDB cùng chiếm một cổng trên cùng host. Không thay đổi mặc định trong code
ở lần sắp xếp này.

## Chưa thể chạy toàn hệ thống bằng Docker

Hai Dockerfile và `docker-compose.yml` hiện vẫn rỗng. A3 cần triển khai và kiểm
tra frontend + backend + VectorDB trước khi hướng dẫn `docker compose up --build`.
Không xem sự tồn tại của các file này là bằng chứng hệ thống đã đóng gói xong.
