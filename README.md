# DSA Document Assistant Using Hybrid RAG

Intelligent Document Workspace là trợ lý học DSA dùng tài liệu của từng phiên. Streamlit quản lý phiên an danh, FastAPI xác minh token và Hybrid Retrieval tìm nguồn trong Qdrant. Chat, Quiz và Study Roadmap dùng Groq.

## Chạy trên máy local

Dùng Python 3.11. Từ thư mục gốc trong PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path backend/.env)) {
    Copy-Item backend/.env.example backend/.env
}
```

Điền `GROQ_API_KEY` trong `backend/.env`. Mặc định dùng `https://api.groq.com/openai/v1` và `openai/gpt-oss-120b`. Xem [cấu hình Groq](backend/app/generation/README.md).

Chạy hai terminal riêng:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py
```

Streamlit mở tại [localhost:8501](http://localhost:8501). Tài liệu API ở [localhost:8000/docs](http://localhost:8000/docs). Dùng chung `backend/.env` nếu chạy `docker compose up --build`. Volume `data/runtime` giữ registry phiên và file tạm qua restart backend.

## Demo

1. Mở Streamlit. Backend tự tạo một phiên rỗng.
2. Tải PDF hoặc DOCX, tối đa 10 file/lượt và 15 MB/file.
3. Hỏi ở Chat và mở nguồn tham khảo. Với câu hỏi nhiều chủ đề, Chat tìm bằng chứng cho từng chủ đề và báo phần chưa đủ căn cứ. Trong Quiz, chọn một chủ đề và 1–5 câu. Trích đoạn và đáp án hiện sau khi chấm.
4. Mở **Kế hoạch học tập**, nhập chủ đề muốn ôn (ví dụ `Tree`) rồi chọn **Tạo lộ trình ôn tập**. Mỗi bước có mục tiêu và kiến thức trọng tâm dựa trên tài liệu trong phiên.
5. Tạo phiên mới để bắt đầu với tài liệu và lịch sử trống. Dùng **Xóa phiên học hiện tại** để thu hồi token và xóa file, embeddings của phiên đó.

Phiên mặc định hết hạn sau 8 giờ. Backend dọn dữ liệu hết hạn lúc khởi động và mỗi 5 phút. Trạng thái học của giao diện thuộc phiên Streamlit. Xem [hợp đồng API](docs/data_contracts.md) và [kiến trúc MVP V2](docs/MVP_V2_IMPLEMENTATION_PLAN.md).

## Kiểm tra chất lượng chunking

Để bật xuất debug, đặt `EXPORT_DEBUG_CHUNKS=true` trong `backend/.env` và khởi động lại backend. Sau khi upload và lập chỉ mục thành công, mở `data/runtime/debug_chunks/<workspace_id>/<document_id>_chunks.jsonl` trong VS Code. File UTF-8 chứa văn bản chunk và thông tin nguồn. Backend cập nhật file khi upload lại và dọn file khi xóa tài liệu, xóa phiên hoặc phiên hết hạn. Git bỏ qua các file xuất trong thư mục runtime.

## Kiểm thử

```powershell
.\.venv\Scripts\python.exe -m pip install pytest -r evaluation/requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Các test tự động dùng provider giả lập. Nếu có `GROQ_API_KEY`, chạy
`python tests/smoke_groq.py` để kiểm tra Chat, Quiz và Roadmap với một đoạn
tài liệu mẫu. Lệnh trả mã 77 khi thiếu key. Hướng dẫn benchmark nằm trong
[Evaluation README](evaluation/README.md).

## Phân công ở mốc Tuần 4

| Mã | Thành viên | Vai trò chính | Phần bàn giao cụ thể Tuần 4 |
|---|---|---|---|
| **A1** | Nguyễn Lê Phú Trọng | Backend & Điều phối | Dựng FastAPI Skeleton (`/health`, `/api/chat` mock, `/api/upload`). |
| **A2** | Nguyễn Văn An | Frontend & UI/UX | Dựng Streamlit UI, **Sidebar Navigation**, Chat UI, Quiz UI & Revision UI khung. |
| **A3** | Nguyễn Phú Nguyên Vũ | Hạ tầng & DevOps | Quản lý Git, cấu hình môi trường `.env.example`, `requirements.txt`, `.vscode/`. |
| **B1** | Nguyễn Thanh Danh | Document Intelligence | Thống nhất Data Contract `DocumentChunk`, chuẩn bị parser PDF. |
| **C1** | Trương Minh Quân | Retrieval & VectorDB | Thống nhất cấu hình **Qdrant**, chuẩn bị Hybrid Search (BM25 + Dense). |
| **C2** | Hứa Kiến Quốc | RAG & Content Generation | Thống nhất Data Contract `RAGResponse` và `Citation` cho API mock. |
| **D1** | Đặng Phan Gia Bảo | Evaluation & Benchmark | Chuẩn bị dữ liệu tài liệu DSA, kế hoạch đánh giá chất lượng RAG. |
| **D2** | Nguyễn Minh Cường | Safety & Reliability | Xây dựng bộ testcase Prompt Injection, thiết kế Guardrails. |
