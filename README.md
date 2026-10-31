# DSA Document Assistant Using Hybrid RAG
**Track C — Intelligent Document Workspace**  
**Trợ lý Học tập & Ôn thi môn Cấu trúc Dữ liệu và Giải thuật (DSA)**

---

## 1. Trạng thái dự án: Mốc bàn giao Tuần 4 (PoC & Khung hạ tầng)

Ở mốc **Tuần 4**, dự án đã hoàn thành khung hệ thống cơ bản, cấu hình môi trường dùng chung và giao diện Proof of Concept (PoC) kết nối **FastAPI Backend (Port 8000)** và **Streamlit Frontend (Port 8501)**. 

### Tập trung vào các file thay đổi cụ thể của A2 và A3:

#### 🟢 Phân công của A3 (Hạ tầng, Môi trường & Tích hợp)
- [`requirements.txt`](requirements.txt): File khai báo thư viện khung ở thư mục gốc, liên kết trực tiếp `-r backend/requirements.txt` và `-r frontend/requirements.txt`, cho phép thiết lập toàn bộ môi trường chỉ với một câu lệnh.
- [`.env.example`](.env.example): Template biến môi trường chuẩn ở thư mục gốc (`HOST=0.0.0.0`, `PORT=8000`, `BACKEND_URL=http://127.0.0.1:8000`, và cấu hình Vector DB `QDRANT_URL=http://localhost:6333` — thay thế hoàn toàn ChromaDB).
- [`.vscode/launch.json`](.vscode/launch.json) & [`.vscode/tasks.json`](.vscode/tasks.json): Cung cấp sẵn cấu hình Run & Debug và Tasks cho **Visual Studio / VS Code**, hỗ trợ chạy song song cả Backend và Frontend chỉ với một phím tắt (`F5`).

#### 🟢 Phân công của A2 (Giao diện người dùng Streamlit)
- [`frontend/app.py`](frontend/app.py): Điểm vào chính của ứng dụng Streamlit, hoàn thiện **Sidebar Navigation** 4 phân hệ chuyển trang và widget giám sát trạng thái Backend (`/health`) theo thời gian thực.
- [`frontend/screens/chat_ui.py`](frontend/screens/chat_ui.py): Màn hình Hỏi đáp kiến thức DSA. Tích hợp nút bấm 1-click gửi nhanh câu hỏi mẫu Cây AVL, nút xóa lịch sử chat, hiển thị trích dẫn tài liệu giả lập (**Mock Citation** theo Data Contract) và thời gian phản hồi API (`X-Process-Time-MS`).
- [`frontend/screens/upload_ui.py`](frontend/screens/upload_ui.py): Màn hình Quản lý & Tải lên tài liệu PDF độc lập, kiểm tra định dạng PDF hợp lệ (header `%PDF-`), giới hạn kích thước ≤10MB và lưu trữ file an toàn tại `data/runtime/uploads/`.
- [`frontend/screens/quiz_ui.py`](frontend/screens/quiz_ui.py): Dựng khung giao diện (Skeleton) cho tính năng **Luyện tập Trắc nghiệm DSA** với câu hỏi mẫu và trích dẫn lời giải (chuẩn bị tích hợp Core & API ở Tuần 7).
- [`frontend/screens/revision_ui.py`](frontend/screens/revision_ui.py): Dựng khung giao diện (Skeleton) cho tính năng **Lộ trình Ôn tập DSA** với sơ đồ cây chủ đề kiến thức (Mindmap) và kế hoạch từng bước (chuẩn bị tích hợp ở Tuần 7).
- [`frontend/api_client.py`](frontend/api_client.py): Module HTTP Client xử lý kết nối từ giao diện Streamlit sang API FastAPI (`/health`, `/api/chat`, `/api/upload`).

---

## 2. Hướng dẫn chạy dự án trên Terminal của Visual Studio Code

### 2.1. Yêu cầu môi trường
- **Python**: Phiên bản **3.11.x** (khuyến nghị dùng thống nhất trong toàn nhóm).
- Khởi chạy trực tiếp từ Terminal tích hợp của VS Code (PowerShell trên Windows).

---

### 2.2. Bước 1: Khởi tạo Virtual Environment và cài đặt thư viện

Mở terminal trong VS Code tại thư mục gốc của dự án (`DSA-Document-Assistant-Using-Hybrid-RAG`):

```powershell
# 1. Tạo môi trường ảo với Python 3.11
py -3.11 -m venv .venv

# 2. Nâng cấp pip
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# 3. Cài đặt thư viện: Có thể cài riêng từng phần:
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
.\.venv\Scripts\python.exe -m pip install -r frontend/requirements.txt

# (Hoặc cài đặt đồng thời toàn bộ qua file requirements.txt gốc):
# .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

### 2.3. Bước 2: Tạo file cấu hình môi trường (.env)

Chạy lệnh sau tại terminal để sao chép file cấu hình mẫu:

```powershell
# Tạo .env ở thư mục gốc và thư mục backend
Copy-Item .env.example .env
if (-not (Test-Path backend/.env)) {
    Copy-Item backend/.env.example backend/.env
}
```

---

### 2.4. Bước 3: Khởi chạy dự án bằng lệnh Terminal trên VS Code

Dự án gồm 2 dịch vụ độc lập, chạy trên 2 cửa sổ Terminal riêng biệt trong VS Code:

#### 🔹 Cửa sổ Terminal 1 — Khởi động Backend (FastAPI):
Nhấn biểu tượng `+` hoặc `Ctrl + Shift + \`` để mở terminal mới và chạy:

```powershell
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Docs (Swagger UI): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Kiểm tra Health: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

#### 🔹 Cửa sổ Terminal 2 — Khởi động Frontend (Streamlit):
Mở tiếp một terminal thứ hai trong VS Code và chạy:

```powershell
cd frontend
..\.venv\Scripts\python.exe -m streamlit run app.py
```
- Giao diện Web Streamlit: [http://127.0.0.1:8501](http://127.0.0.1:8501)

*(Lưu ý: Nếu bạn đã kích hoạt môi trường `.venv` trong terminal, bạn có thể gọi trực tiếp lệnh ngắn: `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload` và `python -m streamlit run app.py`).*

---

### 2.5. (Cách khác) Khởi chạy một chạm bằng Debugger của VS Code

Dự án đã được cấu hình sẵn file [`.vscode/launch.json`](.vscode/launch.json):
1. Nhấn tổ hợp phím `Ctrl + Shift + D` trên VS Code để mở tab **Run & Debug**.
2. Trên thanh menu dropdown chọn: **`Run Both (Backend + Frontend)`**.
3. Nhấn **`F5`** (hoặc nút Play xanh). VS Code sẽ tự động khởi động đồng thời cả FastAPI và Streamlit trong 2 tiến trình riêng.

---

## 3. Kiểm thử nhanh các chức năng đã bàn giao

1. **Kiểm tra kết nối**: Quan sát Sidebar bên trái của Streamlit, biểu tượng hiển thị `● FastAPI: Hoạt động (healthy)`.
2. **Hỏi đáp Cây AVL (Mock)**:
   - Vào mục **💬 Trò chuyện & Hỏi đáp**.
   - Bấm nút **👉 Thử ngay: "Cây AVL là gì và độ phức tạp tìm kiếm là bao nhiêu?"**.
   - Xem câu trả lời giải thích cây AVL và mở mục **📌 Nguồn minh họa (MOCK Trích dẫn)** để xem trích dẫn nguồn slide và thời gian phản hồi API.
3. **Tải lên PDF**:
   - Chuyển sang mục **📤 Tải lên tài liệu**.
   - Chọn một file `.pdf` bất kỳ (≤ 10MB) và nhấn **Tải lên Backend** để kiểm tra cơ chế nhận diện và lưu file tạm.
4. **Khung màn hình Tuần 7**:
   - Chuyển sang **📝 Luyện tập Trắc nghiệm** và **🗺️ Lộ trình ôn tập** để xem trước bộ khung giao diện mẫu.

---

## 4. Phân công trách nhiệm nhóm

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
