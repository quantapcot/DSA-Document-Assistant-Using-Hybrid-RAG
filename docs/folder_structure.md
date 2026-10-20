# Bản đồ thư mục ứng dụng

**Đọc trước:** đây là cấu trúc tổ chức công việc. Ngoài phần API đã có, các file
mới chỉ có mô tả nhiệm vụ hoặc còn rỗng; chúng chưa cung cấp chức năng thực tế.

```text
repo/
├── backend/
│   ├── app/
│   │   ├── main.py                  Điểm khởi tạo FastAPI hiện có
│   │   ├── api/
│   │   │   ├── routes/              A1: health, chat, upload, quiz, revision
│   │   │   └── middleware/          A1: observability.py hiện có
│   │   ├── core/                   config.py hiện có, đọc biến môi trường
│   │   ├── schemas/                contracts.py là nguồn schema hiện tại
│   │   ├── orchestration/          A1: intent và điều phối các module
│   │   ├── document_processing/    B1: PDF → DocumentChunk
│   │   ├── retrieval/              C1: embedding, index, hybrid search, rerank
│   │   ├── generation/             C2: RAG, quiz, revision, citation, abstention
│   │   │   └── prompts/            C2: mẫu hướng dẫn cho mô hình
│   │   └── safety/                 D2: guardrail và chính sách an toàn
│   ├── .env.example                Biến môi trường mẫu cho backend
│   ├── requirements.txt            Giữ nguyên dependencies hiện có
│   └── Dockerfile                  Chưa triển khai
├── frontend/
│   ├── app.py                      Điểm chạy UI, hiện rỗng
│   ├── api_client.py               A2: vị trí dành cho gọi backend
│   ├── screens/                    A2: bốn màn hình chính
│   ├── components/                 A2: thành phần UI dùng lại
│   ├── requirements.txt            Chưa khai báo dependencies UI
│   └── Dockerfile                  Chưa triển khai
├── configs/                        Cấu hình thuật toán mẫu, chưa được nạp
├── data/
│   ├── corpus/                     D1: tài liệu gốc của bộ đánh giá
│   └── runtime/
│       ├── uploads/                Tài liệu người dùng tải lên
│       ├── processed/              Chunk/metadata sinh ra
│       └── indexes/                Chỉ mục local nếu hệ thống sử dụng
├── evaluation/
│   ├── datasets/                   Benchmark có ground truth và red-team
│   ├── metrics/                    Định nghĩa cách tính chỉ số
│   ├── scripts/                    Chạy benchmark, ablation, red-team
│   ├── results/                    Số liệu từng lần chạy, Git bỏ qua
│   └── reports/                    Báo cáo tổng hợp đã được kiểm tra
├── tests/
│   ├── unit/                       Kiểm thử từng module
│   ├── integration/                Kiểm thử kết nối các module/API
│   └── fixtures/                   Dữ liệu nhỏ dùng cho kiểm thử
└── docker-compose.yml              Chưa triển khai
```

## Quy tắc đặt code

- Route nhận request, gọi điều phối và trả response.
- Orchestration nối các bước; không viết lại parser, retrieval hoặc generation.
- Xử lý PDF kết thúc bằng `DocumentChunk`; C1 tiếp nhận chunk để lập chỉ mục.
- `generation/` sở hữu logic sinh nội dung; `safety/` sở hữu quy tắc an toàn.
- `screens/` chứa màn hình, `components/` chứa phần giao diện tái sử dụng.
- `tests/` kiểm tra tính đúng của code; `evaluation/` đo chất lượng hệ thống.
- README giữ các thư mục chưa có dữ liệu hiện diện trong Git, đồng thời giải thích
  cách sử dụng; không tạo PDF/JSON rỗng để giả lập báo cáo.

Xem [phân công](ownership.md), [cách chạy](setup.md) và
[hợp đồng dữ liệu](data_contracts.md).
