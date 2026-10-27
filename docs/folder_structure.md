# Cấu trúc thư mục hiện tại

Tài liệu này mô tả cây thư mục đang có trong repo và ranh giới công việc theo phân công A1–D2. Tên folder thể hiện **nơi phụ trách công việc**, không chứng minh chức năng đã được triển khai.

**Trạng thái thực tế:** Backend có FastAPI, endpoint health và chat trả dữ liệu giả lập. Nhiều file còn rỗng hoặc chỉ chứa mô tả nhiệm vụ. Frontend, xử lý PDF, hybrid retrieval, RAG, guardrail, đánh giá và Docker chưa có triển khai đầy đủ.

## Cây thư mục

Cây dưới đây lược bỏ .git, cache Python, các file __init__.py và README nhỏ bên trong từng folder để dễ nhìn. Các đường dẫn đều tính từ gốc repo.

~~~text
DSA-Document-Assistant-Using-Hybrid-RAG/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── chat.py
│   │   │   │   ├── health.py
│   │   │   │   ├── upload.py
│   │   │   │   ├── quiz.py
│   │   │   │   └── revision.py
│   │   │   └── middleware/
│   │   │       └── observability.py
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── schemas/
│   │   │   └── contracts.py
│   │   ├── orchestration/
│   │   ├── document_processing/
│   │   ├── retrieval/
│   │   ├── generation/
│   │   │   └── prompts/
│   │   └── safety/
│   ├── .env.example
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── screens/
│   ├── components/
│   ├── requirements.txt
│   └── Dockerfile
├── configs/
│   ├── retrieval.example.yaml
│   └── generation.example.yaml
├── data/
│   ├── corpus/
│   └── runtime/
│       ├── uploads/
│       ├── processed/
│       └── indexes/
├── evaluation/
│   ├── datasets/
│   ├── metrics/
│   ├── scripts/
│   ├── results/
│   └── reports/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/
├── .gitignore
├── docker-compose.yml
└── README.md
~~~

## Backend: máy chủ và các module xử lý

Folder backend/app là gói ứng dụng của FastAPI. Nhiều nhóm cùng làm trong backend; đây không phải folder riêng của Team A.

| Folder hoặc file | Tác dụng và ranh giới | Phụ trách | Trạng thái hiện tại |
| --- | --- | --- | --- |
| backend/ | Chứa ứng dụng phía máy chủ, danh sách thư viện và Dockerfile của backend. | A1 quản lý API; A3 phối hợp tích hợp | Mới có backend cơ bản; Dockerfile rỗng |
| backend/app/main.py | Khởi tạo FastAPI, gắn middleware và đăng ký router. | A1 | Có code; hiện chỉ đăng ký health và chat |
| backend/app/api/ | Tầng giao tiếp HTTP. Nhận request, trả response; không phải nơi viết thuật toán PDF/RAG. | A1 | Có khung router và middleware |
| backend/app/api/routes/ | Mỗi file đại diện cho một nhóm endpoint. | A1 | chat.py trả mock, health.py trả thông tin ứng dụng; upload.py, quiz.py, revision.py còn rỗng |
| backend/app/api/middleware/ | Xử lý chung quanh request. | A1; D1 dùng số liệu khi đánh giá | observability.py đo thời gian cho đường dẫn bắt đầu bằng /api/ và gắn response header |
| backend/app/core/ | Cấu hình chung của ứng dụng. | A1, A3 | config.py đã khai báo tên ứng dụng, môi trường, host/port và biến LLM/Qdrant; chưa có kết nối LLM/VectorDB |
| backend/app/schemas/ | Hợp đồng dữ liệu dùng chung giữa các module và API. | A1 điều phối; B1/C1/C2/D2 xác nhận phần liên quan | contracts.py đã định nghĩa DocumentChunk, Citation, ChatRequest, RAGResponse, GuardrailResult |
| backend/app/orchestration/ | Vị trí dự kiến để phân loại ý định và nối API với các module xử lý; không tự làm thay thuật toán của B1/C1/C2/D2. | A1 | Mới có khung intent_router.py, request_flow.py, ingestion_flow.py; chưa được API gọi |
| backend/app/document_processing/ | Vị trí dự kiến đọc PDF, làm sạch, chia đoạn và gắn filename/page/section để tạo DocumentChunk. Không quản lý VectorDB. | B1 | document_parser.py còn rỗng; text_cleaner.py, chunker.py, metadata.py mới có mô tả nhiệm vụ |
| backend/app/retrieval/ | Vị trí dự kiến tạo embedding, lập chỉ mục, tìm theo từ khóa/vector, kết hợp kết quả và rerank. | C1 | Các file hiện chỉ là khung; chưa có tìm kiếm thực tế |
| backend/app/generation/ | Vị trí dự kiến tạo câu trả lời, quiz, lộ trình ôn tập và gắn trích dẫn; xử lý trường hợp không đủ bằng chứng. | C2 | rag_engine.py, quiz_generator.py, revision_generator.py còn rỗng; các file khác là khung |
| backend/app/generation/prompts/ | Nơi đặt prompt khi C2 triển khai generation. | C2 | Mới có README hướng dẫn, chưa có prompt hoạt động |
| backend/app/safety/ | Vị trí dự kiến viết quy tắc và module kiểm tra an toàn; A1 sẽ gọi từ luồng điều phối. | D2 viết logic; A1 tích hợp | guardrail.py còn rỗng; policies.py mới là khung |
| backend/.env.example | Danh sách biến môi trường mẫu, không chứa khóa thật. | A3 phối hợp A1/C1/C2 | Có file mẫu; biến LLM/Qdrant chưa được dùng trong luồng xử lý |
| backend/requirements.txt | Khai báo thư viện backend. | A1, A3 phối hợp | Có các phụ thuộc FastAPI, Uvicorn, Pydantic và phụ thuộc hỗ trợ |
| backend/Dockerfile | Vị trí đóng gói backend. | A3 | File rỗng |

Các file __init__.py trong backend là dấu mốc package Python. File rỗng ở vị trí này là bình thường; nó không có nghĩa module nghiệp vụ đã được viết.

## Frontend: giao diện

| Folder hoặc file | Tác dụng và ranh giới | Phụ trách | Trạng thái hiện tại |
| --- | --- | --- | --- |
| frontend/ | Gom toàn bộ phần giao diện phía người dùng. | A2 | Chưa có giao diện hoạt động |
| frontend/app.py | Vị trí dự kiến làm điểm vào ứng dụng Streamlit. | A2 | File rỗng |
| frontend/api_client.py | Vị trí dự kiến gom các lời gọi HTTP đến backend để màn hình không tự viết lại logic kết nối. | A2 | Chỉ có mô tả nhiệm vụ |
| frontend/screens/ | Bốn file màn hình chat, upload, quiz, revision. | A2 | Cả bốn file màn hình còn rỗng |
| frontend/components/ | Thành phần UI dùng lại ở nhiều màn hình, ví dụ phần hiển thị trích dẫn khi được triển khai. | A2 | Mới có README hướng dẫn |
| frontend/requirements.txt, frontend/Dockerfile | Nơi khai báo phụ thuộc và đóng gói frontend. | A2 phối hợp A3 | Cả hai file rỗng |

## Cấu hình, dữ liệu và đánh giá

| Folder | Tác dụng và ranh giới | Phụ trách | Trạng thái hiện tại |
| --- | --- | --- | --- |
| configs/ | Chứa hai YAML **mẫu** cho retrieval và generation; không phải cấu hình đang được ứng dụng nạp. | C1 cho retrieval, C2 cho generation; D1 ghi nhận cấu hình thí nghiệm | Có retrieval.example.yaml, generation.example.yaml và README; chưa tích hợp |
| data/ | Phân biệt bộ tài liệu gốc dùng để đánh giá với dữ liệu sinh khi người dùng sử dụng ứng dụng. | D1 quản lý corpus; A1/A3 phối hợp lưu trữ runtime | Hiện chỉ có README trong các nhánh dữ liệu |
| data/corpus/ | Vị trí dự kiến chứa tài liệu DSA chuẩn phục vụ đánh giá. | D1 | Chưa có corpus |
| data/runtime/ | Dữ liệu phát sinh trong quá trình vận hành, không dùng thay corpus chuẩn. | A1, A3; B1/C1 thống nhất định dạng phần mình | Mới có folder và README; dữ liệu runtime được .gitignore loại trừ |
| data/runtime/uploads/ | Vị trí dự kiến chứa file người dùng tải lên. | A1 phối hợp A3 | Chưa có file upload |
| data/runtime/processed/ | Vị trí dự kiến chứa văn bản/chunk/metadata đã xử lý nếu nhóm chọn lưu ra đĩa. | B1 phối hợp A1 | Chưa có dữ liệu |
| data/runtime/indexes/ | Vị trí dự kiến chứa chỉ mục local nếu C1 chọn lưu theo cách này. | C1 phối hợp A3 | Chưa có chỉ mục |
| evaluation/ | Đo chất lượng và an toàn bằng dữ liệu, chỉ số và báo cáo; khác với tests kiểm tra code. | D1 phụ trách chất lượng, D2 phụ trách safety | Chưa có lần đánh giá hoàn chỉnh |
| evaluation/datasets/ | Vị trí dự kiến cho câu hỏi benchmark có đáp án/nguồn chuẩn và testcase red-team. | D1 cho benchmark, D2 cho red-team | Mới có README, chưa có dataset |
| evaluation/metrics/ | Vị trí dự kiến tính chỉ số retrieval, chất lượng câu trả lời và safety. | D1 cho retrieval/answer; D2 cho safety | Các file Python mới có mô tả nhiệm vụ, chưa có phép tính |
| evaluation/scripts/ | Điểm vào dự kiến cho benchmark, ablation và red-team. | D1 cho benchmark/ablation; D2 cho red-team | Ba script còn rỗng |
| evaluation/results/ | Vị trí dự kiến lưu số liệu từng lần chạy. | D1/D2 | Chưa có kết quả; dữ liệu chạy được .gitignore loại trừ |
| evaluation/reports/ | Vị trí dự kiến lưu báo cáo tổng hợp đã có nội dung và được kiểm tra. | D1/D2 | Mới có README, chưa có báo cáo |

## Kiểm thử, tài liệu và file gốc

| Folder hoặc file | Tác dụng và ranh giới | Phụ trách | Trạng thái hiện tại |
| --- | --- | --- | --- |
| tests/unit/ | Vị trí dự kiến kiểm tra hành vi từng module riêng lẻ. | Người làm module tương ứng | Mới có README, chưa có test |
| tests/integration/ | Vị trí dự kiến kiểm tra kết nối giữa các module và API. | A3 phối hợp A1/B1/C1/C2/D2 | Mới có README, chưa có test |
| tests/fixtures/ | Dữ liệu nhỏ, ổn định để dùng trong kiểm thử. | Các owner viết test | Mới có README, chưa có fixture |
| docs/ | Tài liệu kiến trúc, phân công, hợp đồng dữ liệu và cấu trúc. Mỗi tài liệu cần owner nghiệp vụ xác nhận khi phần đó thay đổi. | A3 giữ bố cục chung; A1/B1/C1/C2/D1/D2 xác nhận nội dung liên quan | Có các tài liệu Markdown; một số mô tả thiết kế dự kiến |
| README.md | Giới thiệu trạng thái project, tech stack và luồng chat giả lập hiện có. | Nhóm duy trì chung | Có nội dung |
| .gitignore | Bỏ qua cache, môi trường cục bộ, secret và dữ liệu runtime/kết quả sinh ra. | A3 | Có nội dung |
| docker-compose.yml | Vị trí dự kiến ghép các dịch vụ khi được triển khai. | A3 | File rỗng |

## Cách đọc phân công

- **A1 → B1 → C1 → C2** là ranh giới công việc dự kiến: API/điều phối → xử lý tài liệu → truy xuất → sinh nội dung. Đây **chưa phải pipeline đang chạy**.
- **D2** sở hữu logic guardrail; **A1** quyết định chỗ gọi nó trong API. **D1** đo chất lượng; A3 hỗ trợ môi trường và tích hợp.
- Luồng đang có trong code chỉ là: POST /api/chat → kiểm tra câu hỏi → trả lời AVL và trích dẫn giả lập. Không có PDF parsing, VectorDB, hybrid search hay LLM được gọi trong luồng này.
