# Hợp đồng dữ liệu

## Nguồn hiện hành

`backend/app/schemas/contracts.py` được giữ nguyên. Các module mới
phải dùng các class hiện có tại đây, không sao chép sang file schema khác.

| Class đang có | Mục đích | Trường hiện có |
| --- | --- | --- |
| DocumentChunk | B1 bàn giao chunk cho C1 | chunk_id, document_id, filename, page, section, text, metadata |
| Citation | Bằng chứng gửi về UI | filename, page, section, snippet |
| ChatRequest | Câu hỏi gửi vào API | query |
| RAGResponse | Phản hồi hỏi đáp | query, answer, citations, confidence_score, is_safe |
| GuardrailResult | Kết quả kiểm tra an toàn | is_safe, risk_category, refusal_message |

Các mô tả này phản ánh schema hiện có, không cam kết toàn bộ ràng buộc dữ liệu
đã được kiểm tra bằng validator. Chat vẫn trả giá trị giả lập.

## Cần thống nhất trước khi tích hợp

- C1 → C2: cách trả chunk sau retrieval/rerank, điểm số và thứ hạng; phân biệt
  điểm BM25, vector, fusion và reranker nếu cần phân tích.
- Upload: mã tài liệu, trạng thái xử lý/lập chỉ mục, thông tin lỗi.
- Quiz: câu hỏi, phương án, đáp án, giải thích và bằng chứng.
- Revision: cấu trúc chủ đề/kiến thức, thứ tự và nguồn tham chiếu.
- Từ chối: phân biệt thiếu bằng chứng, yêu cầu ngoài phạm vi và vi phạm chính sách.
- Citation: liên kết ổn định tới tài liệu/chunk, quy ước đánh số trang.
- Đánh giá: câu hỏi, đáp án chuẩn, các nguồn/chunk liên quan dùng tính Recall/MRR.

Đây là các quyết định còn mở, chưa phải schema đã được thêm vào code. A1 điều phối
thống nhất cùng B1/C1/C2/D2. Chỉ tách `contracts.py` khi được phép cập nhật cả import
và bên sử dụng; lần sắp xếp này không tách hoặc sửa file đó.
