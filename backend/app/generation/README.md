# Sinh nội dung C2

Ba endpoint POST `/api/chat`, `/api/quiz`, `/api/revision` nhận
`workspace_id`, `corpus_id` tùy chọn. Mặc định là `default-workspace` và
`user-uploads`. Khi reranker lỗi, RRF được gọi trong cùng scope.

Quiz và Revision gọi FIT-AI qua giao diện tương thích OpenAI. Đặt
`LLM_BASE_URL` bằng URL được cấp, `LLM_MODEL` bằng ID Qwen-27B trên gateway
và `OPENAI_API_KEY` bằng key tương ứng trong `backend/.env`. Chat tiếp tục
dùng `LLM_PROVIDER` và có chế độ trích xuất nếu model không khả dụng.

- Quiz nhận một `topic` và `num_questions` (1–5). API chấp nhận `query`
  của client cũ và bỏ qua `difficulty`. Từ khóa như `sort`, `graph`,
  `binary search` và `AVL` sử dụng alias Việt/Anh để lọc nguồn trong scope.
  Checker giải độc lập từ câu hỏi, lựa chọn và nguồn. Đoạn trích được đối
  chiếu bằng code. Lượt kiểm tra tiếp theo kiểm tra giải thích và câu hỏi
  trùng ý. Có tối đa hai lượt sinh bù cho các vị trí thiếu. API báo
  `generated_count` trên `requested_count` và lý do khi thiếu câu.
  Streamlit hiển thị giải thích và đáp án sau khi nộp, sử dụng đáp án
  từ API để chấm.
- Revision nhận `exam_date` hoặc `days_remaining` (1–90), `minutes_per_day`
  (15–240), `level` và `query` hoặc `all_topics=true`. Bộ lập lịch Python
  ưu tiên theo quan hệ tiên quyết suy ra từ tài liệu và chọn thời lượng theo
  độ phức tạp, trình độ. FIT-AI sinh nội dung theo batch tối đa ba nhiệm vụ.
  Batch lỗi JSON hoặc bị cắt được thử lại theo từng nhiệm vụ với cùng ID
  và nguồn. Các mục chưa xếp lịch được báo riêng. Checklist chỉ lưu trong
  phiên Streamlit.
- Nếu thiếu FIT-AI hoặc không có đầu ra đạt kiểm tra, Quiz/Revision trả
  `generation_mode=unavailable` cùng thông báo. Workspace rỗng trả
  `generation_mode=abstained`.

Các thiết lập vận hành trong `backend/.env`:

| Nhóm | Biến | Mặc định |
| --- | --- | --- |
| Timeout mỗi lần gọi model (giây) | `LLM_CHAT_TIMEOUT`, `LLM_QUIZ_GENERATION_TIMEOUT`, `LLM_QUIZ_VERIFICATION_TIMEOUT`, `LLM_REVISION_TIMEOUT` | 30, 90, 40, 45 |
| Ngân sách mỗi HTTP request (giây) | `CHAT_REQUEST_TIMEOUT`, `QUIZ_REQUEST_TIMEOUT`, `REVISION_REQUEST_TIMEOUT` | 90, 240, 240 |
| Giới hạn phạm vi lập lịch | `REVISION_MAX_TOPICS`, `REVISION_SOURCE_TOKEN_BUDGET` | 20, 12000 |

Streamlit đọc ngân sách HTTP từ `/health`. Output FIT-AI cho Quiz và Revision
được chặn ở 4.000 token mỗi lần gọi. Ngân sách nguồn của Revision dùng ước
lượng bảo thủ theo kích thước UTF-8. Hết hạn mức không được retry. Chat giữ
fallback trích xuất và gửi lý do cho giao diện qua `generation_error`.

Log `study_json` và `study_output` ghi giai đoạn xử lý, slot, nhãn nguồn
và mã từ chối. Log Quiz ghi thêm thời gian mỗi bước và số lần gọi FIT-AI
thực tế, bao gồm retry của client. Log không ghi prompt, phản hồi thô hoặc
nội dung tài liệu.
Bộ đọc chấp nhận biểu diễn tương đương như `"1"` hoặc `"[1]"` cho nhãn nguồn
và vẫn kiểm tra nhãn trong phạm vi bằng chứng. Quote được đối chiếu theo
chuỗi token liền nhau sau chuẩn hóa Unicode và định dạng PDF. Thay từ,
đổi số hoặc đổi toán tử vẫn làm đoạn trích bị loại.

`confidence_score` là điểm hỗ trợ từ nguồn tính theo độ phủ trích dẫn và
relevance trên thang của phương thức retrieval, có phạt câu trả lời mơ hồ.
Điểm này chưa được hiệu chuẩn thành xác suất đúng. Trích dẫn giữ nguyên nhãn
nguồn để giao diện có thể hiển thị [2] dù [1] không được viện dẫn.
