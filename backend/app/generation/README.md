# Sinh nội dung bằng Groq

Chat, Quiz và Roadmap dùng cùng một client OpenAI-compatible. Điền `GROQ_API_KEY` trong `backend/.env`. Giá trị mặc định của `GROQ_BASE_URL` và `GROQ_MODEL` là `https://api.groq.com/openai/v1` và `openai/gpt-oss-120b`.

Groq GPT OSS dùng `max_completion_tokens` và reasoning effort `low`. Quiz và Roadmap gửi JSON Schema strict cho GPT OSS 120B/20B, rồi kiểm tra Pydantic và nội dung. Nếu đặt model khác, client dùng JSON Object Mode và giữ validation phía Python. [Groq Structured Outputs](https://console.groq.com/docs/structured-outputs), [Groq Reasoning](https://console.groq.com/docs/reasoning).

- Chat gọi một lần với ngữ cảnh. Khi provider lỗi hoặc đáp án thiếu nguồn, Chat hiển thị đoạn trích kèm nguyên nhân nếu có.
- Quiz lấy nguồn liên quan theo alias Việt–Anh. Draft, giải độc lập, và kiểm tra giải thích/trùng ý vẫn dùng ba prompt riêng để checker không xem đáp án hay giải thích từ draft. Code kiểm tra ngữ cảnh ẩn, nhãn nguồn và trích đoạn. Tối đa hai lượt sinh bù cho vị trí thiếu.
- Roadmap đọc toàn bộ inventory trong scope, gộp chủ đề trùng và sắp tiên quyết bằng Python. Groq nhận các mục đã xác định, thường một batch cho tài liệu nhỏ. Một batch lỗi JSON hoặc bị cắt được thử lại một lần. Mục sai nguồn hoặc chưa có mục tiên quyết hợp lệ được bỏ và báo trong `uncovered_topics`.

| Cấu hình | Mặc định |
| --- | --- |
| `LLM_CHAT_TIMEOUT`, `LLM_QUIZ_GENERATION_TIMEOUT`, `LLM_QUIZ_VERIFICATION_TIMEOUT`, `LLM_REVISION_TIMEOUT` | 30, 90, 40, 45 giây/lần |
| `CHAT_REQUEST_TIMEOUT`, `QUIZ_REQUEST_TIMEOUT`, `REVISION_REQUEST_TIMEOUT` | 90, 240, 240 giây/request |
| `ROADMAP_MAX_TOPICS`, `ROADMAP_SOURCE_BUDGET` | 20 mục, 14.000 byte nguồn ước lượng mỗi batch |

Timeout, lỗi tạm thời và rate limit có tối đa hai retry cấp HTTP, trong ngân sách request. Auth và quota cứng trả lỗi ngay. Khi có `Retry-After` lớn hơn 5 giây, client trả lỗi để giao diện phản hồi sớm. Log hiển thị thời gian, số lần gọi, retry và lý do loại nội dung. Token, API key, prompt và phản hồi thô không được ghi trong các log này.
