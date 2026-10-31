"""Chat UI with bottom input, mock AVL question, and citation viewer."""
import streamlit as st

from api_client import ask_question

SAMPLE_PROMPT = "Cây AVL là gì và độ phức tạp tìm kiếm là bao nhiêu?"


def _show_message(item: dict) -> None:
    with st.chat_message(item["role"]):
        if item.get("error"):
            st.warning(item["content"])
        else:
            st.markdown(item["content"])

        if item.get("citations"):
            with st.expander("📌 Nguồn minh họa (MOCK Trích dẫn)", expanded=True):
                st.caption(
                    "Các trích dẫn này là dữ liệu giả lập theo Data Contract Tuần 4, "
                    "chưa lấy trực tiếp từ PDF upload."
                )

                for index, citation in enumerate(item["citations"], start=1):
                    st.markdown(
                        f"**[{index}] {citation.get('filename', 'Unknown')} — "
                        f"Trang {citation.get('page', '?')}**"
                    )

                    if citation.get("section"):
                        st.caption(f"Mục: {citation['section']}")

                    st.markdown(f"> {citation.get('snippet', '')}")

        if item.get("process_time_ms"):
            st.caption(
                f"⏱️ Thời gian phản hồi API: {item['process_time_ms']} ms · Dữ liệu Mock PoC"
            )


def render_chat_ui() -> None:
    st.header("💬 Trò chuyện & Hỏi đáp môn DSA")
    st.caption("Khung giao diện hỏi đáp tài liệu môn Cấu trúc Dữ liệu & Giải thuật (Track C).")

    if "poc_messages" not in st.session_state:
        st.session_state["poc_messages"] = []

    # Thanh công cụ hỗ trợ
    col_help, col_clear = st.columns([4, 1])
    with col_clear:
        if st.session_state["poc_messages"]:
            if st.button("🗑️ Xóa hội thoại"):
                st.session_state["poc_messages"] = []
                st.rerun()

    prompt_to_send = None

    # Gợi ý câu hỏi mẫu khi chưa có tin nhắn
    if not st.session_state["poc_messages"]:
        with st.container():
            st.info(
                "💡 **Gợi ý nhanh (PoC Tuần 4):** Backend hiện đang chạy mock endpoint `/api/chat`. "
                "Bạn có thể nhấn vào nút bên dưới để gửi ngay câu hỏi mẫu mà không cần nhập thủ công."
            )
            if st.button(f"👉 Thử ngay: \"{SAMPLE_PROMPT}\""):
                prompt_to_send = SAMPLE_PROMPT

    # Hiển thị các tin nhắn đã có
    for item in st.session_state["poc_messages"]:
        _show_message(item)

    # Chat input ở cuối màn hình
    question = st.chat_input(
        f"Nhập câu hỏi (Ví dụ: {SAMPLE_PROMPT})"
    )

    if question:
        prompt_to_send = question

    if not prompt_to_send:
        return

    # Xử lý gửi câu hỏi
    user_item = {
        "role": "user",
        "content": prompt_to_send,
    }
    st.session_state["poc_messages"].append(user_item)
    _show_message(user_item)

    # Kiểm tra khớp câu hỏi mẫu cho PoC
    if " ".join(prompt_to_send.split()) != SAMPLE_PROMPT:
        assistant_item = {
            "role": "assistant",
            "content": (
                f"⚠️ Bản PoC Tuần 4 hiện chỉ hỗ trợ câu hỏi mẫu cố định:\n\n"
                f"**\"{SAMPLE_PROMPT}\"**\n\n"
                f"*(Các câu hỏi tự do kết nối Hybrid RAG thực tế sẽ được hỗ trợ từ Tuần 5 & 6)*."
            ),
            "error": True,
        }
    else:
        with st.spinner("Đang gửi yêu cầu đến FastAPI Backend..."):
            ok, data = ask_question(SAMPLE_PROMPT)

        if ok:
            assistant_item = {
                "role": "assistant",
                "content": data.get("answer", ""),
                "citations": data.get("citations", []),
                "process_time_ms": data.get("process_time_ms"),
            }
        else:
            assistant_item = {
                "role": "assistant",
                "content": (
                    "❌ Không kết nối được với FastAPI Backend:\n"
                    f"`{data.get('error', 'Unknown error')}`\n\n"
                    "Vui lòng đảm bảo Backend đang chạy tại `http://127.0.0.1:8000`."
                ),
                "error": True,
            }

    st.session_state["poc_messages"].append(assistant_item)
    st.rerun()
