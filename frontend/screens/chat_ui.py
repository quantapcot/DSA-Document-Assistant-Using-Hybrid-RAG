"""Chat UI with bottom input, real Hybrid RAG retrieval, and citation viewer."""
import streamlit as st

from api_client import ask_question

SAMPLE_PROMPTS = [
    "Cây AVL là gì và độ phức tạp tìm kiếm là bao nhiêu?",
    "Giải thích thao tác quay cây trong Cây AVL?",
    "Cây nhị phân tìm kiếm (BST) là gì?",
    "So sánh thuật toán Quick Sort và Merge Sort?",
]


def _show_message(item: dict) -> None:
    with st.chat_message(item["role"]):
        if item.get("error"):
            st.warning(item["content"])
        else:
            st.markdown(item["content"])

        if item.get("citations"):
            with st.expander(f"📌 Nguồn minh chứng ({len(item['citations'])} trích dẫn)", expanded=True):
                st.caption(
                    "Các trích dẫn này được truy xuất trực tiếp từ kho ngữ liệu slide/tài liệu DSA "
                    "bằng động cơ Hybrid Search (BM25 + Qdrant Dense + BGE Reranker)."
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
                f"⏱️ Thời gian phản hồi API: {item['process_time_ms']} ms · Hybrid RAG Engine"
            )


def render_chat_ui() -> None:
    st.header("💬 Trò chuyện & Hỏi đáp môn DSA")
    st.caption("Trợ lý hỏi đáp tài liệu môn Cấu trúc Dữ liệu & Giải thuật dựa trên Hybrid RAG (Track C).")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # Thanh công cụ hỗ trợ
    col_help, col_clear = st.columns([4, 1])
    with col_clear:
        if st.session_state["chat_messages"]:
            if st.button("🗑️ Xóa hội thoại"):
                st.session_state["chat_messages"] = []
                st.rerun()

    prompt_to_send = None

    # Gợi ý câu hỏi mẫu khi chưa có tin nhắn
    if not st.session_state["chat_messages"]:
        with st.container():
            st.info(
                "💡 **Gợi ý câu hỏi:** Bạn có thể bấm vào các gợi ý dưới đây hoặc tự nhập bất kỳ câu hỏi nào về môn DSA vào khung chat bên dưới:"
            )
            cols = st.columns(2)
            for i, p in enumerate(SAMPLE_PROMPTS):
                with cols[i % 2]:
                    if st.button(f"👉 {p}", key=f"sample_{i}"):
                        prompt_to_send = p

    # Hiển thị các tin nhắn đã có
    for item in st.session_state["chat_messages"]:
        _show_message(item)

    # Chat input ở cuối màn hình
    question = st.chat_input("Nhập câu hỏi DSA của bạn (Ví dụ: Cây AVL là gì?)...")

    if question:
        prompt_to_send = question

    if not prompt_to_send:
        return

    # Xử lý gửi câu hỏi
    user_item = {
        "role": "user",
        "content": prompt_to_send,
    }
    st.session_state["chat_messages"].append(user_item)
    _show_message(user_item)

    with st.spinner("🔍 Đang tìm kiếm bằng chứng qua Hybrid Search & Reranking..."):
        ok, data = ask_question(prompt_to_send)

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

    st.session_state["chat_messages"].append(assistant_item)
    st.rerun()
