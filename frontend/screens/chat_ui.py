"""Chat UI with bottom input, real Hybrid RAG retrieval, and citation viewer."""
import streamlit as st

from api_client import ask_question, generation_error_message

SAMPLE_PROMPTS = [
    "Cây AVL là gì và độ phức tạp tìm kiếm là bao nhiêu?",
    "Giải thích thao tác quay cây trong Cây AVL?",
    "Cây nhị phân tìm kiếm (BST) là gì?",
    "So sánh thuật toán Quick Sort và Merge Sort?",
]


def show_study_metadata(item: dict) -> None:
    if item.get("generation_mode") == "extractive":
        reason = generation_error_message(item.get("generation_error"))
        st.warning("Đang dùng chế độ trích xuất trực tiếp từ tài liệu." + (f" {reason}" if reason else ""))
    confidence = item.get("confidence_score")
    if confidence is not None and item.get("generation_mode") != "abstained":
        st.caption(f"Điểm hỗ trợ từ nguồn · {confidence:.2f}/1")
    citations = item.get("citations") or []
    if citations:
        with st.expander(f"Nguồn minh chứng ({len(citations)})"):
            for index, citation in enumerate(citations, 1):
                label = citation.get("label") or index
                st.markdown(f"**[{label}] {citation['filename']} · Trang {citation['page']}**")
                if citation.get("section"):
                    st.caption(citation["section"])
                st.write(citation["snippet"])


def _show_message(item: dict) -> None:
    with st.chat_message(item["role"]):
        if item.get("error") or item.get("generation_mode") == "abstained":
            st.warning(item["content"])
        else:
            st.markdown(item["content"])

        show_study_metadata(item)

        if item.get("process_time_ms"):
            st.caption(
                f"⏱️ Thời gian phản hồi API: {item['process_time_ms']} ms · Hybrid RAG Engine"
            )


def render_chat_ui(workspace_id="default-workspace", corpus_id="user-uploads") -> None:
    st.header("💬 Trò chuyện & Hỏi đáp môn DSA")
    st.caption("Trợ lý hỏi đáp tài liệu môn Cấu trúc Dữ liệu & Giải thuật dựa trên Hybrid RAG (Track C).")

    history_key = f"chat_messages:{(workspace_id, corpus_id)!r}"
    messages = st.session_state.setdefault(history_key, [])

    # Thanh công cụ hỗ trợ
    col_help, col_clear = st.columns([4, 1])
    with col_clear:
        if messages:
            if st.button("🗑️ Xóa hội thoại"):
                messages.clear()
                st.rerun()

    prompt_to_send = None

    # Gợi ý câu hỏi mẫu khi chưa có tin nhắn
    if not messages:
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
    for item in messages:
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
    messages.append(user_item)
    _show_message(user_item)

    with st.spinner("🔍 Đang tìm kiếm bằng chứng qua Hybrid Search & Reranking..."):
        ok, data = ask_question(prompt_to_send, workspace_id, corpus_id)

    if ok:
        assistant_item = {
            "role": "assistant",
            "content": data.get("answer", ""),
            "citations": data.get("citations", []),
            "process_time_ms": data.get("process_time_ms"),
            "confidence_score": data.get("confidence_score"),
            "generation_mode": data.get("generation_mode"),
            "generation_error": data.get("generation_error"),
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

    messages.append(assistant_item)
    st.rerun()
