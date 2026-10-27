
"""Chat-like UI with a bottom input and one backend-supported mock AVL question."""
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
            with st.expander("Nguồn minh họa (MOCK)", expanded=False):
                st.caption(
                    "Các trích dẫn này là dữ liệu giả lập, "
                    "không lấy từ PDF đã tải lên."
                )

                for index, citation in enumerate(item["citations"], start=1):
                    st.markdown(
                        f"**[{index}] {citation.get('filename', 'Unknown')} — "
                        f"Trang {citation.get('page', '?')}**"
                    )

                    if citation.get("section"):
                        st.caption(citation["section"])

                    st.markdown(f"> {citation.get('snippet', '')}")

        if item.get("process_time_ms"):
            st.caption(
                f"API: {item['process_time_ms']} ms · Phản hồi giả lập"
            )


def render_chat_ui() -> None:
    if "poc_messages" not in st.session_state:
        st.session_state["poc_messages"] = []

    for item in st.session_state["poc_messages"]:
        _show_message(item)

    # Chat input nằm ở cuối phần nội dung chính.
    question = st.chat_input(
        "Thử hỏi: Cây AVL là gì và độ phức tạp tìm kiếm là bao nhiêu?"
    )

    if not question:
        return

    user_item = {
        "role": "user",
        "content": question,
    }

    st.session_state["poc_messages"].append(user_item)
    _show_message(user_item)

    if " ".join(question.split()) != SAMPLE_PROMPT:
        assistant_item = {
            "role": "assistant",
            "content": (
                f"Bản PoC hiện chỉ hỗ trợ câu hỏi mẫu: "
                f"**{SAMPLE_PROMPT}**"
            ),
            "error": True,
        }
    else:
        with st.spinner("Đang hỏi FastAPI..."):
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
                    "Không gọi được FastAPI: "
                    f"{data.get('error', 'Unknown error')}"
                ),
                "error": True,
            }

    st.session_state["poc_messages"].append(assistant_item)
    _show_message(assistant_item)
