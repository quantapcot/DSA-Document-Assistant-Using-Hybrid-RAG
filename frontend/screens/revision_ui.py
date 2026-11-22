"""Study outline and flashcards from the scoped revision API."""
import streamlit as st

from api_client import create_revision
from screens.chat_ui import show_study_metadata


def render_revision_ui(workspace_id="default-workspace", corpus_id="user-uploads"):
    st.header("Ôn tập DSA")
    scope_key = f"revision:{(workspace_id, corpus_id)!r}"
    with st.form("revision_generate"):
        topic = st.text_input("Chủ đề hoặc mục tiêu ôn tập")
        generate = st.form_submit_button("Tạo nội dung ôn tập")
    if generate:
        if not topic.strip():
            st.warning("Nhập chủ đề cần ôn tập.")
            return
        with st.spinner("Đang tổng hợp tài liệu…"):
            ok, data = create_revision(topic.strip(), workspace_id, corpus_id)
        st.session_state[scope_key] = data if ok else {"error": data["error"]}

    data = st.session_state.get(scope_key)
    if data is None:
        return
    if data.get("error"):
        st.error(data["error"])
        return
    if data.get("generation_mode") == "abstained":
        st.warning(data["answer"])
        return

    show_study_metadata(data)
    outline, cards = st.tabs(["Dàn bài", "Flashcards"])
    with outline:
        for section in data.get("sections", []):
            st.subheader(section["title"])
            st.write(section["summary"])
            st.caption("Nguồn " + " ".join(f"[{i}]" for i in section["sources"]))
    with cards:
        for card in data.get("flashcards", []):
            with st.expander(card["question"]):
                st.write(card["answer"])
                st.caption("Nguồn " + " ".join(f"[{i}]" for i in card["sources"]))
