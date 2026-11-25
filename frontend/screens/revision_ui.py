"""Topic-focused study roadmap for the current session."""
import streamlit as st

from api_client import create_roadmap


def render_revision_ui(workspace_id="default-workspace", corpus_id="user-uploads", *, token=None):
    st.header("Lộ trình ôn tập")
    scope_key = f"roadmap:{(workspace_id, corpus_id)!r}"
    with st.form(f"{scope_key}:form"):
        topic = st.text_input("Chủ đề muốn ôn tập", placeholder="Tree, Graph, Sorting, Dijkstra", max_chars=150)
        submitted = st.form_submit_button("Tạo lộ trình ôn tập", type="primary")
    if submitted:
        if not topic.strip():
            st.session_state.pop(scope_key, None)
            st.warning("Nhập chủ đề muốn ôn tập.")
            return
        with st.spinner("Đang tìm tài liệu phù hợp và sắp xếp lộ trình…"):
            ok, data = create_roadmap(topic.strip(), workspace_id, corpus_id, token=token)
        st.session_state[scope_key] = data if ok else {"error": data["error"]}

    data = st.session_state.get(scope_key)
    if data is None:
        return
    if data.get("error"):
        st.error(data["error"])
        return
    items = data.get("items", [])
    if not items:
        st.warning(data["answer"])
        return
    st.write(data["answer"])
    for warning in data.get("warnings", []):
        st.warning(warning)
    for index, item in enumerate(items, start=1):
        with st.container(border=True):
            st.subheader(f"Bước {index} — {item['title']}")
            st.write(f"Mục tiêu: {item['objective']}")
            st.markdown("**Kiến thức trọng tâm**")
            for point in item["key_knowledge"]:
                st.markdown(f"- {point}")
