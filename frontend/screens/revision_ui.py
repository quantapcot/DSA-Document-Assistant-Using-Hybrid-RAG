"""Study Roadmap with session-local progress and a Quiz handoff."""
import re

import streamlit as st

from api_client import create_roadmap


def _record_completion(progress_key, topic_id, widget_key):
    st.session_state.setdefault(progress_key, {})[topic_id] = st.session_state[widget_key]


def _open_quiz(workspace, corpus, title):
    topic = title.partition("·")[0].strip()
    if re.fullmatch(r"Tr\.\s*\d+(?:[–-]\d+)?", topic, re.IGNORECASE):
        topic = ""
    st.session_state[f"quiz_topic:{(workspace, corpus)!r}"] = topic
    st.session_state["study_navigation"] = "📝 Luyện tập Trắc nghiệm"


def render_revision_ui(workspace_id="default-workspace", corpus_id="user-uploads", *, token=None):
    st.header("Kế hoạch học tập")
    scope_key = f"roadmap:{(workspace_id, corpus_id)!r}"
    progress_key = f"{scope_key}:progress"
    if st.button("Tạo kế hoạch học tập", type="primary"):
        with st.spinner("Đang xây dựng lộ trình từ tài liệu trong phiên…"):
            ok, data = create_roadmap(workspace_id, corpus_id, token=token)
        version = st.session_state.get(scope_key, (0, None))[0] + 1
        st.session_state[scope_key] = (version, data if ok else {"error": data["error"]})
        if ok and data.get("items"):
            st.session_state.pop(progress_key, None)

    version, data = st.session_state.get(scope_key, (0, None))
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
    if data.get("uncovered_topics"):
        with st.expander("Chủ đề chưa đưa vào lộ trình"):
            for title in data["uncovered_topics"]:
                st.write(title)

    by_id = {item["id"]: item["title"] for item in items}
    citations = {source["label"]: source for source in data["citations"]}
    progress = st.session_state.get(progress_key, {})
    st.caption(f"Đã hoàn thành {sum(progress.values())}/{len(items)} chủ đề trong phiên này.")
    for item in items:
        widget_key = f"{scope_key}:{version}:{item['id']}"
        with st.container(border=True):
            st.checkbox(item["title"], value=progress.get(item["id"], False), key=widget_key,
                        on_change=_record_completion, args=(progress_key, item["id"], widget_key))
            if item["prerequisites"]:
                st.caption("Học trước " + ", ".join(by_id[identifier] for identifier in item["prerequisites"]))
            st.write(item["objective"])
            for point in item["knowledge"]:
                st.markdown(f"- {point}")
            st.write(item["activity"])
            for label in item["sources"]:
                source = citations[label]
                with st.popover(f"[{label}] {source['filename']} · Trang {source['page']}"):
                    st.write(source["snippet"])
            st.button("Luyện quiz cho mục này", key=f"{widget_key}:quiz",
                      on_click=_open_quiz, args=(workspace_id, corpus_id, item["title"]))
