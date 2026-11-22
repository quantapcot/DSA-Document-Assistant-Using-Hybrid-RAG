"""Calendar, references, and session-local study progress."""
from datetime import datetime, timedelta, timezone

import streamlit as st

from api_client import create_revision


def _record_completion(progress_key, task_id, widget_key):
    progress = st.session_state.setdefault(progress_key, {})
    progress[task_id] = st.session_state[widget_key]


def _open_quiz(workspace, corpus, title):
    st.session_state[f"quiz_topic:{(workspace, corpus)!r}"] = title
    st.session_state["study_navigation"] = "📝 Luyện tập Trắc nghiệm"


def render_revision_ui(workspace_id="default-workspace", corpus_id="user-uploads"):
    st.header("Kế hoạch ôn tập DSA")
    scope_key = f"revision:{(workspace_id, corpus_id)!r}"
    all_topics = st.checkbox("Ôn tất cả tài liệu đã tải")
    mode = st.radio("Mốc thời gian", ["Số ngày còn lại", "Ngày thi"], horizontal=True)
    today = datetime.now(timezone(timedelta(hours=7))).date()
    with st.form("revision_generate"):
        topic = "" if all_topics else st.text_input("Chủ đề hoặc mục tiêu ôn tập")
        by_date = mode == "Ngày thi"
        exam = (st.date_input("Ngày thi", today + timedelta(days=7),
                              min_value=today + timedelta(days=1),
                              max_value=today + timedelta(days=90)) if by_date else None)
        days = None if by_date else st.number_input("Số ngày còn lại", 1, 90, 7)
        minutes = st.number_input("Phút học mỗi ngày", 15, 240, 45, step=5)
        level = st.selectbox("Trình độ hiện tại", ["Beginner", "Intermediate", "Advanced"], index=1)
        generate = st.form_submit_button("Tạo kế hoạch học")
    if generate:
        if not all_topics and not topic.strip():
            st.warning("Nhập chủ đề hoặc chọn ôn tất cả tài liệu.")
            return
        with st.spinner("Đang lập lịch ôn tập theo tài liệu…"):
            ok, data = create_revision(
                topic.strip(), workspace_id, corpus_id,
                exam_date=exam.isoformat() if exam else None, days_remaining=days,
                minutes_per_day=minutes, level=level, all_topics=all_topics,
            )
        version = st.session_state.get(scope_key, (0, None))[0] + 1
        st.session_state[scope_key] = (version, data if ok else {"error": data["error"]})
        st.session_state.pop(f"{scope_key}:progress", None)

    version, data = st.session_state.get(scope_key, (0, None))
    if data is None:
        return
    if data.get("error"):
        st.error(data["error"])
        return
    plan = data.get("plan")
    if not plan:
        st.warning(data["answer"])
        return

    st.subheader("Lịch học")
    st.write(data["answer"])
    st.caption(f"Thi ngày {plan['exam_date']} · {plan['minutes_per_day']} phút tối đa mỗi ngày · {plan['level']}")
    for warning in data.get("warnings", []):
        st.warning(warning)
    if data.get("uncovered_topics"):
        with st.expander("Các mục chưa xếp lịch"):
            for title in data["uncovered_topics"]:
                st.write(title)
    topics = {topic["id"]: topic["title"] for topic in plan["topics"]}
    sources = {source["label"]: source for source in data.get("citations", [])}
    progress_key = f"{scope_key}:progress"
    progress = st.session_state.get(progress_key, {})
    total = sum(len(day["tasks"]) for day in plan["days"])
    st.caption(f"Đã hoàn thành {sum(progress.values())}/{total} nhiệm vụ trong phiên này.")
    labels = {"learn": "Học", "review": "Ôn lại", "mock_test": "Kiểm tra"}
    for day in plan["days"]:
        duration = sum(task["minutes"] for task in day["tasks"])
        with st.expander(f"{day['date']} · {duration} phút", expanded=True):
            for i, task in enumerate(day["tasks"]):
                task_id = f"{day['date']}:{i}"
                widget_key = f"{scope_key}:{version}:{task_id}"
                title = topics[task["topic_id"]]
                covered = task.get("covered_topic_ids") or [task["topic_id"]]
                coverage_title = ", ".join(topics[item] for item in covered)
                st.checkbox(
                    f"{labels[task['kind']]} · {coverage_title} · {task['minutes']} phút",
                    value=progress.get(task_id, False), key=widget_key,
                    on_change=_record_completion, args=(progress_key, task_id, widget_key),
                )
                st.write(task["objective"])
                for point in task["knowledge"]:
                    st.markdown(f"- {point}")
                if task["prerequisites"]:
                    st.caption("Học trước: " + ", ".join(topics[item] for item in task["prerequisites"]))
                st.write(task["activity"])
                st.write(task["practice"])
                for label in task["sources"]:
                    source = sources.get(label)
                    if source:
                        with st.popover(f"[{label}] {source['filename']} · Trang {source['page']}"):
                            st.write(source["snippet"])
                st.button("Luyện quiz cho mục này", key=f"{widget_key}:quiz",
                          on_click=_open_quiz, args=(workspace_id, corpus_id, title))
