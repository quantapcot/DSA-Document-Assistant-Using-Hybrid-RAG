"""Interactive study quiz backed by the scoped quiz API."""
import streamlit as st

from api_client import create_quiz
from screens.chat_ui import show_study_metadata


def render_quiz_ui(workspace_id="default-workspace", corpus_id="user-uploads", *, token=None):
    st.header("Luyện tập trắc nghiệm DSA")
    scope_key = f"quiz:{(workspace_id, corpus_id)!r}"
    prefill_key = f"quiz_topic:{(workspace_id, corpus_id)!r}"
    with st.form("quiz_generate"):
        topic = st.text_input("Chủ đề", key=prefill_key)
        count = st.number_input("Số câu hỏi (tối đa 5)", min_value=1, max_value=5, value=5)
        generate = st.form_submit_button("Tạo bài trắc nghiệm")
    if generate:
        if not topic.strip():
            st.warning("Nhập chủ đề cần luyện tập.")
            return
        if any(separator in topic for separator in ",;，；"):
            st.warning("Mỗi bài Quiz chỉ nhận một chủ đề. Bỏ dấu phẩy hoặc chấm phẩy.")
            return
        with st.spinner("Đang tạo câu hỏi từ tài liệu…"):
            ok, data = create_quiz(topic.strip(), count, workspace_id, corpus_id, token=token)
        version = st.session_state.get(scope_key, (0, None))[0] + 1
        st.session_state[scope_key] = (version, data if ok else {"error": data["error"]})

    version, data = st.session_state.get(scope_key, (0, None))
    if data is None:
        return
    if data.get("error"):
        st.error(data["error"])
        return
    questions = data.get("questions", [])
    if not questions:
        st.warning(data.get("answer") or "Tài liệu chưa đủ nội dung để tạo câu hỏi.")
        return
    if data.get("generated_count", len(questions)) < data.get("requested_count", len(questions)):
        st.warning(data["answer"])

    attempt_key = f"{scope_key}:{version}"
    with st.form(attempt_key):
        answers = [
            st.radio(
                f"{i + 1}. {question['question']}", range(4), index=None,
                format_func=lambda choice, options=question["options"]: options[choice],
                key=f"{attempt_key}:{i}",
            )
            for i, question in enumerate(questions)
        ]
        if st.form_submit_button("Chấm bài"):
            if None in answers:
                st.warning("Chọn đáp án cho tất cả câu hỏi.")
            else:
                st.session_state[f"{attempt_key}:submitted"] = answers
    submitted = st.session_state.get(f"{attempt_key}:submitted")
    if submitted is not None:
        score = sum(choice == question["correct_index"] for choice, question in zip(submitted, questions))
        st.success(f"Kết quả · {score}/{len(questions)} câu đúng")
        review_topics = dict.fromkeys(
            question["topic"] for choice, question in zip(submitted, questions)
            if choice != question["correct_index"]
        )
        if review_topics:
            st.write("Chủ đề cần ôn thêm: " + ", ".join(review_topics))
        for i, question in enumerate(questions, 1):
            correct = question["options"][question["correct_index"]]
            sources = " ".join(f"[{label}]" for label in question["sources"])
            result = "Đúng" if submitted[i - 1] == question["correct_index"] else "Sai"
            st.markdown(f"**Câu {i} · {result} · Đáp án: {correct}**")
            st.write(question["explanation"], sources)
            for evidence in question.get("evidence", []):
                with st.expander(f"[{evidence['label']}] {evidence['filename']} · Trang {evidence['page']}"):
                    st.write(evidence["snippet"])
        if not any(question.get("evidence") for question in questions):
            show_study_metadata(data)
