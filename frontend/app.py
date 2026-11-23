"""Streamlit frontend for one anonymous study workspace at a time."""
import time

import streamlit as st

from api_client import check_health, create_session, delete_session
from screens.chat_ui import render_chat_ui
from screens.quiz_ui import render_quiz_ui
from screens.revision_ui import render_revision_ui
from screens.upload_ui import render_upload_ui

st.set_page_config(page_title="DSA Study Workspace", page_icon="📚", layout="wide")

_SESSION = "active_study_session"
_MENU = {
    "💬 Trò chuyện & Hỏi đáp": render_chat_ui,
    "📤 Tải lên tài liệu": render_upload_ui,
    "📝 Luyện tập Trắc nghiệm": render_quiz_ui,
    "🗺️ Kế hoạch học tập": render_revision_ui,
}


def _replace_session(session=None, notice=None):
    st.session_state.clear()
    if session is not None:
        st.session_state[_SESSION] = session
    if notice:
        st.session_state["session_notice"] = notice


with st.sidebar:
    st.title("📚 DSA Study Workspace")
    online, _ = check_health()
    if not online:
        st.error("Chưa kết nối được Backend.")
        st.stop()

    session = st.session_state.get(_SESSION)
    if session is None or session["expires_at"] <= time.time():
        ok, data = create_session()
        if not ok:
            st.error(data["error"])
            st.stop()
        if session is not None:
            _replace_session(data, "Phiên trước đã hết hạn. Bạn đang ở một phiên mới.")
        else:
            st.session_state[_SESSION] = data
        session = data

    if notice := st.session_state.pop("session_notice", None):
        st.success(notice)
    selected = st.radio("Chức năng", list(_MENU), key="study_navigation")
    if st.button("Tạo phiên học mới"):
        ok, data = create_session()
        if ok:
            _replace_session(data, "Đã mở phiên học mới.")
            st.rerun()
        st.error(data["error"])

    with st.expander("Xóa phiên học hiện tại"):
        st.caption("Xóa tài liệu và kết thúc phiên này.")
        if st.button("Xác nhận xóa phiên"):
            ok, data = delete_session(session["token"])
            if ok:
                message = "Phiên đã kết thúc. Backend sẽ hoàn tất việc dọn dữ liệu." if data["pending"] else "Đã xóa phiên học."
                _replace_session(notice=message)
                st.rerun()
            st.error(data["error"])

_MENU[selected](session["workspace_id"], session["corpus_id"], token=session["token"])
