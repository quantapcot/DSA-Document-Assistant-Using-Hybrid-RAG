"""Streamlit Main Application: DSA Document Assistant.
Tuần 4: Dựng khung Streamlit UI với Sidebar Navigation và các màn hình khung (PoC).
"""
import streamlit as st

from api_client import BACKEND_URL, check_health
from screens.chat_ui import render_chat_ui
from screens.upload_ui import render_upload_ui
from screens.quiz_ui import render_quiz_ui
from screens.revision_ui import render_revision_ui

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="DSA Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# SIDEBAR NAVIGATION & SYSTEM MONITORING
# ==========================================
with st.sidebar:
    st.title("📚 DSA Assistant")
    st.caption("AI Agent Hỗ trợ Học tập & Ôn thi DSA (Track C)")
    st.divider()

    st.subheader("🧭 Điều hướng chức năng")
    menu_options = [
        "💬 Trò chuyện & Hỏi đáp",
        "📤 Tải lên tài liệu",
        "📝 Luyện tập Trắc nghiệm",
        "🗺️ Lộ trình ôn tập",
    ]
    selected_page = st.radio(
        "Chọn màn hình làm việc:",
        menu_options,
        index=0,
        label_visibility="collapsed",
    )

    st.divider()

    # Trạng thái kết nối FastAPI Backend
    st.subheader("🔌 Trạng thái Backend")
    online, info = check_health()
    if online:
        st.success(f"● FastAPI: Hoạt động ({info.get('status', 'healthy')})")
        st.caption(f"Phiên bản: {info.get('version', '1.0.0')} · Môi trường: {info.get('environment', 'dev')}")
    else:
        st.error("● FastAPI: Mất kết nối")
        st.caption(f"Lỗi: {info.get('error', 'Chưa khởi động')}")
        st.info("💡 Chạy lệnh sau tại thư mục `backend/`:\n```bash\nuvicorn app.main:app --reload\n```")

    st.caption(f"API Target: `{BACKEND_URL}`")
    if st.button("🔄 Kiểm tra kết nối lại", use_container_width=True):
        st.rerun()

    st.divider()
    st.caption("© 2026 Nhóm Track C — PoC Tuần 4")


# ==========================================
# MAIN CONTENT AREA
# ==========================================
if selected_page == "💬 Trò chuyện & Hỏi đáp":
    render_chat_ui()
elif selected_page == "📤 Tải lên tài liệu":
    render_upload_ui()
elif selected_page == "📝 Luyện tập Trắc nghiệm":
    render_quiz_ui()
elif selected_page == "🗺️ Lộ trình ôn tập":
    render_revision_ui()
