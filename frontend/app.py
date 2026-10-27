"""Streamlit PoC: PDF upload + one fixed AVL chat question."""
import streamlit as st

from api_client import BACKEND_URL, check_health
from screens.chat_ui import render_chat_ui
from screens.upload_ui import render_upload_ui

st.set_page_config(page_title="DSA Assistant - PoC", page_icon="📚")

with st.sidebar:
    st.subheader("Trạng thái Backend")
    online, info = check_health()
    if online:
        st.success(f"FastAPI đang chạy: {info.get('status', 'OK')}")
    else:
        st.error("Không kết nối được FastAPI")
        st.caption(info.get("error", "Hãy chạy backend tại port 8000."))
    st.caption(f"API: {BACKEND_URL}")
    if st.button("Kiểm tra lại"):
        st.rerun()

st.title("📚 DSA Document Assistant - PoC")
st.info(
    "Đây là bản demo kết nối Streamlit ↔ FastAPI. "
    "PDF chỉ được lưu, chưa phân tích hoặc lập chỉ mục. "
    "Câu trả lời và trích dẫn AVL là dữ liệu mẫu cố định."
)

render_upload_ui()
st.divider()
render_chat_ui()