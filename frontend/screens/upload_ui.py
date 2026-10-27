"""Upload a real PDF to FastAPI without parsing or indexing it."""
import streamlit as st

from api_client import upload_pdf


def render_upload_ui() -> None:
    st.subheader("1. Upload PDF")
    file = st.file_uploader("Chọn một file PDF thật", type=["pdf"])

    if st.button("📤 Upload lên Backend", disabled=file is None):
        if file is not None:
            with st.spinner("Đang lưu PDF..."):
                ok, data = upload_pdf(file.name, file.getvalue())
            st.session_state["poc_upload_result"] = (ok, data)

    result = st.session_state.get("poc_upload_result")
    if result is None:
        return

    ok, data = result
    if ok:
        st.success(data.get("message", "Đã upload thành công."))
        st.write(f"**Tên file:** {data.get('filename')}")
        st.write(f"**Kích thước:** {data.get('size_kb')} KB")
        st.caption("Mới lưu file; chưa trích xuất text, tạo chunk, BM25 hay Qdrant.")
    else:
        st.error(data.get("error", "Upload thất bại."))