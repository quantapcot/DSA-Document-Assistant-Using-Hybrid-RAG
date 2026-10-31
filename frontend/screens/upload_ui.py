"""Upload a real PDF to FastAPI without parsing or indexing it (Tuần 4 PoC)."""
import streamlit as st

from api_client import upload_pdf


def render_upload_ui() -> None:
    st.header("📤 Tải lên & Quản lý Tài liệu PDF")
    st.caption("Quản lý nguồn tài liệu slide, giáo trình môn Cấu trúc Dữ liệu & Giải thuật.")

    st.info(
        "ℹ️ **Trạng thái PoC Tuần 4**: File tải lên sẽ được kiểm tra định dạng PDF hợp lệ "
        "và lưu trữ tại `data/runtime/uploads/`. Quá trình bóc tách text (PyMuPDF), "
        "chunking và lập chỉ mục Hybrid Search (BM25 + Qdrant) sẽ được thực hiện tại **Tuần 5**."
    )

    with st.container():
        file = st.file_uploader(
            "Chọn file PDF tài liệu môn DSA (Tối đa 10 MB)",
            type=["pdf"],
            help="Hệ thống chỉ chấp nhận định dạng PDF chuẩn có header %PDF-.",
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            btn_upload = st.button("📤 Tải lên Backend", disabled=file is None, use_container_width=True)

        if btn_upload and file is not None:
            with st.spinner(f"Đang gửi '{file.name}' lên máy chủ FastAPI..."):
                ok, data = upload_pdf(file.name, file.getvalue())
            st.session_state["poc_upload_result"] = (ok, data)

    result = st.session_state.get("poc_upload_result")
    if result is not None:
        ok, data = result
        st.divider()
        if ok:
            st.success(f"✅ {data.get('message', 'Đã tải lên tệp thành công.')}")
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Tên tệp gốc", data.get("filename", "N/A"))
            with col_info2:
                st.metric("Kích thước", f"{data.get('size_kb', 0)} KB")
            with col_info3:
                st.metric("Thời gian xử lý", f"{data.get('process_time_ms', 'N/A')} ms")
            st.caption("📌 Lưu ý: Tệp đã được lưu an toàn tại máy chủ; chưa qua bóc tách văn bản, tạo chunk, BM25 hay chỉ mục Qdrant (PoC).")
        else:
            st.error(f"❌ Tải lên thất bại: {data.get('error', 'Lỗi không xác định')}")
