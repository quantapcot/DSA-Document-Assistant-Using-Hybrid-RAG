"""A2: Upload & Document Management UI — Tuần 5.

Hai tab:
  📤 Nạp tài liệu  — Upload PDF/DOCX và chạy pipeline B1 → C1.
  🗂️ Quản lý kho   — Xem danh sách, xóa tài liệu đã nạp.
"""
import streamlit as st

from api_client import upload_pdf, list_documents, delete_document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_upload_tab() -> None:
    st.subheader("📤 Tải lên tài liệu mới")
    st.caption("Hỗ trợ định dạng PDF và DOCX — Tối đa 15 MB.")

    st.info(
        "⚡ **Pipeline Ingestion:** File tải lên sẽ tự động chạy qua: "
        "**B1 Preprocessing** (bóc tách PyMuPDF/python-docx, chuẩn hóa Unicode NFC, "
        "cắt chunk 512 ký tự) "
        "→ **C1 Indexing** (Embedding vector + Qdrant upsert + BM25 cache invalidation)."
    )

    file = st.file_uploader(
        "Chọn file PDF hoặc DOCX tài liệu DSA",
        type=["pdf", "docx"],
        help="Slide bài giảng, giáo trình, đề cương môn Cấu trúc Dữ liệu & Giải thuật.",
        key="uploader_file",
    )

    col1, _ = st.columns([1, 4])
    with col1:
        btn_upload = st.button(
            "📤 Nạp vào Hệ thống",
            disabled=file is None,
            use_container_width=True,
            key="btn_upload",
        )

    if btn_upload and file is not None:
        with st.spinner(
            f"Đang bóc tách, làm sạch và lập chỉ mục '{file.name}'… "
            "(có thể mất vài chục giây tùy kích thước file)"
        ):
            ok, data = upload_pdf(file.name, file.getvalue())
        st.session_state["upload_result"] = (ok, data)
        # Làm mới danh sách tài liệu ở tab quản lý
        st.session_state.pop("doc_list_cache", None)

    result = st.session_state.get("upload_result")
    if result is not None:
        ok, data = result
        st.divider()
        if ok:
            st.success(f"✅ {data.get('message', 'Đã nạp tài liệu thành công!')}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📄 Tên tệp", data.get("filename", "N/A"))
            c2.metric("💾 Kích thước", f"{data.get('size_kb', 0)} KB")
            c3.metric("🔖 Chunks tạo ra", data.get("total_chunks", 0))
            c4.metric("✅ Chunks đã lập chỉ mục", data.get("indexed_chunks", 0))
            st.info(
                f"🎉 **Chỉ mục sẵn sàng:** Đã lưu `{data.get('indexed_chunks', 0)}` chunks "
                f"vào kho `{data.get('workspace_id', 'default-workspace')}`. "
                "Chuyển sang tab **💬 Trò chuyện** để hỏi đáp về tài liệu này ngay!"
            )
        else:
            st.error(f"❌ Nạp tài liệu thất bại: {data.get('error', 'Lỗi không xác định')}")


def _render_management_tab() -> None:
    st.subheader("🗂️ Kho tài liệu đã nạp")
    st.caption("Danh sách tài liệu đang được lập chỉ mục trong hệ thống RAG.")

    col_refresh, col_ws, col_corp = st.columns([1, 2, 2])
    with col_ws:
        workspace_id = st.text_input(
            "Workspace ID", value="default-workspace", key="mgmt_ws"
        )
    with col_corp:
        corpus_id = st.text_input("Corpus ID", value="user-uploads", key="mgmt_corpus")
    with col_refresh:
        st.write("")  # padding
        if st.button("🔄 Tải lại danh sách", use_container_width=True):
            st.session_state.pop("doc_list_cache", None)
            st.rerun()

    # Cache danh sách trong session để tránh gọi API liên tục
    cache_key = f"doc_list_{workspace_id}_{corpus_id}"
    if cache_key not in st.session_state:
        with st.spinner("Đang tải danh sách tài liệu…"):
            ok, data = list_documents(workspace_id=workspace_id, corpus_id=corpus_id)
        st.session_state[cache_key] = (ok, data)
    else:
        ok, data = st.session_state[cache_key]

    if not ok:
        st.error(f"❌ Không thể tải danh sách: {data.get('error', 'Lỗi không xác định')}")
        return

    documents = data.get("documents", [])
    total = data.get("total", 0)

    if total == 0:
        st.info(
            "📭 Kho tài liệu đang trống. "
            "Hãy chuyển sang tab **📤 Nạp tài liệu** để tải lên tài liệu DSA đầu tiên!"
        )
        return

    st.success(f"📚 Tổng cộng **{total} tài liệu** trong kho `{workspace_id}/{corpus_id}`.")
    st.divider()

    # Thanh tìm kiếm nhanh
    search_query = st.text_input(
        "🔍 Tìm nhanh theo tên file", placeholder="Ví dụ: chuong_4_cay", key="mgmt_search"
    )
    if search_query:
        documents = [d for d in documents if search_query.lower() in d["filename"].lower()]
        if not documents:
            st.warning(f"Không tìm thấy tài liệu nào khớp với '{search_query}'.")
            return

    # Render bảng danh sách
    for doc in documents:
        with st.container(border=True):
            col_info, col_meta, col_del = st.columns([4, 3, 1])
            with col_info:
                file_icon = "📄" if doc.get("file_type") == "pdf" else "📝"
                st.markdown(f"**{file_icon} {doc['filename']}**")
                st.caption(f"Document ID: `{doc['document_id']}`")
            with col_meta:
                st.markdown(
                    f"🔖 **{doc['chunk_count']}** chunks · "
                    f"📃 **{doc['page_count']}** trang · "
                    f"🏷️ `{doc.get('file_type', '?').upper()}`"
                )
            with col_del:
                # Dùng confirm bằng session_state để tránh xóa nhầm
                confirm_key = f"confirm_del_{doc['document_id']}"
                if st.session_state.get(confirm_key):
                    if st.button(
                        "✅ Xác nhận xóa",
                        key=f"confirm_yes_{doc['document_id']}",
                        type="primary",
                        use_container_width=True,
                    ):
                        with st.spinner(f"Đang xóa '{doc['filename']}'…"):
                            del_ok, del_data = delete_document(
                                doc["document_id"],
                                workspace_id=workspace_id,
                                corpus_id=corpus_id,
                            )
                        st.session_state.pop(confirm_key, None)
                        st.session_state.pop(cache_key, None)  # làm mới cache
                        if del_ok:
                            st.success(f"✅ {del_data.get('message', 'Đã xóa thành công!')}")
                        else:
                            st.error(f"❌ {del_data.get('error', 'Xóa thất bại')}")
                        st.rerun()
                    if st.button(
                        "↩️ Hủy",
                        key=f"confirm_no_{doc['document_id']}",
                        use_container_width=True,
                    ):
                        st.session_state.pop(confirm_key, None)
                        st.rerun()
                else:
                    if st.button(
                        "🗑️ Xóa",
                        key=f"del_{doc['document_id']}",
                        use_container_width=True,
                    ):
                        st.session_state[confirm_key] = True
                        st.rerun()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_upload_ui() -> None:
    st.header("📤 Tải lên & Quản lý Tài liệu DSA")
    st.caption("Nạp slide, giáo trình môn Cấu trúc Dữ liệu & Giải thuật vào hệ thống RAG.")

    tab_upload, tab_manage = st.tabs(["📤 Nạp tài liệu", "🗂️ Quản lý kho"])

    with tab_upload:
        _render_upload_tab()

    with tab_manage:
        _render_management_tab()
