"""A2: Upload & Document Management UI — Tuần 5.

Hai tab:
  📤 Nạp tài liệu  — Upload PDF/DOCX và chạy pipeline B1 → C1.
  🗂️ Quản lý kho   — Xem danh sách, xóa tài liệu đã nạp.
"""
from datetime import datetime

import streamlit as st

from api_client import upload_pdf, list_documents, delete_document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_upload_tab(workspace_id, corpus_id, cache_key) -> None:
    st.subheader("📤 Tải lên tài liệu mới")
    st.caption("Hỗ trợ định dạng PDF và DOCX — Tối đa 15 MB.")

    file = st.file_uploader(
        "Chọn file PDF hoặc DOCX tài liệu DSA",
        type=["pdf", "docx"],
        help="Slide bài giảng, giáo trình, đề cương môn Cấu trúc Dữ liệu & Giải thuật.",
        key="uploader_file",
    )
    if file is not None:
        ok, listing = _load_documents(workspace_id, corpus_id, cache_key)
        if ok and any(doc["filename"] == file.name for doc in listing.get("documents", [])):
            st.warning("Tài liệu trùng tên. Tải lên sẽ thay thế bản hiện tại trong kho này.")

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
            ok, data = upload_pdf(file.name, file.getvalue(), workspace_id, corpus_id)
        st.session_state[f"upload_result_{cache_key}"] = (ok, data)
        # Làm mới danh sách tài liệu ở tab quản lý
        st.session_state.pop(cache_key, None)
        if ok and data.get("replaced"):
            st.warning("Đã thay thế tài liệu cùng tên.")

    result = st.session_state.get(f"upload_result_{cache_key}")
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


def _load_documents(workspace_id, corpus_id, cache_key):
    if cache_key not in st.session_state:
        with st.spinner("Đang tải danh sách tài liệu…"):
            result = list_documents(workspace_id=workspace_id, corpus_id=corpus_id)
        if result[0]:
            st.session_state[cache_key] = result
        return result
    return st.session_state[cache_key]


def _render_management_tab(workspace_id, corpus_id, cache_key) -> None:
    st.subheader("🗂️ Kho tài liệu đã nạp")
    st.caption("Danh sách tài liệu đang được lập chỉ mục trong hệ thống RAG.")

    if st.button("🔄 Tải lại danh sách"):
        st.session_state.pop(cache_key, None)
    notice = st.session_state.pop(f"delete_notice_{cache_key}", None)
    if notice:
        ok, message = notice
        (st.success if ok else st.error)(message)
    ok, data = _load_documents(workspace_id, corpus_id, cache_key)

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
                size = doc.get("size_kb")
                uploaded = doc.get("uploaded_at")
                uploaded_date = datetime.fromisoformat(uploaded).strftime("%d/%m/%Y %H:%M UTC") if uploaded else "—"
                st.caption(f"Dung lượng: {f'{size:g} KB' if size is not None else '—'}")
                st.caption(f"Ngày tải: {uploaded_date}")
                st.markdown(
                    f"🔖 **{doc['chunk_count']}** chunks · "
                    f"📃 **{doc['page_count']}** trang · "
                    f"🏷️ `{doc.get('file_type', '?').upper()}`"
                )
            with col_del:
                # Dùng confirm bằng session_state để tránh xóa nhầm
                confirm_key = f"confirm_del_{cache_key}_{doc['document_id']}"
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
                        message = del_data.get("message", "Đã xóa tài liệu.") if del_ok else del_data.get("error", "Xóa thất bại.")
                        st.session_state[f"delete_notice_{cache_key}"] = (del_ok, message)
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

    col_ws, col_corpus = st.columns(2)
    workspace_id = col_ws.text_input("Workspace ID", value="default-workspace", key="mgmt_ws").strip()
    corpus_id = col_corpus.text_input("Corpus ID", value="user-uploads", key="mgmt_corpus").strip()
    if not workspace_id or not corpus_id:
        st.warning("Nhập Workspace ID và Corpus ID để tiếp tục.")
        return
    cache_key = f"doc_list_{(workspace_id, corpus_id)!r}"
    tab_upload, tab_manage = st.tabs(["📤 Nạp tài liệu", "🗂️ Quản lý kho"])

    with tab_upload:
        _render_upload_tab(workspace_id, corpus_id, cache_key)

    with tab_manage:
        _render_management_tab(workspace_id, corpus_id, cache_key)
