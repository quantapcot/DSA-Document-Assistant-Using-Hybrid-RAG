"""A2: Upload & Document Management UI — Tuần 5.

Hai tab:
  📤 Nạp tài liệu  — Upload PDF/DOCX và chạy pipeline B1 → C1.
  🗂️ Quản lý kho   — Xem danh sách, xóa tài liệu đã nạp.
"""
from datetime import datetime

import streamlit as st

from api_client import upload_pdf, list_documents, delete_document

MAX_FILES_PER_UPLOAD = 10
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_upload_tab(workspace_id, corpus_id, cache_key, token) -> None:
    st.subheader("📤 Tải lên tài liệu mới")
    st.caption("Chọn tối đa 10 file PDF hoặc DOCX mỗi lượt, tối đa 15 MB mỗi file.")

    files = st.file_uploader(
        "Chọn file PDF hoặc DOCX tài liệu DSA",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Slide bài giảng, giáo trình, đề cương môn Cấu trúc Dữ liệu & Giải thuật.",
        key="uploader_file",
    )
    if files:
        st.write(f"Đã chọn {len(files)} file:")
        for file in files:
            st.text(f"{file.name} · {file.size / (1024 * 1024):.2f} MB")
        if any(file.size > MAX_UPLOAD_BYTES for file in files):
            st.warning("File vượt 15 MB sẽ được bỏ qua.")
        ok, listing = _load_documents(workspace_id, corpus_id, cache_key, token)
        names = {file.name for file in files}
        existing = {doc["filename"] for doc in listing.get("documents", [])} if ok else set()
        if duplicates := sorted(names & existing):
            st.warning("Sẽ thay thế tài liệu trùng tên trong kho này: " + ", ".join(duplicates))
        if len(names) < len(files):
            st.warning("File trùng tên trong lượt chọn sẽ được xử lý theo thứ tự; bản tải thành công cuối cùng sẽ được giữ.")

    too_many = len(files) > MAX_FILES_PER_UPLOAD
    if too_many:
        st.warning(f"Chỉ được chọn tối đa {MAX_FILES_PER_UPLOAD} file mỗi lượt. Hãy bỏ bớt file.")
    upload_clicked = st.button(
        "📤 Nạp vào Hệ thống",
        disabled=not files or too_many,
        key="btn_upload",
    )

    result_key = f"upload_batch_result_{cache_key}"
    if upload_clicked and files and not too_many:
        progress = st.progress(0, text=f"Đã xử lý 0/{len(files)} file")
        results = []
        for index, file in enumerate(files, start=1):
            if file.size > MAX_UPLOAD_BYTES:
                status, data = "skipped", {"error": "Vượt giới hạn 15 MB."}
            else:
                try:
                    ok, data = upload_pdf(file.name, file.getvalue(), workspace_id, corpus_id, token=token)
                    status = "success" if ok else "failed"
                except OSError:
                    status, data = "failed", {"error": "Không đọc được file hoặc kết nối bị gián đoạn."}
            results.append((file.name, status, data))
            progress.progress(index / len(files), text=f"Đã xử lý {index}/{len(files)} file")
        st.session_state[result_key] = results
        # The management tab reloads this scope later in the same run.
        st.session_state.pop(cache_key, None)

    if results := st.session_state.get(result_key):
        _show_upload_results(results)


def _show_upload_results(results):
    succeeded = [data for _, status, data in results if status == "success"]
    failed = sum(status == "failed" for _, status, _ in results)
    skipped = sum(status == "skipped" for _, status, _ in results)
    chunks = sum(data.get("indexed_chunks", 0) for data in succeeded)
    st.subheader("Kết quả lượt tải gần nhất")
    st.write(
        f"Thành công: {len(succeeded)} · Thất bại: {failed} · Bỏ qua: {skipped} "
        f"· Chunks đã lập chỉ mục: {chunks}"
    )
    for name, status, data in results:
        if status == "success":
            replacement = " · Đã thay thế bản trước" if data.get("replaced") else ""
            st.success(f"{name} · Thành công · {data.get('indexed_chunks', 0)} chunks{replacement}")
        elif status == "failed":
            st.error(f"{name} · Thất bại · {data.get('error', 'Lỗi không xác định')}")
        else:
            st.warning(f"{name} · Bỏ qua · {data['error']}")


def _load_documents(workspace_id, corpus_id, cache_key, token):
    if cache_key not in st.session_state:
        with st.spinner("Đang tải danh sách tài liệu…"):
            result = list_documents(workspace_id=workspace_id, corpus_id=corpus_id, token=token)
        if result[0]:
            st.session_state[cache_key] = result
        return result
    return st.session_state[cache_key]


def _render_management_tab(workspace_id, corpus_id, cache_key, token) -> None:
    st.subheader("🗂️ Kho tài liệu đã nạp")
    st.caption("Danh sách tài liệu đang được lập chỉ mục trong hệ thống RAG.")

    if st.button("🔄 Tải lại danh sách"):
        st.session_state.pop(cache_key, None)
    notice = st.session_state.pop(f"delete_notice_{cache_key}", None)
    if notice:
        ok, message = notice
        (st.success if ok else st.error)(message)
    ok, data = _load_documents(workspace_id, corpus_id, cache_key, token)

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

    st.success(f"📚 **{total} tài liệu** trong phiên học hiện tại.")
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
                                token=token,
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

def render_upload_ui(workspace_id="default-workspace", corpus_id="user-uploads", *, token=None) -> None:
    st.header("📤 Tải lên & Quản lý Tài liệu DSA")
    st.caption("Nạp slide, giáo trình môn Cấu trúc Dữ liệu & Giải thuật vào hệ thống RAG.")

    cache_key = f"doc_list_{(workspace_id, corpus_id)!r}"
    tab_upload, tab_manage = st.tabs(["📤 Nạp tài liệu", "🗂️ Quản lý kho"])

    with tab_upload:
        _render_upload_tab(workspace_id, corpus_id, cache_key, token)

    with tab_manage:
        _render_management_tab(workspace_id, corpus_id, cache_key, token)
