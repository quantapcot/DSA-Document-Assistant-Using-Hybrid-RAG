"""Quiz UI framework for DSA Document Assistant.
Tuần 4: Dựng khung giao diện (Skeleton).
Tuần 7: Tích hợp hoàn thiện với Backend API (/api/quiz) và Quiz Engine.
"""
import streamlit as st


def render_quiz_ui() -> None:
    st.header("📝 Luyện tập Trắc nghiệm DSA")
    st.caption("Khung giao diện sinh câu hỏi trắc nghiệm tự động từ slide bài giảng DSA.")

    st.info(
        "ℹ️ **Trạng thái**: Khung giao diện (Skeleton) cho Tuần 4. "
        "Tính năng tạo câu hỏi tự động bằng AI sẽ được kết nối vào **Tuần 7** "
        "(phối hợp giữa Team C - Quiz Engine và Team A - API `/api/quiz`)."
    )

    with st.expander("⚙️ Tùy chọn sinh đề trắc nghiệm", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            topic = st.selectbox(
                "Chủ đề kiến thức",
                [
                    "Chương 4: Cây nhị phân tìm kiếm & Cây AVL",
                    "Chương 5: Đồ thị & Các thuật toán duyệt đồ thị",
                    "Chương 6: Các thuật toán sắp xếp (QuickSort, MergeSort)",
                    "Chương 7: Bảng băm (Hash Table) & Xử lý xung đột",
                    "Chương 8: Quy hoạch động (Dynamic Programming)",
                ],
            )
            difficulty = st.select_slider(
                "Độ khó",
                options=["Nhận biết", "Thông hiểu", "Vận dụng", "Vận dụng cao"],
                value="Thông hiểu",
            )
        with col2:
            num_questions = st.number_input(
                "Số lượng câu hỏi",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
            )
            mode = st.radio(
                "Chế độ ôn luyện",
                ["Làm bài có bấm giờ", "Xem ngay đáp án & giải thích"],
                index=1,
            )

        btn_generate = st.button("🚀 Khởi tạo bài trắc nghiệm mẫu (Mock)", use_container_width=True)

    if btn_generate:
        st.session_state["show_sample_quiz"] = True

    if st.session_state.get("show_sample_quiz", True):
        st.subheader(f"📋 Bài ôn tập mẫu: {topic}")
        st.markdown(
            "**Câu 1 (Mẫu)**: Trong cây AVL, hệ số cân bằng (Balance Factor) của một nút được tính bằng công thức nào sau đây?"
        )

        user_choice = st.radio(
            "Chọn đáp án đúng:",
            [
                "A. Chiều cao của cây con trái trừ đi chiều cao của cây con phải",
                "B. Số lượng nút của cây con trái trừ đi số lượng nút của cây con phải",
                "C. Chiều cao cây con phải cộng với chiều cao cây con trái",
                "D. Bậc của nút con lớn nhất trừ bậc của nút con nhỏ nhất",
            ],
            index=None,
            key="sample_quiz_q1",
        )

        if st.button("Nộp câu trả lời"):
            if user_choice is None:
                st.warning("Vui lòng chọn một đáp án trước khi nộp.")
            elif user_choice.startswith("A."):
                st.success("✅ **Chính xác!** Hệ số cân bằng = Height(Left) - Height(Right).")
                st.markdown(
                    "> **Trích dẫn nguồn slide (MOCK):**\n"
                    "> - *Tài liệu:* `Chuong4_Trees_DSA.pdf` — Trang 12\n"
                    "> - *Mục:* `4.2 Cây AVL - Định nghĩa và Tính chất`\n"
                    "> - *Giải thích:* Giá trị hệ số cân bằng chỉ nhận các giá trị {-1, 0, 1}. Nếu chênh lệch vượt quá giá trị này, cây cần thực hiện phép quay."
                )
            else:
                st.error("❌ **Chưa chính xác.** Hãy kiểm tra lại định nghĩa hệ số cân bằng trong slide.")
                st.markdown("Đáp án đúng là **A**.")
