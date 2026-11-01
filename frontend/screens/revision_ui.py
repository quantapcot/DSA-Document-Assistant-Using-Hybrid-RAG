"""Revision Roadmap UI framework for DSA Document Assistant.
Tuần 4: Dựng khung giao diện (Skeleton).
Tuần 7: Tích hợp hoàn thiện với Backend API (/api/revision) và Revision Engine.
"""
import streamlit as st


def render_revision_ui() -> None:
    st.header("🗺️ Lộ trình Ôn tập DSA (Revision Roadmap)")
    st.caption("Khung giao diện sơ đồ cây chủ đề và lộ trình ôn tập cá nhân hóa từ tài liệu.")

    st.info(
        "ℹ️ **Trạng thái**: Khung giao diện (Skeleton) cho Tuần 4. "
        "Tính năng tổng hợp lộ trình ôn thi và tạo sơ đồ kiến thức sẽ được hoàn thiện vào **Tuần 7** "
        "(phối hợp giữa Team C - Revision Engine và Team A - API `/api/revision`)."
    )

    with st.expander("🎯 Thiết lập mục tiêu ôn tập", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            target_scope = st.selectbox(
                "Phạm vi kiến thức cần ôn tập",
                [
                    "Toàn bộ chương trình Cấu trúc dữ liệu & Giải thuật",
                    "Chuyên đề 1: Cấu trúc Cây & Cây cân bằng (AVL, Red-Black)",
                    "Chuyên đề 2: Lý thuyết Đồ thị & Các giải thuật duyệt đồ thị (BFS, DFS, Dijkstra)",
                    "Chuyên đề 3: Các thuật toán Sắp xếp & Tìm kiếm",
                    "Chuyên đề 4: Quy hoạch động & Đệ quy có nhớ",
                ],
            )
            exam_date = st.selectbox(
                "Thời gian còn lại đến kỳ thi",
                ["3 ngày tới (Ôn cấp tốc)", "1 tuần tới (Ôn trọng tâm)", "2-4 tuần (Ôn toàn diện)"],
                index=1,
            )
        with col2:
            weak_points = st.multiselect(
                "Các phần còn chưa vững",
                [
                    "Phép quay cây AVL (Single / Double rotation)",
                    "Độ phức tạp thuật toán (Big-O)",
                    "Duyệt đồ thị BFS / DFS",
                    "Thuật toán tìm đường đi ngắn nhất Dijkstra",
                    "Bảng băm và giải quyết đụng độ",
                ],
                default=["Phép quay cây AVL (Single / Double rotation)"],
            )

        btn_create = st.button("📌 Tạo lộ trình gợi ý mẫu (Mock)", use_container_width=True)

    if btn_create or True:
        st.subheader("📚 Sơ đồ cây kiến thức & Lộ trình đề xuất (Mock)")

        tab1, tab2 = st.tabs(["🌳 Cây chủ đề kiến thức", "📅 Kế hoạch từng bước"])

        with tab1:
            st.markdown(
                """
```text
Cấu trúc Dữ liệu & Giải thuật
├── 1. Danh sách tuyến tính (Linked List, Stack, Queue)
├── 2. Cây (Tree)
│   ├── Cây nhị phân (Binary Tree)
│   ├── Cây nhị phân tìm kiếm (BST)
│   └── Cây AVL (Tự cân bằng) 📍 [Trọng tâm cần ôn]
│       ├── Điều kiện cân bằng (|Balance Factor| <= 1)
│       ├── Các phép quay đơn: Left-Left (LL), Right-Right (RR)
│       └── Các phép quay kép: Left-Right (LR), Right-Left (RL)
├── 3. Đồ thị (Graph)
│   ├── Biểu diễn: Ma trận kề, Danh sách kề
│   └── Thuật toán: BFS, DFS, Dijkstra, Prim, Kruskal
└── 4. Kỹ thuật thiết kế giải thuật
    ├── Chia để trị (Divide and Conquer)
    └── Quy hoạch động (Dynamic Programming)
```
                """
            )

        with tab2:
            st.markdown(
                """
1. **Bước 1 — Nắm vững lý thuyết cơ bản (Ngày 1)**:
   - Đọc slide `Chuong4_Trees_DSA.pdf` từ trang 1 đến trang 15.
   - Nắm rõ công thức tính Balance Factor = Height(L) - Height(R).
2. **Bước 2 — Thực hành các phép biến đổi cây (Ngày 2)**:
   - Làm bài tập vẽ từng bước chèn các phần tử vào cây AVL và cân bằng lại.
   - Ôn tập 4 trường hợp mất cân bằng và các phép quay tương ứng.
3. **Bước 3 — Luyện câu hỏi trắc nghiệm & Red-Teaming (Ngày 3)**:
   - Chuyển sang tab **Luyện tập Trắc nghiệm** để kiểm tra mức độ ghi nhớ.
                """
            )
