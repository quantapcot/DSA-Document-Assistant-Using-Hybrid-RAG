# Tìm kiếm và lập chỉ mục — C1

- `embeddings.py`: chuyển text/query thành vector.
- `indexing.py`: tiếp nhận DocumentChunk và xây/cập nhật các chỉ mục.
- `vector_store.py`: giao tiếp với VectorDB được chọn.
- `keyword_search.py`: tìm kiếm từ khóa BM25.
- `vector_search.py`: tìm kiếm theo vector.
- `hybrid_search.py`: kết hợp và xếp hạng kết quả từ hai nguồn.
- `reranker.py`: xếp hạng lại ứng viên bằng mô hình reranker.

Hợp đồng kết quả C1 → C2 cần được thống nhất trước khi tích hợp. Các file hiện
chỉ mô tả nhiệm vụ; không có thuật toán hoặc kết nối dịch vụ.
