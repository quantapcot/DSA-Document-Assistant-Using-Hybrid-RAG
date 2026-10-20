# Hợp đồng dữ liệu dùng chung

`contracts.py` là nguồn duy nhất hiện có cho DocumentChunk, Citation, ChatRequest,
RAGResponse và GuardrailResult. Giữ nguyên file và import của code hiện tại.

A1 điều phối thay đổi; B1/C1/C2/D2 xác nhận contract liên quan. Không tạo class
trùng trong module riêng. Xem `docs/data_contracts.md` tại gốc repo để biết
những contract còn cần thống nhất.
