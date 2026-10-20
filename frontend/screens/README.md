# Bốn màn hình — A2

`chat_ui.py`, `upload_ui.py`, `quiz_ui.py`, `revision_ui.py` được chuyển từ
`components/` khi còn rỗng. Đây là module màn hình để `app.py` gọi sau này,
không phải cấu hình tự động của Streamlit multipage.

Gọi backend qua `api_client.py`; không import backend trực tiếp. Hiện chưa có UI.
