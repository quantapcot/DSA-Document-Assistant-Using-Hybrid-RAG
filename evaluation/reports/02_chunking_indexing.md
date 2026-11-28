# BÁO CÁO GIAI ĐOẠN 02 — PHÂN ĐOẠN KÉP (DUAL CHUNKING) VÀ LẬP CHỈ MỤC ĐÁNH GIÁ

> **Dự án:** DSA Document Assistant Using Hybrid RAG  
> **Giai đoạn:** Stage 02 (E1-01, E1-02 Adaptive Spec, E2-01..E2-03 & AC-03 Provenance Correction)  
> **Branch:** `feat/final-readiness` | **Checkpoint:** `0012a0cdfbda39c9b5117ee91e85f60e0bad23d7` | **Immutable `main`:** `b84f09450f213b03ad57ef8b388f482ee0f0eaa2`  
> **Trạng thái:** **ACCEPTED (100% Provenance & Coverage Partition Verified)**

---

## 1. Mục tiêu

Xây dựng hai kho ngữ liệu chuẩn đóng băng (frozen benchmark corpora) từ cùng một tập 54 tài liệu DSA gốc bằng hai chiến lược phân đoạn độc lập — **Profile A (`recursive_512_100`)** và **Profile B (`structure_adaptive`, `ADAPTIVE_CONFIG_V1`)** — kèm truy vết nguồn chính xác đến từng ký tự (`per-character provenance`), kiểm chứng tính tất định (determinism), và lập chỉ mục vào kho Qdrant đánh giá cô lập (`data/runtime/qdrant_eval/`) mà không sửa đổi mã nguồn hay dữ liệu production.

## 2. Phương pháp thực hiện

- Áp dụng kiến trúc **Option B (Evaluation-only)** nằm hoàn toàn trong `evaluation/` và `tests/unit/`.
- Thiết kế **Shared Benchmark Preprocessor** (`evaluation/corpus/shared_preprocessor.py`) dùng chung cho cả hai profile để loại trừ nhiễu tiền xử lý: chuẩn hóa Unicode NFC, bảo toàn thụt lề code/pseudocode, trích xuất đủ **17/17 bảng DOCX** theo thứ tự XML `doc.element.body`, sinh `document_id = sha256(normalized_rel_path)[:16]` (0 đụng độ), phân loại ngôn ngữ deterministic (`vi`/`en`/`mixed`).
- **Profile A (`recursive_512_100`):** Ánh xạ chính xác thuật toán production `chunk_text(512, 100)` + `is_valid_chunk_text(15)` với cơ chế theo dõi nguồn từng ký tự `_TrackedString` (tách bạch `is_overlap`, ghi nhận `recursive_separator_consumed`).
- **Profile B (`structure_adaptive`, `ADAPTIVE_CONFIG_V1`):** Phân loại 10 dạng nội dung cấu trúc (code/pseudocode, công thức, bảng, tiêu đề, danh sách, đoạn văn...), tách khối hỗn hợp, gắn `heading_prefix` / `table_header` / `code_signature` dạng `SyntheticSegment(source_backed=True)` tách biệt khỏi `source_spans` gốc, và ràng buộc cứng `<= 128` token theo tokenizer MiniLM.
- Lập chỉ mục vào bộ sưu tập cô lập `eval_dsa_chunks_v1` trong `data/runtime/qdrant_eval/` (tách biệt hoàn toàn với `data/runtime/qdrant_local/`) và chạy smoke test 8 tổ hợp `A1–A4, B1–B4`.

## 3. Dữ liệu đầu vào

- **54 tài liệu DSA gốc** trong `data/corpus/dataset_raw/` (`41` PDF + `13` DOCX, `80.294.690` bytes, `RAW_V1_SHA256 = cc44dfdf537710e23e4426dc71d0d28fab8fd09ca8ca7ef180a090979f8903f6`).
- **Biểu diễn chuẩn hóa dùng chung (`CanonicalDocument`):** `1.733` trang chuẩn hóa, `1.916.622` ký tự Unicode NFC, `canonical_fingerprint_sha256 = 712f3847a6370156258f18297eabb41a657b4ac109dbe6290b9b50d9a0c6bd1d`.

## 4. Nội dung triển khai

- `evaluation/corpus/shared_preprocessor.py`: Tiền xử lý chuẩn hóa dùng chung cho PDF (`get_text("dict")`) và DOCX (paragraphs + 17 XML tables).
- `evaluation/corpus/provenance.py`: Hợp đồng `SourceSpan`, `OverlapSegment`, `SyntheticSegment`, `FallbackInfo` và hàm kiểm định chặt `validate_chunk_provenance()`.
- `evaluation/chunking/profiles.py`, `recursive_profile.py`, `structure_adaptive.py`: Hai chiến lược phân đoạn và cấu hình bất biến.
- `evaluation/corpus/benchmark_builder.py`, `evaluation/scripts/build_benchmark_corpora.py`, `evaluation/scripts/smoke_eval_combos.py`: Trình dựng corpus, kiểm tra determinism 2 vòng, đối chiếu phân hoạch độ phủ không đếm trùng (`exact_match = True`), lập chỉ mục Qdrant và kiểm thử smoke 8 tổ hợp.
- **Sửa lỗi mục tiêu AC-03 (Provenance Correction):** Loại bỏ toàn bộ `1.468` lỗi coarse parent-span ở Profile A (`1.468 → 0`) và `17` lỗi lệch span ở Profile B (`17 → 0`), loại bỏ hoàn toàn `61.368` ký tự đếm trùng `dropped_but_covered`, khôi phục công thức ngắn (`formula`: `342 → 1.110`), và đưa `uncovered_other_meaningful_chars = 0` trên cả hai profile.

## 5. Kết quả

| Chỉ số chẩn đoán Corpus (Post-Correction Final) | Profile A (`recursive_512_100`) | Profile B (`structure_adaptive`) |
|---|---:|---:|
| Tổng số tài liệu / Đụng độ `document_id` | `54` / `0` | `54` / `0` |
| Tổng số chunks | **`6.366`** | **`13.390`** (`2,10×`) |
| Độ dài ký tự (Min / Median / Mean / P95 / Max) | `100 / 421 / 386,7 / 509 / 512` | `1 / 112 / 145,5 / 323 / 1.871` |
| Phân loại cấu trúc (Code / Formula / Table) | `0 / 0 / 0` *(không gán nhãn cấu trúc)* | `3.434 / 1.110 / 89` |
| Chunks vượt ngưỡng MiniLM `128` tokens (Truncation) | **`2.750` (`43,20%`)** (`114.961` tokens bị bỏ) | **`0` (`0,00%`)** (`0` tokens bị bỏ) |
| Cắt cụt cặp câu hỏi–chunk ở BGE Reranker (`512` tok) | `0,00%` (Max `363`) | `0,00%` (Max `190`) |
| Độ phủ nguồn hiệu dụng (`effective_source_coverage_pct`) | **`91,36%`** (`1.751.048` ký tự) | **`88,93%`** (`1.704.477` ký tự) |
| Dư thừa chồng lấn (`overlap_redundancy_pct`) | `20,33%` | **`1,06%`** |
| Lỗi truy vết provenance (`validate_chunk_provenance`) | **`0 / 6.366`** | **`0 / 13.390`** |
| Phân hoạch độ phủ khớp tuyệt đối (`== 1.916.622`) | **TRUE** (`0` unexplained) | **TRUE** (`0` unexplained) |
| Điểm chỉ mục trong Qdrant cô lập (`eval_dsa_chunks_v1`) | `6.366` points | `13.390` points (Tổng `19.756`) |
| `corpus_sha256` đóng băng | `e39811da17d99201e4ad26f54c4a53614a087e76e3e0cb6d031a733862af0649` | `81c568275754e35d7323b1e51fc247b885abc873eef4c5caa5c1ec4daa0b5d63` |

- **Lưu trữ:** JSONL Profile A `9,96 MiB`, Profile B `15,53 MiB`; dung lượng vector ước lượng lý thuyết (`float32`, `384d`) lần lượt `9,33 MiB` và `19,61 MiB`, tách biệt với dung lượng vật lý QdrantLocal SQLite dùng chung thực tế **`100,82 MiB`** (`19.756` points).
- Biểu đồ tổng hợp trực quan xem [Hình 01](../figures/consolidated/png/fig01_danh_gia_chunking.png) và bảng dữ liệu [table_01_chunking_summary.csv](../tables/table_01_chunking_summary.csv).

## 6. Kiểm thử và xác minh

- Kiểm tra determinism 2 vòng độc lập từ raw files: `canonical_fingerprint` và `corpus_sha256` trùng khớp 100%.
- `8/8` tổ hợp smoke test (`A1–A4`, `B1–B4`) trên GPU RTX 5060 đạt `PASSED`, `0` rò rỉ chéo profile, `0` OOM.
- Unit tests `tests/unit/test_adaptive_chunking.py` (`8/8 PASS`); hồi quy tại checkpoint Stage 02 đạt `294 passed, 0 failed`, `pip check` exit `0`.

## 7. Acceptance Criteria

- Toàn bộ tiêu chí Stage 02 (`AC-01` đến `AC-06`, bao gồm `AC-03` sau vòng Targeted Correction) đều đạt **PASS**.

## 8. Vấn đề và giới hạn

- `46,8%` chunk của Profile B ngắn dưới `100` ký tự (trung vị `112`) do chia nhỏ theo ranh giới cấu trúc và trần `128` token MiniLM, dẫn đến hiện tượng phân mảnh ngữ cảnh ở các câu hỏi đa ý (được kiểm chứng định lượng ở Stage 04).
- Trích xuất bảng từ PDF dùng `get_text("dict")` theo thứ tự đọc thay cho `page.find_tables()` vì `find_tables()` trên 41 PDF gốc tốn ~8 phút và nhận nhầm code listing đóng khung thành bảng; toàn bộ 17 bảng DOCX được trích xuất nguyên vẹn dạng Markdown pipe table.

## 9. Kết luận

Hai kho ngữ liệu đánh giá đóng băng (`6.366` Recursive chunks và `13.390` Adaptive chunks) cùng chỉ mục Qdrant cô lập (`19.756` points) đạt 100% tính tất định, 0 lỗi provenance và bảo toàn tuyệt đối dữ liệu/mã nguồn production.

## 10. Checkpoint và tài liệu liên quan

- **Checkpoint Git:** `0012a0cdfbda39c9b5117ee91e85f60e0bad23d7` (`feat(evaluation): build dual benchmark corpora and isolated retrieval indexes`).
- **Frozen Artifacts (giữ nguyên đường dẫn gốc để bảo toàn SHA-256):** `evaluation/artifacts/recursive_512_100/{chunks.jsonl,manifest.json}`, `evaluation/artifacts/structure_adaptive/{chunks.jsonl,manifest.json}`, `evaluation/artifacts/diagnostics.json`, `evaluation/artifacts/smoke_test_8_combos.json`.
- **Tài liệu liên quan:** [Báo cáo Stage 01](01_environment.md) | [Báo cáo Stage 03](03_dataset_ground_truth.md) | [Hình 01](../figures/consolidated/png/fig01_danh_gia_chunking.png).
