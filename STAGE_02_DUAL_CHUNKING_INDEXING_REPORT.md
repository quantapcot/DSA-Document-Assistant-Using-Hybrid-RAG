# STAGE 02 – DUAL CHUNKING & BENCHMARK INDEXING REPORT

> **Dự án:** DSA Document Assistant Using Hybrid RAG
> **Giai đoạn:** `Stage 02` – Dual Chunking & Benchmark Indexing *(gộp E1-02, E2-01, E2-02, E2-03; đã bao gồm AC-03 Targeted Correction & Final Verification)*
> **Workspace:** `D:\DSA-Document-Assistant-Using-Hybrid-RAG`
> **Branch:** `feat/final-readiness`
> **HEAD (bất biến):** `ff2187eb996b380d817d30ded85db42880273d6c`
> **Local `main` (bất biến):** `b84f09450f213b03ad57ef8b388f482ee0f0eaa2`
> **Ngày hoàn tất:** 2026-11-27

---

## 1. Executive Summary

Stage 02 đã hoàn thành đầy đủ theo các quyết định phê duyệt (**Option B** `Evaluation-only`, **`ADAPTIVE_CONFIG_V1`**, **Shared Benchmark Preprocessor**, isolated **`data/runtime/qdrant_eval/`**, giữ nguyên 100% production pipeline), và đã hoàn tất vòng **Targeted AC-03 Correction & Final Verification** xử lý triệt để các lỗi provenance và lọc nội dung phát hiện ở Final Acceptance Check:

- **Shared Preprocessing (bất biến giữa pre/post-correction):** Cùng **54 tài liệu gốc** (`41` PDF + `13` DOCX, `80,294,690` bytes, `1,733` canonical pages, `1,916,622` canonical chars, **17/17 bảng DOCX** theo XML order, bảo toàn code indentation, `0` `document_id` collisions, bộ phân loại ngôn ngữ deterministic không cài `langdetect`), `canonical_fingerprint_sha256 = 712f3847a6370156258f18297eabb41a657b4ac109dbe6290b9b50d9a0c6bd1d` không đổi 100%.
- **Targeted AC-03 Provenance & Filtering Correction (post-correction):**
  - **Profile A (`recursive_512_100`):** Thay fallback coarse parent-span ở `_locate_subsegment()` bằng theo dõi **per-character origin** `(canonical_idx, is_overlap)` qua toàn bộ recursive split/re-split; đạt **`1,733/1,733` exact parity** với production `chunk_text(512, 100)` + `is_valid_chunk_text(15)`, **`6,366` chunks**, **`0 / 6,366` provenance errors** *(giảm từ `1,468`)*, **`0` `dropped_but_covered_chars`** *(giảm từ `61,368`)*, **`0` `uncovered_other_meaningful_chars`**, `coverage_partition_exact_match = true`.
  - **Profile B (`structure_adaptive`):** Áp dụng **per-character origin** `(block_id, canonical_idx) | None` cho mọi phép ghép/tách non-table và overlap, lưu tọa độ source-backed trên `SyntheticSegment` (`heading_prefix`, `table_header`, `code_signature`) mà không trộn vào `original source_spans`, mở rộng bảo toàn công thức/gán ngắn (`formula` tăng `342 → 1,110`) và gắn ký hiệu ngắn sau code/formula khi phù hợp; đạt **`13,390` chunks**, **`0 / 13,390` provenance errors** *(giảm từ `17`)*, **`0` chunks (`0.00%`) vượt `128` MiniLM tokens** (`max = 128`), **`0` `uncovered_other_meaningful_chars`** *(giảm từ `2,371`)*, `coverage_partition_exact_match = true`.
  - **Benchmark Corpus Revision Note:** Do sửa chính xác `source_spans` (thành phần `primary_span` trong `chunk_id`) và khôi phục short formulas ở Profile B, hai corpus JSONL là bản **Stage 02 Post-Correction Revision** (`corpus_sha256` mới cho cả Profile A & B) được dựng sạch hoàn toàn trước khi tạo Ground Truth ở Stage 03.
- **Isolated Benchmark Indexing & 8-Combo Smoke Readiness:** Rebuild sạch `data/runtime/qdrant_eval/` (`eval_dsa_chunks_v1`, tổng **`19,756` points = `6,366` Scope A + `13,390` Scope B**, `0` stale points, `0` cross-profile overlap, `check_model_exhaustive = PASSED`). Chạy thành công **8/8 tổ hợp (`A1–A4` và `B1–B4`)** trên **RTX 5060 GPU** (`cuda:0`, `0` OOM, `0` CPU fallback).
- **Zero Regression & Production Safety:** Full `pytest -q` đạt **`294 passed, 0 failed, 0 skipped, 1 warning (fitz)`** (`exit 0`), `pip check` exit `0`, toàn bộ checksums của raw dataset, production `chunks_all.jsonl` và production `qdrant_local` (`0` points) giữ nguyên 100%.

---

## 2. Implemented Files and Commands

### 2.1. Danh sách file Stage 02 (100% nằm trong `evaluation/` và `tests/unit/`, 0 file production bị sửa)

| Nhóm | File | Vai trò |
|---|---|---|
| Shared Preprocessing & Provenance | [evaluation/corpus/__init__.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/corpus/__init__.py) | Khởi tạo package corpus |
| | [evaluation/corpus/provenance.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/corpus/provenance.py) | Hợp đồng `SourceSpan`, `OverlapSegment`, `SyntheticSegment` *(mở rộng `source_backed`, `page`, `char_start`, `char_end`)*, `FallbackInfo`, `validate_chunk_provenance()` kiểm tra chặt bounds + text correspondence cho spans/overlaps/source-backed synthetic segments, `benchmark_manifest_v1` |
| | [evaluation/corpus/shared_preprocessor.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/corpus/shared_preprocessor.py) | Shared PDF (`get_text("dict")`) & DOCX (paragraphs + 17 XML tables) parser/cleaner adapter, code-indent guard, deterministic language detection (`vi`/`en`/`mixed`/hint fallback) *(giữ nguyên trong AC-03 correction)* |
| Dual Chunkers | [evaluation/chunking/__init__.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/chunking/__init__.py) | Khởi tạo package chunking |
| | [evaluation/chunking/profiles.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/chunking/profiles.py) | Cấu hình chuẩn `RECURSIVE_CONFIG` & `ADAPTIVE_CONFIG_V1` + `chunking_config_hash` |
| | [evaluation/chunking/recursive_profile.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/chunking/recursive_profile.py) | Profile A adapter gọi đúng `chunker.chunk_text(512, 100)` + `is_valid_chunk_text(15)` với **per-character origin mirror** (`_TrackedString`, tách bạch `is_overlap`, ghi nhận `recursive_separator_consumed`, loại bỏ hoàn toàn coarse parent-span fallback) |
| | [evaluation/chunking/structure_adaptive.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/chunking/structure_adaptive.py) | Profile B `ADAPTIVE_CONFIG_V1`: 10 content types, mixed-content split, **per-character origin tracking** cho multi-span/heading/overlap/window-split, source-backed `heading_prefix`/`table_header`/`code_signature`, bảo toàn compact formulas & pure-symbol attach, hard `<= 128` tokenizer verify |
| Builder, Indexing & Smoke | [evaluation/corpus/benchmark_builder.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/corpus/benchmark_builder.py) | Build 2 corpus, kiểm tra determinism 2 lần từ raw files, đối chiếu phân hoạch coverage không double-count (`_merge_intervals`, `_interval_difference`, `_interval_intersection`), nạp `data/runtime/qdrant_eval/` và kiểm chứng isolation |
| | [evaluation/scripts/build_benchmark_corpora.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/scripts/build_benchmark_corpora.py) | CLI build + verify determinism + index |
| | [evaluation/scripts/smoke_eval_combos.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/scripts/smoke_eval_combos.py) | Smoke test 8 tổ hợp `A1–A4` & `B1–B4` trên `data/runtime/qdrant_eval/` |
| Evaluation Unit Tests | [tests/unit/test_adaptive_chunking.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/tests/unit/test_adaptive_chunking.py) | `8` unit/integration tests (tăng từ `6`): bổ sung regression test cho nested overlapped recursive re-split parity & merged non-contiguous headings, source-backed repeated context, short formulas, pure-symbol attach, và exact coverage partition |

### 2.2. Generated Benchmark Artifacts (Post-Correction Revision)

- **Corpus A:** [evaluation/artifacts/recursive_512_100/chunks.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/recursive_512_100/chunks.jsonl) & [manifest.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/recursive_512_100/manifest.json)
  - Pre-correction `corpus_sha256`: `5160e8db57a4bc8fcb3f7b65b642c99ed7f318db205c498121971b4dd9143129`
  - Post-correction `corpus_sha256`: `e39811da17d99201e4ad26f54c4a53614a087e76e3e0cb6d031a733862af0649`
- **Corpus B:** [evaluation/artifacts/structure_adaptive/chunks.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/structure_adaptive/chunks.jsonl) & [manifest.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/structure_adaptive/manifest.json)
  - Pre-correction `corpus_sha256`: `d92a256ebe24ed67da43015d87ab4176dd4bfe34c5440e50ac95cd1e86d48412`
  - Post-correction `corpus_sha256`: `81c568275754e35d7323b1e51fc247b885abc873eef4c5caa5c1ec4daa0b5d63`
- **Diagnostics & Smoke Outputs:** [evaluation/artifacts/diagnostics.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/diagnostics.json), [smoke_test_8_combos.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/smoke_test_8_combos.json)
- **Isolated Qdrant Storage (gitignored):** `D:\DSA-Document-Assistant-Using-Hybrid-RAG\data\runtime\qdrant_eval\collection\eval_dsa_chunks_v1\storage.sqlite`

### 2.3. Reproducibility Commands (đã chạy thực tế sau AC-03 Correction)

```powershell
Set-Location 'D:\DSA-Document-Assistant-Using-Hybrid-RAG'
.\.venv\Scripts\python.exe -m pytest tests/unit/test_adaptive_chunking.py tests/unit/test_reranker.py tests/unit/test_embeddings.py tests/unit/test_keyword_search.py tests/unit/test_hybrid_search.py tests/unit/test_optimizations.py -v
.\.venv\Scripts\python.exe evaluation/scripts/build_benchmark_corpora.py --verify-determinism --index
.\.venv\Scripts\python.exe evaluation/scripts/smoke_eval_combos.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

---

## 3. Shared Preprocessing Results

- **Đầu vào:** Cùng `54` tài liệu gốc trong `data/corpus/dataset_raw/` (`80,294,690` bytes, `RAW_V1_SHA256 = cc44dfdf537710e23e4426dc71d0d28fab8fd09ca8ca7ef180a090979f8903f6`).
- **Canonical Representation:** `1,733` canonical pages, tổng `1,916,622` ký tự Unicode NFC, `canonical_fingerprint_sha256 = 712f3847a6370156258f18297eabb41a657b4ac109dbe6290b9b50d9a0c6bd1d` *(bất biến hoàn toàn trước và sau AC-03 correction)*.
- **Những sửa lỗi áp dụng đồng đều cho CẢ Profile A và Profile B:**
  1. `document_id = sha256(normalized_rel_path)[:16]` → **`54` `document_id` duy nhất (`0` collisions)**.
  2. Trích xuất toàn bộ **`17/17` bảng DOCX** theo đúng thứ tự XML `doc.element.body` dưới dạng Markdown pipe table (`0` trùng lặp).
  3. Bảo toàn indentation của dòng code/pseudocode trước khi nén khoảng trắng ngang (`clean_canonical_block(..., preserve_indent=True)`).
  4. Bộ phân loại ngôn ngữ deterministic quy tắc (`vi` / `en` / `mixed` / folder hint fallback; `.venv` không có `langdetect`).
  5. Giữ nguyên tất cả source occurrences hợp lệ có vị trí khác nhau (`dropped_duplicate_records_in_preprocessing = 0`).
- **Ghi chú thiết kế thực tế về PDF tables:** `pymupdf` `page.find_tables()` trên 41 PDF gốc mất ~8 phút/lượt (~69.5s trên riêng `001_train_file_Vie.pdf` với `lines_strict`) và nhận nhầm code listing đóng khung (vd `def merge_sort`) thành bảng; PDF sử dụng `get_text("dict")` nhanh theo reading order (~3.3s toàn tập) còn toàn bộ 17 bảng DOCX được trích xuất bằng `doc.tables` XML order (`table = 89` chunks ở Profile B).

---

## 4. AC-03 Correction: Root-Cause Fix & Exact Source Coverage Reconciliation

### 4.1. Pre-Correction vs Post-Correction Comparison

| Chỉ số kiểm chứng AC-03 | Profile A (Pre) | Profile A (Post) | Profile B (Pre) | Profile B (Post) |
|---|---:|---:|---:|---:|
| **Canonical total chars** | `1,916,622` | `1,916,622` | `1,916,622` | `1,916,622` |
| **Chunk count** | `6,366` | **`6,366`** | `13,100` | **`13,390`** |
| **Production Recursive parity (`1,733` pages)** | `1,733/1,733` | **`1,733/1,733`** | — | — |
| **Strict provenance mismatches (`validate_chunk_provenance`)** | `1,468` (`23.06%`) | **`0` (`0.00%`)** | `17` (`0.13%`) | **`0` (`0.00%`)** |
| **Chunks `> 128` MiniLM tokens** | `2,750` (`43.20%`) | `2,750` (`43.20%`) | `0` (`0.00%`) | **`0` (`0.00%`)** |
| **Original `source_spans` union (`canonical_retained_chars_union`)** | `1,800,865` (`93.96%`) | **`1,751,048` (`91.36%`)** | `1,816,907` (`94.80%`) | **`1,701,085` (`88.75%`)** |
| **Repeated context only (`source_backed synthetic_segments \ covered`)** | `0` | **`0`** | *(chưa tách tọa độ)* | **`3,392`** *(tổng `13,034`)* |
| **Effective source coverage (`covered + repeated_context_only`)** | `93.96%` | **`91.36%`** | — | **`88.93%`** (`1,704,477`) |
| **Dropped & truly uncovered (`dropped \ kept`)** | `66,180` | **`65,564`** | `37,494` | **`37,026`** |
| **Dropped but covered (`dropped ∩ kept`, false double-count)** | `61,368` | **`0`** | `290` | **`0`** |
| **Uncovered whitespace / separators** | `49,577` | **`100,010`** | `57,539` | **`175,119`** |
| **Uncovered other meaningful content** | `0` | **`0`** | `2,371` *(+`306` tbl hdr + `2,005` parser hdg)* | **`0`** |
| **Exact non-double-counted partition sum (`== 1,916,622`)** | *(lệch overlap)* | **`True`** (`1,916,622`) | *(lệch `61,931`)* | **`True`** (`1,916,622`) |

### 4.2. Giải thích minh bạch về biến động số liệu Coverage sau khi sửa Provenance

1. **Profile A (`1,800,865 → 1,751,048`, `dropped_but_covered: 61,368 → 0`):** Trước correction, `_locate_subsegment()` fallback về coarse parent span khi chunk đã ghép overlap vượt `512` ký tự bị tách đệ quy lại — dẫn đến `1,468` chunk mang span giả bao trùm cả khoảng trắng liên khối (`\n\n`) và vùng thuộc chunk bị lọc (`dropped_but_covered = 61,368`). Sau khi chuyển sang theo dõi per-character origin `(canonical_idx, is_overlap)` và chỉ tính ký tự không phải whitespace ở biên khối, `6,366` chunk giữ nguyên văn bản 100% với production `chunk_text`, `0` lỗi provenance, `0` dropped-but-covered, `67` dấu chấm `'.'` bị `str.split(". ")` của Recursive tiêu thụ được ghi nhận tường minh ở `recursive_separator_consumed`, và `uncovered_other_meaningful_chars = 0`.
2. **Profile B (`1,816,907 → 1,701,085 + 3,392 repeated_context_only`, `uncovered_other_meaningful: 2,371 → 0`):** Trước correction, `_split_non_table_unit()` tính tuyến tính theo `unit.spans[0]` trên các khối đã ghép nhiều span không liên tiếp (gây lệch `17` heading spans và kéo khoảng trắng liên khối vào dải `[start, end)`), đồng thời `heading_prefix` và `table_header` lặp lại không mang tọa độ nguồn. Sau correction:
   - Per-character origin map `(block_id, canonical_idx) | None` loại bỏ toàn bộ `17` lỗi lệch span (`0 / 13,390` errors) và không còn kéo khoảng trắng liên khối vào `source_spans` (vì vậy `uncovered_whitespace_chars` phản ánh trung thực `175,119` ký tự khoảng trắng/xuống dòng giữa các khối/dòng, còn **`uncovered_other_meaningful_chars = 0`**).
   - Toàn bộ `heading_prefix` (`source_backed=True`) và `table_header` (`source_backed=True`) lưu tọa độ canonical riêng trên `synthetic_segments`, không thổi phồng `original source_spans` (`repeated_context_only_chars_union = 3,392`, tổng `13,034` chars).
   - Mở rộng `_FORMULA_RE` giữ lại các công thức/phép gán ngắn (`i=0`, `k=i+1`, `m=0; t=2;`, `Z(S)=[...]`, `𝑗=1÷𝑚{𝑓𝑝,𝑗}`), nâng số chunk `formula` từ `342` lên `1,110`, giảm `noise_lt_3_tokens` (`1,434 → 1,412`) và `no_bm25_indexable_terms` (`50 → 45`) khi gắn ký hiệu ngắn sau code/formula.

---

## 5. Profile A vs Profile B Corpus Diagnostics (Post-Correction Final)

So sánh trực tiếp trên cùng `CanonicalDocument` (`54` documents, `1,733` pages, `1,916,622` canonical chars):

| Metric | Profile A (`recursive_512_100`) | Profile B (`structure_adaptive`) | Ghi chú đối chiếu |
|---|---:|---:|---|
| **Documents / `document_id` collisions** | `54` / **`0`** | `54` / **`0`** | Đóng 13 collisions cũ |
| **Determinism (2 full runs from raw)** | **PASS** | **PASS** | `canonical_fingerprint` & JSONL khớp 100% |
| **Total chunks** | **`6,366`** | **`13,390`** | Không áp đặt trước số chunk |
| **Duplicate `chunk_id` / Strict provenance errors** | **`0` / `0`** | **`0` / `0`** | Kiểm tra 100% chunks (bounds + text match) |
| **Char length (`min / mean / med / p95 / max`)** | `100 / 361.18 / 421.0 / 509.0 / 512` | `1 / 141.50 / 112.0 / 323.0 / 1871` | Char max của B từ chuỗi ký tự đặc biệt nhỏ hơn 128 MiniLM tokens |
| **Embedding token length (`MiniLM`, with special)** | `19 / 119.35 / 120.0 / 226.0 / 398` | `3 / 47.87 / 39.0 / 111.0 / 128` | Đo trực tiếp bằng tokenizer `e8f8c211...` |
| **Embedding truncation (`> 128` tokens)** | **`2,750` (`43.20%`)** | **`0` (`0.00%`)** | Profile B thỏa tuyệt đối hard budget `<= 128` |
| **Original Source Coverage (interval union)** | **`91.36%`** (`1,751,048` chars) | **`88.75%`** (`1,701,085` chars) | Không gồm khoảng trắng liên khối hay overlap |
| **Effective Source Coverage (+ source-backed repeated ctx)** | **`91.36%`** (`+0`) | **`88.93%`** (`+3,392` chars) | Tách bạch `synthetic_segments` khỏi `source_spans` |
| **Dropped spans & chars (union, `0` overlap w/ kept)** | `599` spans (`65,564` chars): `MIN_CHUNK_SIZE_lt_100` (115), `min_words_lt_15` (374), `non_alpha_gt_80pct` (43), `recursive_separator_consumed` (67) | `1,457` spans (`37,026` chars): `noise_lt_3_tokens` (1,412), `no_bm25_indexable_terms` (45) | `dropped_but_covered_chars = 0` ở cả hai profile |
| **Uncovered whitespace / Uncovered other meaningful** | `100,010` / **`0`** | `175,119` / **`0`** | Partition sum `= 1,916,622` chính xác tuyệt đối |
| **Overlap redundancy (`%` final chars)** | `20.33%` (`467,358` chars) | `1.06%` (`20,095` chars) | Profile A phản ánh đúng toàn bộ overlap thực tế sau per-char tracking |
| **Synthetic prefix (`%` final chars)** | `0.00%` (`0` chars) | `1.36%` (`25,684` chars) | Heading, table header & code signature lặp |
| **Forced splits / fallbacks** | `0` | `2` (`oversized_unit_token_span: 2`) | Đều tiến sạch `<= 128` tokens |
| **Repeated `content_hash` groups (`same_doc / cross_doc`)** | `24 / 193` | `173 / 592` | Đo theo `body_text_for_hash` chính xác sau sửa provenance |
| **Content type distribution** | `recursive_segment`: `6,366` | `narrative`: 6,034; `code_block`: 3,416; `algorithm_steps`: 1,337; `formula`: 1,110; `complexity_analysis`: 879; `heading`: 310; `definition`: 171; `table`: 89; `algorithm_explanation`: 26; `pseudocode`: 18 | Đủ 10/10 loại nội dung |

---

## 6. Indexing Results (`data/runtime/qdrant_eval/`)

Lập chỉ mục lại hoàn toàn (`rebuild=True`) bằng `backend/app/retrieval/indexing.py:build_index()` trên `RetrievalConfig.from_yaml("configs/retrieval.mvp.yaml")` chỉ override `qdrant_mode="local"`, `qdrant_path="D:\\DSA-Document-Assistant-Using-Hybrid-RAG\\data\\runtime\\qdrant_eval"`, `collection="eval_dsa_chunks_v1"`:

| Kiểm tra Isolation & Index | Kết quả xác minh thực tế (Post-Correction) |
|---|---|
| **Collection & Storage** | `eval_dsa_chunks_v1` tại `data/runtime/qdrant_eval/` (tách biệt vật lý 100% khỏi `data/runtime/qdrant_local/`) |
| **Scope A (`eval-recursive-512-v1`, `workspace_id="evaluation"`)** | `indexed = 6,366`, Qdrant `count = 6,366`, `failed = 0` (`100%` khớp `chunks.jsonl` IDs, `0` stale points) |
| **Scope B (`eval-structure-adaptive-v1`, `workspace_id="evaluation"`)** | `indexed = 13,390`, Qdrant `count = 13,390`, `failed = 0` (`100%` khớp `chunks.jsonl` IDs, `0` stale points) |
| **Total Collection Points & Sum Match** | **`19,756`** (`6,366 + 13,390 == 19,756`, `sum_match = True`; thay bản pre-correction `19,466` points) |
| **Cross-profile Chunk ID Overlap** | **`0`** |
| **`check_model_exhaustive()`** | **`PASSED`** (`paraphrase-multilingual-MiniLM-L12-v2`, `dim=384`, `COSINE`) |
| **Payload Isolation Check** | 100% points trong Scope A mang `profile_id="recursive_512_100"`; 100% points trong Scope B mang `profile_id="structure_adaptive"` |

---

## 7. Eight-Combination Smoke-Test Results (Post-Correction)

Chạy lại trên `RTX 5060 Laptop GPU` (`embedding = cuda:0` với `hits=48, misses=1`; `reranker = cuda:0` với `hits=8, misses=1`), `top_k=15`, `candidate_top_k=30`, `reranker_batch_size=4`, `reranker_max_length=512` với 4 smoke queries (`smoke_vi_avl`, `smoke_en_bst`, `smoke_code_sort`, `smoke_complexity`):

*(Ghi chú: Đây là **smoke test kiểm chứng khả năng truy xuất**, KHÔNG tính Precision/Recall/MRR/nDCG do chưa có Ground Truth, và smoke latency không phải official benchmark. `A1`/`B1` `mean_smoke_latency_ms` bao gồm lượt cold đầu tiên dựng `BM25Okapi` + PyVi trong RAM: `12,466.74 ms` ở A1 và `19,357.22 ms` ở B1; 3 lượt kế tiếp chạy trong `437–451 ms` ở A1 và `533–591 ms` ở B1.)*

| Combo | Profile | Method | Scope | Returned/Query | Cross-Profile Leak | Valid Ranks & Finite Scores | Mean Smoke Latency (ms) | Status |
|---|---|---|---|---:|---:|---|---:|---|
| **A1** | `recursive_512_100` | `bm25` | `eval-recursive-512-v1` | `15` | `0` | Yes | `3,449.06` *(warm ~443 ms)* | **PASS** |
| **A2** | `recursive_512_100` | `dense` | `eval-recursive-512-v1` | `15` | `0` | Yes | `675.32` | **PASS** |
| **A3** | `recursive_512_100` | `hybrid_rrf` | `eval-recursive-512-v1` | `15` | `0` | Yes | `954.65` | **PASS** |
| **A4** | `recursive_512_100` | `hybrid_rerank` | `eval-recursive-512-v1` | `15` | `0` | Yes (`0` OOM) | `1,502.03` | **PASS** |
| **B1** | `structure_adaptive` | `bm25` | `eval-structure-adaptive-v1` | `15` | `0` | Yes | `5,265.40` *(warm ~568 ms)* | **PASS** |
| **B2** | `structure_adaptive` | `dense` | `eval-structure-adaptive-v1` | `15` | `0` | Yes | `586.88` | **PASS** |
| **B3** | `structure_adaptive` | `hybrid_rrf` | `eval-structure-adaptive-v1` | `15` | `0` | Yes | `1,092.40` | **PASS** |
| **B4** | `structure_adaptive` | `hybrid_rerank` | `eval-structure-adaptive-v1` | `15` | `0` | Yes (`0` OOM) | `1,237.26` | **PASS** |

---

## 8. Test Results (Post-Correction)

| Bộ kiểm thử | Lệnh chạy | Kết quả thực tế | Ghi chú |
|---|---|---|---|
| **Targeted Stage 02 + Retrieval Suite** | `.\.venv\Scripts\python.exe -m pytest tests/unit/test_adaptive_chunking.py tests/unit/test_reranker.py tests/unit/test_embeddings.py tests/unit/test_keyword_search.py tests/unit/test_hybrid_search.py tests/unit/test_optimizations.py -v` | **`68 passed in 4.05s`** (`exit 0`) | Gồm `8/8` tests ở `test_adaptive_chunking.py` (tăng `+2` regression tests cho AC-03) |
| **Full Regression (`pytest -q`)** | `.\.venv\Scripts\python.exe -m pytest -q` | **`294 passed, 0 failed, 0 skipped, 1 warning (fitz) in 18.33s`** (`PYTEST_EXIT=0`) | `+8` test mới so với `286` total trước Stage 02; `test_llm_reliability.py` không bị chỉnh sửa; `0` regression mới |
| **Environment Dependency Check** | `.\.venv\Scripts\python.exe -m pip check` | **`No broken requirements found.`** (`PIP_CHECK_EXIT=0`) | Không cài thêm dependency ngoài |

---

## 9. Data & Git Integrity

| Kiểm tra | Pre-Stage-02 | Post-Correction (Final) | Trạng thái |
|---|---|---|---|
| **Raw Dataset (`54` files, `80,294,690` B, `benchmark_manifest_v1`)** | `cc44dfdf537710e23e4426dc71d0d28fab8fd09ca8ca7ef180a090979f8903f6` | `cc44dfdf537710e23e4426dc71d0d28fab8fd09ca8ca7ef180a090979f8903f6` | **IDENTICAL** |
| **Production Corpus (`chunks_all.jsonl`, `4,100,882` B)** | `dccdecdcf7ef8fd81860f44860e553e1026ee19a4e56e39d2dfe95b38b872c90` | `dccdecdcf7ef8fd81860f44860e553e1026ee19a4e56e39d2dfe95b38b872c90` | **IDENTICAL** |
| **Production Qdrant (`qdrant_local/.../storage.sqlite`, `0` pts)** | `165bd70ef44a7c9f5ceb262bb75e20e9169437a93b9170c99d33d13c1f3c3f80` | `165bd70ef44a7c9f5ceb262bb75e20e9169437a93b9170c99d33d13c1f3c3f80` | **IDENTICAL** |
| **Git Branch / `HEAD` / `main`** | `feat/final-readiness` / `ff2187e...` / `b84f094...` | `feat/final-readiness` / `ff2187e...` / `b84f094...` | **UNCHANGED** |
| **Modified tracked files (`git diff --stat`)** | `0` | `0` (`0` file production bị sửa) | **CLEAN** |
| **Staged files (`git diff --cached --stat`)** | `0` | `0` | **CLEAN** |
| **`git diff --check`** | Exit `0` | Exit `0` | **CLEAN** |
| **Gitignored / Untracked Artifacts** | — | `data/runtime/qdrant_eval/` (gitignored); `evaluation/{artifacts,chunking,corpus}/`, `evaluation/scripts/{build_benchmark_corpora,smoke_eval_combos}.py`, `tests/unit/test_adaptive_chunking.py`, `STAGE_02_DUAL_CHUNKING_INDEXING_REPORT.md` + các audit/spec docs (untracked) | **CLEAN** |

---

## 10. Acceptance Criteria Matrix (`AC-01` – `AC-06`)

| AC ID | Tiêu chí | Trạng thái đề xuất | Bằng chứng xác minh (Post-Correction) |
|---|---|---|---|
| **AC-01** | Shared benchmark preprocessing | **PASS** | `shared_preprocessor.py` không đổi, chạy chung cho 54 raw files (`canonical_fingerprint = 712f3847...`, 17 DOCX tables, code indent, deterministic lang, `0` collisions) |
| **AC-02** | Dual deterministic benchmark corpora | **PASS** | `build_benchmark_corpora.py --verify-determinism` xác minh 2 runs giống hệt nhau từ raw files; sinh đủ `chunks.jsonl`, `manifest.json`, `diagnostics.json` |
| **AC-03** | Corpus integrity, source provenance và hard token budget | **PASS** | `RECURSIVE_PROD_EQUIVALENCE: 1733/1733 (mismatches=0)`; **`0 / 6,366` (A)** và **`0 / 13,390` (B)** strict provenance errors; `0` duplicate chunk IDs; `0` `dropped_but_covered_chars`; `0` `uncovered_other_meaningful_chars`; `coverage_partition_exact_match = True` trên cả A & B; Profile B **`0 / 13,390` (`0.00%`) vượt `128` MiniLM tokens** (`max = 128`) |
| **AC-04** | Isolated benchmark indexes | **PASS** | `data/runtime/qdrant_eval/` (`eval_dsa_chunks_v1`): **`19,756 = 6,366 + 13,390` points**, `0` stale points, `0` cross-profile overlap, `check_model_exhaustive = PASSED`, `qdrant_local` giữ `0` points |
| **AC-05** | Eight-combination Retrieval readiness | **PASS** | `smoke_eval_combos.py`: `A1–A4` & `B1–B4` **`8/8 PASS`** trên RTX 5060 (`cuda:0`), `0` OOM, `0` cross-profile leaks |
| **AC-06** | Production, regression và Git integrity | **PASS** | Full `pytest -q`: **`294 passed, 0 failed`** (`PYTEST_EXIT=0`), `pip check` exit `0`, raw/clean/qdrant SHA-256 bất biến, `0` production code modified, `HEAD/main` bất biến, `0` staged |

**Tổng hợp Acceptance Criteria (đề xuất):** **`6 / 6 PASS`** (`0` FAIL, `0` BLOCKED, `0` UNVERIFIED) — chờ Người quản lý nghiệm thu chính thức.

---

## 11. Known Limitations and Reproducibility Instructions

1. **Chưa đo chất lượng & latency chính thức (Stage 03–04):** Stage 02 chỉ chứng minh hạ tầng corpus, provenance, index cô lập và smoke retrieval; **không** tính Precision/Recall/MRR/nDCG và không kết luận chiến lược nào tốt hơn khi chưa có Ground Truth.
2. **Remaining short slide-diagram fragments trong `noise_lt_3_tokens`:** Sau khi mở rộng `_FORMULA_RE` bảo toàn các phương trình/phép gán chuẩn (`i=0`, `k=i+1`, `m=0; t=2;`, `Z(S)=[...]`, `𝑗=1÷𝑚{𝑓𝑝,𝑗}`), còn `28` mảnh đồ họa/slide siêu ngắn chứa `=` nhưng không hoàn chỉnh hoặc mang tên nhãn biến số (`'='`, `'t ='`, `',i ='`, `'j6=i'`, `'v0=12'`, `'p+=r;'`) vẫn thuộc `noise_lt_3_tokens` (`1,412` spans) và được báo cáo minh bạch trong `dropped_and_uncovered_chars`.
3. **Phân loại ngôn ngữ rule-based (`langdetect` vắng mặt trong `.venv`):** Sử dụng quy tắc NFC + thư mục nguồn hoàn toàn deterministic; chưa có Ground Truth nhãn ngôn ngữ riêng.
4. **PDF `find_tables()`:** Không bật trên PDF vì tốn ~8 phút/lượt và hay nhận nhầm code listing đóng khung thành bảng; `17/17` bảng DOCX được trích xuất đầy đủ theo XML order.
5. **Qdrant local warning > 20,000 points tạm thời khi rebuild:** Trong lúc upsert Scope B trước khi hoàn tất, Qdrant local phát cảnh báo vượt 20k điểm (tổng cuối cùng sau rebuild là `19,756` points = `6,366 + 13,390`); hoạt động ổn định trong đánh giá offline cục bộ.
6. **Hồ sơ `evaluation/artifacts/`:** Thư mục này hiện đang ở trạng thái untracked (còn `data/runtime/qdrant_eval/` đã được gitignore); cần quyết định giữ ignored hay commit khi chốt checkpoint Stage 02.
7. **Dừng theo Stop Condition:** Chưa tạo official questions/Ground Truth, chưa chạy benchmark chính thức, chưa stage/commit/push/merge và chưa bắt đầu **Stage 03**.

---

## 12. Official Acceptance Appendix

- **Official Acceptance Decision:** **APPROVED – `6 / 6` Acceptance Criteria PASS** (`AC-01`–`AC-06`).
- **AC-03 Provenance Correction Verified:**
  - Recursive (`recursive_512_100`) provenance errors: **`1,468 → 0`** (`0 / 6,366`).
  - Adaptive (`structure_adaptive`) provenance errors: **`17 → 0`** (`0 / 13,390`).
  - Recursive production parity: **`1,733 / 1,733` pages** (`mismatches = 0`).
- **Final Frozen Benchmark Corpora (Post-Correction Revision):**
  - Canonical fingerprint (`shared_preprocessor.py`): `712f3847a6370156258f18297eabb41a657b4ac109dbe6290b9b50d9a0c6bd1d`.
  - Profile A (`recursive_512_100`): `6,366` chunks, `corpus_sha256 = e39811da17d99201e4ad26f54c4a53614a087e76e3e0cb6d031a733862af0649` (đóng băng & sao lưu cục bộ tại `data/runtime/eval_corpus_backups/stage02_frozen/recursive_512_100_chunks.jsonl`).
  - Profile B (`structure_adaptive`): `13,390` chunks, `corpus_sha256 = 81c568275754e35d7323b1e51fc247b885abc873eef4c5caa5c1ec4daa0b5d63` (đóng băng & sao lưu cục bộ tại `data/runtime/eval_corpus_backups/stage02_frozen/structure_adaptive_chunks.jsonl`).
- **Retrieval Readiness & Full Regression:**
  - 8-combination smoke test (`A1–A4`, `B1–B4`): **`8 / 8 PASS`** trên `cuda:0` *(lưu ý: smoke latency không phải official benchmark)*.
  - Full regression (`pytest -q`): **`294 passed, 0 failed, 1 warning (fitz)`** (`exit 0`).
- **Known Limitations Preserved:** Bảo toàn đầy đủ các giới hạn đã nêu tại Mục 11, bao gồm `28` short fragments chứa dấu `=` còn bị lọc dưới `noise_lt_3_tokens`; **chưa tạo Ground Truth** và chưa chạy official evaluation benchmark (thuộc Stage 03–04).
