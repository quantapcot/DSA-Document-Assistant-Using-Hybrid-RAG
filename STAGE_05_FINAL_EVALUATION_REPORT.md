# STAGE 05 – BÁO CÁO ĐÁNH GIÁ TỔNG KẾT, PHÂN TÍCH SO SÁNH VÀ KẾT LUẬN HỆ THỐNG (FINAL EVALUATION REPORT)

> **Dự án:** DSA Document Assistant Using Hybrid RAG  
> **Giai đoạn:** `Stage 05` – Visualization, Comparative Analysis & Final Evaluation Conclusion *(Giai đoạn cuối của Lộ trình Evaluation)*  
> **Workspace:** `D:\DSA-Document-Assistant-Using-Hybrid-RAG`  
> **Branch:** `feat/final-readiness`  
> **Accepted Stage 04 Checkpoint (HEAD bất biến):** `966ecfa415b274680a521cce1a65229d8ac32d40`  
> **Local `main` (bất biến):** `b84f09450f213b03ad57ef8b388f482ee0f0eaa2`  
> **Dataset & Ground Truth:** `stage03_dsa_40q_v1.1_ai_reviewed` (`llm_generated_pooled_relevance_judgments`, `human_validated = false`, `human_approval_status = "DRAFT"`)  
> **Trạng thái các giai đoạn trước:** Stage 01 (`ACCEPTED`), Stage 02 (`ACCEPTED`), Stage 03 (`ACCEPTED`), Stage 04 (`ACCEPTED WITH LIMITATIONS — 5 PASS, 1 PARTIAL`, giữ nguyên `AC-04 = PARTIAL`)  
> **Ngày hoàn tất:** 2026-11-28

---

## 1. Executive Summary

Giai đoạn 05 chuyển hóa toàn bộ kết quả thực nghiệm đo đạc thực tế của Stage 04 (`59` artifacts đã kiểm chứng checksum, không chạy lại thí nghiệm và không tạo dữ liệu giả) thành **28 biểu đồ logic đạt chuẩn công bố học thuật** (`28` ảnh PNG `300 DPI` + `28` tệp vector SVG, phông chữ `Arial`), **10 bảng tổng hợp truy nguyên trực tiếp** và phân tích định lượng–thống kê toàn diện cho **8 tổ hợp thực nghiệm (`A1–A4, B1–B4`)** trên **40 câu hỏi DSA dùng chung** tại **9 mốc `K = [1, 3, 5, 10, 15, 20, 30, 45, 60]`** (`candidate_top_k = 120`):

- **Chiến lược phân đoạn (Chunking — Hình 1–5):** `recursive_512_100` (Profile A, `6.366` chunks) vượt trội có ý nghĩa thống kê ở các chỉ số xếp hạng và bao phủ bằng chứng (`A4` vs `B4`: `nDCG@5 = 0,7053` vs `0,5591`, `Holm p = 0,0010`; `Evidence Recall@15 = 0,9083` vs `0,7500`, `Holm p = 0,0035`) nhờ giữ đủ ngữ cảnh dài tại `K` cố định và BGE Reranker (`max_length = 512`) có `0,00%` pair truncation. Ngược lại, `structure_adaptive` (Profile B, `13.390` chunks) triệt tiêu hoàn toàn cắt cụt embedding MiniLM (`0,00%` vs `43,20%` và `114.961` token bị bỏ của Profile A), giảm dư thừa chồng lấn xuống `1,06%` (vs `20,33%`), giảm `Evidence Redundancy@5` xuống `0,0075` (vs `0,0608`), đạt `Pooled Recall@5` cao hơn nhẹ ở `B4` (`0,3509` vs `0,3474`) và rút ngắn **riêng giai đoạn reranking** khoảng **`41,2%`** (`1.364,8 ms` vs `2.320,4 ms` ở `K=5, pool=120`).
- **Truy xuất & Bằng chứng nguồn (Retrieval & Evidence — Hình 6–15):** `320/320` lượt truy xuất khớp `100,0%` với Stage 03 (`0` ranking drift, `Judged@K = 1,000`). **Không có một cấu hình duy nhất thắng mọi chỉ số:** `A4` dẫn đầu `nDCG@5 (0,7053)`, `MRR@5 (0,9875)`, `Precision@5 (0,7000)`, `Evidence Recall@5 (0,7208)`, `Evidence Recall@15 (0,9083)` và `Complete Evidence@15 (80,0%)`; `B4` dẫn đầu `Pooled Recall@5 (0,3509)` và mật độ trùng lặp thấp nhất; `A1 (Recursive + BM25)` vượt mạnh Dense thuần (`0,5584` vs `0,3921`) nhờ khớp chính xác thuật ngữ DSA/ký hiệu phức tạp/code tiếng Việt và đạt độ trễ thấp nhất (`P95 = 607,5 ms`, `0 MB` VRAM). `Hybrid RRF` cải thiện có ý nghĩa thống kê so với Dense (`+0,1115 nDCG@5`, `Holm p = 0,0032`), nhưng **không vượt BM25 tại `K=5`** (`-0,0547`, `Holm p = 0,1314`, chưa có ý nghĩa thống kê); giá trị chính của RRF là mở rộng `Pooled Recall@60` (`0,7956 → 0,8445`) làm đầu vào cho Reranker (`+40,0%` `nDCG@5` ở Profile A).
- **Hiệu năng & Tài nguyên (Performance — Hình 16–20):** Các giao thức đo được tách bạch hoàn toàn (Main Warm `14.400` lượt, Stage Breakdown `640` lượt, Production-Parameter Profile `1.600` lượt, Cold-Start `24` lượt, Candidate Pool Sensitivity `640` lượt). `QdrantLocal` SQLite chiếm `~461–645 ms` mỗi nhánh BM25/Dense và `115` lần tái tạo cache BM25 (TTL `300s`) làm `Mean > P50`. Toàn bộ `16.664` lượt đo đạt `0` lỗi và `0` CUDA OOM trên RTX 5060 Laptop GPU (`peak_gpu_allocated = 2.686,82 MB`).
- **Sinh câu trả lời (Controlled Agent Generation — Hình 21–22):** Do Production Groq API và HTTP E2E đang ở trạng thái **`BLOCKED`** (Stage 04 `AC-04 = PARTIAL`), chất lượng sinh được đánh giá bằng Controlled Agent (`400` câu trả lời, `budget = 2.048`, `flash` gen + blind `pro` judge, `human_validated = false`). **`NO_RAG` đạt `Correctness (0,9688)` và `Relevancy (1,0000)` cao hơn `A4` (`0,9250` và `0,9563`)** (chênh lệch `-0,0438`, `p = 0,2381`, chưa có ý nghĩa thống kê ở `α = 0,05` — không suy diễn là chứng minh tương đương). Trong 8 tổ hợp RAG, `A4` dẫn đầu `Correctness (0,9250)`, `Completeness (0,9167)` và giảm `Contradiction Rate` về `0,0000` (so với `0,0175` của `NO_RAG`), trong khi `A1` cao hơn nhẹ ở `Faithfulness (0,9250 vs 0,9187)` và `Citation Accuracy (0,8500 vs 0,8187)`.
- **Biên Pareto & Khuyến nghị thăm dò (Hình 23–25):** Biên Pareto (`K=5 & K=15, pool=120`, không dùng trọng số chủ quan) gồm đúng **`{A1, B4, A4}`**. Trong thử nghiệm độ nhạy single-pass tách biệt (`candidate_pool_sensitivity.csv`), **`A4` với `candidate_top_k = 30`, `Top-K = 5–10`, `token_budget = 2.048`** giữ **`99,8%` `nDCG@5`** của `pool=120` (`0,7039` vs `0,7053`) đồng thời cắt `~53%` độ trễ trung bình (`1.628 ms` vs `3.486 ms`). Mọi khuyến nghị đều là **lựa chọn thăm dò (exploratory)**, chưa qua kiểm chứng độc lập trên tập held-out hay môi trường production Groq thật.

---

## 2. Research Objectives

Hệ thống đánh giá hướng tới trả lời có bằng chứng định lượng 8 câu hỏi nghiên cứu cốt lõi (`RQ1`–`RQ8`) cho hệ thống trợ lý học tập Cấu trúc Dữ liệu & Giải thuật (DSA) sử dụng Hybrid RAG:

1. **RQ1 (Chunking Strategy):** So sánh toàn diện `recursive_512_100` (Profile A) và `structure_adaptive` (Profile B) về phân bố độ dài, cắt cụt tokenizer, độ phủ nguồn, phân mảnh ngữ cảnh, chất lượng truy xuất, bằng chứng và sinh câu trả lời.
2. **RQ2 (Retrieval Method):** Đánh giá hiệu quả của bốn phương pháp `BM25`, `Dense`, `Hybrid RRF` và `Hybrid Rerank` trên miền DSA song ngữ Việt–Anh.
3. **RQ3 (Hybrid RRF Contribution):** Kiểm định đóng góp thực sự của hợp nhất thứ hạng `RRF (k=60)` so với từng bộ truy xuất đơn lẻ (BM25 và Dense).
4. **RQ4 (Reranker Contribution):** Định lượng mức tăng chất lượng xếp hạng/bằng chứng của Cross-Encoder `BAAI/bge-reranker-v2-m3` đi kèm chi phí độ trễ và VRAM.
5. **RQ5 (Top-K Selection):** Xác định quy luật đánh đổi `Precision@K` – `Evidence Recall@K` – `Complete Evidence Rate@K` trên 9 mốc `K = [1..60]` để tìm vùng vận hành thực dụng.
6. **RQ6 (Generation Quality & RAG Value):** So sánh 8 cấu hình RAG với baseline `NO_RAG` trên 10 thước đo chất lượng, trung thực ngữ cảnh, trích dẫn và tính đúng đắn học thuật DSA (độ phức tạp & mã nguồn).
7. **RQ7 (Efficiency & Trade-off):** Phân tích độ trễ warm, phân rã từng công đoạn, khởi động nguội, cấu hình tham số production, tiêu thụ GPU/RAM và biên tối ưu Pareto.
8. **RQ8 (Robustness & Reliability):** Kiểm tra độ bền vững của hệ thống trước câu hỏi ngoài miền (OOD), không thể trả lời, prompt injection, ngữ cảnh nhiễu, diễn đạt lại (paraphrase) và câu hỏi tiếng Việt không dấu/sai chính tả, cùng các chức năng hỗ trợ học tập (Quiz, Study Roadmap).

---

## 3. Experimental Setup

Toàn bộ dữ liệu trực quan hóa và phân tích trong Stage 05 kế thừa nguyên trạng thiết lập phần cứng và phần mềm từ Stage 02–04 (không sửa mã nguồn production hay chạy lại benchmark):

- **Phần cứng:** Windows 10/11 (`14` nhân vật lý / `20` luồng CPU, `15,71 GB` RAM), **NVIDIA GeForce RTX 5060 Laptop GPU (`7,96 GB` VRAM)**, Python `3.11.9`, PyTorch `2.11.0+cu128`, CUDA `12.8`.
- **Mô hình & Cấu hình truy xuất (đóng băng):**
  - Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (`max_seq_length = 128`, `384` chiều, `cuda:0`).
  - Cross-Encoder Reranker: `BAAI/bge-reranker-v2-m3` (`max_length = 512`, `batch_size = 4`, `cuda:0`).
  - Lexical Retrieval: `BM25` (`k1 = 1,5`, `b = 0,75`, `accent_insensitive = false`, `bm25_cache_ttl = 300s`).
  - Rank Fusion: `RRF` (`rrf_k = 60`), `candidate_top_k = 120` (thí nghiệm chính), `top_k = 60`.
- **Tám tổ hợp thực nghiệm chuẩn hóa:**
  - **Profile A (`recursive_512_100`):** `A1` (Recursive + BM25), `A2` (Recursive + Dense), `A3` (Recursive + Hybrid RRF), `A4` (Recursive + Hybrid Rerank).
  - **Profile B (`structure_adaptive`):** `B1` (Adaptive + BM25), `B2` (Adaptive + Dense), `B3` (Adaptive + Hybrid RRF), `B4` (Adaptive + Hybrid Rerank).
- **Môi trường trực quan hóa Stage 05:** `matplotlib 3.11.2`, `pandas 3.1.1`, `numpy 2.4.6`, `scipy 1.17.1`, `Pillow 13.0.0`, phông chữ `Arial` (cài sẵn trên Windows, dự phòng `DejaVu Sans`), xuất đồng bộ `300 DPI PNG` và vector `SVG`, bảng màu thân thiện người mù màu (Okabe–Ito) kết hợp nét liền (`Profile A`) / nét đứt & gạch chéo (`Profile B`).

---

## 4. Dataset and Ground Truth

Mọi đầu vào đóng băng đã được kiểm tra lại SHA-256 trước khi sinh biểu đồ Stage 05 (`100%` khớp tuyệt đối, xem chi tiết trong [manifest.json](evaluation/artifacts/stage05/manifest.json)):

- **Tập câu hỏi dùng chung (`questions_40.jsonl`):** `40` câu hỏi DSA (`24` tiếng Việt, `16` tiếng Anh; độ khó: `12` Easy, `18` Medium, `10` Hard; trải đều `8` chủ đề DSA và `10` dạng câu hỏi; gồm tổng cộng `91` đơn vị kiến thức bắt buộc — Knowledge Units).
- **Bằng chứng nguồn xác thực (`source_evidence.jsonl`):** `107` đoạn nguồn chuẩn (canonical spans) được ánh xạ tọa độ ký tự chính xác về `1.733` trang tài liệu chuẩn hóa.
- **Hai kho chunk đóng băng (Stage 02):**
  - Profile A (`recursive_512_100`): `6.366` chunks, SHA-256 `e39811da17d99201e4ad26f54c4a53614a087e76e3e0cb6d031a733862af0649`.
  - Profile B (`structure_adaptive`): `13.390` chunks, SHA-256 `81c568275754e35d7323b1e51fc247b885abc873eef4c5caa5c1ec4daa0b5d63`.
  - Chỉ mục đánh giá tách biệt (`data/runtime/qdrant_eval`, collection `eval_dsa_chunks_v1`): `19.756` points dùng chung vật lý (`100,82 MiB`), không đụng chạm Qdrant production (`0` points).
- **Nhãn liên quan tự động (Stage 03):** Phiên bản `stage03_dsa_40q_v1.1_ai_reviewed` gán nhãn theo thang `0 / 1 / 2` trên toàn bộ pool `9.068` cặp câu hỏi–chunk (`4.526` cho Profile A: `0=4.057, 1=300, 2=169`; `4.542` cho Profile B: `0=4.151, 1=266, 2=125`; `0` câu hỏi zero-positive). **Trạng thái quản trị:** `human_validated = false`, `human_approval_status = "DRAFT"`.

---

## 5. Evaluation Methodology

Để đảm bảo tính chặt chẽ khoa học và tránh lẫn lộn định nghĩa thước đo hoặc giao thức đo, Stage 05 duy trì 5 nguyên tắc phương pháp luận nghiêm ngặt:

1. **Phân biệt Độ liên quan mức Chunk (Pooled Relevance) với Độ phủ Bằng chứng Nguồn (Source-Backed Evidence):** Các chỉ số `Precision@K`, `Pooled Recall@K`, `MRR@K`, `nDCG@K`, `MAP@K`, `F1@K`, `Hit@K` đánh giá trên pool ứng viên đã gán nhãn Stage 03 (`Pooled Recall` chia cho số chunk dương trong pool, **không phải** tổng số chunk liên quan trên toàn bộ corpus). Ngược lại, nhóm chỉ số Evidence (`Evidence Recall`, `Evidence Precision`, `Complete Evidence Rate`, `Partial Evidence Coverage`, `KU Coverage`, `Evidence Redundancy`) đối chiếu trực tiếp hợp khoảng ký tự nguồn xác thực (`source_spans ∪ overlap_segments ∪ source_backed synthetic_segments`) với ngưỡng hoàn chỉnh `overlap_chars ≥ min(20, max(8, floor(0,25 × evidence_span_len)))`, hỗ trợ đầy đủ 3 chế độ `single`, `alternative`, `composite` và luôn thỏa mãn bất biến `Complete ≤ EvRecall ≤ Partial ≤ KU_Any`.
2. **Tách biệt 5 Giao thức Đo Độ trễ:** Không bao giờ gộp chung kết quả từ: (1) Main Warm Latency (`14.400` lượt, `5` repeats xen kẽ, `pool=120`, `ms`), (2) Stage Breakdown (`640` lượt, `K=5 & 60`, `pool=120`, `ms`), (3) Production-Parameter Profile trên evaluation index (`1.600` lượt, `K=5, pool=10`, `ms`), (4) Cold-Start (`24` lượt subprocess độc lập, đơn vị `giây`), và (5) Candidate Pool Sensitivity (`640` lượt single-pass, `pool=[10, 30, 60, 120]`, `ms`). Mọi độ trễ đều là độ trễ bộ truy xuất nội bộ, **không phải** độ trễ HTTP API đầu-cuối.
3. **Thí nghiệm Sinh có kiểm soát (Controlled Agent Generation):** Tách bạch hoàn toàn với Production Groq API (`BLOCKED`); dùng `antigravity_agent_flash` sinh `400` câu trả lời (`320` RAG `budget=2.048` + `40` `NO_RAG` + `40` Context-Budget) và chấm điểm mù 10 chỉ số bằng `antigravity_agent_pro` (đã qua `3/3` ca kiểm chứng tổng hợp PASS và kiểm tra chéo 10% đạt `100,0%` correctness / `96,77%` faithfulness trong ngưỡng ±1 điểm). Giá trị `N/A` của `NO_RAG` ở các chỉ số phụ thuộc ngữ cảnh được giữ nguyên.
4. **Suy luận Thống kê Chuẩn hóa:** Khoảng tin cậy `95%` và kiểm định ghép cặp dùng bootstrap `10.000` mẫu (`seed = 42`) kèm hiệu chỉnh đa kiểm định `Holm-Bonferroni`. Chênh lệch không có ý nghĩa thống kê (`p ≥ 0,05`) được diễn giải đúng bản chất là *chưa đủ bằng chứng bác bỏ giả thuyết không* (với `n = 40`), tuyệt đối không diễn giải là chứng minh tương đương.
5. **Truy nguyên 100% và Kiểm định Biểu đồ Tự động:** Mọi biểu đồ đều được sinh bởi mã Python tái lập (`evaluation/scripts/generate_final_visualizations.py`), đăng ký nguồn trong [figure_manifest.json](evaluation/artifacts/stage05/figure_manifest.json), kiểm tra tự động bằng [chart_validation.json](evaluation/artifacts/stage05/chart_validation.json) và rà soát trực quan từng ảnh PNG trước khi bàn giao.

---

## 6. Chunking Results

**Bảng tổng hợp:** [table_01_chunking_summary.csv](evaluation/artifacts/stage05/tables/table_01_chunking_summary.csv) | **Nguồn Stage 04:** [chunking_metrics.csv](evaluation/artifacts/stage04/chunking/chunking_metrics.csv), [chunking_diagnostics.json](evaluation/artifacts/stage04/chunking/chunking_diagnostics.json).

| Thước đo | Profile A (`recursive_512_100`) | Profile B (`structure_adaptive`) | Nhận xét chính |
|---|---:|---:|---|
| Tổng số chunks | `6.366` | `13.390` (`2,10×`) | Adaptive sinh gấp 2,1 lần số đoạn ngắn hơn |
| Độ dài ký tự (Mean / Median / P95 / Max) | `361,2 / 421 / 509 / 512` | `141,5 / 112 / 323 / 1.871` | 46,8% chunk Profile B `< 100` ký tự; bảo toàn khối code/bảng dài tới `1.871` |
| Embedding Tokens (Mean / P95 / Max, giới hạn `128`) | `119,4 / 226 / 398` | `47,9 / 111 / 128` | Profile B kiểm soát nghiêm ngặt trần `128` tokens |
| Tỷ lệ cắt cụt Embedding / Token bị loại bỏ | **`43,20%`** (`114.961` = `15,13%`) | **`0,00%`** (`0` = `0,00%`) | Ưu thế tuyệt đối của Profile B ở lớp mã hóa MiniLM |
| Reranker Pair Tokens (Mean / Max, giới hạn `512`) | `168,0 / 363` (`0,0%` cắt cụt) | `101,0 / 190` (`0,0%` cắt cụt) | BGE Reranker đọc trọn 100% văn bản trên cả hai profiles |
| Token ngữ cảnh truy xuất / câu (`K=5 / 15 / 60`) | `654,5 / 1.939,3 / 7.532,9` | `309,2 / 842,1 / 3.347,6` | Cùng `K`, Profile A mang mật độ thông tin gấp `~2,1–2,3×` |
| Độ phủ nguồn hiệu dụng / Bảo toàn KU | `91,36%` / `91/91 (100%)` | `88,93%` / `91/91 (100%)` | 0 ký tự mất phủ không giải thích được; 0 thiếu provenance |
| Dư thừa chồng lấn / Tiền tố tổng hợp | `20,33%` / `0,00%` | `1,06%` / `1,36%` | Profile B giảm `19×` dư thừa overlap nhưng có `2.012` excess duplicate chunks |
| Phân loại cấu trúc (Code+Pseudo / Công thức / Bảng) | `0 / 0 / 0` (untyped) | `3.434 / 1.110 / 89` | Profile B giữ siêu dữ liệu ngữ nghĩa cấu trúc DSA |
| Dung lượng Corpus JSONL / Vector lý thuyết (`f32 384d`)* | `9,96 MiB` / `9,33 MiB` | `15,53 MiB` / `19,61 MiB` | Dung lượng vật lý QdrantLocal dùng chung (A + B) = **`100,82 MiB`** |

*\*Lưu ý tách biệt lưu trữ: `100,82 MiB` (`105.722.471` bytes) là kích thước vật lý thực tế của một cơ sở dữ liệu SQLite `QdrantLocal` chứa chung cả hai profile; dung lượng vector từng profile (`9,33 MiB` vs `19,61 MiB`) là ước lượng lý thuyết `chunk_count × 384 × 4 bytes`.*

- **Hình 1 ([PNG](evaluation/artifacts/stage05/figures/png/fig01_chunk_count_and_storage.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig01_chunk_count_and_storage.svg)):** *So sánh tổng số lượng chunk, phân loại cấu trúc (code/pseudocode, công thức, bảng) và dung lượng lưu trữ giữa Profile A (`6.366` chunks) và Profile B (`13.390` chunks, `2,10×`), tách bạch dung lượng vector lý thuyết từng profile khỏi QdrantLocal vật lý dùng chung (`100,82 MiB`).*
- **Hình 2 ([PNG](evaluation/artifacts/stage05/figures/png/fig02_chunk_length_distribution.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig02_chunk_length_distribution.svg)):** *Phân bố độ dài ký tự theo 5 nhóm và thống kê mô tả (`Min, Median, Mean, P95, Max`): Profile A tập trung ở `400–512` ký tự (`52,8%`), trong khi Profile B nghiêng mạnh về đoạn ngắn `< 100` ký tự (`46,8%`) kèm đuôi dài cho khối cấu trúc nguyên vẹn.*
- **Hình 3 ([PNG](evaluation/artifacts/stage05/figures/png/fig03_token_distribution_and_context_volume.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig03_token_distribution_and_context_volume.svg)):** *Chẩn đoán hai tokenizer độc lập và dung lượng ngữ cảnh truy xuất: MiniLM cắt cụt `43,2%` chunk Profile A nhưng BGE Reranker (`limit = 512`) đạt `0%` cắt cụt cặp câu hỏi–chunk ở cả hai profile, giúp Profile A tận dụng trọn ngữ cảnh dài gấp `2,1–2,3×` tại mỗi mốc `K`.*
- **Hình 4 ([PNG](evaluation/artifacts/stage05/figures/png/fig04_truncation_rate_and_discarded_tokens.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig04_truncation_rate_and_discarded_tokens.svg)):** *Tỷ lệ cắt cụt Embedding so với Reranking và tổng số token bị loại bỏ (`114.961` tokens = `15,13%` ở Profile A so với `0` ở Profile B).*
- **Hình 5 ([PNG](evaluation/artifacts/stage05/figures/png/fig05_source_coverage_ku_and_redundancy.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig05_source_coverage_ku_and_redundancy.svg)):** *Độ phủ văn bản nguồn, tỷ lệ bảo toàn đơn vị kiến thức (`91/91 = 100%` ở cả hai) và đặc trưng dư thừa chồng lấn (`20,33%` vs `1,06%`) so với tỷ lệ chunk văn bản ngắn trùng lặp (`3,25%` vs `15,03%`).*

**Tổng kết đánh đổi Chunking:** Profile A chịu nhược điểm cắt cụt nặng ở bộ nhúng MiniLM (`43,2%`) và chồng lấn lớn (`20,33%`), nhưng chính kích thước chunk đầy đặn kết hợp khả năng đọc trọn `512` token của BGE Reranker giúp nó tránh phân mảnh thông tin. Profile B giải quyết xuất sắc bài toán cắt cụt và dư thừa chồng lấn, song vì tách nhỏ quá mức (`median = 112` ký tự, nhiều tiêu đề/đoạn rời rạc) nên mất ngữ cảnh xung quanh ở các mốc `K` cố định.

---

## 7. Retrieval Results

**Bảng tổng hợp:** [table_02_retrieval_k5_k15_k60.csv](evaluation/artifacts/stage05/tables/table_02_retrieval_k5_k15_k60.csv) | **Nguồn Stage 04:** [retrieval_metrics_summary.csv](evaluation/artifacts/stage04/retrieval/retrieval_metrics_summary.csv), [strict_relevance_metrics.csv](evaluation/artifacts/stage04/retrieval/strict_relevance_metrics.csv), [ranking_drift.json](evaluation/artifacts/stage04/retrieval/ranking_drift.json).

| Tổ hợp | P@5 | Pooled R@5 | MRR@5 | nDCG@5 | MAP@5 | F1@5 | Hit@5 | nDCG@15 | Pooled R@15 | nDCG@60 | Pooled R@60 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A1** (Rec+BM25) | `0,6000` | `0,2899` | `0,9104` | `0,5584` | `0,5566` | `0,3650` | `1,000` | `0,5672` | `0,5227` | `0,6476` | `0,7956` |
| **A2** (Rec+Dense) | `0,4700` | `0,2340` | `0,7079` | `0,3921` | `0,3816` | `0,2950` | `0,900` | `0,4099` | `0,4235` | `0,4769` | `0,6198` |
| **A3** (Rec+RRF) | `0,5700` | `0,2845` | `0,8467` | `0,5037` | `0,5003` | `0,3525` | `1,000` | `0,5351` | `0,5157` | `0,6407` | `0,8445` |
| **A4** (Rec+Rerank) | **`0,7000`** | `0,3474` | **`0,9875`** | **`0,7053`** | **`0,6638`** | **`0,4374`** | **`1,000`** | **`0,7335`** | **`0,6745`** | **`0,8097`** | **`0,9447`** |
| **B1** (Ada+BM25) | `0,4700` | `0,2677` | `0,8563` | `0,4414` | `0,4373` | `0,3165` | `0,950` | `0,4470` | `0,4320` | `0,5398` | `0,6975` |
| **B2** (Ada+Dense) | `0,3400` | `0,1932` | `0,6246` | `0,3128` | `0,2872` | `0,2335` | `0,775` | `0,3227` | `0,3177` | `0,3948` | `0,5406` |
| **B3** (Ada+RRF) | `0,4350` | `0,2629` | `0,7863` | `0,4163` | `0,3894` | `0,3047` | `0,925` | `0,4389` | `0,4312` | `0,5414` | `0,7297` |
| **B4** (Ada+Rerank) | `0,6000` | **`0,3509`** | `0,9008` | `0,5591` | `0,5638` | `0,4141` | `1,000` | `0,5908` | `0,5677` | `0,6745` | `0,8454` |

- **Hình 6 ([PNG](evaluation/artifacts/stage05/figures/png/fig06_precision_at_k.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig06_precision_at_k.svg)):** *`Precision@K` của 8 tổ hợp qua 9 mốc `K` và đối chiếu `K=5, 15`: giảm đơn điệu khi `K` tăng; `A4` đạt cao nhất (`0,7000` tại `K=5`), còn `A1` (`0,6000`) vượt xa `A2` (`0,4700`).*
- **Hình 7 ([PNG](evaluation/artifacts/stage05/figures/png/fig07_pooled_recall_at_k.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig07_pooled_recall_at_k.svg)):** *`Pooled Recall@K` (tính trên mẫu số pool ứng viên Stage 03, **không phải** recall trên toàn bộ corpus): tại `K=5`, **`B4` (`0,3509`) nhỉnh hơn nhẹ `A4` (`0,3474`)**; từ `K=10..60`, `A4` dẫn đầu (`0,6745` tại `K=15`; `0,9447` tại `K=60`).*
- **Hình 8 ([PNG](evaluation/artifacts/stage05/figures/png/fig08_mrr_at_k.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig08_mrr_at_k.svg)):** *`MRR@K`: Reranking đưa tài liệu đúng đầu tiên lên sát top-1 (`A4 = 0,9875`; `B4 = 0,9008`), cải thiện vượt trội so với Dense đơn lẻ (`A2 = 0,7079`; `B2 = 0,6246`).*
- **Hình 9 ([PNG](evaluation/artifacts/stage05/figures/png/fig09_ndcg_at_k.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig09_ndcg_at_k.svg)):** *`nDCG@K` phân bậc `0/1/2` (thứ hạng `A4 > B4 ≈ A1 > A3 > B1 > B3 > A2 > B2` được giữ nguyên hoàn toàn dưới phân tích nhạy cảm Strict Grade-2 trong `strict_relevance_metrics.csv`).*
- **Hình 10 ([PNG](evaluation/artifacts/stage05/figures/png/fig10_map_at_k.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig10_map_at_k.svg)):** *`MAP@K`: Hai tổ hợp Reranker (`A4 = 0,6638`, `B4 = 0,5638` tại `K=5`) vượt rõ nhóm không rerank về mật độ tài liệu liên quan ở đầu danh sách.*
- **Hình 11 ([PNG](evaluation/artifacts/stage05/figures/png/fig11_f1_at_k.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig11_f1_at_k.svg)):** *`F1@K` đạt đỉnh tự nhiên quanh `K = 10–15` (`A4` cực đại `~0,520` tại `K=15`), củng cố vùng chọn `K` hợp lý.*
- **Hình 12 ([PNG](evaluation/artifacts/stage05/figures/png/fig12_hit_rate_and_judged_at_k.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig12_hit_rate_and_judged_at_k.svg)):** *`Hit Rate@K` (`A1, A3, A4, B4` đạt `1,000` ngay từ `K=5`) và xác minh `Judged@60 = 1,000` (`0` vị trí chưa gán nhãn, `320/320` khớp thứ hạng Stage 03).*

**Giải thích cơ chế quan trọng:**
1. *Vì sao BM25 vượt trội Dense trên tập DSA?* Câu hỏi DSA dày đặc thuật ngữ chuyên ngành chính xác (`AVL`, `Dijkstra`, `QuickSort`, `O(n log n)`, tên hàm/biến mã giả, cụm từ tiếng Việt chuẩn trong giáo trình). BM25 khớp từ vựng chính xác các thuật ngữ này, trong khi bộ nhúng đa ngôn ngữ nhỏ gọn `MiniLM-L12-v2` (384 chiều, giới hạn `128` token) vừa có năng lực phân biệt thuật ngữ kỹ thuật yếu hơn, vừa cắt cụt `43,2%` chunk Profile A.
2. *Vì sao Hybrid RRF không luôn vượt BM25 ở `K=5`?* RRF cộng trọng số bằng nhau cho thứ hạng của BM25 và Dense; khi Dense kém BM25 tới `-0,166 nDCG@5` (Profile A), thứ hạng nhiễu từ Dense kéo một số tài liệu tốt của BM25 xuống ngoài top-5 (`A3 = 0,5037` vs `A1 = 0,5584`, `Holm p = 0,1314`). Tuy nhiên ở chiều sâu `K=60`, RRF kéo thêm các ứng viên ngữ nghĩa mà BM25 bỏ sót (`Pooled Recall@60`: `0,7956 → 0,8445`).
3. *Vì sao Reranking cải thiện đột phá?* `BAAI/bge-reranker-v2-m3` tương tác chú ý chéo (cross-attention) toàn bộ cặp câu hỏi–chunk trong giới hạn `512` token (`0%` cắt cụt), lọc sạch nhiễu của Dense trong pool RRF và sắp xếp chính xác các ứng viên liên quan lên top-5 (`+40,0%` `nDCG@5` ở Profile A; `+34,3%` ở Profile B).

---

## 8. Evidence Results

**Bảng tổng hợp:** [table_03_evidence_summary.csv](evaluation/artifacts/stage05/tables/table_03_evidence_summary.csv) | **Nguồn Stage 04:** [evidence_metrics_summary.csv](evaluation/artifacts/stage04/evidence/evidence_metrics_summary.csv), [evidence_redundancy.csv](evaluation/artifacts/stage04/evidence/evidence_redundancy.csv).

| Tổ hợp | EvRecall@5 | EvPrec@5 | CompleteEv@5 | PartialEv@5 | KU_Any@5 | SemKU@5 | EvRedund@5 | EvRecall@15 | CompleteEv@15 | EvRecall@60 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A1** | `0,6583` | `0,3200` | `0,4500` | `0,6831` | `0,7125` | `0,8292` | `0,0469` | `0,8000` | `0,6000` | `0,9417` |
| **A2** | `0,4958` | `0,2200` | `0,2750` | `0,5080` | `0,5208` | `0,7375` | `0,0415` | `0,6500` | `0,4250` | `0,7667` |
| **A3** | `0,6042` | `0,2700` | `0,3500` | `0,6331` | `0,6625` | `0,8042` | `0,0380` | `0,7750` | `0,6250` | `0,9292` |
| **A4** | **`0,7208`** | **`0,3550`** | **`0,5500`** | **`0,7521`** | **`0,7833`** | **`0,8958`** | `0,0608` | **`0,9083`** | **`0,8000`** | **`0,9708`** |
| **B1** | `0,3917` | `0,2100` | `0,1750` | `0,4300` | `0,4708` | `0,6625` | `0,0075` | `0,5542` | `0,3250` | `0,7750` |
| **B2** | `0,3000` | `0,1550` | `0,1000` | `0,3271` | `0,3542` | `0,5417` | **`0,0000`** | `0,3917` | `0,2250` | `0,6042` |
| **B3** | `0,3750` | `0,2000` | `0,1500` | `0,4021` | `0,4292` | `0,6875` | **`0,0000`** | `0,5375` | `0,3000` | `0,7458` |
| **B4** | `0,5125` | `0,2850` | `0,3000` | `0,5521` | `0,5917` | `0,8083` | `0,0075` | `0,7500` | `0,4750` | `0,8500` |

- **Hình 13 ([PNG](evaluation/artifacts/stage05/figures/png/fig13_evidence_recall_and_semantic_ku_at_k.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig13_evidence_recall_and_semantic_ku_at_k.svg)):** *`Evidence Recall@K` (dựa trên giao khoảng nguồn chuẩn xác thực) tăng dốc từ `K=1` tới `K=15` (`A4`: `0,475 → 0,721@5 → 0,908@15`) và bão hòa sau `K=30`; đối chiếu với `Semantic KU Coverage@5` (nhãn LLM Stage 03) cho thấy độ phủ ngữ nghĩa luôn cao hơn độ phủ span chuẩn vì giáo trình có nhiều đoạn diễn giải tương đương nằm ở vị trí khác.*
- **Hình 14 ([PNG](evaluation/artifacts/stage05/figures/png/fig14_evidence_precision_and_redundancy_at_k.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig14_evidence_precision_and_redundancy_at_k.svg)):** *`Evidence Precision@K` và `Evidence Redundancy@K` (`K=5, 15, 60`): Profile B gần như không có trùng lặp bằng chứng (`0,0000–0,0075` tại `K=5`), trong khi Profile A có mức dư thừa `0,0380–0,0608` (tăng lên `0,128` tại `K=60` ở `A4`) do cửa sổ chồng lấn `100` ký tự.*
- **Hình 15 ([PNG](evaluation/artifacts/stage05/figures/png/fig15_complete_vs_partial_evidence_coverage.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig15_complete_vs_partial_evidence_coverage.svg)):** *`Complete Evidence Rate@K` (`A4` đạt `55,0%` tại `K=5`, `80,0%` tại `K=15`, `92,5%` tại `K=60`) và phân tầng tường minh 4 chỉ số `Complete ≤ EvRecall ≤ Partial ≤ KU_Any` xác minh trên toàn bộ 72 dòng dữ liệu, phản ánh chính xác cài đặt `evidence_metrics.py` và không đánh đồng giao ký tự tối thiểu với phủ bằng chứng hoàn chỉnh.*

**Vì sao thước đo mức chunk và thước đo mức bằng chứng nguồn khác nhau?** Một chunk dài có thể được gán nhãn liên quan (`grade 1/2`), giúp `Precision@5` của `A4` đạt `0,7000`, nhưng chỉ `0,3550` số chunk trong Top-5 thực sự giao trực tiếp với đoạn nguồn chuẩn (`source_evidence.jsonl`). Ngược lại, ở Profile B (`B4`), `Pooled Recall@5` (`0,3509`) cao hơn `A4` (`0,3474`), nhưng `Complete Evidence Rate@5` chỉ đạt `30,0%` (so với `55,0%` của `A4`) vì mỗi chunk ngắn thường chỉ chứa một mảnh của KU đa câu.

---

## 9. Latency and Resource Results

**Bảng tổng hợp:** [table_04_latency_protocols.csv](evaluation/artifacts/stage05/tables/table_04_latency_protocols.csv) | **Nguồn Stage 04:** [latency_summary.csv](evaluation/artifacts/stage04/performance/latency_summary.csv), [stage_latency.csv](evaluation/artifacts/stage04/performance/stage_latency.csv), [production_profile_latency.csv](evaluation/artifacts/stage04/performance/production_profile_latency.csv), [cold_start_latency.csv](evaluation/artifacts/stage04/performance/cold_start_latency.csv), [resource_usage.csv](evaluation/artifacts/stage04/performance/resource_usage.csv).

| Tổ hợp | Warm K=5 Mean / P50 / P95 (`ms`, `pool=120`) | Stage Rerank K=5 (`ms`, `pool=120`) | Prod-Param K=5 P50 / P95 (`ms`, `pool=10`) | Cold-Start Mean (`s`, 24 trials) | Cold BM25 Build (`s`) | Peak GPU Alloc (`MB`) | Cold RSS (`MB`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **A1** | `417,3 / 360,4 / 607,5` | `0,0` | `352,0 / 447,0` | `15,42` | `9,57` | `0,0` | `1.246,0` |
| **A2** | `407,8 / 389,4 / 626,5` | `0,0` | `371,9 / 479,6` | `10,44` | `0,00` | `458,5` | `2.064,0` |
| **A3** | `820,4 / 757,6 / 1.321,8` | `0,0` | `720,3 / 943,6` | `21,03` | `10,86` | `458,5` | `2.120,0` |
| **A4** | `3.209,5 / 3.097,9 / 3.693,1` | `2.320,4` | `988,7 / 1.230,9` | `32,13` | `10,66` | `2.662,0` | `2.556,9` |
| **B1** | `525,3 / 452,4 / 756,7` | `0,0` | `429,5 / 553,8` | `20,49` | `16,02` | `0,0` | `1.255,9` |
| **B2** | `484,2 / 457,6 / 754,0` | `0,0` | `433,4 / 586,1` | `9,62` | `0,00` | `458,5` | `2.063,5` |
| **B3** | `1.008,4 / 896,1 / 1.308,4` | `0,0` | `868,3 / 1.151,1` | `24,92` | `15,21` | `458,5` | `2.131,4` |
| **B4** | `2.502,6 / 2.366,8 / 3.214,3` | `1.364,8` | `1.045,7 / 1.368,1` | `34,32` | `15,49` | `2.648,3` | `2.578,9` |

- **Hình 16 ([PNG](evaluation/artifacts/stage05/figures/png/fig16_mean_latency_at_k.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig16_mean_latency_at_k.svg)):** *Giao thức 1 (Main Warm Latency, `14.400` lượt, `pool=120`): Độ trễ `Mean` và `P50` gần như phẳng theo `K` vì hệ thống chấm điểm cả `120` ứng viên trước khi cắt Top-K; `115` lần tái tạo cache BM25 (TTL `300s`) làm `Mean` cao hơn `P50` rõ rệt trên các tổ hợp BM25/Hybrid.*
- **Hình 17 ([PNG](evaluation/artifacts/stage05/figures/png/fig17_p95_latency_at_k.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig17_p95_latency_at_k.svg)):** *Giao thức 1: `P95@K` và phân vị `P50 / Mean / P95 / P99` tại `K=5`. `A1` nhanh nhất (`P95 = 607,5 ms`); trên tổng warm retrieval (`pool=120`), `B4` nhanh hơn `A4` **`13,0%` tại P95 K=5** (`3.214,3` vs `3.693,1 ms`) và **`23,6%` tại P50** (`2.366,8` vs `3.097,9 ms`).*
- **Hình 18 ([PNG](evaluation/artifacts/stage05/figures/png/fig18_retrieval_latency_breakdown.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig18_retrieval_latency_breakdown.svg)):** *Giao thức 2 (Stage Breakdown, `640` lượt, `K=5 & K=60`, `pool=120`): Làm rõ hai điểm nghẽn kiến trúc: (1) Mức cải thiện tốc độ **`41,2%`** của `B4` áp dụng **riêng cho giai đoạn BGE Reranking** (`1.364,8 ms` vs `2.320,4 ms`) nhờ chuỗi token cặp ngắn hơn (`101` vs `168` tokens); (2) Chi phí `QdrantLocal` SQLite (`~461–645 ms` mỗi nhánh BM25/Dense) chiếm gần `~1 giây` ở nhóm Hybrid (`A3/B3/A4/B4`).*
- **Hình 19 ([PNG](evaluation/artifacts/stage05/figures/png/fig19_cold_vs_warm_and_prod_profile_latency.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig19_cold_vs_warm_and_prod_profile_latency.svg)):** *Tách biệt đơn vị và giao thức: Giao thức 4 Cold-Start (`24` lượt, **giây**, xây BM25 Profile B tốn `15,2–16,0s` vs `9,6–10,9s` của Profile A) đối chiếu với Giao thức 1 Warm `pool=120` và Giao thức 3 Production-Parameter Eval Profile (`1.600` lượt, `K=5, pool=10`, **ms**: `A4 P95` giảm từ `3.693 ms` xuống `1.231 ms`).*
- **Hình 20 ([PNG](evaluation/artifacts/stage05/figures/png/fig20_resource_usage.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig20_resource_usage.svg)):** *Tài nguyên GPU VRAM (`0 MB` BM25; `458,5 MB` Dense/RRF; `2.648–2.687 MB` Rerank — dưới `34%` dung lượng RTX 5060 `8 GB`, `0` OOM) và RAM tiến trình (`1.246–2.579 MB` ở cold-start độc lập; ổn định `965–993 MB` xuyên suốt 5 lượt warm).*

---

## 10. Answer Generation Results

**Bảng tổng hợp:** [table_05_generation_summary.csv](evaluation/artifacts/stage05/tables/table_05_generation_summary.csv) | **Nguồn Stage 04:** [answer_metrics_summary.csv](evaluation/artifacts/stage04/generation/answer_metrics_summary.csv), [answer_quality_audit.json](evaluation/artifacts/stage04/generation/answer_quality_audit.json), [citation_validation.csv](evaluation/artifacts/stage04/generation/citation_validation.csv).

> **Cảnh báo phạm vi bắt buộc:** Thí nghiệm này chạy ở chế độ **Controlled Agent Generation** (`antigravity_agent_flash` sinh + `antigravity_agent_pro` chấm mù, `token_budget = 2.048`, `human_validated = false`). Kết quả **không đại diện cho hiệu năng live Groq production** (đang **`BLOCKED`**, giữ nguyên Stage 04 `AC-04 = PARTIAL`).

| Tổ hợp | Correctness | Relevancy | Faithfulness | Completeness | Cit. Acc. | Cit. Valid. | Unsup. Rate | Contra. Rate | Complexity | Code |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **A1** | `0,8125` | `0,8500` | **`0,9250`** | `0,7917` | **`0,8500`** | `1,0000` | **`0,0590`** | `0,0027` | `0,9231` | `0,7778` |
| **A2** | `0,7063` | `0,7688` | `0,8562` | `0,6667` | `0,8187` | `1,0000` | `0,1125` | **`0,0000`** | `0,8462` | `0,8889` |
| **A3** | `0,8187` | `0,8500` | `0,8500` | `0,8083` | `0,8187` | `1,0000` | `0,1225` | `0,0027` | `0,9062` | `0,8333` |
| **A4** | `0,9250` | `0,9563` | `0,9187` | `0,9167` | `0,8187` | `1,0000` | `0,0638` | **`0,0000`** | **`1,0000`** | **`0,9500`** |
| **B1** | `0,6625` | `0,7438` | `0,8562` | `0,6000` | `0,8000` | `1,0000` | `0,1263` | `0,0035` | `0,9091` | `0,5625` |
| **B2** | `0,4813` | `0,5938` | `0,8750` | `0,3500` | `0,8438` | `1,0000` | `0,1205` | `0,0043` | `0,6154` | `0,3333` |
| **B3** | `0,6562` | `0,7312` | `0,8313` | `0,5917` | `0,8063` | `1,0000` | `0,1439` | `0,0020` | `0,8462` | `0,7778` |
| **B4** | `0,7500` | `0,8000` | `0,8500` | `0,7333` | `0,8125` | `1,0000` | `0,1131` | `0,0136` | `0,9583` | `0,7000` |
| **NO_RAG** | **`0,9688`** | **`1,0000`** | `N/A` | **`0,9208`** | `N/A` | `N/A` | `N/A` | `0,0175` | **`1,0000`** | **`0,9500`** |

- **Hình 21 ([PNG](evaluation/artifacts/stage05/figures/png/fig21_answer_quality_comparison.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig21_answer_quality_comparison.svg)):** *So sánh toàn bộ 10 chỉ số chất lượng sinh cho 8 tổ hợp RAG và `NO_RAG`, giữ nguyên giá trị `N/A` cho các chỉ số không áp dụng với `NO_RAG`.*

**Các phát hiện cốt lõi cần công bố minh bạch:**
1. **`NO_RAG` vượt `A4` về Correctness và Relevancy:** `NO_RAG` đạt `Correctness = 0,9688` và `Relevancy = 1,0000`, cao hơn `A4` (`0,9250` và `0,9563`; chênh lệch `-0,0438`, `p = 0,2381` chưa có ý nghĩa thống kê). Nguyên nhân trực tiếp là DSA đại cương là miền tri thức phổ quát đã nằm sẵn trong tham số tiền huấn luyện của LLM, trong khi các cấu hình RAG buộc tuân thủ nghiêm quy tắc *chỉ trả lời dựa trên ngữ cảnh được cấp và phải từ chối/giới hạn khi thiếu bằng chứng*.
2. **Giá trị thực chất của RAG (`A4`):** Cung cấp neo bằng chứng xác thực (`Faithfulness = 0,9187`), trích dẫn truy nguyên được về đúng giáo trình môn học (`Citation Validity = 1,0000`), và đưa `Contradiction Rate` về **`0,0000`** (so với `0,0175` ở `NO_RAG`).
3. **`A1` vượt `A4` về Faithfulness và Citation Accuracy:** `A1` đạt `Faithfulness = 0,9250` (vs `0,9187` của `A4`), `Citation Accuracy = 0,8500` (vs `0,8187`) và `Unsupported Claim Rate = 0,0590` (vs `0,0638`), cho thấy `A4` dẫn đầu nhóm RAG về `Correctness/Completeness/Complexity/Code` nhưng không dẫn đầu tuyệt đối mọi chỉ số sinh.
4. **`Citation Validity = 100%` không đồng nghĩa `Citation Accuracy = 100%`:** Mọi cấu hình RAG đều đạt `100%` mã trích dẫn hợp lệ về mặt cú pháp (`[1]..[k]` tồn tại trong ngữ cảnh), nhưng độ chính xác ngữ nghĩa của trích dẫn chỉ đạt `0,8000–0,8500` (đôi khi trích sai đoạn trong danh sách ngữ cảnh).
5. **Tác hại của phân mảnh ngữ cảnh lên sinh mã nguồn:** Khi chuyển từ `A4` sang `B4` (cùng ngân sách `2.048` tokens), `Correctness` giảm mạnh từ `0,9250 → 0,7500` và `Code Correctness` giảm từ `0,9500 → 0,7000` (còn `B2` chỉ đạt `0,4813` và `0,3333`) vì retriever yếu buộc mô hình phải abstention hoặc ghép mảnh thiếu ngữ cảnh.

---

## 11. Robustness Results

**Bảng tổng hợp:** [table_08_robustness_and_study.csv](evaluation/artifacts/stage05/tables/table_08_robustness_and_study.csv) | **Nguồn Stage 04:** [robustness_metrics.csv](evaluation/artifacts/stage04/robustness/robustness_metrics.csv), [robustness_results.jsonl](evaluation/artifacts/stage04/robustness/robustness_results.jsonl).

- **Hình 27 — Panel A ([PNG](evaluation/artifacts/stage05/figures/png/fig27_robustness_and_study_features.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig27_robustness_and_study_features.svg)):** *Đánh giá độ bền vững trên `20` ca độc lập (hiển thị cỡ mẫu `n` đi kèm tỷ lệ %).*

| Nhóm kiểm thử | Số ca (`n`) | Xử lý đúng (`%`) | Tỷ lệ Abstention | Unsupported Rate | Suy giảm `EvRecall@5` | Điểm chất lượng (`0–1`) |
|---|---:|---:|---:|---:|---:|---:|
| `out_of_domain` | `4` | **`100,0%`** (`4/4`) | `100,0%` | `0,0%` | `—` | `1,0000` |
| `unanswerable_in_corpus` | `4` | **`100,0%`** (`4/4`) | `100,0%` | `0,0%` | `—` | `1,0000` |
| `prompt_injection`* | `3` | **`100,0%`** (`3/3`) | `0,0%` | `0,0%` | `0,0000` | `0,9167` |
| `distractor_context` | `3` | `66,7%` (`2/3`) | `0,0%` | `0,0%` | `0,0000` | `0,7500` |
| `semantic_paraphrase` | `3` | `33,3%` (`1/3`) | `33,3%` | `0,0%` | **`-0,5556`** | `0,5000` |
| `typo_noisy` (Tiếng Việt không dấu) | `3` | **`0,0%`** (`0/3`) | `66,7%` | `0,0%` | **`-0,6667`** | `0,1667` |

- **Điểm mạnh:** Từ chối chuẩn xác `4/4` câu ngoài miền và `4/4` câu không có trong giáo trình với `0%` bịa đặt; kháng được `3/3` ca prompt injection thử nghiệm. **Cảnh báo bắt buộc:** *Không được suy rộng kết quả `3/3` ca prompt injection thành chứng nhận an toàn bảo mật toàn hệ thống*, đặc biệt khi kiểm toán mã nguồn cho thấy `backend/app/safety/guardrail.py` hiện là file rỗng `0 B` và an toàn đang phụ thuộc hoàn toàn vào `abstention.py` + luật prompt.
- **Điểm yếu nghiêm trọng nhất đã định lượng:** Truy vấn tiếng Việt không dấu/sai chính tả (`n=3`) thất bại `0/3` (`EvRecall@5` sụt `-0,6667`) do thiết lập `accent_insensitive: false` trong BM25 kết hợp năng lực yếu của Dense MiniLM. Truy vấn diễn đạt lại (`semantic_paraphrase`, `n=3`) cũng chỉ đạt `33,3%` (`EvRecall@5` sụt `-0,5556`).

---

## 12. Study Feature Results

**Nguồn Stage 04:** [functional_test_results.json](evaluation/artifacts/stage04/study_features/functional_test_results.json), [study_features_summary.csv](evaluation/artifacts/stage04/study_features/study_features_summary.csv), [quiz_evaluation.jsonl](evaluation/artifacts/stage04/study_features/quiz_evaluation.jsonl), [roadmap_evaluation.jsonl](evaluation/artifacts/stage04/study_features/roadmap_evaluation.jsonl).

- **Hình 27 — Panel B ([PNG](evaluation/artifacts/stage05/figures/png/fig27_robustness_and_study_features.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig27_robustness_and_study_features.svg)):** *Tách bạch nghiêm ngặt giữa kiểm thử hợp đồng module backend thật và đánh giá nội dung học tập bằng Controlled Agent:*
  1. **Real Backend Module Contract Checks:** **`11 / 11 PASSED`** (`plan_chat_query` nhận diện đúng `comparison/overview/explanation`; abstention trên workspace rỗng cho Chat/Quiz/Roadmap; `low_score_abstention`; sắp xếp tô-pô DAG `study_planner` hợp lệ trên `4` chủ đề DSA lớn gồm `18–179` topics; `citation_builder`).
  2. **Controlled Agent Quiz (`n = 8` chủ đề DSA):** `schema_valid = 1,00`, `content_correctness = 0,6875`, `source_grounding = 0,6250`, `pedagogical_quality = 0,6250`, `duplicate_rate = 0,0`.
  3. **Controlled Agent Study Roadmap (`n = 4` kịch bản):** `schema_valid = 1,00`, `content_correctness = 0,6250`, `source_grounding = 0,4375`, `pedagogical_quality = 0,5000`, `duplicate_rate = 0,0`.
- **Nhận xét:** Tính toàn vẹn cấu trúc JSON và logic điều phối backend đạt mức tin cậy cao (`100%`), nhưng mức độ bám sát nguồn giáo trình của lộ trình học tập (`0,4375`) và câu hỏi trắc nghiệm (`0,6250`) ở chế độ sinh tự động còn nhiều dư địa cải thiện (chủ yếu do truy xuất ngữ cảnh rộng đa chủ đề dễ kéo vào tiêu đề bị cắt hoặc mảnh kiến thức rời rạc).

---

## 13. Ablation Study

**Bảng tổng hợp:** [table_06_ablation_and_stats.csv](evaluation/artifacts/stage05/tables/table_06_ablation_and_stats.csv) | **Nguồn Stage 04:** [ablation_results.csv](evaluation/artifacts/stage04/ablation/ablation_results.csv), [rag_contribution.csv](evaluation/artifacts/stage04/ablation/rag_contribution.csv).

- **Hình 22 ([PNG](evaluation/artifacts/stage05/figures/png/fig22_rag_vs_norag_and_ablation_study.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig22_rag_vs_norag_and_ablation_study.svg)):** *Biểu đồ ghép cặp đóng góp của RAG so với `NO_RAG` (Panel A) và nghiên cứu cắt bỏ từng thành phần trên `nDCG@5` với khoảng tin cậy 95% bootstrap và `Holm p` (Panel B).*

| Thành phần kiểm định | So sánh (`Mới − Nền`) | `Δ nDCG@5` (Tương đối) | `95% CI` (`nDCG@5`) | `Holm p` | `Δ EvRecall@15` (Tương đối) | `Holm p` |
|---|---|---:|---|---:|---:|---:|
| **BM25 vs Dense (Rec)** | `A1 − A2` | `+0,1662` (`+42,4%`) | `[+0,084, +0,249]` | *(direct)* | `+0,1500` (`+23,1%`) | *(direct)* |
| **BM25 vs Dense (Ada)** | `B1 − B2` | `+0,1286` (`+41,1%`) | `[+0,058, +0,198]` | *(direct)* | `+0,1625` (`+41,5%`) | *(direct)* |
| **RRF vs Dense (Rec)** | `A3 − A2` | **`+0,1115`** (`+28,4%`) | `[+0,058, +0,169]` | **`0,0032`*** | **`+0,1250`** (`+19,2%`) | **`0,0060`*** |
| **RRF vs Dense (Ada)** | `B3 − B2` | **`+0,1035`** (`+33,1%`) | `[+0,043, +0,164]` | **`0,0054`*** | **`+0,1458`** (`+37,2%`) | **`0,0060`*** |
| **RRF vs BM25 (Rec)** | `A3 − A1` | `-0,0547` (`-9,8%`) | `[-0,109, -0,006]` | `0,1314` (ns) | `-0,0250` (`-3,1%`) | `1,0000` (ns) |
| **RRF vs BM25 (Ada)** | `B3 − B1` | `-0,0251` (`-5,7%`) | `[-0,076, +0,024]` | `0,3402` (ns) | `-0,0167` (`-3,0%`) | `1,0000` (ns) |
| **Reranker (Rec)** | `A4 − A3` | **`+0,2016`** (`+40,0%`) | `[+0,148, +0,257]` | **`0,0010`*** | **`+0,1333`** (`+17,2%`) | **`0,0291`*** |
| **Reranker (Ada)** | `B4 − B3` | **`+0,1428`** (`+34,3%`) | `[+0,073, +0,214]` | **`0,0032`*** | **`+0,2125`** (`+39,5%`) | **`0,0036`*** |
| **Adaptive vs Rec (BM25)** | `B1 − A1` | **`-0,1170`** (`-20,9%`) | `[-0,193, -0,043]` | **`0,0315`*** | **`-0,2458`** (`-30,7%`) | **`0,0018`*** |
| **Adaptive vs Rec (Rerank)** | `B4 − A4` | **`-0,1461`** (`-20,7%`) | `[-0,216, -0,078]` | **`0,0010`*** | **`-0,1583`** (`-17,4%`) | **`0,0035`*** |

**Kết luận Ablation:** (1) **Reranker** là thành phần tạo cải thiện lớn nhất và chắc chắn nhất trên cả hai profile; (2) **BM25** đóng góp tín hiệu nền mạnh hơn hẳn Dense (`+41–42%` `nDCG@5`); (3) **Hybrid RRF** cứu Dense rõ rệt nhưng không vượt BM25 ở Top-5 nếu thiếu Reranker; (4) **Adaptive Chunking** làm giảm đáng kể `nDCG@5` và `EvRecall@15` so với Recursive ở hầu hết các nhánh (`Holm p < 0,05` trên 4/4 nhánh `EvRecall@15`).

---

## 14. Sensitivity Analysis

**Bảng tổng hợp:** [table_07_sensitivity_summary.csv](evaluation/artifacts/stage05/tables/table_07_sensitivity_summary.csv) | **Nguồn Stage 04:** [candidate_pool_sensitivity.csv](evaluation/artifacts/stage04/sensitivity/candidate_pool_sensitivity.csv), [top_k_sensitivity.csv](evaluation/artifacts/stage04/sensitivity/top_k_sensitivity.csv), [context_budget_sensitivity.csv](evaluation/artifacts/stage04/sensitivity/context_budget_sensitivity.csv).

- **Hình 23 ([PNG](evaluation/artifacts/stage05/figures/png/fig23_candidate_pool_and_topk_sensitivity.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig23_candidate_pool_and_topk_sensitivity.svg)):**
  - *Candidate Pool Sensitivity (Giao thức 5, single-pass `640` lượt, `top_k = 5`, `pool = [10, 30, 60, 120]`, tách biệt hoàn toàn với Main Warm Latency):* **`A4` tại `pool = 30`** đạt `nDCG@5 = 0,7039` — bảo toàn **`99,80%`** điểm `nDCG@5` của `pool = 120` (`0,7053`) và thậm chí đạt `EvRecall@5 = 0,7375` (so với `0,7208`), trong khi độ trễ trung bình single-pass giảm hơn một nửa từ `3.486 ms` (`P95 = 3.980 ms`) xuống **`1.628 ms` (`P95 = 1.777 ms`)**. Giảm xuống `pool = 10` làm tụt mạnh chất lượng (`0,6363`).
  - *Top-K Sensitivity (`K = 1..60`):* `K = 5–10` cân bằng tốt `Precision` và `Evidence Recall` cho hỏi đáp tương tác; `K = 15` là điểm gãy đạt `Complete Evidence Rate = 80,0%` (`EvRecall = 0,9083`) ở `A4`; mở rộng quá `K = 30` hầu như không mang thêm bằng chứng mới (`0,963 → 0,971`).
- **Hình 24 ([PNG](evaluation/artifacts/stage05/figures/png/fig24_context_budget_sensitivity.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig24_context_budget_sensitivity.svg)):**
  - *Context Budget Sensitivity (mẫu thăm dò nhỏ `n = 10` câu hỏi × `A4, B4` tại `1.024 / 2.048 / 4.096` tokens):* **`2.048` tokens** là điểm vận hành hợp lý nhất cho `A4` (`Correctness = 0,950`, `Faithfulness = 0,925`, `Citation Accuracy = 0,850`). Tăng lên `4.096` tokens nhồi thêm ngữ cảnh nhiễu làm `Faithfulness` tụt xuống `0,825` và `Citation Accuracy` tụt xuống `0,700` (hiệu ứng lost-in-the-middle / quá tải ngữ cảnh). *Cần nhấn mạnh đây là mẫu thăm dò `n = 10`, không khẳng định tối ưu toàn cục.*

---

## 15. Statistical Analysis

**Nguồn Stage 04:** [confidence_intervals.csv](evaluation/artifacts/stage04/statistics/confidence_intervals.csv), [paired_bootstrap_results.csv](evaluation/artifacts/stage04/statistics/paired_bootstrap_results.csv), [win_tie_loss.csv](evaluation/artifacts/stage04/statistics/win_tie_loss.csv), [annotation_sensitivity.csv](evaluation/artifacts/stage04/statistics/annotation_sensitivity.csv), [subgroup_metrics.csv](evaluation/artifacts/stage04/statistics/subgroup_metrics.csv).

- **Hình 26 ([PNG](evaluation/artifacts/stage05/figures/png/fig26_statistical_ci_subgroups_and_audit.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig26_statistical_ci_subgroups_and_audit.svg)):**
  1. **95% Bootstrap CIs (`10.000` mẫu, `seed = 42`):** `A4` có khoảng tin cậy `nDCG@5` (`[0,643, 0,762]`) tách biệt gần như hoàn toàn khỏi nhóm không rerank và cao hơn rõ rệt `B4` (`[0,486, 0,628]`); `EvRecall@15` của `A4` đạt `[0,846, 0,963]`.
  2. **Win–Tie–Loss ghép cặp (`n = 40`, ngưỡng hòa `1e-4`):** Trên `nDCG@5`, `A4` thắng `A3` ở **`36/40`** câu (`0` hòa, `4` thua); `B4` thắng `B3` ở `30/40`; `A3` thắng `A2` ở `28/40`; ngược lại `A3` thua `A1` ở `22/40` (`11` thắng, `7` hòa), và `B4` thua `A4` ở **`28/40`** (`10` thắng, `2` hòa).
  3. **Độ nhạy gán nhãn AI (Annotation Sensitivity):** So sánh nhãn sơ cấp (`ai_annotations_primary`) với nhãn chung cuộc (`ai_annotations_final`) cho thấy dịch chuyển tuyệt đối tối đa `|Δ nDCG| ≤ 0,011` (`B1@5 = -0,0110`; `A4@5 = -0,0088`), **không làm đảo bất kỳ thứ hạng kết luận nào**. *Lưu ý: Độ ổn định giữa các vòng gán nhãn AI không đồng nghĩa với độ chính xác đã được chuyên gia con người xác thực (`human_validated = false`).*
  4. **Phân nhóm ngôn ngữ & độ khó (`nDCG@5`):** `A4` vận hành ổn định nhất qua cả hai ngôn ngữ (`EN = 0,720`, `VI = 0,695`) và mọi mức độ khó (`Easy = 0,701`, `Medium = 0,711`, `Hard = 0,700`). Ngược lại, `B4` giảm khá mạnh trên câu hỏi tiếng Việt (`0,509` vs `0,635` ở EN) và câu khó (`0,533` vs `0,616` ở Easy), phản ánh mức độ nhạy cao hơn của chunk ngắn đối với truy vấn đa ý tiếng Việt.

---

## 16. Error Analysis

**Nguồn Stage 04:** [failure_taxonomy_summary.csv](evaluation/artifacts/stage04/error_analysis/failure_taxonomy_summary.csv), [error_analysis.jsonl](evaluation/artifacts/stage04/error_analysis/error_analysis.jsonl), [representative_case_studies.json](evaluation/artifacts/stage04/error_analysis/representative_case_studies.json).

- **Hình 28 ([PNG](evaluation/artifacts/stage05/figures/png/fig28_error_distribution_and_taxonomy.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig28_error_distribution_and_taxonomy.svg)):** Phân bố `85` lượt gắn nhãn lỗi thực đo trên `40` câu hỏi (`A4 / B4`) cùng bản đồ phân tán từng câu hỏi gắn các ca đại diện:

| Loại lỗi thực đo | Số lượt (`A4/B4`) | Hiện tượng đo được (Measured) vs Nguyên nhân hợp lý (Causal Explanation) |
|---|---:|---|
| `2. Relevant Evidence Ranked Too Low` | **`29`** | **Đo được:** Bằng chứng có mặt trong Top-60 nhưng xếp ngoài Top-5 (chiếm nhiều nhất). **Giải thích:** Truy vấn đa KU đòi hỏi nhiều đoạn nguồn rải rác hoặc điểm reranker ưu tiên đoạn định nghĩa chung hơn đoạn chi tiết phụ. |
| `1. Relevant Evidence Not Retrieved` | `17` | **Đo được:** Không có mặt trong Top-60 (`Q007, Q022, Q029` yếu trên mọi phương pháp). **Giải thích:** Khác biệt từ vựng/ngôn ngữ giữa câu hỏi và slide nguồn hoặc đoạn bị lọc/phân tán. |
| `9. Missing Knowledge Unit` | `17` | **Đo được:** Ít nhất 1 KU bắt buộc không đạt ngưỡng phủ nguồn tại `K=5`. **Giải thích:** Câu hỏi tổng hợp nhiều khía cạnh (`multi_evidence = 40/40`) vượt quá số slot hữu dụng của Top-5. |
| `10. Irrelevant Context Overload` | `7` | **Đo được:** Nhiều chunk điểm `0` chen vào Top-5 làm giảm độ chính xác ngữ cảnh. **Giải thích:** Khớp từ khóa chung (`thuật toán`, `độ phức tạp`) không đúng cấu trúc dữ liệu mục tiêu. |
| `7. Context Fragmentation` | `6` | **Đo được:** Rõ nhất trên Profile B (`B4` có mảnh nhỏ dưới `100` ký tự không đạt ngưỡng giao hoàn chỉnh). **Giải thích:** Tách cấu trúc quá nhỏ tách rời tiêu đề/điều kiện khỏi thân thuật toán. |
| `4. Semantic Mismatch` | `5` | **Đo được:** Truy xuất đúng chủ đề nhưng lệch khía cạnh hỏi (ví dụ độ phức tạp không gian vs thời gian). **Giải thích:** Dense MiniLM yếu trong phân biệt sắc thái kỹ thuật hẹp. |
| `8. Duplicate Evidence` | `4` | **Đo được:** Nhiều chunk lặp gần giống cùng một bằng chứng chiếm dụng slot Top-5 (chủ yếu ở Profile A do overlap `100` ký tự). **Giải thích:** Chưa có khử trùng lặp mức ngữ nghĩa sau rerank. |

- **Các ca điển hình (`representative_case_studies.json`):** `Q007, Q022, Q029` yếu trên mọi phương pháp; `Q001, Q003, Q004` thể hiện lợi thế rõ của Recursive (`A4 > B4`); `Q011, Q014, Q015` là các ca mà Adaptive vượt Recursive (`B4 > A4`) nhờ cô lập gọn công thức/mã giả không bị cắt cụt; `Q006, Q007, Q008` là các ca hiếm hoi mà Reranker làm giảm nhẹ `nDCG@5` so với RRF.

---

## 17. Pareto Trade-off

**Bảng tổng hợp:** [table_09_pareto_and_recommendations.csv](evaluation/artifacts/stage05/tables/table_09_pareto_and_recommendations.csv) | **Nguồn Stage 04:** [pareto_frontier.csv](evaluation/artifacts/stage04/tradeoff/pareto_frontier.csv), [quality_latency_tradeoff.csv](evaluation/artifacts/stage04/tradeoff/quality_latency_tradeoff.csv).

- **Hình 25 ([PNG](evaluation/artifacts/stage05/figures/png/fig25_quality_latency_pareto_tradeoff.png) | [SVG](evaluation/artifacts/stage05/figures/svg/fig25_quality_latency_pareto_tradeoff.svg)):** Biên Pareto Chất lượng (`nDCG@5` & `Evidence Recall@15`) – Độ trễ Warm `P95` – Tài nguyên GPU VRAM trên **cùng Giao thức 1 (`pool=120`)**, không dùng hàm tổng hợp trọng số chủ quan:
  - **`A1` (Non-dominated — Cực nhanh & Không cần GPU):** `P95 = 607,5 ms` (`K=5`), `0 MB` VRAM, `nDCG@5 = 0,5584`, `EvRecall@15 = 0,8000`, dẫn đầu `Faithfulness (0,9250)` và `Citation Accuracy (0,8500)`. Trội hoàn toàn (dominates) `A2, B1, B2, B3` và cả `A3` về hầu hết trục đánh đổi chất lượng–độ trễ!
  - **`B4` (Non-dominated — Điểm trung gian Rerank nhanh & 0% cắt cụt):** `P95 = 3.214,3 ms` (`K=5`, nhanh hơn `A4` `13,0%`; `K=15`: `3.110,4 ms`, nhanh hơn `16,8%`), `nDCG@5 = 0,5591`, `EvRecall@15 = 0,7500`, `0%` cắt cụt embedding, `0,0075` redundancy.
  - **`A4` (Non-dominated — Chất lượng truy xuất & bằng chứng cao nhất):** `nDCG@5 = 0,7053`, `EvRecall@15 = 0,9083`, `CompleteEv@15 = 80,0%`, dẫn đầu `Correctness` nhóm RAG (`0,9250`), đổi lại `P95 = 3.693,1 ms` (`K=5, pool=120`) và `2,66 GB` VRAM.
- **Không ép một cấu hình thắng duy nhất:** `A1`, `B4` và `A4` phục vụ ba bài toán vận hành khác nhau; việc chọn điểm nào phụ thuộc trực tiếp vào ràng buộc SLA độ trễ, phần cứng GPU và yêu cầu độ phủ bằng chứng.

---

## 18. Configuration Recommendations

**Artifact có cấu trúc:** [configuration_recommendations.json](evaluation/artifacts/stage05/configuration_recommendations.json).

> **Trạng thái quản trị:** Toàn bộ đề xuất dưới đây là **khuyến nghị thăm dò (Exploratory Recommendations)** rút ra từ tập `40` câu hỏi benchmark (`human_validated = false`, không có tập kiểm chứng held-out độc lập, Production Groq API đang `BLOCKED`). Không coi là cấu hình production đã nghiệm thu.

| Kịch bản vận hành | Cấu hình đề xuất *(Exploratory)* | Số liệu đo được làm cơ sở | Ràng buộc & Lưu ý |
|---|---|---|---|
| **1. Ưu tiên chất lượng (Quality-oriented)** | **`A4`** (`recursive_512_100` + `BM25 + Dense + RRF + BGE Rerank`, `K=5–15`) | `nDCG@5 = 0,7053`, `MRR@5 = 0,9875`, `EvRecall@15 = 0,9083`, `CompleteEv@15 = 80,0%`, RAG `Correctness = 0,9250`, `Contradiction = 0,0000` | Độ trễ cao nhất ở `pool=120` (`P95 = 3.693 ms`), cần `~2,66 GB` VRAM, chịu `43,2%` embedding truncation; không thắng mọi chỉ số (`NO_RAG` > `A4` ở Correctness; `A1` > `A4` ở Faithfulness/Cit. Acc.). |
| **2. Ưu tiên độ trễ thấp / không GPU (Low-latency)** | **`A1`** (`recursive_512_100` + `BM25`) | Warm `P95 = 607,5 ms` (`Prod-Param pool=10`: `447,0 ms`), **`0 MB` VRAM**, `nDCG@5 = 0,5584`, `EvRecall@15 = 0,8000`, `Faithfulness = 0,9250`, `Cit. Acc. = 0,8500` | Nhạy cảm với lệch từ vựng, paraphrase và câu hỏi tiếng Việt không dấu (`accent_insensitive = false`). |
| **3. Phương án phân đoạn thay thế (Alternative Chunking)** | **`B4`** (`structure_adaptive` + `Hybrid Rerank`) | `0,00%` embedding truncation, `0,0075` `EvRedund@5`, `Pooled R@5 = 0,3509`, **giai đoạn rerank nhanh hơn `41,2%`** (`1.365` vs `2.320 ms`), tổng Warm `P95` thấp hơn `13,0–16,8%` | Phân mảnh ngữ cảnh làm giảm `nDCG@5` (`-20,7%`) và RAG `Correctness` (`0,7500` vs `0,9250`), tốn `1,5×` thời gian build BM25 và `2,1×` vector lý thuyết. |
| **4. Cân bằng thực dụng (Practical Balanced — Exploratory)** | **`A4` tinh chỉnh:** `Recursive 512/100` + `BM25 + Dense + RRF (k=60)` + `BGE Reranker` + **`candidate_top_k = 30`** + **`Top-K = 5–10`** + **`Context Budget = 2.048` tokens** | Ở single-pass `pool=30`: giữ **`99,80%` `nDCG@5`** (`0,7039` vs `0,7053`), `EvRecall@5 = 0,7375`, cắt `~53%` độ trễ trung bình (`1.628 ms`, `P95 = 1.777 ms`); ở `budget=2.048` (`n=10`): `Correctness = 0,950`, `Faithfulness = 0,925`. | Dữ liệu `pool=30` thuộc Giao thức 5 single-pass và `budget=2.048` dựa trên mẫu nhỏ `n=10`; cần kiểm chứng held-out và đo end-to-end với Groq API trước khi triển khai production. |

---

## 19. Research Question Conclusions (`RQ1` – `RQ8`)

Mỗi câu hỏi nghiên cứu được kết luận đầy đủ theo cấu trúc 6 phần bắt buộc (1. Kết quả đo đạc, 2. Cơ sở đối chiếu, 3. Biểu đồ/Artifact chứng minh, 4. Bằng chứng thống kê, 5. Diễn giải thực tiễn, 6. Giới hạn liên quan):

### RQ1 — Chiến lược phân đoạn (`recursive_512_100` vs `structure_adaptive`)
1. **Kết quả đo đạc:** `A4` đạt `nDCG@5 = 0,7053`, `EvRecall@15 = 0,9083`, `CompleteEv@15 = 80,0%`, `RAG Correctness = 0,9250`; `B4` đạt `0,00%` embedding truncation, `0,0075` `EvRedund@5`, `Pooled R@5 = 0,3509`, và `1.364,8 ms` cho riêng giai đoạn rerank.
2. **Cơ sở đối chiếu:** `B4` vs `A4` (`nDCG@5`: `0,5591` vs `0,7053`, `-20,7%`; `EvRecall@15`: `0,7500` vs `0,9083`, `-17,4%`; `RAG Correctness`: `0,7500` vs `0,9250`); `A4` vs `B4` về cắt cụt (`43,20%` vs `0,00%`) và rerank (`2.320,4` vs `1.364,8 ms`, chậm hơn `70%` / `B4` nhanh hơn `41,2%`).
3. **Biểu đồ/Artifact:** Hình 1–5, Hình 9, Hình 15, Hình 18, Hình 22; [table_01_chunking_summary.csv](evaluation/artifacts/stage05/tables/table_01_chunking_summary.csv), [ablation_results.csv](evaluation/artifacts/stage04/ablation/ablation_results.csv).
4. **Bằng chứng thống kê:** `B4 − A4` giảm có ý nghĩa thống kê trên `nDCG@5` (`-0,1461`, `95% CI [-0,216, -0,078]`, `Holm p = 0,0010`, thua `28/40` câu) và `EvRecall@15` (`-0,1583`, `Holm p = 0,0035`); toàn bộ 4/4 cặp `B − A` đều giảm `EvRecall@15` với `Holm p ≤ 0,0035`.
5. **Diễn giải thực tiễn:** Với pipeline có Cross-Encoder `512` token, `recursive_512_100` vượt trội về chất lượng đầu-cuối vì giữ mạch văn đầy đủ; `structure_adaptive` hiện phân mảnh quá nhỏ (`median = 112` ký tự) nên chưa chuyển hóa được lợi thế `0%` truncation thành chất lượng truy xuất cao hơn.
6. **Giới hạn liên quan:** Đánh giá cố định theo số chunk `K` và `token_budget = 2.048` với ngưỡng min-size mặc định của Stage 02; chưa thử nghiệm phiên bản Adaptive có bước gộp đơn vị tối thiểu `80–120` tokens.

### RQ2 — Phương pháp truy xuất (`BM25`, `Dense`, `Hybrid RRF`, `Hybrid Rerank`)
1. **Kết quả đo đạc:** Tại `K=5`: `A4 = 0,7053 > B4 = 0,5591 ≈ A1 = 0,5584 > A3 = 0,5037 > B1 = 0,4414 > B3 = 0,4163 > A2 = 0,3921 > B2 = 0,3128` (`nDCG@5`).
2. **Cơ sở đối chiếu:** `BM25` vượt mạnh `Dense` trên cả hai profiles (`A1 − A2 = +0,1662 / +42,4%`; `B1 − B2 = +0,1286 / +41,1%`); `Hybrid Rerank` đứng đầu tuyệt đối trong mỗi profile.
3. **Biểu đồ/Artifact:** Hình 6–12, Hình 22; [table_02_retrieval_k5_k15_k60.csv](evaluation/artifacts/stage05/tables/table_02_retrieval_k5_k15_k60.csv).
4. **Bằng chứng thống kê:** Khoảng tin cậy `95%` của `A1 − A2` (`[+0,084, +0,249]`) và `B1 − B2` (`[+0,058, +0,198]`) hoàn toàn dương; thứ hạng 8 tổ hợp bất biến dưới Strict Grade-2 và Annotation Sensitivity (`|Δ| ≤ 0,011`).
5. **Diễn giải thực tiễn:** Miền DSA có mật độ thuật ngữ chính xác cao khiến truy xuất từ vựng (BM25) mạnh hơn embedding đa ngôn ngữ nhỏ gọn (`MiniLM`), nhưng kết hợp Hybrid + Cross-Encoder vẫn là kiến trúc tối ưu về độ chính xác xếp hạng.
6. **Giới hạn liên quan:** Chỉ thử một mô hình dense (`MiniLM-L12-v2`, `128` token); một bộ nhúng chuyên miền dài hơn (`512+` token) có thể thu hẹp khoảng cách Dense vs BM25.

### RQ3 — Đóng góp của hợp nhất `Hybrid RRF`
1. **Kết quả đo đạc:** `A3` đạt `nDCG@5 = 0,5037`, `EvRecall@15 = 0,7750`, `Pooled R@60 = 0,8445`; `B3` đạt `nDCG@5 = 0,4163`, `EvRecall@15 = 0,5375`, `Pooled R@60 = 0,7297`.
2. **Cơ sở đối chiếu:** So với Dense (`A2, B2`), RRF tăng mạnh (`+0,1115` và `+0,1035` `nDCG@5`); so với BM25 (`A1, B1`), điểm mẫu của RRF thấp hơn ở `K=5` (`-0,0547` và `-0,0251`) nhưng cao hơn ở `Pooled Recall@60` (`+0,0489` ở Profile A; `+0,0322` ở Profile B).
3. **Biểu đồ/Artifact:** Hình 7, Hình 9, Hình 22, Hình 26 (Panel B); [table_06_ablation_and_stats.csv](evaluation/artifacts/stage05/tables/table_06_ablation_and_stats.csv).
4. **Bằng chứng thống kê:** `A3 vs A2` (`Holm p = 0,0032`) và `B3 vs B2` (`Holm p = 0,0054`) có ý nghĩa thống kê; `A3 vs A1` (`Holm p = 0,1314`, thua `22/40` câu) và `B3 vs B1` (`Holm p = 0,3402`) chưa đạt ý nghĩa thống kê ở `α = 0,05`.
5. **Diễn giải thực tiễn:** Không nên triển khai `Hybrid RRF` đơn độc (không có Reranker) để thay BM25 nếu chỉ quan tâm Top-5; vai trò đúng của RRF là *bộ tạo tập ứng viên đa dạng* (`candidate_top_k = 30–120`) cho tầng Reranker.
6. **Giới hạn liên quan:** Dùng `rrf_k = 60` đồng trọng số (`1:1`); chưa tối ưu trọng số bất đối xứng nghiêng về BM25.

### RQ4 — Đóng góp của `Cross-Encoder Reranker` (`BAAI/bge-reranker-v2-m3`)
1. **Kết quả đo đạc:** `A4 − A3` tăng **`+0,2016 nDCG@5` (`+40,0%`)** và **`+0,1333 EvRecall@15` (`+17,2%`)**; `B4 − B3` tăng **`+0,1428 nDCG@5` (`+34,3%`)** và **`+0,2125 EvRecall@15` (`+39,5%`)**. Chi phí riêng giai đoạn rerank (`pool=120`): `+2.320,4 ms` (`A4`) và `+1.364,8 ms` (`B4`), `+2,2 GB` VRAM.
2. **Cơ sở đối chiếu:** So trực tiếp với `Hybrid RRF` (`A3, B3`) trên cùng pool đầu vào `120`, và đối chiếu chi phí khi giảm pool xuống `10` (`production_profile_latency.csv`: `A4 − A3` chỉ thêm `~268 ms` P50) hoặc `30` (`candidate_pool_sensitivity.csv`: `A4 mean = 1.628 ms`).
3. **Biểu đồ/Artifact:** Hình 9, Hình 13, Hình 18, Hình 22, Hình 23; [stage_latency.csv](evaluation/artifacts/stage04/performance/stage_latency.csv).
4. **Bằng chứng thống kê:** Có ý nghĩa thống kê cao trên mọi cặp chính: `A4 vs A3` (`Holm p = 0,0010`, thắng **`36/40`** câu `nDCG@5`; `EvRecall@15` `Holm p = 0,0291`), `B4 vs B3` (`Holm p = 0,0032`, thắng `30/40`; `EvRecall@15` `Holm p = 0,0036`).
5. **Diễn giải thực tiễn:** Reranker là khoản đầu tư mang lại tỷ suất chất lượng cao nhất trong toàn bộ hệ thống; chi phí độ trễ lớn ở `pool=120` có thể kiểm soát hiệu quả bằng việc giảm `candidate_top_k` xuống `30`.
6. **Giới hạn liên quan:** Đòi hỏi GPU (~`2,66 GB` VRAM trên RTX 5060); tốc độ rerank trên CPU thuần chưa được đo và sẽ chậm hơn nhiều lần.

### RQ5 — Lựa chọn `Top-K` tối ưu thực dụng
1. **Kết quả đo đạc:** Với `A4`, `Precision@K` giảm từ `0,975@1 → 0,700@5 → 0,550@10 → 0,473@15 → 0,180@60`, trong khi `EvRecall@K` tăng `0,475@1 → 0,721@5 → 0,846@10 → 0,908@15 → 0,963@30 → 0,971@60` và `CompleteEv@K` đạt `55,0%@5 → 70,0%@10 → 80,0%@15 → 92,5%@60`.
2. **Cơ sở đối chiếu:** So sánh đồng thời 9 mốc `K = [1..60]` giữa `A4` và `B4`, kết hợp thử nghiệm `Context Budget` (`1.024 / 2.048 / 4.096`).
3. **Biểu đồ/Artifact:** Hình 6, Hình 11, Hình 13, Hình 15, Hình 23 (Panel B), Hình 24; [top_k_sensitivity.csv](evaluation/artifacts/stage04/sensitivity/top_k_sensitivity.csv).
4. **Bằng chứng thống kê:** `F1@K` đạt đỉnh quanh `K=10–15`; độ trễ warm retrieval hầu như không đổi theo `K` (`A4 P95`: `3.693 ms@5` vs `3.692 ms@60` ở `pool=120`), nên chi phí tăng `K` nằm chủ yếu ở số token đầu vào LLM và suy giảm Faithfulness khi vượt `2.048` tokens.
5. **Diễn giải thực tiễn:** Chọn `K = 5–10` (đóng gói theo `token_budget = 2.048`) cho chat học tập tiêu chuẩn và `K = 15` cho tác vụ tổng hợp sâu cần `Complete Evidence ≥ 80%`.
6. **Giới hạn liên quan:** Mốc `Context Budget` mới kiểm định trên mẫu thăm dò `n = 10` câu bằng Controlled Agent.

### RQ6 — Chất lượng sinh câu trả lời và giá trị của RAG
1. **Kết quả đo đạc:** `NO_RAG` đạt `Correctness = 0,9688`, `Relevancy = 1,0000`, `Completeness = 0,9208`, `Contradiction = 0,0175`. Trong 8 cấu hình RAG, `A4` dẫn đầu `Correctness (0,9250)`, `Relevancy (0,9563)`, `Completeness (0,9167)`, `Contradiction (0,0000)`, `Complexity (1,0000)`, `Code (0,9500)`; `A1` dẫn đầu `Faithfulness (0,9250)` và `Citation Accuracy (0,8500)`.
2. **Cơ sở đối chiếu:** So 8 tổ hợp RAG trực tiếp với `NO_RAG` trên cùng 40 câu hỏi và cùng khung chấm mù 10 chỉ số (`N/A` bảo toàn).
3. **Biểu đồ/Artifact:** Hình 21, Hình 22 (Panel A); [table_05_generation_summary.csv](evaluation/artifacts/stage05/tables/table_05_generation_summary.csv), [rag_contribution.csv](evaluation/artifacts/stage04/ablation/rag_contribution.csv).
4. **Bằng chứng thống kê:** `A4 − NO_RAG` chênh `-0,0438` Correctness (`p = 0,2381`, ns), `-0,0042` Completeness (`p = 0,9009`, ns) — *không chứng minh tương đương*; ngược lại 7 cấu hình RAG còn lại (`A1–A3, B1–B4`) đều thấp hơn `NO_RAG` có ý nghĩa thống kê (`p ≤ 0,0006`).
5. **Diễn giải thực tiễn:** Trên miền DSA kinh điển, RAG chỉ mang lại giá trị dương rõ ràng (neo giáo trình + trích dẫn xác thực + `0%` mâu thuẫn ngữ cảnh mà không hy sinh nặng Correctness) khi retriever đủ mạnh (`A4`); retriever yếu (`B2`) làm hệ thống từ chối nhiều khiến chất lượng sụt nghiêm trọng.
6. **Giới hạn liên quan:** Controlled Agent (`flash/pro`), rủi ro family-level bias, `human_validated = false`, và Production Groq API vẫn `BLOCKED`.

### RQ7 — Hiệu năng, độ trễ và đánh đổi tài nguyên
1. **Kết quả đo đạc:** Warm `K=5, pool=120`: `A1 P95 = 607,5 ms` (`0 MB` VRAM), `B4 P95 = 3.214,3 ms` (`2,65 GB`), `A4 P95 = 3.693,1 ms` (`2,66 GB`). Stage Breakdown: rerank `B4` (`1.365 ms`) nhanh hơn `A4` (`2.320 ms`) `41,2%`; `QdrantLocal` SQLite tốn `~461–645 ms`/nhánh. Prod-Param `pool=10`: `A4 P95 = 1.230,9 ms`. Single-pass `pool=30`: `A4 mean = 1.628 ms` (`P95 = 1.777 ms`).
2. **Cơ sở đối chiếu:** So sánh tách bạch 5 giao thức đo độ trễ và biên Pareto 3 tiêu chí (`Quality`, `P95`, `VRAM`).
3. **Biểu đồ/Artifact:** Hình 16–20, Hình 23, Hình 25; [table_04_latency_protocols.csv](evaluation/artifacts/stage05/tables/table_04_latency_protocols.csv), [table_09_pareto_and_recommendations.csv](evaluation/artifacts/stage05/tables/table_09_pareto_and_recommendations.csv).
4. **Bằng chứng thống kê:** Dựa trên `14.400` lượt warm (`200` mẫu/combo/K, `0` lỗi), `640` lượt stage breakdown, `1.600` lượt prod-profile, `24` lượt cold-start; biên Pareto `{A1, B4, A4}` bất biến ở cả `K=5` và `K=15`.
5. **Diễn giải thực tiễn:** Hai đòn bẩy giảm độ trễ lớn nhất là giảm `candidate_top_k` từ `120 → 30` (giảm `~53%` thời gian trung bình mà giữ `99,8%` `nDCG@5`) và thay `QdrantLocal` SQLite bằng Qdrant Server (mục tiêu tương lai để cắt `~900 ms` overhead).
6. **Giới hạn liên quan:** Phụ thuộc phần cứng Windows + RTX 5060, chưa xóa OS disk cache khi đo cold-start, chứa `115` lần BM25 TTL rebuild tự nhiên, và chưa bao gồm độ trễ mạng/LLM HTTP API.

### RQ8 — Độ bền vững và tính năng học tập
1. **Kết quả đo đạc:** Robustness (`n=20`): OOD `4/4 (100%)`, Unanswerable `4/4 (100%)`, Prompt Injection `3/3 (100%)`, Distractor `2/3 (66,7%)`, Semantic Paraphrase `1/3 (33,3%, ΔEvRecall = -0,556)`, VI không dấu `0/3 (0%, ΔEvRecall = -0,667)`. Study Features: `11/11` real backend contract checks PASS; Controlled Agent Quiz (`n=8`, Correctness `0,688`, Grounding `0,625`), Roadmap (`n=4`, Correctness `0,625`, Grounding `0,438`).
2. **Cơ sở đối chiếu:** So sánh trực tiếp độ suy giảm `EvRecall@5` của ca biến đổi so với câu gốc (`linked_orig_ev_recall_k5`) và tách riêng kiểm thử contract thật với đánh giá nội dung Agent.
3. **Biểu đồ/Artifact:** Hình 27; [table_08_robustness_and_study.csv](evaluation/artifacts/stage05/tables/table_08_robustness_and_study.csv), [functional_test_results.json](evaluation/artifacts/stage04/study_features/functional_test_results.json).
4. **Bằng chứng thống kê:** Thống kê mô tả kèm cỡ mẫu tường minh (`n = 3–4`/nhóm robustness, `n = 4–8` study scenarios); không đủ cỡ mẫu để chạy kiểm định suy luận diện rộng.
5. **Diễn giải thực tiễn:** Cơ chế abstention chống ngoài miền và thiếu thông tin hoạt động rất tốt, nhưng hệ thống **chưa sẵn sàng cho người dùng thực tế gõ tiếng Việt không dấu** khi chưa bật `accent_insensitive` và cần hoàn thiện `guardrail.py` đang là scaffold `0 B`.
6. **Giới hạn liên quan:** Cỡ mẫu nhỏ (`n=20` robustness, `n=3` prompt injection không thể suy rộng thành an toàn bảo mật tổng thể); Quiz/Roadmap đánh giá bằng Controlled Agent, không phải Groq production.

---

## 20. Limitations and Threats to Validity

Báo cáo giữ nguyên toàn bộ 15 giới hạn bắt buộc và mối đe dọa tính hợp lệ, không che giấu kết quả bất lợi và **giữ nguyên trạng thái Stage 04 `AC-04 = PARTIAL`**:

1. **Cỡ mẫu câu hỏi đánh giá giới hạn:** `40` câu hỏi dùng chung (các phân nhóm nhỏ chỉ có `n = 3–18`), dẫn đến lực thống kê hạn chế đối với các chênh lệch vừa/nhỏ (`non-significant ≠ proven equivalence`).
2. **Nhãn liên quan do AI tạo (AI-generated Pooled Judgments):** Dựa trên Stage 03 LLM-as-a-Judge trên pool Top-60, không phải đánh giá toàn bộ corpus.
3. **Chưa xác nhận bởi chuyên gia con người:** `human_validated = false`.
4. **Trạng thái phê duyệt Ground Truth:** `human_approval_status = "DRAFT"`.
5. **Nguy cơ thiên vị cùng họ mô hình (Model-Family Bias):** Cả bộ sinh (`antigravity_agent_flash`) và bộ chấm mù (`antigravity_agent_pro`) thuộc cùng hệ sinh thái Agent (dù khác tier và đã kiểm tra `3/3` synthetic controls + `10%` audit).
6. **Không có tập kiểm chứng độc lập (No Held-Out Validation):** Mọi lựa chọn cấu hình (`A4`, `pool=30`, `K=5–10`, `budget=2.048`) đều là *exploratory selection* trên cùng 40 câu hỏi.
7. **Production Groq API ở trạng thái `BLOCKED`:** `backend/.env` chỉ chứa placeholder `GROQ_API_KEY`.
8. **Production HTTP End-to-End API ở trạng thái `BLOCKED`:** Không có API key thật và `data/runtime/qdrant_local` có `0` points (cấm ghi dữ liệu evaluation vào production storage); giữ nguyên Stage 04 **`AC-04 = PARTIAL`**.
9. **Controlled Agent Generation không phải Production Generation:** Không phản ánh chất lượng hay độ trễ sinh thực tế của mô hình Groq production.
10. **4 unit test lỗi sẵn trên Windows (Pre-existing Failures):** `tests/unit/test_llm_reliability.py::test_timeout_per_task_and_remaining_budget[*]` (`4` failures do sai số dấu phẩy động `time.monotonic()` trên Windows: `0.8000000000029104 <= 0.8`), giữ nguyên ngoài phạm vi Evaluation và không tuyên bố full pytest suite PASS.
11. **Cỡ mẫu Robustness nhỏ:** Chỉ `20` ca tổng cộng (`3–4` ca/nhóm), đặc biệt `3/3` ca prompt injection không chứng minh an toàn bảo mật tổng thể.
12. **Độ trễ phụ thuộc phần cứng và hệ điều hành:** Đo trên một máy Windows + RTX 5060 Laptop GPU, không khóa nhiệt độ/tiến trình nền và không xóa OS disk cache giữa các lượt cold-start.
13. **Hiệu ứng tái tạo cache BM25 (`TTL = 300s`):** `115` lần rebuild xảy ra tự nhiên trong phiên warm 5 giờ, làm kéo đuôi `Mean` và `P99` của nhóm BM25/Hybrid.
14. **Chi phí lưu trữ và truy vấn `QdrantLocal` SQLite:** Chiếm `~461–645 ms` mỗi nhánh truy vấn và `100,82 MiB` dung lượng vật lý dùng chung cho cả hai profile.
15. **Độ phủ bằng chứng nguồn một phần (Partial Source Evidence Coverage):** Ở `K=5`, ngay cả `A4` cũng chỉ đạt `55,0%` câu hỏi đủ `100%` KU bắt buộc (`PartialEv@5 = 0,7521` vs `CompleteEv@5 = 0,5500`); chỉ có bằng chứng một phần không được diễn giải là bằng chứng đầy đủ.

---

## 21. Future Work

Các đề xuất cải tiến dưới đây **không được triển khai vào mã nguồn production trong Stage 05** mà được ghi nhận thành lộ trình kỹ thuật tiếp theo dựa trực tiếp trên bằng chứng đo đạc:

1. **Hỗ trợ truy vấn tiếng Việt không dấu cho BM25:** Bật `accent_insensitive = true` hoặc dùng chỉ mục hai trường (có dấu + bỏ dấu) để khắc phục lỗi `0,0%` (`-0,6667 EvRecall@5`) trên câu hỏi không dấu/sai chính tả.
2. **Chuyển từ `QdrantLocal` SQLite sang Qdrant Server (HNSW) và tối ưu cache BM25:** Nhằm loại bỏ `~461–645 ms`/nhánh overhead của SQLite và tránh `115` lần rebuild đồng bộ do TTL `300s`.
3. **Cải tiến `structure_adaptive` chunking:** Gộp các đơn vị cấu trúc quá ngắn (mục tiêu `80–120` tokens tối thiểu) kèm ngữ cảnh tiêu đề cha để vừa giữ `0%` cắt cụt embedding vừa khắc phục phân mảnh ngữ cảnh.
4. **Hoàn thiện lớp an toàn `guardrail.py` và nâng độ bám nguồn cho Study Features:** Triển khai thực tế `backend/app/safety/guardrail.py` (hiện `0 B`) và tăng kiểm chứng trích dẫn cho Quiz/Roadmap (`grounding = 0,44–0,63`).
5. **Xác nhận Ground Truth bởi chuyên gia & Benchmark Production thật:** Thực hiện human audit (chuyển `human_validated = false → true`), đánh giá trên tập held-out mới và chạy benchmark HTTP E2E với Groq API thật khi được cấp quyền.

---

## 22. Final Evaluation Conclusion

Qua toàn bộ 5 giai đoạn của lộ trình Evaluation (`Stage 01` → `Stage 05`), hệ thống **DSA Document Assistant Using Hybrid RAG** đã được đánh giá một cách định lượng, minh bạch và có khả năng tái lập hoàn toàn:

- **Kiến trúc truy xuất Hybrid + Cross-Encoder Reranking trên `recursive_512_100` (`A4`)** chứng minh hiệu quả cao nhất về xếp hạng (`nDCG@5 = 0,7053`, `MRR@5 = 0,9875`), độ phủ bằng chứng nguồn (`EvRecall@15 = 0,9083`, `CompleteEv@15 = 80,0%`) và độ chính xác câu trả lời trong nhóm RAG (`0,9250`, `0%` mâu thuẫn), nhờ `BGE-reranker-v2-m3` khắc phục hoàn toàn điểm yếu cắt cụt embedding và lọc nhiễu hiệu quả từ tập ứng viên RRF.
- **Tuy nhiên, không có cấu hình toàn năng duy nhất:** `A1 (Recursive + BM25)` là điểm tối ưu Pareto cho môi trường không GPU/độ trễ thấp (`P95 = 607,5 ms`, `0 MB` VRAM, `Faithfulness = 0,9250`, `Cit. Acc. = 0,8500`); `B4 (Adaptive + Rerank)` là điểm tối ưu Pareto trung gian không cắt cụt embedding và rerank nhanh hơn `41,2%`; còn **`A4` với `candidate_top_k = 30`, `Top-K = 5–10`, `token_budget = 2.048`** nổi lên như ứng viên cân bằng thực dụng nhất ở mức *thăm dò* (`99,8%` `nDCG@5` với nửa độ trễ).
- **Tính trung thực khoa học được bảo toàn tuyệt đối:** Báo cáo ghi nhận rõ `NO_RAG` đạt `Correctness` cao hơn `A4` trên miền DSA tổng quát (đổi lại không có trích dẫn và có tỷ lệ mâu thuẫn cao hơn), lỗ hổng nghiêm trọng với tiếng Việt không dấu (`0%`), trạng thái `human_validated = false`, và trạng thái `BLOCKED` của Production Groq API & HTTP E2E (giữ nguyên Stage 04 `AC-04 = PARTIAL`).

---

## 23. Presentation-Ready Results

Để phục vụ báo cáo bảo vệ đồ án hoặc tổng kết nhóm *(không tạo slide deck riêng theo yêu cầu)*, **8 biểu đồ cốt lõi nhất** đại diện cho toàn bộ câu chuyện thực nghiệm gồm:

1. **Hình 1 & Hình 4 ([fig01](evaluation/artifacts/stage05/figures/png/fig01_chunk_count_and_storage.png), [fig04](evaluation/artifacts/stage05/figures/png/fig04_truncation_rate_and_discarded_tokens.png)):** Nghịch lý Chunking — Adaptive triệt tiêu `43,2%` cắt cụt MiniLM nhưng tạo `2,1×` chunk ngắn, trong khi BGE Reranker có `0%` cắt cụt ở cả hai.
2. **Hình 9 ([fig09](evaluation/artifacts/stage05/figures/png/fig09_ndcg_at_k.png)):** Bức tranh tổng thể `nDCG@K` của 8 tổ hợp — `A4 (0,705)` dẫn đầu rõ rệt, `BM25` vượt mạnh `Dense`, Reranker tạo bước nhảy lớn nhất (`+40%`).
3. **Hình 15 ([fig15](evaluation/artifacts/stage05/figures/png/fig15_complete_vs_partial_evidence_coverage.png)):** Độ phủ bằng chứng nguồn xác thực — `A4` đạt `80,0% Complete Evidence` ở `K=15` so với `47,5%` của `B4`.
4. **Hình 18 ([fig18](evaluation/artifacts/stage05/figures/png/fig18_retrieval_latency_breakdown.png)):** Phân rã độ trễ — giải thích chính xác hai nút cổ chai (`BGE Rerank` và `QdrantLocal SQLite ~461–645 ms`) cùng mức nhanh hơn `41,2%` của `B4` ở riêng tầng rerank.
5. **Hình 21 & Hình 22 ([fig21](evaluation/artifacts/stage05/figures/png/fig21_answer_quality_comparison.png), [fig22](evaluation/artifacts/stage05/figures/png/fig22_rag_vs_norag_and_ablation_study.png)):** Sinh câu trả lời & Ablation — công bố minh bạch `NO_RAG > A4` về Correctness, giá trị grounding/`0%` contradiction của `A4`, và kiểm định Holm cho từng thành phần.
6. **Hình 23 ([fig23](evaluation/artifacts/stage05/figures/png/fig23_candidate_pool_and_topk_sensitivity.png)):** Điểm uốn kỹ thuật — **`A4 @ pool=30`** giữ **`99,8%` `nDCG@5`** trong khi cắt quá nửa độ trễ (`3.486 → 1.628 ms`).
7. **Hình 25 ([fig25](evaluation/artifacts/stage05/figures/png/fig25_quality_latency_pareto_tradeoff.png)):** Biên tối ưu Pareto `{A1, B4, A4}` theo Chất lượng – Độ trễ P95 – GPU VRAM.
8. **Hình 27 ([fig27](evaluation/artifacts/stage05/figures/png/fig27_robustness_and_study_features.png)):** Độ bền vững & khoảng trống thực tế — `100%` OOD/Unanswerable nhưng `0%` tiếng Việt không dấu (định hướng cải tiến ưu tiên số 1).

---

## 24. Reproducibility Instructions

Toàn bộ `28` biểu đồ (`PNG 300 DPI + SVG`), `10` bảng tổng hợp, các tệp manifest và kiểm định chất lượng tự động có thể tái tạo từ các bảng tổng hợp Stage 04 mà không cần GPU hay API key.

> **Lưu ý quan trọng về phạm vi kho Git và tệp raw cục bộ:** Một bản `git clone` sạch từ repository chỉ chứa mã nguồn, báo cáo, 28 biểu đồ, 10 bảng tổng hợp và các tệp summary/per-query đã commit; **bản clone sạch KHÔNG chứa các tệp raw dung lượng lớn đã loại trừ khỏi Git**. Để chạy kiểm định toàn vẹn đầy đủ (`verify_inputs_and_integrity`) hoặc tái lập hoàn toàn từ dữ liệu thô, cần có các tệp cục bộ sau (được bảo toàn cục bộ và sao lưu kiểm chứng SHA-256 trong `data/runtime/eval_stage04_backups/`):
> - Hai frozen corpora Stage 02: `evaluation/artifacts/recursive_512_100/chunks.jsonl` (`6.366` chunks) và `evaluation/artifacts/structure_adaptive/chunks.jsonl` (`13.390` chunks).
> - Năm tệp raw Stage 04 được sao lưu trong `data/runtime/eval_stage04_backups/` (kèm `backup_checksums.json`): `packed_contexts_main.jsonl` (`3,28 MB`), `latency_raw.jsonl` (`3,02 MB`), `retrieval_rankings.jsonl` (`1,57 MB`), `generated_answers.jsonl` (`1,31 MB`), và `retrieval_metrics_per_query.csv` (`544 KB`).
> - Các tệp chuẩn bị trung gian Stage 04 (`packed_contexts_budget.jsonl`, `robustness_prepared.jsonl`, `study_prepared.json`) và chỉ mục `data/runtime/qdrant_eval` nếu muốn tái chạy benchmark truy xuất Stage 04.

```powershell
Set-Location 'D:\DSA-Document-Assistant-Using-Hybrid-RAG'

# 1. Sinh lại toàn bộ 28 biểu đồ (56 tệp PNG+SVG), 10 bảng CSV, manifests và chạy QA
.\.venv\Scripts\python.exe evaluation/scripts/generate_final_visualizations.py --mode all

# 2. Chạy kiểm định tự động ở chế độ chỉ-đọc (read-only, không ghi đè manifest đã commit)
.\.venv\Scripts\python.exe evaluation/scripts/generate_final_visualizations.py --mode validate --require-report

# 3. Chạy bộ kiểm thử đơn vị Stage 05 và hồi quy Evaluation (không sửa production)
.\.venv\Scripts\python.exe -m pytest tests/unit/test_stage05_visualizations.py tests/unit/test_comprehensive_evaluation.py tests/unit/test_retrieval_metrics.py tests/unit/test_ground_truth.py -q
```

---

## 25. Artifact and Integrity Summary

### 25.1 Đối chiếu và Hòa giải Tiêu chí Nghiệm thu Stage 04 (`5 PASS, 1 PARTIAL`)

Bảng dưới đây đối chiếu tường minh tên gọi chuẩn hóa gốc (Stage 04 Section 27 / `STAGE_04_COMPREHENSIVE_EVALUATION_REPORT.md` Mục 21) với nhãn rút gọn từng xuất hiện trong tóm tắt checkpoint, bảo toàn nguyên vẹn ngữ nghĩa gốc và kết quả `5 PASS / 1 PARTIAL` ([table_10_stage04_ac_reconciliation.csv](evaluation/artifacts/stage05/tables/table_10_stage04_ac_reconciliation.csv)):

| AC ID | Tên chuẩn hóa gốc Stage 04 | Nhãn rút gọn ở bản tóm tắt checkpoint | Trạng thái | Ý nghĩa nghiệm thu được bảo toàn |
|---|---|---|---|---|
| **AC-01** | Benchmark Infrastructure & Integrity | Chunking & Retrieval Quality | **PASS** | Đầu vào đóng băng bất biến, nâng cấp `retrieval_metrics.py` tương thích ngược, `0` sửa đổi production. |
| **AC-02** | Retrieval & Evidence Evaluation | Evidence Coverage | **PASS** | Đủ 8 tổ hợp × 40 câu × 9 mốc K × 8 chỉ số retrieval + 6 chỉ số evidence, `0` drift, `Judged@K = 1,0`. |
| **AC-03** | Performance Evaluation | Latency & Resource Benchmark | **PASS** | `14.400` warm + `640` stage + `1.600` prod-profile + `24` cold-start + RAM/VRAM đầy đủ. |
| **AC-04** | Generation, Robustness & Study Features | Generation & Production E2E | **PARTIAL** | Controlled Agent (`400` answers + audit), `20` robustness, `11/11` contract checks & Quiz/Roadmap hoàn tất; Production Groq API & HTTP E2E **BLOCKED**. |
| **AC-05** | Comprehensive Experimental Analysis | Robustness & Study Features | **PASS** | Ablation, 3 độ nhạy (Top-K/Pool/Budget), bootstrap `10.000` + Holm, Error Analysis, Pareto frontier. |
| **AC-06** | Regression, Reproducibility & Final Handover | Statistics, Trade-Off & Data Contract | **PASS** | `53/53` targeted PASS (`0` regression mới), `pip check = 0`, validator sạch, Stage 05 Data Contract sẵn sàng. |

### 25.2 Danh mục Artifacts Stage 05 & Kiểm tra Toàn vẹn

- **Entrypoint & Module tái lập:** [generate_final_visualizations.py](evaluation/scripts/generate_final_visualizations.py), `evaluation/stage05/{__init__,style,data_loader,charts_chunking,charts_retrieval_evidence,charts_performance,charts_generation_ablation,charts_analysis,tables_and_manifests,validator}.py`.
- **Thư mục đầu ra Stage 05:**
  - `evaluation/artifacts/stage05/figures/png/` — `28` tệp PNG `300 DPI` (`fig01`–`fig28`).
  - `evaluation/artifacts/stage05/figures/svg/` — `28` tệp vector SVG (`fig01`–`fig28`).
  - `evaluation/artifacts/stage05/tables/` — `10` bảng CSV truy nguyên (`table_01`–`table_10`).
  - [figure_manifest.json](evaluation/artifacts/stage05/figure_manifest.json) — Đăng ký 28 biểu đồ, nguồn dữ liệu, giao thức, SHA-256 và chú thích tiếng Việt.
  - [configuration_recommendations.json](evaluation/artifacts/stage05/configuration_recommendations.json) — Khuyến nghị thăm dò theo 4 kịch bản + lộ trình tương lai.
  - [chart_validation.json](evaluation/artifacts/stage05/chart_validation.json) — Kết quả QA tự động trên 28 PNG, 28 SVG, dữ liệu nguồn, `N/A` và liên kết báo cáo.
  - [manifest.json](evaluation/artifacts/stage05/manifest.json) — Manifest tổng thể Stage 05 và chứng thực checksum đầu vào Stage 02/03/04 + 5 bản sao lưu cục bộ.
- **Kiểm tra Tiêu chí Nghiệm thu Stage 05 (`AC-01` – `AC-06`):**
  - **AC-01 (Data Integrity):** PASS — Git baseline (`feat/final-readiness`, `HEAD = 966ecfa4...`, `main = b84f0945...`), 9 checksums đóng băng, 59 artifacts Stage 04 và 5 backups cục bộ bất biến.
  - **AC-02 (Visualization Completeness):** PASS — Đủ `28/28` biểu đồ logic Data Contract (`56` tệp `PNG+SVG`), phủ trọn 11 nhóm `A`–`K`, đạt `300 DPI` và đã kiểm tra trực quan.
  - **AC-03 (Comparative Analysis):** PASS — Phân tích đầy đủ 8 tổ hợp `A1–A4, B1–B4` và trả lời cấu trúc 6 phần cho `RQ1`–`RQ8`.
  - **AC-04 (Statistical & Scientific Validity):** PASS — Bảo toàn bootstrap `10.000`, `Holm p`, `N/A`, tách biệt 5 giao thức độ trễ, công bố kết quả bất lợi (`NO_RAG > A4`, `0%` typo VI) và giữ Stage 04 `AC-04 = PARTIAL`.
  - **AC-05 (Final Report & Recommendations):** PASS — Báo cáo 25 mục hoàn chỉnh bằng tiếng Việt cùng khuyến nghị thăm dò đa kịch bản.
  - **AC-06 (Reproducibility & Handover):** PASS — Mã nguồn tái lập, unit tests, manifests, kiểm định tự động và `100%` liên kết tương đối hợp lệ.
