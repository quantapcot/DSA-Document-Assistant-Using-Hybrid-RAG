# STAGE 03 – DATASET & GROUND TRUTH REPORT

> **Dự án:** DSA Document Assistant Using Hybrid RAG
> **Giai đoạn:** `Stage 03` – Dataset & Ground Truth *(Preparation + Fully Automated LLM-as-a-Judge Completion)*
> **Workspace:** `D:\DSA-Document-Assistant-Using-Hybrid-RAG`
> **Branch:** `feat/final-readiness`
> **Accepted HEAD (bất biến):** `0012a0cdfbda39c9b5117ee91e85f60e0bad23d7`
> **Local `main` (bất biến):** `b84f09450f213b03ad57ef8b388f482ee0f0eaa2`
> **Dataset Version:** `stage03_dsa_40q_v1.1_ai_reviewed` *(nâng từ `v1` với 5 minimal fixes, 0 query text changes)*
> **Human Validation:** **`FALSE` (`human_validated = false`, `human_approval_status = "DRAFT"`)**
> **Ngày hoàn tất:** 2026-11-28

---

## PART I – HISTORICAL PREPARATION RESULTS (Pre-Automated `v1` Baseline)

## 1. Executive Summary

Stage 03 đã hoàn tất toàn bộ khâu chuẩn bị dữ liệu, pooling Top-60/Pool-120, xây dựng workbook gán nhãn mù phương pháp (blind-method) và Ground Truth Importer trên hai frozen benchmark corpora từ Stage 02 mà **không sửa production code**, **không rebuild corpus/index**, và **dừng chính xác tại Human Annotation Gate**:

- **Shared 40-Question Dataset (`Q001–Q040`, trạng thái `DRAFT`):** Đúng **40 câu hỏi chung** dùng thống nhất cho cả Profile A (`recursive_512_100`) và Profile B (`structure_adaptive`), gồm **24 tiếng Việt + 16 tiếng Anh** (độc lập, không dịch nguyên nghĩa), **12 Easy + 18 Medium + 10 Hard**, đủ **8/8 chủ đề DSA** và **10/10 dạng câu hỏi** (mỗi dạng 4 câu), đi kèm **40 Reference Answers**, **91 Required Knowledge Units (KUs)** và **107 canonical source evidence records** khớp tuyệt đối tọa độ `[char_start, char_end)` trên `canonical_fingerprint_sha256 = 712f3847...`.
- **320 Official Pooling Calls (`top_k=60`, `candidate_top_k=120`):** Hoàn tất `320/320` lượt truy xuất trên **RTX 5060 GPU** (`cuda:0`, `batch_size=4`, `max_length=512`, `0` OOM, `0` CPU fallback, `0` lỗi, `0` lượt trả `< 60` kết quả):
  - **Raw candidates:** `19,200` (`9,600` Profile A + `9,600` Profile B).
  - **Unique retrieval candidates:** `9,011` (`4,515` Profile A + `4,496` Profile B; loại bỏ `10,189` trùng lặp, tỷ lệ deduplication **`53.07%`**).
  - **Evidence-added candidates:** `57` (`11` Profile A + `46` Profile B; áp dụng chính sách đồng đều trên mọi đoạn `source_backed`).
  - **Final annotation workload:** **`9,068` rows** (`4,526` Profile A + `4,542` Profile B), **100% ô `Relevance Grade` để trống (`UNJUDGED`)**.
- **Human Annotation Workbook & Importer:** Xuất thành công [ground_truth_annotation.xlsx](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/ground_truth_annotation.xlsx) (`2,271,141` B, `5` sheets, ZIP integrity `PASSED`, blind-method verified) cùng Ground Truth Importer ([importer.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/stage03/importer.py), [import_ground_truth.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/scripts/import_ground_truth.py)) được kiểm thử đầy đủ `14/14` unit tests trên synthetic fixtures (chưa sinh `ground_truth_recursive.jsonl` và `ground_truth_adaptive.jsonl`).
- **Zero Regression & Integrity:** Full `pytest -q` đạt **`308 passed, 0 failed, 0 skipped, 1 warning (fitz)`** (`PYTEST_EXIT=0`), `pip check` exit `0`, toàn bộ checksums raw/production/frozen corpora/Qdrant và Git baseline bất biến 100%.

---

## 2. Frozen Benchmark Verification

Kiểm tra đọc trước và sau Stage 03 xác nhận toàn bộ benchmark đã đóng băng ở Stage 02 còn nguyên vẹn tuyệt đối:

| Thành phần | Kỳ vọng (Stage 02 Frozen) | Thực tế xác minh (Stage 03) | Trạng thái |
|---|---|---|---|
| **Canonical Source (`54` files, `1,733` pages)** | `712f3847a6370156258f18297eabb41a657b4ac109dbe6290b9b50d9a0c6bd1d` | `712f3847a6370156258f18297eabb41a657b4ac109dbe6290b9b50d9a0c6bd1d` | **IDENTICAL** |
| **Frozen Corpus A (`recursive_512_100`)** | `6,366` chunks, `e39811da17d99201e4ad26f54c4a53614a087e76e3e0cb6d031a733862af0649` | `6,366` chunks, `e39811da17d99201e4ad26f54c4a53614a087e76e3e0cb6d031a733862af0649` | **IDENTICAL** |
| **Frozen Corpus B (`structure_adaptive`)** | `13,390` chunks, `81c568275754e35d7323b1e51fc247b885abc873eef4c5caa5c1ec4daa0b5d63` | `13,390` chunks, `81c568275754e35d7323b1e51fc247b885abc873eef4c5caa5c1ec4daa0b5d63` | **IDENTICAL** |
| **Isolated Qdrant (`eval_dsa_chunks_v1`)** | `19,756 = 6,366 + 13,390` points, `check_model_exhaustive = PASSED` | `19,756 = 6,366 + 13,390` points, `check_model_exhaustive = PASSED` | **IDENTICAL** |
| **Top-60 / Candidate-120 Capacity Smoke** | 8/8 combos (`A1–A4`, `B1–B4`) bằng smoke queries cũ, `0` prod code edits | `8/8 PASS` (`60/60` hits, `cuda:0`, `0` OOM, `0` leak, `0` prod edits) | **PASS** |

---

## 3. Final 40-Question Distribution

File dữ liệu: [evaluation/datasets/questions_40.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/questions_40.jsonl) (`SHA-256 = c0e9c7b234e9b8a95f4ec4f18f608b13d2bf2ac1a209325a36f3d459ef603020`). Tất cả 40 câu hỏi đều ở trạng thái `human_approval_status = "DRAFT"`, không tái sử dụng 4 smoke queries của Stage 02.

### 3.1. Phân bố theo Chủ đề, Ngôn ngữ, Độ khó & Loại câu hỏi

| Topic | VI | EN | Total | Easy | Medium | Hard |
|---|---:|---:|---:|---:|---:|---:|
| Array & Linked List | 3 | 2 | **5** | 2 | 2 | 1 |
| Stack & Queue | 3 | 2 | **5** | 2 | 3 | 0 |
| Tree, BST & AVL | 4 | 2 | **6** | 2 | 2 | 2 |
| Graph Algorithms | 4 | 2 | **6** | 0 | 3 | 3 |
| Searching & Sorting | 4 | 2 | **6** | 1 | 5 | 0 |
| Time & Space Complexity | 2 | 2 | **4** | 2 | 1 | 1 |
| Hashing | 2 | 2 | **4** | 2 | 1 | 1 |
| Algorithm Design Techniques | 2 | 2 | **4** | 1 | 1 | 2 |
| **Total** | **24** | **16** | **40** | **12** | **18** | **10** |

**Phân bố 10 Question Types (4 câu mỗi loại = 40):** `definition (4)`, `concept_explanation (4)`, `algorithm_steps (4)`, `comparison (4)`, `time_complexity (4)`, `space_complexity (4)`, `code_interpretation (4)`, `pseudocode_interpretation (4)`, `application_scenario (4)`, `multi_evidence_reasoning (4)`.

### 3.2. Danh sách đầy đủ 40 Shared Questions (`Q001–Q040`)

| ID | Topic | Lang | Diff | Type | Query | Source Docs |
|---|---|---|---|---|---|---|
| Q001 | Array & Linked List | vi | easy | definition | Mảng động (Dynamic Array) lưu trữ dữ liệu trong bộ nhớ như thế nào và cơ chế mở rộng dung lượng hoạt động ra sao? | `vie_docx/003` (p.2), `vie_pdf/020` (p.3) |
| Q002 | Array & Linked List | vi | medium | comparison | So sánh ưu điểm và nhược điểm giữa mảng và danh sách liên kết về tổ chức bộ nhớ, truy cập ngẫu nhiên và chèn xóa phần tử. | `vie_docx/003` (p.2–3) |
| Q003 | Array & Linked List | vi | easy | code_interpretation | Trong C++, cấu trúc một Node của danh sách liên kết đơn được khai báo như thế nào và cần lưu ý gì về quản lý bộ nhớ khi tạo node bằng toán tử new? | `vie_docx/005` (p.3) |
| Q004 | Array & Linked List | en | medium | concept_explanation | How does a Doubly Linked List differ structurally from a Singly Linked List, and what traversal capability does it enable? | `end_docx/005` (p.3) |
| Q005 | Array & Linked List | en | hard | time_complexity | Why does shrinking a Dynamic Array immediately when fill ratio drops to r = 1/2 hurt alternating insertions and deletions, and what threshold policy achieves amortized O(1) time with Theta(n) space? | `eng_pdf/002` (p.4) |
| Q006 | Stack & Queue | vi | easy | definition | Hàng đợi hai đầu (Deque) khác gì hàng đợi thông thường và hỗ trợ những thao tác cơ sở nào? | `vie_pdf/008` (p.17) |
| Q007 | Stack & Queue | vi | medium | pseudocode_interpretation | Trong mã giả cài đặt Queue bằng danh sách móc nối với hai con trỏ F (front) và R (rear), thao tác ENQUEUE và DEQUEUE xử lý trường hợp danh sách rỗng hoặc chỉ có một phần tử như thế nào? | `vie_pdf/008` (p.16) |
| Q008 | Stack & Queue | vi | medium | code_interpretation | Tại sao khi cài đặt Queue (FIFO) trong Python nên dùng collections.deque thay vì list, và trong C++ cần cấu hình std::priority_queue như thế nào để tạo Min-Heap? | `vie_docx/005` (p.4), `vie_docx/007` (p.4) |
| Q009 | Stack & Queue | en | medium | comparison | Compare implementing a Priority Queue with an unordered dynamic array versus a sorted dynamic array in terms of insert and delete_max complexities, and identify which sorting algorithm each corresponds to. | `eng_pdf/008` (p.2) |
| Q010 | Stack & Queue | en | easy | application_scenario | What are the governing ordering principles and classic practical applications of Stacks versus Queues? | `end_docx/005` (p.4) |
| Q011 | Tree, BST & AVL | vi | easy | definition | Trình bày thứ tự thăm nút của ba phép duyệt cây nhị phân (Duyệt trước, Duyệt giữa, Duyệt sau) và phép duyệt nào cho ra dãy khóa tăng dần trên BST? | `vie_docx/004` (p.3), `vie_pdf/021` (p.70) |
| Q012 | Tree, BST & AVL | vi | hard | multi_evidence_reasoning | Trên cây tìm kiếm 2-3, giải thuật SEARCH-2-3(T, X) rẽ nhánh qua các con trỏ LPTR, MPTR, RPTR như thế nào và các trường hợp ban đầu khi bổ sung nút mới là gì? | `vie_pdf/014` (p.3–4) |
| Q013 | Tree, BST & AVL | vi | easy | application_scenario | Tại sao Heap thường được cài đặt bằng mảng thay vì con trỏ và quy tắc biểu diễn cây nhị phân hoàn chỉnh lên mảng thực hiện như thế nào? | `vie_docx/004` (p.4), `vie_pdf/007` (p.22) |
| Q014 | Tree, BST & AVL | vi | medium | code_interpretation | Trong hàm insert_case5 của cây Đỏ-Đen (Red-Black Tree), khi nút cha P là đỏ, nút bác U là đen và nút mới N nằm cùng phía với P so với nút ông G, hàm thực hiện đổi màu và phép quay nào? | `vie_pdf/004` (p.121) |
| Q015 | Tree, BST & AVL | en | medium | concept_explanation | In a binary tree, how do you find the traversal-order successor of a node X, and how is a non-leaf node X deleted without violating traversal order? | `eng_pdf/010` (p.2–3) |
| Q016 | Tree, BST & AVL | en | hard | multi_evidence_reasoning | In an AVL tree, how is node skew defined, why does height-balance guarantee O(log n) height, and what rotations rebalance a node B with skew +2 depending on the skew of its right child F? | `eng_pdf/009` (p.2–3) |
| Q017 | Graph Algorithms | vi | medium | comparison | So sánh tính chất của ma trận kề trên đồ thị vô hướng với đồ thị có hướng, và nêu ưu điểm của biểu diễn đồ thị bằng danh sách kề. | `vie_pdf/021` (p.105, 107) |
| Q018 | Graph Algorithms | vi | hard | algorithm_steps | Trình bày thuật toán DFS tìm tất cả các cạnh cầu (bridge) của đồ thị vô hướng offline bằng hai mảng tin[] và fup[], kèm điều kiện nhận biết cầu và độ phức tạp. | `vie_pdf/002` (p.3–4) |
| Q019 | Graph Algorithms | vi | medium | pseudocode_interpretation | Trong giải thuật mã giả DFS(G, v), hai loại cung DISCOVERY và BACK được gán nhãn dựa trên điều kiện nào khi duyệt các cung liên thuộc với đỉnh v? | `vie_pdf/016` (p.9) |
| Q020 | Graph Algorithms | vi | hard | code_interpretation | Trong đoạn code C++ cài đặt thuật toán Dijkstra sử dụng priority_queue<pair<int,int>>, tại sao lại lấy cur_d = -q.top().first và kiểm tra điều kiện if (cur_d > d[v]) continue? | `vie_pdf/004` (p.188), `vie_docx/007` (p.4) |
| Q021 | Graph Algorithms | en | medium | space_complexity | Why does an adjacency-list graph representation require Theta(\|V\| + \|E\|) space, and why does storing a Shortest Paths Tree via parent pointers take O(\|V\|) instead of Omega(\|V\|^2) space? | `eng_pdf/007` (p.2–3) |
| Q022 | Graph Algorithms | en | hard | algorithm_steps | How does Bellman-Ford use graph duplication into \|V\| + 1 levels and negative-cycle witnesses to compute shortest paths and detect vertices with distance -infinity? | `eng_pdf/019` (p.2–3) |
| Q023 | Searching & Sorting | vi | easy | algorithm_steps | Trình bày các bước hoạt động của thuật toán Sắp xếp chèn (Insertion Sort) theo cách ví von xếp các lá bài. | `vie_pdf/021` (p.23) |
| Q024 | Searching & Sorting | vi | medium | pseudocode_interpretation | Giải thích hoạt động của hàm đệ quy BINARY-SEARCH(A, l, r, key) và điều kiện nào của dữ liệu đầu vào là bắt buộc để thuật toán này đúng? | `vie_pdf/013` (p.4) |
| Q025 | Searching & Sorting | vi | medium | time_complexity | Trong giải thuật Bubble Sort (Procedure BUBBLE-SORT), số phép so sánh và đổi chỗ ở trường hợp tốt nhất và xấu nhất là bao nhiêu? | `vie_pdf/012` (p.10) |
| Q026 | Searching & Sorting | vi | medium | comparison | So sánh tính ổn định (stability), bộ nhớ phụ và độ phức tạp trường hợp xấu nhất giữa Quick Sort và Merge Sort. | `vie_pdf/021` (p.33, 35–36) |
| Q027 | Searching & Sorting | en | medium | algorithm_steps | How does Radix Sort combine base-n tuple representation with Counting Sort, and why must the auxiliary Counting Sort be stable and process digits from least significant to most significant? | `eng_pdf/005` (p.3–4) |
| Q028 | Searching & Sorting | en | medium | time_complexity | Why is the worst-case running time of any comparison-based sorting algorithm lower bounded by Omega(n log n)? | `eng_pdf/005` (p.2) |
| Q029 | Time & Space Complexity | vi | easy | concept_explanation | Phân biệt ba ký hiệu tiệm cận Big O, Big Omega và Big Theta trong đánh giá hiệu năng thuật toán, và quy tắc rút gọn hàm f(n) theo Big O là gì? | `vie_docx/006` (p.2), `vie_pdf/019` (p.5) |
| Q030 | Time & Space Complexity | vi | easy | space_complexity | Độ phức tạp không gian đo lường đại lượng nào, và sự khác biệt về cấp phát bộ nhớ giữa Stack Memory và Heap Memory cùng nguyên nhân gây ra lỗi Stack Overflow khi đệ quy là gì? | `vie_docx/006` (p.2–4) |
| Q031 | Time & Space Complexity | en | medium | time_complexity | What is the formal definition of amortized cost T(n) over a sequence of operations, and why does inserting into a Dynamic Array achieve Theta(1) amortized time even though individual reallocations cost Theta(n)? | `eng_pdf/002` (p.3–4) |
| Q032 | Time & Space Complexity | en | hard | space_complexity | How are complexity classes P, NP, NP-hard, and NP-complete formally defined, and how do polynomial-time reductions establish that a problem is NP-complete? | `eng_pdf/012` (p.2–3) |
| Q033 | Hashing | vi | easy | definition | Đụng độ (collision) trong bảng băm là gì và phương pháp nối kết trực tiếp (chaining) giải quyết đụng độ như thế nào? | `vie_pdf/021` (p.126), `vie_docx/004` (p.2) |
| Q034 | Hashing | vi | medium | application_scenario | Tại sao dò tuyến tính (Linear Probing) dễ gây phân bố dồn cụm, và dò bậc hai (Quadratic Probing) cùng băm kép (Double Hashing) khắc phục bằng công thức hàm băm lại nào? | `vie_pdf/021` (p.130), `vie_pdf/014` (p.21–22) |
| Q035 | Hashing | en | easy | concept_explanation | Why must collisions inevitably occur when hashing a large key universe u into a table of size m << u, and why does a fixed division hash function h(k) = k mod m have a Theta(n) worst-case chain? | `eng_pdf/004` (p.2–3) |
| Q036 | Hashing | en | hard | space_complexity | Why is a Direct Access Array impractical in space when n << u, and how does Universal Hashing H(p, m) bound expected chain length to O(1) while using Theta(n) space? | `eng_pdf/004` (p.2, 4) |
| Q037 | Algorithm Design Techniques | vi | hard | multi_evidence_reasoning | Kỹ thuật Quy hoạch động bao lồi (Convex Hull Trick) giải quyết bài toán cực trị tập hàm bậc nhất như thế nào, giảm độ phức tạp từ bao nhiêu xuống bao nhiêu và áp dụng ra sao cho bài toán Commando (APIO 2010)? | `vie_docx/002` (p.2–4, gồm DOCX pipe table) |
| Q038 | Algorithm Design Techniques | vi | easy | application_scenario | Ý tưởng chia để trị (Divide and Conquer) trong Quick Sort thực hiện phân hoạch mảng xung quanh phần tử trục (pivot) như thế nào và điều kiện dừng đệ quy là gì? | `vie_pdf/021` (p.31) |
| Q039 | Algorithm Design Techniques | en | medium | pseudocode_interpretation | In the bottom-up dynamic programming algorithm lcs(A, B) for Longest Common Subsequence, what do the subproblems x[i][j], the recurrence relation, the base cases, and the loop iteration order represent? | `eng_pdf/015` (p.2–3) |
| Q040 | Algorithm Design Techniques | en | hard | multi_evidence_reasoning | In Dynamic Programming for Longest Increasing Subsequence (LIS), why does simply defining subproblems on suffixes A[i:] fail, what extra constraint fixes it, and how are the recurrence and final answer formulated? | `eng_pdf/015` (p.4) |

---

## 4. Reference Answer and Source Evidence Validation

- **Artifacts:** [source_evidence.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/source_evidence.jsonl) (`SHA-256 = 53ffa1415b4c9cdac2cedd34746d55d2b3927eba39d657fffa23832b0be2235a`), [dataset_manifest.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/dataset_manifest.json).
- **Canonical Coordinate Verification:** `107 / 107` evidence records được định vị tự động qua `locate_canonical_spans()` khớp nguyên văn 100% trên `canonical_text` của 54 tài liệu (`canonical_fingerprint_sha256 = 712f3847...`), mang đủ `document_id`, `source_revision`, `source_path`, `canonical_page`, `source_spans` (`page`, `block_id`, `char_start`, `char_end`) và `evidence_text`.
- **Evidence Modes:** Phân biệt rõ `single`, `alternative` (vd `Q001_KU1`, `Q011_KU1`, `Q033_KU1`, `Q033_KU2` có nhiều nguồn tương đương) và `composite` (vd `Q002_KU2`, `Q016_KU3`, `Q017_KU1`, `Q020_KU1`, `Q026_KU2`, `Q027_KU2`, `Q034_KU1`, `Q037_KU2/KU3`, `Q039_KU1/KU3` cần kết hợp nhiều spans, bao gồm bảng DOCX `| Commando.inp | Commando.out |` ở `Q037`).
- **Frozen Corpus KU Retention:**
  - Profile A (`recursive_512_100`): **`91 / 91` KUs retained** (`missing_ku_count = 0`).
  - Profile B (`structure_adaptive`): **`91 / 91` KUs retained** (`missing_ku_count = 0`).

---

## 5. Pooling Execution Results

- **Script chạy:** [evaluation/scripts/generate_annotation_pools.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/scripts/generate_annotation_pools.py)
- **Cấu hình thực thi:** `configs/retrieval.mvp.yaml` override sang `data/runtime/qdrant_eval/`, collection `eval_dsa_chunks_v1`, `workspace_id = "evaluation"`, `top_k = 60`, `candidate_top_k = 120`, `reranker_batch_size = 4`, `reranker_max_length = 512` (`retrieval_config_fingerprint = 54980e5829c2bf70`).
- **Thiết bị thực tế:** `embedding_device = cuda:0`, `reranker_device = cuda:0` (`NVIDIA GeForce RTX 5060 Laptop GPU`), `0` OOM, `0` CPU fallback, `0` silent method fallback.
- **Tổng số calls:** **`320 / 320` official pooling calls thành công** (`160` Profile A + `160` Profile B), `0` calls trả ít hơn 60 kết quả, `0` duplicate chunk IDs trong cùng ranking, `0` cross-profile contamination.
- **Artifacts lưu tại `evaluation/artifacts/stage03/`:**
  - [raw_retrieval_results.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage03/raw_retrieval_results.jsonl) (`SHA-256 = db0b2700...`, `320` records)
  - [method_candidate_mapping.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage03/method_candidate_mapping.jsonl) (`SHA-256 = 7514da93...`, `9,068` traceability records — ẩn hoàn toàn khỏi workbook)
  - [candidate_pool_recursive.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage03/candidate_pool_recursive.jsonl) (`SHA-256 = 96b71712...`, `4,526` candidates)
  - [candidate_pool_adaptive.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage03/candidate_pool_adaptive.jsonl) (`SHA-256 = 64454c64...`, `4,542` candidates)
  - [pool_diagnostics.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage03/pool_diagnostics.json)

---

## 6. Deduplication Statistics & 7. Final Annotation Workload

Deduplication áp dụng chặt theo khóa `(query_id, profile_id, chunk_id)` (không gộp giữa hai profiles hay giữa hai câu hỏi khác nhau); Evidence-based Pool Augmentation áp dụng đồng đều cho cả hai profiles trên mọi đoạn `source_backed` (`source_spans`, `overlap_segments`, và `synthetic_segments` có `source_backed = true`) với ngưỡng giao `overlap_chars >= min(20, max(8, floor(0.25 * evidence_span_len)))`:

| Metric | Profile A (`recursive_512_100`) | Profile B (`structure_adaptive`) | Total |
|---|---:|---:|---:|
| **Actual official retrieval calls** | `160` | `160` | **`320`** |
| **Calls returning `< 60` hits** | `0` | `0` | **`0`** |
| **Raw candidates** | `9,600` | `9,600` | **`19,200`** |
| **Unique retrieval candidates** | `4,515` | `4,496` | **`9,011`** |
| **Duplicates removed** | `5,085` | `5,104` | **`10,189`** |
| **Deduplication rate** | `52.97%` | `53.17%` | **`53.07%`** |
| **Evidence-added candidates (`evidence_augmentation`)** | `11` | `46` | **`57`** |
| **Final annotation workload (rows)** | **`4,526`** | **`4,542`** | **`9,068`** |

---

## 8. Workbook Structure and Validation

- **Đường dẫn:** [evaluation/datasets/ground_truth_annotation.xlsx](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/ground_truth_annotation.xlsx)
- **Kích thước & Checksum:** `2,271,141` bytes, `SHA-256 = cb3a2944bdfca1d94c0015e09417f5869f392cd06ada81955ad92cb916222162`.
- **Cấu trúc 5 sheets đã xác minh (`zip_integrity = PASSED`, `sanitized_xml_control_chars = 0`):**
  1. `Instructions`: Hướng dẫn gán nhãn chi tiết bằng tiếng Việt (mục tiêu, 2 profiles, quy trình review, thang điểm `0/1/2`, phân biệt `0` vs ô trống `UNJUDGED`, blind-method note, filter, hoàn thành).
  2. `Questions`: Đủ `40` câu hỏi `Q001–Q040`, tất cả `Human Approval Status = "DRAFT"` có DataValidation `"DRAFT,APPROVED,NEEDS_REVISION"`.
  3. `Profile_A_Annotation`: `4,526` rows, **`4,526` ô `Relevance Grade` trống (`0` pre-filled)**, DataValidation `"0,1,2"` (`allow_blank=True`), conditional formatting cho blank/`0`/`1`/`2`, freeze panes `A2`, auto-filter, `data_type="s"` bảo toàn nguyên văn text.
  4. `Profile_B_Annotation`: `4,542` rows, **`4,542` ô `Relevance Grade` trống (`0` pre-filled)**, cấu hình tương tự Profile A.
  5. `Annotation_Progress`: Bảng tổng hợp Profile A / Profile B / Total và bảng 40 dòng `Q001–Q040` dùng công thức `COUNTIF`/`COUNTIFS` (`Judged = G0 + G1 + G2`, `Unjudged = Total - Judged`) bảo đảm **ô `0` được tính là Judged** và **ô trống được tính là Unjudged**.
- **Blind-Method Guarantee:** Đã xác minh không có bất kỳ cột Method, Rank, Score, Frequency hay `candidate_origin` nào trong hai sheet gán nhãn; thứ tự dòng trong mỗi `(query_id, profile_id)` xáo trộn deterministic theo `sha256("stage03_blind|query_id|profile_id|chunk_id")`. *(Lưu ý trung thực: đây là blind-method annotation; người gán nhãn vẫn phân biệt Profile A/B do tách 2 sheet.)*

---

## 9. Importer Implementation, Stage 04 Contract & Tests

### 9.1. Importer & Validation Coverage

- **Modules:** [evaluation/stage03/importer.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/stage03/importer.py), CLI [evaluation/scripts/import_ground_truth.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/scripts/import_ground_truth.py).
- **Bảo vệ Gate:** Chưa chạy tạo [ground_truth_recursive.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/) hay `ground_truth_adaptive.jsonl` (`OFFICIAL_GT_ABSENT_BEFORE_GATE = PASS`).
- **Kiểm thử tự động ([tests/unit/test_ground_truth.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/tests/unit/test_ground_truth.py)):** **`14 / 14 PASSED`** bao phủ đầy đủ 12 tình huống importer (valid complete, missing grade, invalid grade, duplicate annotation ID, unknown chunk ID, wrong-profile chunk, missing candidate row, modified query/chunk text, corpus SHA mismatch, dataset version mismatch, all-zero judged query `zero_positive=True`, incomplete approval `DRAFT`), cộng thêm deduplication, source-backed augmentation, literal `=` preservation và progress formulas.
- **Full Regression & Environment:**
  - `.\.venv\Scripts\python.exe -m pytest -q` → **`308 passed, 0 failed, 0 skipped, 1 warning (fitz) in 18.68s`** (`PYTEST_EXIT=0`, tăng `+14` tests so với `294` baseline Stage 02).
  - `.\.venv\Scripts\python.exe -m pip check` → **`No broken requirements found.`** (`PIP_CHECK_EXIT=0`).

### 9.2. Stage 04 Evaluation Contract & Existing Module Incompatibilities

Kiểm tra các module đánh giá cũ ghi nhận 2 giới hạn cố định (giữ nguyên không sửa trong Stage 03, đề xuất nâng cấp ở Stage 04):
1. **Zero-positive query incompatibility:** `evaluation/metrics/retrieval_metrics.py:39-42` (`recall_at_k`) và `evaluation/retrieval_results.py:48-49` (`validate_labels`) hiện raise `ValueError` khi `relevant_ids` rỗng. Importer Stage 03 vẫn bảo toàn câu zero-positive với `relevant_ids=[]`, `zero_positive=True`, `labels_complete=True`; Stage 04 cần hỗ trợ chính sách zero-positive tường minh (báo cáo số câu zero-positive, macro-average Recall trên tập có positive hoặc quy ước rõ ràng).
2. **Hardcoded Top-15 / Pool-30:** `evaluation/retrieval_results.py:12-14` khóa `KS = (1, 3, 5, 10, 15)`, `RETRIEVE_TOP_K = 15`, `DEFAULT_CANDIDATE_POOL = 30`. Stage 04 cần triển khai runner mới/mở rộng cho `KS = [1, 3, 5, 10, 15, 20, 30, 45, 60]` với fixed maximum-depth quality ranking (`top_k=60`, `candidate_top_k=120`, prefix cắt tại từng `K`), thêm `Judged@K` đối chiếu ranking drift với `raw_retrieval_results.jsonl`, và tách riêng latency benchmark với requested `top_k` thực tế cho từng `K` (giữ `candidate_top_k=120`, warmup + repeated interleaved runs + GPU sync).
3. **Evidence Metrics Contract cho Stage 04:**
   - **Evidence Recall@K:** Tỷ lệ `required_knowledge_units` của câu hỏi được phủ bởi ít nhất 1 chunk trong Top-K (với KU `single`/`alternative`: phủ ít nhất 1 evidence thuộc KU; với KU `composite`: tập các chunk trong Top-K cùng nhau phủ đủ mọi evidence thành phần của KU), tính trên hợp khoảng source-backed (`source_spans` ∪ `overlap_segments` ∪ `source_backed synthetic_segments`), không double-count cùng KU.
   - **Evidence Precision@K:** Tỷ lệ chunk trong Top-K có ít nhất 1 đoạn source-backed khớp hợp lệ với một `source_evidence` của câu hỏi (phân biệt complete vs partial evidence coverage, không dùng page-level overlap đơn thuần).

---

## 10. Data & Git Integrity

| Kiểm tra | Baseline (End of Stage 02) | Post-Stage-03 Preparation | Trạng thái |
|---|---|---|---|
| **Raw Dataset (`54` files, `80,294,690` B)** | `cc44dfdf537710e23e4426dc71d0d28fab8fd09ca8ca7ef180a090979f8903f6` | `cc44dfdf537710e23e4426dc71d0d28fab8fd09ca8ca7ef180a090979f8903f6` | **IDENTICAL** |
| **Production `chunks_all.jsonl`** | `dccdecdcf7ef8fd81860f44860e553e1026ee19a4e56e39d2dfe95b38b872c90` | `dccdecdcf7ef8fd81860f44860e553e1026ee19a4e56e39d2dfe95b38b872c90` | **IDENTICAL** |
| **Production `qdrant_local` (`0` pts)** | `165bd70ef44a7c9f5ceb262bb75e20e9169437a93b9170c99d33d13c1f3c3f80` | `165bd70ef44a7c9f5ceb262bb75e20e9169437a93b9170c99d33d13c1f3c3f80` | **IDENTICAL** |
| **Frozen Corpus A / Corpus B / `qdrant_eval`** | `e39811da...` / `81c56827...` / `19,756` pts | `e39811da...` / `81c56827...` / `19,756` pts | **IDENTICAL** |
| **Git Branch / `HEAD` / `main`** | `feat/final-readiness` / `0012a0c...` / `b84f094...` | `feat/final-readiness` / `0012a0c...` / `b84f094...` | **UNCHANGED** |
| **Modified tracked / Staged files** | `0` / `0` | `0` / `0` (`git diff --check` exit `0`) | **CLEAN** |

---

## 11. Acceptance Criteria (`AC-01` – `AC-06`)

| AC ID | Tiêu chí | Trạng thái hiện tại | Bằng chứng xác minh |
|---|---|---|---|
| **AC-01** | Shared Evaluation Dataset | **DRAFT READY / HUMAN APPROVAL PENDING** | Đủ 40 câu hỏi chung (`Q001–Q040`, `DRAFT`), 91 KUs (`91/91` retained ở cả A & B), 107 verified canonical source evidence records; chờ con người review & đánh dấu `APPROVED` |
| **AC-02** | Candidate Pool Integrity | **PASS** | `320/320` pooling calls (`top_k=60`, `pool=120`, `cuda:0`), `19,200` raw → `9,011` unique (`53.07%` dedup) + `57` evidence-added = `9,068` rows (`4,526` A + `4,542` B), `0` cross-profile leaks, mapping riêng đầy đủ |
| **AC-03** | Human Annotation Workbook | **PASS** | [ground_truth_annotation.xlsx](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/ground_truth_annotation.xlsx): 5 sheets, ZIP `PASSED`, blind-method verified, `9,068/9,068` ô Relevance Grade trống, DataValidation `0/1/2`, live progress formulas |
| **AC-04** | Ground Truth Importer | **PASS** | [importer.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/stage03/importer.py) & [import_ground_truth.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/scripts/import_ground_truth.py): `14/14` synthetic tests PASS (đủ 12 invalid/edge cases + zero-positive), từ chối thiếu nhãn/chỉnh sửa immutable, chưa sinh Ground Truth giả |
| **AC-05** | Human-validated Ground Truth | **PENDING** | Chờ con người hoàn tất review 40 câu hỏi và gán nhãn `9,068` candidates, sau đó chạy official import |
| **AC-06** | Regression, Frozen Benchmark & Git Integrity | **PASS** | Full `pytest -q`: `308 passed, 0 failed` (`PYTEST_EXIT=0`), `pip check` exit `0`, toàn bộ frozen corpora/indexes/production data/Git `HEAD` & `main` bất biến, `0` staged |

---

## 12. Human Annotation Instructions & Official Import Command

1. Mở workbook: [evaluation/datasets/ground_truth_annotation.xlsx](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/ground_truth_annotation.xlsx).
2. Đọc sheet `Instructions`, sau đó mở sheet `Questions` để review 40 câu hỏi (`Q001–Q040`), Reference Answers, Required Knowledge Units và Source Evidence; cập nhật cột `Human Approval Status` thành `APPROVED` (hoặc `NEEDS_REVISION` kèm ghi chú ở `Reviewer Notes` nếu cần chỉnh sửa câu hỏi và regenerate pool riêng câu đó).
3. Chấm điểm `Relevance Grade` (`0` = Irrelevant, `1` = Partially Relevant, `2` = Highly Relevant; tuyệt đối không coi ô trống là `0`) cho toàn bộ `4,526` dòng trên `Profile_A_Annotation` và `4,542` dòng trên `Profile_B_Annotation`. Không chỉnh sửa các cột cố định (`Annotation ID`, `Query ID`, `Question`, `Reference Answer`, `Chunk ID`, `Chunk Text`, `Source Document`, `Source Page/Location`).
4. Theo dõi sheet `Annotation_Progress` cho tới khi `Questions Approved = 40/40` và `Completion % = 100.0%` (`0` Unjudged).
5. Sau khi hoàn tất, chạy lệnh importer chính thức:

```powershell
Set-Location 'D:\DSA-Document-Assistant-Using-Hybrid-RAG'
.\.venv\Scripts\python.exe evaluation/scripts/import_ground_truth.py
```

---

## 13. Known Limitations, Blockers & Stop Condition

- **Blockers hiện tại:** **Không có blocker kỹ thuật** (hạ tầng Top-60/Pool-120 chạy ổn định trên RTX 5060 không cần sửa production code). Hai điểm không tương thích của module đánh giá cũ (`retrieval_metrics.py` & `retrieval_results.py` với `K > 15` và zero-positive queries) đã được ghi nhận rõ trong Mục 9.2 để nâng cấp ở Stage 04.
- **Giới hạn đã biết:**
  - Đây là **blind-method annotation** (ẩn Retrieval Method, Rank, Score, origin), không phải double-blind chunking profile vì hai profiles tách ở 2 sheets riêng.
  - Recall trên judged pool Top-60 + evidence augmentation là **pooled Recall**, không phải corpus-wide exhaustive Recall; do đó Stage 04 sẽ kết hợp thêm Evidence Recall@K và Evidence Precision@K trên cùng 91 KUs chung.
  - Thời gian chạy 320 pooling calls không được dùng làm official latency benchmark.
- **Đã dừng đúng Human Annotation Gate:** Không tự gán nhãn relevance, không đánh dấu `APPROVED`, không sinh `ground_truth_recursive.jsonl`/`ground_truth_adaptive.jsonl`, không tính P/R/MRR/nDCG, không stage/commit/push/merge, và không bắt đầu Stage 04.

---

## PART II – FULLY AUTOMATED LLM-AS-A-JUDGE GROUND TRUTH COMPLETION

Theo phê duyệt chính thức của người quản lý chuyển Stage 03 từ Human Annotation sang **Fully Automated, Source-Grounded LLM-as-a-Judge** (thay thế Human Annotation Gate & Pilot Human Approval Gate bằng **Automated Quality Gate**), toàn bộ `9,068` candidates trên 40 shared questions đã được gán nhãn, kiểm toán chéo và import tự động.

> [!IMPORTANT]
> **Tính minh bạch phương pháp (`human_validated = false`):**
> - Toàn bộ nhãn relevance trong Ground Truth là **`llm_generated_pooled_relevance_judgments`**, KHÔNG phải nhãn do con người gán thủ công.
> - Trường `human_approval_status` của 40 câu hỏi vẫn giữ nguyên `"DRAFT"` (chuyển sang `automated_review_status = "AI_VALIDATED"`).
> - Workbook ban đầu [ground_truth_annotation.xlsx](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/ground_truth_annotation.xlsx) (ứng với `v1`, `SHA-256 = cb3a2944...`) được bảo toàn nguyên vẹn 100% và tách biệt khỏi workbook tự động [ground_truth_automated.xlsx](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/ground_truth_automated.xlsx) (ứng với `v1.1_ai_reviewed`).

---

## 14. Automated Dataset Review & Reference Verification

- **Artifact:** [automated_question_review.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage03/automated_question_review.json) (đánh giá bởi `antigravity_agent_pro`).
- **Kết quả kiểm tra 40/40 câu hỏi:** Phát hiện và áp dụng **5 minimal objective revisions**, **0 sửa đổi query text** (bảo toàn 100% `9,068` candidate pools từ `320` pooling calls):
  - `Q007`: Sửa `reference_answer` & `KU2` để `Call Dispose(p)` thực thi sau `if/else` cho cả hai nhánh `DEQUEUE`.
  - `Q015`: Làm rõ trong ngoặc ở `reference_answer` rằng "traversal order" của MIT 6.006 là *in-order traversal*.
  - `Q029`: Bổ sung diễn giải chính xác về ký hiệu tiệm cận song song với cách ghép worst/best/average trong giáo trình.
  - `Q032`: Sửa metadata `question_type` từ `space_complexity` → `concept_explanation`.
  - `Q038`: Sửa metadata `question_type` từ `application_scenario` → `algorithm_steps`.
- **Cập nhật file:** [questions_40.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/questions_40.jsonl) (`SHA-256 = c620d728460fa724226fc2c8331c8a168a25ad75edc424f09912ffcd659465d3`), [dataset_manifest.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/dataset_manifest.json) (`dataset_version = "stage03_dsa_40q_v1.1_ai_reviewed"`).
- **Actual Question Type Distribution (`v1.1_ai_reviewed`):** `definition (4)`, `comparison (4)`, `code_interpretation (4)`, `concept_explanation (5)`, `time_complexity (4)`, `pseudocode_interpretation (4)`, `application_scenario (3)`, `multi_evidence_reasoning (4)`, `algorithm_steps (5)`, `space_complexity (3)`. (Part I ở trên phản ánh snapshot `v1` trước review.)

---

## 15. LLM-as-a-Judge Architecture, Models & Calibration

- **Môi trường LLM:** Không có Groq/OpenAI/Anthropic API key ngoài (`backend/.env` chỉ chứa placeholder) và không cài Ollama; hệ thống sử dụng trực tiếp **Antigravity `run_workflow` Agent LLM** trên 2 model tiers tách biệt, ép cấu trúc JSON Schema nghiêm ngặt (`BATCH_ANNOTATION_SCHEMA`, `ADJUDICATION_SCHEMA`) trong [judge.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/stage03/judge.py):
  - **Primary Judge:** `antigravity_agent_flash` (`stage03_llm_judge_v1`).
  - **Blind Secondary Judge:** `antigravity_agent_pro` (mù hoàn toàn với `ai_grade`, `confidence_indicator` và lý do của Primary).
  - **Evidence-Grounded Adjudicator:** `antigravity_agent_pro_adjudicator`.
- **Nguyên tắc chống rò rỉ & chống bịa:** Input vào Judge chỉ gồm `annotation_id`, `chunk_text`, câu hỏi, `reference_answer`, `required_knowledge_units` và `verified source_evidence`; ẩn tuyệt đối Retrieval Method, Rank, Score, `candidate_origin` và Chunking Profile. Mọi nhãn `1`/`2` bắt buộc trích `supporting_text_from_chunk` verbatim và được kiểm chứng tự động qua `verify_supporting_text()` (bao gồm chuẩn hóa NFKC + PDF PUA glyph `\uf8ff`).
- **Pilot Round 1 & Calibration ([pilot_round_1.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage03/pilot_round_1.json), [calibration_history.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage03/calibration_history.json)):**
  - **Synthetic Controls (6/6):** Khớp 100% (`6/6`) trên cả `flash` và `pro`, chặn thành công prompt injection `SYN_05` (`grade = 0`).
  - **160 Stratified Pilot Candidates (`80` A + `80` B):** `100%` hợp lệ (`0` retries, `0` invalids), Exact Agreement = **`89.38%`** (`≥ 85%`), Quadratic Weighted Kappa = **`0.9334`** (`≥ 0.75`), Linear Weighted Kappa = **`0.8885`**, Severe `0-vs-2` conflicts = **`0`** (`runtime = 735.94s`). Đạt 4/4 chỉ tiêu ngay Round 1 và khóa `stage03_llm_judge_v1`.

---

## 16. Full Annotation, Secondary Cross-Check, Adjudication & Quality Audit

- **Primary Annotation ([ai_annotations_primary.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage03/ai_annotations_primary.jsonl)):** Hoàn tất **`9,068 / 9,068` (`100.0%`, `missing = 0`)** qua 3 đợt checkpointed (`998.32s` Wave 2A + `513.31s` Wave 2B + `5.54s` 2 PUA-glyph retries).
- **Blind Secondary Cross-Check & Adjudication ([ai_annotations_secondary.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage03/ai_annotations_secondary.jsonl), [ai_adjudications.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage03/ai_adjudications.jsonl), [ai_annotations_final.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage03/ai_annotations_final.jsonl), [automated_quality_audit.json](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/artifacts/stage03/automated_quality_audit.json)):**
  - Tổng hợp đánh giá Secondary (`pro`, `0` invalids, `0` retries): **`1,073` unique candidates** = hợp của **`907`** (10% Stratified Random Audit) và **`199`** (Mandatory Targeted Cross-Check), với phần giao **`33` candidates** (`907 + 199 - 33 = 1,073`). Đây là hai nhóm thống kê riêng có giao nhau, không cộng trực tiếp.
  - **Pilot vs Random Audit Overlap Reconciliation:** Trong `160` pilot candidates, có **`24` candidates** xuất hiện trong tập Random Audit `907` (`27` trong union `1,073`). Tính riêng trên **`883` non-pilot Random Audit candidates**: Exact Agreement = **`96.60%`**, QWK = **`0.9153`**, LWK = **`0.8796`** — xác nhận chất lượng nhất quán ngoài tập pilot.

| Nhánh kiểm toán | Số mẫu | Exact Agreement | QWK | LWK | Disagreements | `0-vs-2` | Ghi chú |
|---|---:|---:|---:|---:|---:|---:|---|
| **10% Stratified Random Audit (Overall)** | `907` | **`96.58%`** (`876/907`) | **`0.9314`** | **`0.8967`** | `31` (`3.42%`) | `0` | Giao `33` mẫu với Targeted |
| — Non-Pilot Random Audit Subset | `883` | `96.60%` | `0.9153` | `0.8796` | — | `0` | Loại trừ `24` pilot candidates |
| — Profile A (`recursive_512_100`) | `453` | `97.79%` (`443/453`) | `0.9618` | `0.9414` | `10` (`2.21%`) | `0` | Confusion matrix `[[378,7,0],[1,39,0],[0,2,26]]` |
| — Profile B (`structure_adaptive`) | `454` | `95.37%` (`433/454`) | `0.8886` | `0.8368` | `21` (`4.63%`) | `0` | Confusion matrix `[[392,6,0],[9,27,0],[0,6,14]]` |
| **Mandatory Targeted Cross-Check** | `199` | `69.35%` (`138/199`) | `0.1680` | `0.1438` | `61` (`30.65%`) | `0` | Gồm `177` med-conf Grade 1, `39` composite, `18` short, `10` low-conf, `5` ev-aug Grade 0 |

- **Evidence-Grounded Adjudication:** Tổng disagreements là `31 + 61 - 8 (giao) =` **`84` unique disagreements**; phân xử **`84 / 84` (`100%`)** trong `5` batches (`0` retries), với **`68`** thay đổi so với Primary (`6.34%` trên `1,073` mẫu cross-check; `0.75%` trên toàn bộ `9,068`). Lưu ý: độ đồng thuận giữa hai LLM tiers phản ánh *inter-judge consistency*, **không** phải *human-validated accuracy*.
- **Kiểm tra nhất quán & bằng chứng:**
  - `supporting_text_validation_rate_pct`: **`100.0%`** (`9,068 / 9,068`).
  - `cross_profile_identical_text_groups`: `33` nhóm, **`cross_profile_inconsistency_count = 0`** sau adjudication.
  - Runtime cross-check + adjudication: `2,214.5s`.

### Phân bố Final Grades & Annotation Sources (`9,068` candidates)

| Profile | Total | Grade `0` | Grade `1` | Grade `2` | `AI-generated` | `AI-cross-checked` | `AI-adjudicated` | Zero-positive queries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Profile A (`recursive_512_100`)** | `4,526` | `4,057` (`89.6%`) | `300` (`6.6%`) | `169` (`3.7%`) | `3,993` | `501` | `32` | **`0`** (`[]`) |
| **Profile B (`structure_adaptive`)** | `4,542` | `4,151` (`91.4%`) | `266` (`5.9%`) | `125` (`2.8%`) | `4,002` | `488` | `52` | **`0`** (`[]`) |
| **Total** | **`9,068`** | **`8,208`** | **`566`** | **`294`** | **`7,995`** | **`989`** | **`84`** | **`0`** |

---

## 17. Automated Workbook & Official Ground Truth Deliverables

- **Automated Workbook:** [evaluation/datasets/ground_truth_automated.xlsx](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/ground_truth_automated.xlsx) (`2,549,877` bytes, `SHA-256 = 353ac7fb3d4a63657af07879e5ef281a96db2cf831cd0f5b6f4dd9dda96d8350`, 6 sheets: `Instructions`, `Questions`, `Profile_A_Annotation`, `Profile_B_Annotation`, `Quality_Audit`, `Annotation_Progress`, ZIP `PASSED`).
- **Official Ground Truth Profile A:** [evaluation/datasets/ground_truth_recursive.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/ground_truth_recursive.jsonl) (`40` queries, `4,526` judged candidates, `SHA-256 = 0a0717b0320e53cb5b77fb141c8bf0f021ed693cef0ddfedfb9beaad2025e34b`, `human_validated = false`).
- **Official Ground Truth Profile B:** [evaluation/datasets/ground_truth_adaptive.jsonl](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/ground_truth_adaptive.jsonl) (`40` queries, `4,542` judged candidates, `SHA-256 = 91bdcba2c5fb3604f1549b546719bad3affed53c386ae0f985d28bf2c6e03865`, `human_validated = false`).

---

## 18. Updated Acceptance Criteria (`AC-01` – `AC-06`, Automated Mode)

| AC ID | Tiêu chí | Trạng thái | Bằng chứng xác minh |
|---|---|---|---|
| **AC-01** | Shared Evaluation Dataset (AI-Reviewed) | **PASS (Automated)** | 40 câu hỏi chung (`v1.1_ai_reviewed`), `automated_review_status = "AI_VALIDATED"`, 5 minimal fixes (`0` sửa query text), `human_approval_status = "DRAFT"` |
| **AC-02** | Candidate Pool Integrity | **PASS** | `320/320` pooling calls, `9,068` candidates (`4,526` A + `4,542` B), không đổi sau review |
| **AC-03** | Annotation Workbooks | **PASS** | Giữ nguyên [ground_truth_annotation.xlsx](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/ground_truth_annotation.xlsx) (`cb3a2944...`) + sinh mới [ground_truth_automated.xlsx](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/ground_truth_automated.xlsx) (6 sheets, ZIP `PASSED`) |
| **AC-04** | Ground Truth Importer & Judge Validation | **PASS** | `25/25` unit tests PASS trong [test_ground_truth.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/tests/unit/test_ground_truth.py) (14 human + 11 automated/judge) |
| **AC-05** | Automated Ground Truth Generation | **PASS (Automated)** | `9,068/9,068` (`100%`), `supporting_text_valid = 100%`, 10% random audit `96.58%` agree (`QWK = 0.9314`), `0` `0-vs-2` conflicts, `human_validated = false` |
| **AC-06** | Regression, Frozen Benchmark & Git Integrity | **PASS** | Xem kết quả `pytest -q`, `pip check`, checksums & Git bên dưới |

---

## 19. Regression, Environment & Data/Git Integrity (Post-Automated Stage 03)

- **Stage 03 Unit & Integration Tests ([test_ground_truth.py](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/tests/unit/test_ground_truth.py)):** **`25 / 25 PASSED`** (14 tests preparation/human-importer + 11 tests Judge/NFKC+PUA supporting-text/metrics/automated-workbook/automated-importer).
- **Full `pytest -q`:** **`315 passed, 4 failed, 1 warning (fitz)`** (`21.78s`).
  - **Chi tiết 4 failures (pre-existing Windows `time.monotonic()` float-precision flake, không liên quan Stage 03):** `tests/unit/test_llm_reliability.py::test_timeout_per_task_and_remaining_budget[{chat-2,quiz_generation-3,quiz_verification-4,revision-5}]` thất bại tại `assert 0.8000000000029104 <= 0.8` do sai số tick `time.monotonic()` trên Windows (không chỉnh sửa test ngoài phạm vi Stage 03).
- **Dependency Check (`pip check`):** `No broken requirements found.` (`exit 0`).
- **Frozen Benchmark, Production & Original Workbook Integrity:**
  - `data/corpus/dataset_clean/chunks_all.jsonl` (`dccdecdc...`): **IDENTICAL**
  - `data/runtime/qdrant_local/collection/dsa_chunks_v1/storage.sqlite` (`165bd70e...`): **IDENTICAL**
  - `evaluation/artifacts/recursive_512_100/chunks.jsonl` (`e39811da...`): **IDENTICAL**
  - `evaluation/artifacts/structure_adaptive/chunks.jsonl` (`81c56827...`): **IDENTICAL**
  - `evaluation/datasets/source_evidence.jsonl` (`53ffa141...`): **IDENTICAL**
  - `evaluation/datasets/ground_truth_annotation.xlsx` (`cb3a2944...`): **IDENTICAL**
- **Git Baseline:** Branch `feat/final-readiness`, `HEAD = 0012a0cdfbda39c9b5117ee91e85f60e0bad23d7`, `main = b84f09450f213b03ad57ef8b388f482ee0f0eaa2`, `0` modified tracked files, `0` staged files.
- **Known Limitations (Automated Phase):**
  - Toàn bộ `9,068` nhãn là LLM-generated (`human_validated = false`), sử dụng 2 tiers (`flash` primary, `pro` blind secondary & adjudicator) trong cùng hệ thống Antigravity `run_workflow`.
  - [ground_truth_annotation.xlsx](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/ground_truth_annotation.xlsx) ứng với `v1` (`c0e9c7b2...`), trong khi [ground_truth_automated.xlsx](file:///D:/DSA-Document-Assistant-Using-Hybrid-RAG/evaluation/datasets/ground_truth_automated.xlsx) và hai file Ground Truth JSONL ứng với `v1.1_ai_reviewed` (`c620d728...`).
  - Chưa chạy official P/R/MRR/nDCG hoặc latency benchmarks và chưa bắt đầu Stage 04.
