# PHASE E1-01 – ROUND 02 VERIFICATION REPORT: TARGETED CORRECTION & FINAL VERIFICATION

> **Dự án:** DSA Document Assistant Using Hybrid RAG
> **Nhiệm vụ:** `E1-01` – Retrieval Cache Stabilization (**Round 02 – Targeted Correction & Final Verification**)
> **Workspace:** `D:\DSA-Document-Assistant-Using-Hybrid-RAG`
> **Branch:** `feat/final-readiness`
> **HEAD (bất biến):** `8fe5458adfbff6d291615d64b11dcfa1a213434c`
> **Local `main` (bất biến):** `b84f09450f213b03ad57ef8b388f482ee0f0eaa2`
> **Báo cáo Round 01 (được bảo toàn):** `PHASE_E1_01_IMPLEMENTATION_REPORT.md`
> **Ngày thực hiện:** 2026-11-27

---

## 1. Round 01 Review Reconciliation

Kết quả review chính thức Round 01 của người quản lý dự án: **PARTIAL PASS (`21/24 PASS`, `1 FAIL`, `2 UNVERIFIED`)**.

| AC ID | Trạng thái Round 01 | Vấn đề được chỉ ra khi review Round 01 | Khắc phục trong Round 02 | Trạng thái đề xuất sau Round 02 |
|---|---|---|---|---|
| **AC-21** | **FAIL** | Round 01 tự bổ sung validation từ chối số âm (`ValueError`), `bool` và non-`int` (`TypeError`) trong `_normalize_max_length()`, làm thay đổi hành vi legacy `8fe5458` (đặc biệt số âm trước đây dùng model default). | Khôi phục điều kiện legacy `if max_length and max_length > 0:` kèm `hash(max_length)`, bỏ toàn bộ validation policy mới, chuẩn hóa mọi trường hợp dùng model default về `None`, giữ nguyên giá trị & kiểu của đối số được chuyển tiếp (không ép `int()`), dùng `@lru_cache(maxsize=4, typed=True)`, bổ sung unit tests kiểm chứng toàn bộ ma trận hành vi bằng mocked constructor. | **PASS** |
| **AC-22** | **UNVERIFIED** | Báo cáo Round 01 chỉ nêu đường dẫn scratch tới `verify_e1_01_runtime.py` và tóm tắt số liệu, chưa nhúng nguyên văn script và raw GPU/CPU stdout logs kèm cached HF revisions & công thức percentile. | Đính kèm nguyên văn toàn bộ mã nguồn `verify_e1_01_runtime.py` (Mục 9) và nguyên văn raw JSON outputs + exit codes + commands + env vars + cached HF `refs/main` & snapshot checks + raw latency samples + percentile methodology từ lần chạy kiểm chứng lại của Round 02 (Mục 10). | **PASS** |
| **AC-18** | **UNVERIFIED** | Một số phát biểu trong báo cáo Round 01 chưa phân tách chính xác mức độ xác minh (ví dụ `allclose` vs bitwise equality, targeted 3-file suite vs `test_study_sessions.py`, direct loader vs `/api/chat` HTTP 500, PyTorch CUDA allocator vs total VRAM, full `pytest` status khi còn 4 pre-existing failures, smoke latency vs total Hybrid Rerank Retrieval Latency). | Hiệu chỉnh đầy đủ cả 7 phát biểu kỹ thuật trong Mục 11 và áp dụng ngôn từ chính xác xuyên suốt báo cáo Round 02 (bảo toàn báo cáo Round 01 gốc không ghi đè). | **PASS** |

Toàn bộ **21 tiêu chí đã đạt PASS** ở Round 01 đều được giữ nguyên phạm vi và tái xác nhận không phát sinh regression sau chỉnh sửa Round 02.

---

## 2. AC-21 Root Cause

Kiểm tra trực tiếp mã nguồn tại checkpoint `8fe5458adfbff6d291615d64b11dcfa1a213434c` bằng lệnh read-only `git show 8fe5458...:backend/app/retrieval/reranker.py`:

1. **Hành vi thực tế của loader nội bộ `_get_model()` tại `8fe5458`:**
   ```python
   @lru_cache(maxsize=4)
   def _get_model(name: str, device: str = "cpu", max_length: int = 0):
       resolved = _resolve_device(device)
       ...
       init_kwargs: dict = {"device": resolved}
       if max_length and max_length > 0:
           init_kwargs["max_length"] = max_length
       model = CrossEncoder(name, **init_kwargs)
   ```
   - Khi `max_length` là `0`, `None`, số âm (`-1`, `-5`), `False`, `0.0`, hoặc `""`, điều kiện `if max_length and max_length > 0:` đánh giá là `False` → `init_kwargs` **không chứa** khóa `"max_length"` (sử dụng mặc định của model).
   - Khi `max_length` là giá trị truthy thỏa `max_length > 0` (ví dụ `512`, `256`, `1`, `True`, `512.0`, `512.5`), giá trị được chuyển tiếp nguyên dạng vào `CrossEncoder(name, device=resolved, max_length=max_length)` mà không có bước ép kiểu `int()`.
   - Khi `max_length` là chuỗi không rỗng như `"512"`, biểu thức `"512" > 0` tự nhiên ném `TypeError: '>' not supported between instances of 'str' and 'int'`.
   - Khi `max_length` là đối tượng không hash được như `[]` hoặc `[512]`, decorator `@lru_cache` ném `TypeError: unhashable type: 'list'`.
2. **Đường gọi thực tế từ config/public boundary vào loader:**
   - `RetrievalConfig` (`backend/app/retrieval/config.py`) là `@dataclass(frozen=True)` với `reranker_max_length: int = 0`, không có `__post_init__` validation.
   - `backend/app/main.py:80-85` gọi `_get_model(cfg.reranker_model, cfg.reranker_device, cfg.reranker_max_length)`.
   - `backend/app/retrieval/hybrid_search.py:363-371` gọi `rerank(..., max_length=self.config.reranker_max_length)`, và `rerank()` truyền nguyên `_ml=max_length` vào `_get_model(_name, _dev, _ml)`.
3. **Nguyên nhân sai lệch của Round 01:**
   - Round 01 đã thiết kế `_normalize_max_length()` với kiểm tra chặt `isinstance(max_length, bool) or not isinstance(max_length, int)` (`TypeError`) và `max_length < 0` (`ValueError`).
   - Điều này thay đổi hành vi của loader nội bộ so với `8fe5458` (đặc biệt biến negative integers từ model-default thành `ValueError`) và tự áp đặt chính sách input validation ngoài phạm vi cache stabilization.
4. **Cách giải quyết triệt để ở Round 02 (không có xung đột với cache key isolation):**
   - `_normalize_max_length(max_length)` gọi `hash(max_length)` (bảo toàn hành vi `TypeError` với unhashable args của `@lru_cache` cũ) rồi thực hiện đúng điều kiện legacy `if max_length and max_length > 0: return max_length` else `return None`.
   - `_load_cached_model` dùng `@lru_cache(maxsize=4, typed=True)`: mọi cấu hình dẫn đến cùng lệnh gọi model-default `CrossEncoder(name, device=resolved_device)` dùng chung khóa `(name, resolved_device, None)`; còn mọi đối số được chuyển tiếp vào `init_kwargs["max_length"]` giữ nguyên **cả giá trị lẫn kiểu dữ liệu** nên `1` (`int`) tách biệt hoàn toàn với `True` (`bool`), và `512` (`int`) tách biệt hoàn toàn với `512.0` (`float`).
   - **Phân biệt quan trọng:** Việc loader chuyển tiếp nguyên dạng `True`, `512.0`, `512.5` tới constructor là bảo toàn *constructor forwarding semantics* của wrapper legacy `8fe5458` (được xác minh bằng mocked `CrossEncoder`); báo cáo **không** khẳng định `sentence_transformers.CrossEncoder` thật chấp nhận mọi kiểu dữ liệu được chuyển tiếp này.

---

## 3. Legacy vs Round 01 vs Round 02 Argument Behavior Matrix

| Đối số `max_length` | Legacy (`8fe5458`) | Round 01 (lệch AC-21) | Round 02 (đã khôi phục) | Effective Cache Key (Round 02, `typed=True`) | Ghi chú tương thích |
|---|---|---|---|---|---|
| `512` (`int`, production) | Forward `max_length=512` (`int`) | Forward `max_length=512` (`int`) | Forward `max_length=512` (`int`) | `(name, resolved, 512 [int])` | Khớp 100%; dùng chung giữa `"auto"` & `"cuda"`/`"cpu"` |
| `256` (`int`) | Forward `max_length=256` (`int`) | Forward `max_length=256` (`int`) | Forward `max_length=256` (`int`) | `(name, resolved, 256 [int])` | Tách biệt khỏi `512` và default |
| `0` (`int`) / omitted default | Omit `max_length` (model default) | Omit `max_length` (model default) | Omit `max_length` (model default) | `(name, resolved, None [NoneType])` | Tương đương ngữ nghĩa với `None` |
| `None` | Omit `max_length` (model default) | Omit `max_length` (model default) | Omit `max_length` (model default) | `(name, resolved, None [NoneType])` | Dùng chung cache với `0` |
| `-1` / `-5` (negative `int`) | Omit `max_length` (model default) | **Raised `ValueError`** *(breaking)* | Omit `max_length` (model default) | `(name, resolved, None [NoneType])` | Khôi phục legacy behavior; dùng chung cache default |
| `False` (`bool`) | Omit `max_length` (model default) | **Raised `TypeError`** *(new policy)* | Omit `max_length` (model default) | `(name, resolved, None [NoneType])` | Khôi phục legacy behavior; dùng chung cache default |
| `0.0` (`float`) | Omit `max_length` (model default) | **Raised `TypeError`** *(new policy)* | Omit `max_length` (model default) | `(name, resolved, None [NoneType])` | Khôi phục legacy behavior; dùng chung cache default |
| `""` (empty `str`) | Omit `max_length` (model default) | **Raised `TypeError`** *(new policy)* | Omit `max_length` (model default) | `(name, resolved, None [NoneType])` | Khôi phục legacy behavior; dùng chung cache default |
| `1` (`int`) | Forward `max_length=1` (`int`) | Forward `max_length=1` (`int`) | Forward `max_length=1` (`int`) | `(name, resolved, 1 [int])` | Tách biệt kiểu với `True [bool]` |
| `True` (`bool`) | Forward `max_length=True` (`bool`) | **Raised `TypeError`** *(new policy)* | Forward `max_length=True` (`bool`) | `(name, resolved, True [bool])` | Chỉ bảo toàn forwarding tới constructor (verified by mock); tách biệt với `1 [int]` |
| `512.0` (`float`) | Forward `max_length=512.0` (`float`) | **Raised `TypeError`** *(new policy)* | Forward `max_length=512.0` (`float`) | `(name, resolved, 512.0 [float])` | Không ép `int()`; tách biệt với `512 [int]` |
| `512.5` (`float`) | Forward `max_length=512.5` (`float`) | **Raised `TypeError`** *(new policy)* | Forward `max_length=512.5` (`float`) | `(name, resolved, 512.5 [float])` | Không ép `int()`; chuyển tiếp nguyên dạng tới mocked constructor |
| `"512"` (truthy `str`) | Raised `TypeError` (`'>'` not supported) | Raised `TypeError` (custom) | Raised `TypeError` (`'>'` not supported) | Không cache | Khớp loại lỗi & nguyên nhân legacy |
| `[]`, `[512]` (unhashable) | Raised `TypeError` (unhashable) | Raised `TypeError` (custom) | Raised `TypeError` (unhashable) | Không cache | Khớp loại lỗi & nguyên nhân legacy |
| `device="tpu"` (invalid device) | Raised `ValueError` | Raised `ValueError` | Raised `ValueError` | Không cache | Giữ nguyên semantics `_resolve_device()` |

---

## 4. Minimal Changes

Round 02 **chỉ chỉnh sửa thêm đúng 2 file** trong allowlist được phê duyệt (`backend/app/retrieval/reranker.py` và `tests/unit/test_reranker.py`). Hai file `backend/app/retrieval/embeddings.py` và `tests/unit/test_embeddings.py` từ Round 01 được giữ nguyên hoàn toàn vì không có sai lệch argument semantics nào.

### 4.1. Diff kỹ thuật của `backend/app/retrieval/reranker.py` so với checkpoint `8fe5458`

```diff
diff --git a/backend/app/retrieval/reranker.py b/backend/app/retrieval/reranker.py
index 80c7b6b..6a897ee 100644
--- a/backend/app/retrieval/reranker.py
+++ b/backend/app/retrieval/reranker.py
@@ -34,17 +34,28 @@ def _resolve_device(device: str) -> str:
     raise ValueError(f"Unknown reranker device '{device}'; choose cpu, cuda, or auto")


-@lru_cache(maxsize=4)
-def _get_model(name: str, device: str = "cpu", max_length: int = 0):
-    """Load and cache a CrossEncoder, keyed by (name, resolved_device, max_length).
+def _normalize_max_length(max_length):
+    """Normalize max_length to the effective constructor value without new validation.
+
+    Preserves legacy ``if max_length and max_length > 0:`` semantics:
+    - Values that omit ``max_length`` in ``CrossEncoder`` (``None``, ``0``,
+      negative numbers, ``False``, ``0.0``, ``""``, etc.) normalize to ``None``
+      so semantically equivalent model-default calls share one cache entry.
+    - Values forwarded to ``CrossEncoder`` (positive integers or other truthy
+      values where ``max_length > 0``) are preserved as-is without ``int()``
+      coercion.
+    - Unhashable values raise ``TypeError`` (matching legacy ``@lru_cache``).
+    - Non-comparable truthy values (e.g. ``"512"``) naturally raise ``TypeError``.
+    """
+    hash(max_length)
+    if max_length and max_length > 0:
+        return max_length
+    return None

-    ``max_length`` is set at construction time per the ST v5.7 API.
-    A value of 0 or None uses the model's built-in default.

-    Raises a clear error on CUDA OOM or model load failure — never
-    silently falls back to CPU.
-    """
-    resolved = _resolve_device(device)
+@lru_cache(maxsize=4, typed=True)
+def _load_cached_model(name: str, resolved_device: str, max_length):
+    """Load and cache a CrossEncoder keyed by effective (name, resolved_device, max_length)."""
     try:
         from sentence_transformers import CrossEncoder
     except ImportError as exc:
@@ -52,20 +63,20 @@ def _get_model(name: str, device: str = "cpu", max_length: int = 0):
             "Install sentence-transformers to use reranking"
         ) from exc

-    init_kwargs: dict = {"device": resolved}
-    if max_length and max_length > 0:
+    init_kwargs: dict = {"device": resolved_device}
+    if max_length is not None:
         init_kwargs["max_length"] = max_length

     try:
         logger.info(
             "Loading reranker model %s on %s (max_length=%s)",
-            name, resolved, max_length or "default",
+            name, resolved_device, max_length or "default",
         )
         model = CrossEncoder(name, **init_kwargs)
         try:
             actual_device = str(next(model.model.parameters()).device)
         except Exception:
-            actual_device = resolved
+            actual_device = resolved_device
         logger.info(
             "Reranker model %s initialized successfully on actual device: %s",
             name, actual_device,
@@ -75,12 +86,30 @@ def _get_model(name: str, device: str = "cpu", max_length: int = 0):
         msg = str(exc)
         if "CUDA" in msg or "out of memory" in msg.lower():
             raise RuntimeError(
-                f"Failed to load reranker '{name}' on {resolved}: {msg}. "
+                f"Failed to load reranker '{name}' on {resolved_device}: {msg}. "
                 f"Try setting reranker_device to 'cpu' or freeing GPU memory."
             ) from exc
         raise


+def _get_model(name: str, device: str = "cpu", max_length: int | None = 0):
+    """Resolve effective device and max_length before looking up the LRU cache.
+
+    ``max_length`` is set at construction time per the ST v5.7 API.
+    A value of 0 or None uses the model's built-in default.
+
+    Raises a clear error on CUDA OOM or model load failure — never
+    silently falls back to CPU.
+    """
+    resolved = _resolve_device(device)
+    effective_max_length = _normalize_max_length(max_length)
+    return _load_cached_model(name, resolved, effective_max_length)
+
+
+_get_model.cache_clear = _load_cached_model.cache_clear
+_get_model.cache_info = _load_cached_model.cache_info
+
+
 def rerank(query, candidates, *, model_name=None, scorer=None, top_k=5,
            batch_size=32, device="cpu", max_length=0):
```

### 4.2. Tóm tắt thay đổi trong `tests/unit/test_reranker.py`

- Giữ nguyên toàn bộ test có sẵn trước E1-01 (`test_mock_and_missing_model`).
- Mở rộng `test_auto_and_cuda_share_single_model_instance` kiểm tra cả positional, keyword và explicit `type(...) is int`.
- Cập nhật `test_cache_clear_and_legacy_error_arguments` kiểm tra `cache_clear()`, invalid device `"tpu"` (`ValueError`), `"512"` (`TypeError`), và unhashable `[]`, `[512]` (`TypeError`).
- Bổ sung `test_legacy_default_max_length_values_and_rerank_negative`: kiểm tra default omitted, `0`, `None`, `-1`, `-5`, `False`, `0.0`, `""` chia sẻ đúng 1 instance, không truyền `"max_length"` vào mocked constructor, đồng thời xác minh `rerank(..., max_length=-1)` tái sử dụng cache default và trả về thứ hạng đúng.
- Bổ sung `test_forwarded_max_length_preserves_type_and_isolates_cache`: dùng explicit `type(...) is ...`, constructor call count và object identity (`is not`) để chứng minh `1` (`int`) tách biệt `True` (`bool`), và `512` (`int`) tách biệt `512.0` (`float`) và `512.5` (`float`) không bị ép `int()`.

---

## 5. Targeted Regression Results

- **Exact command:**
  ```powershell
  .\.venv\Scripts\python.exe -m pytest tests/unit/test_reranker.py tests/unit/test_embeddings.py tests/unit/test_optimizations.py -v
  ```
- **Actual exit code:** **`0`**
- **Result:** **`55 passed in 3.70s`** (`0` failed, `0` skipped, `0` warnings; tăng `+2` tests mới so với `53 passed` ở Round 01).
- **Phạm vi file của lệnh targeted này:** chỉ gồm `tests/unit/test_reranker.py` (8 tests), `tests/unit/test_embeddings.py` (4 tests), và `tests/unit/test_optimizations.py` (43 tests). *(Lưu ý hiệu chỉnh theo AC-18: `tests/integration/test_study_sessions.py` nằm trong full regression suite ở Mục 8, không thuộc lệnh targeted 3 file này.)*
- **Các nhóm kiểm tra đã được xác minh PASS:**
  1. Auto/CUDA cache reuse (positional + keyword args) — PASS
  2. Auto/CPU cache reuse (default + positional + keyword args) — PASS
  3. Model isolation — PASS
  4. Device isolation — PASS
  5. Effective `max_length` & type isolation (`1` vs `True`, `512` vs `512.0` vs `512.5`, `512` vs `256` vs default) — PASS
  6. Legacy argument semantics (`-1`, `-5`, `False`, `0.0`, `""`, `"512"`, `[]`, `[512]`, `rerank(..., max_length=-1)`, invalid device `"tpu"`) — PASS
  7. Equivalent default arguments (`0`, `None`, omitted, negative, `False`, `0.0`, `""`) — PASS
  8. Existing ranking behavior & score validation (NaN/Inf rejection, metadata preservation) — PASS
  9. Embedding output behavior (normalized unit vectors, dimension, empty inputs) — PASS
  10. Cache reset behavior (`_get_model.cache_clear()`, `_get_model.cache_info()`) — PASS

---

## 6. GPU Runtime Reverification

Thực hiện lại hoàn toàn sau chỉnh sửa `backend/app/retrieval/reranker.py` ở Round 02 bằng script chuẩn hóa trong chế độ offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, không tải model mới, không gọi Groq, không chạy `/api/chat`):

- **Exit code:** **`0`**
- **Hardware / Runtime:** `NVIDIA GeForce RTX 5060 Laptop GPU`, `torch==2.11.0+cu128`, CUDA runtime `12.8`.
- **Cached Models & Verified Revisions (kiểm tra `refs/main` khớp đúng thư mục `snapshots/<rev>` trên máy):**
  - Embedding: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — `cached_ref_main = e8f8c211226b894fcb81acc59f3b34ba3efd5f42` (`ref_matches_local_snapshot = true`, 1 local snapshot duy nhất).
  - Reranker: `BAAI/bge-reranker-v2-m3` — `cached_ref_main = 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` (`ref_matches_local_snapshot = true`, 1 local snapshot duy nhất).
- **Workload:** Cùng 30 representative corpus chunks (`10 vi`, `10 en`, `10 code/complexity`), cùng query (`"Giải thích thao tác quay cây trong Cây AVL và độ phức tạp thời gian tìm kiếm O(log n)?"`), `batch_size=4`, `max_length=512`, `top_k=15`, `10` lượt tuần tự với `torch.cuda.synchronize()` trước và sau mỗi lượt.
  - Pair token length stats (BGE tokenizer, `add_special_tokens=True`, `truncation=False`): `min=111`, `median=172.0`, `mean=172.6`, `p95=226.6`, `max=268`, `truncated_over_512=0`.
- **Constructor counts & Actual devices:**
  - `SentenceTransformer.__init__` = **`1`**, actual device = **`cuda:0`**
  - `CrossEncoder.__init__` = **`1`**, actual device = **`cuda:0`** (không có silent CPU fallback)
- **Actual Embedding Inference Verification:**
  - Dimension = `384`, `norm_min = 1.0`, `norm_max = 1.0`, toàn bộ hữu hạn.
  - So sánh đầu ra `device="auto"` vs `device="cuda"` trên cùng instance: `auto_cuda_allclose_atol_1e5 = true`, `auto_cuda_max_abs_diff = 0.0`, `auto_cuda_bitwise_equal = true`.
  - Cache info: `hits=9, misses=1, maxsize=4, currsize=1`.
- **Actual Reranker Verification (10 lượt):**
  - `all_10_runs_identical_order = true`, mỗi lượt trả 15 kết quả hợp lệ, `top5_ids = ['7a8db12bc33de10e', 'af353dd60a55b45f', 'f44845be12e1f01b', 'e88e7b109a45131a', 'cadfe9f5bbd6b3ee']`, `top5_scores = [0.3381, 0.2116, 0.1126, 0.0585, 0.0179]`.
  - Cache info: `hits=11, misses=1, maxsize=4, currsize=1`.
- **PyTorch CUDA Allocator Memory** *(chỉ đo bộ cấp phát CUDA của tiến trình PyTorch, không bao gồm CUDA context/driver và các tiến trình GPU khác trên máy)*:
  - Before warmup: `allocated = 0.0 MB`, `reserved = 0.0 MB`.
  - After 10 runs: `allocated = 2,623.89 MB`, `reserved = 2,718.00 MB`, `peak_allocated = 2,670.26 MB`, `peak_reserved = 2,718.00 MB`, `oom_occurred = false`.
- **Isolated Reranker Smoke-Test Latency (10 runs, CUDA-synced)** *(chỉ là thời gian chạy `reranker.rerank()` trên 30 candidates tĩnh, không phải total Hybrid Rerank Retrieval Latency hay official E4 benchmark)*:
  - Raw samples (ms): `[696.7, 639.45, 643.54, 640.95, 643.74, 640.76, 641.3, 642.36, 642.84, 643.51]`
  - Summary: `min = 639.45 ms`, `mean = 647.51 ms`, `p50 = 642.60 ms`, `p95 = 672.87 ms`, `max = 696.70 ms`.
  - Percentile method: `statistics.quantiles(sorted_latencies, n=20, method='inclusive')[-1]`.

---

## 7. CPU Runtime Reverification

Thực hiện trong tiến trình con PowerShell & Python hoàn toàn cách ly với `$env:CUDA_VISIBLE_DEVICES='-1'` được đặt trước khi import PyTorch:

- **Exit code:** **`0`**
- **Isolation verification:**
  - Trong tiến trình con CPU: `environment.CUDA_VISIBLE_DEVICES = "-1"`, **`torch.cuda.is_available() == False`**.
  - Trong shell cha ngay sau khi tiến trình con kết thúc: `PARENT_CUDA_VISIBLE_DEVICES=[]` (không rò rỉ biến môi trường sang các phiên/benchmark GPU về sau).
- **Auto-device fallback:** `resolved_auto_devices = {"embedding": "cpu", "reranker": "cpu"}`.
- **Workload:** Cùng 6 representative chunks đầu tiên của Round 01 (`9a5778b6959843e9`, `3d1ba83588a32ee4`, `e88e7b109a45131a`, `d65d9ddb0bcb5672`, `ad92dee7d65e5998`, `0839f89b382ac156`), `batch_size=4`, `max_length=512`, `top_k=5`, 2 lượt rerank (`device="cpu"` và `device="auto"`).
- **Constructor counts & Actual devices:**
  - `SentenceTransformer.__init__` = **`1`**, actual device = **`cpu`**, `embedding_cache_info = {hits: 5, misses: 1, maxsize: 4, currsize: 1}`.
  - `CrossEncoder.__init__` = **`1`**, actual device = **`cpu`**, `reranker_cache_info = {hits: 3, misses: 1, maxsize: 4, currsize: 1}`.
- **Outputs:** `cpu_vs_auto_identical_order = true`, `reranker_top5_ids = ['e88e7b109a45131a', 'ad92dee7d65e5998', '3d1ba83588a32ee4', '0839f89b382ac156', 'd65d9ddb0bcb5672']`, `reranker_top5_scores = [0.0585, 0.0007, 0.0003, 0.0003, 0.0002]` (điểm `0.0585` của chunk `e88e7b109a45131a` khớp với kết quả GPU).
- **Smoke latencies (6 candidates, CPU):** `cpu_run = 1,546.54 ms`, `auto_run = 1,508.24 ms`.

---

## 8. Full Regression Results

Chạy đúng một lượt duy nhất (không chạy lặp để che giấu failure):

- **Exact commands:**
  ```powershell
  .\.venv\Scripts\python.exe -m pytest -q
  .\.venv\Scripts\python.exe -m pip check
  ```
- **Actual exit codes:** `PYTEST_EXIT=1`, `PIP_CHECK_EXIT=0` (`No broken requirements found.`).

| Metric | Baseline (trước E1-01) | Round 01 | Round 02 | Chênh lệch so với Baseline |
|---|---:|---:|---:|---:|
| Collected | 276 | 284 | **286** | +10 |
| Passed | 272 | 280 | **282** | **+10** |
| Failed | 4 *(pre-existing)* | 4 *(pre-existing)* | **4** *(pre-existing)* | **0 new failures** |
| Skipped | 0 | 0 | **0** | 0 |
| Warnings | 1 (`fitz`) | 1 (`fitz`) | **1** (`fitz`) | 0 |
| `pytest` Exit Code | `1` | `1` | **`1`** | Không đổi |
| `pip check` Exit Code | `0` | `0` | **`0`** | Không đổi |
| Duration | 24.86s | 14.64s | **14.67s** | — |

- **Báo cáo trung thực về trạng thái full suite (theo AC-18):** Full `pytest -q` **không** đạt trạng thái 100% PASS (exit code `1`) do 4 pre-existing failures giữ nguyên ngoài phạm vi E1-01 tại `tests/unit/test_llm_reliability.py:43`:
  - `tests/unit/test_llm_reliability.py::test_timeout_per_task_and_remaining_budget[chat-2]`
  - `tests/unit/test_llm_reliability.py::test_timeout_per_task_and_remaining_budget[quiz_generation-3]`
  - `tests/unit/test_llm_reliability.py::test_timeout_per_task_and_remaining_budget[quiz_verification-4]`
  - `tests/unit/test_llm_reliability.py::test_timeout_per_task_and_remaining_budget[revision-5]`
  - Tất cả 4 case đều lỗi `AssertionError: assert 0.8000000000001819 <= 0.8` (sai số làm tròn IEEE-754 float64 của `time.monotonic()` trên Windows). Không có bất kỳ failure mới nào phát sinh từ E1-01; `tests/integration/test_study_sessions.py` và toàn bộ 282 test còn lại đều PASS.

---

## 9. Verification Script

Toàn bộ nội dung nguyên văn của script kiểm chứng `verify_e1_01_runtime.py` (lưu tại scratch directory và đính kèm đầy đủ tại đây để tái lập độc lập mà không phụ thuộc đường dẫn scratch):

```python
"""Isolated real GPU & CPU runtime verification for E1-01 (HF offline mode, read-only corpus)."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean, median, quantiles

# Force HuggingFace offline mode BEFORE importing transformers/sentence_transformers
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

REPO = Path(r"D:\DSA-Document-Assistant-Using-Hybrid-RAG")
sys.path.insert(0, str(REPO / "backend"))

import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoTokenizer

from app.retrieval import embeddings, reranker
from app.retrieval.config import RetrievalConfig


def inspect_cached_hf_revision(model_id: str) -> dict:
    """Read local HuggingFace hub cached ref/main and verify matching local snapshot directory."""
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_dir = hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
    ref_main_path = repo_dir / "refs" / "main"
    ref_main = ref_main_path.read_text(encoding="utf-8").strip() if ref_main_path.exists() else None
    snapshots_dir = repo_dir / "snapshots"
    snapshots = sorted(p.name for p in snapshots_dir.iterdir() if p.is_dir()) if snapshots_dir.exists() else []
    snapshot_match = bool(ref_main and (snapshots_dir / ref_main).is_dir())
    return {
        "model_id": model_id,
        "cached_ref_main": ref_main,
        "local_snapshots": snapshots,
        "ref_matches_local_snapshot": snapshot_match,
    }


def select_representative_candidates(corpus_path: Path) -> list[dict]:
    """Deterministically select 30 representative chunks (10 vi, 10 en, 10 code/complexity)."""
    rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    code_re = re.compile(r"\b(def |int |void |for\s*\(|while\s*\(|return\b|struct\b|class\s+\w)|O\s*\([^)]+\)")

    vi_theory, en_theory, code_chunks = [], [], []
    for row in rows:
        txt = row["text"]
        lang = (row.get("metadata") or {}).get("language", "")
        is_code = bool(code_re.search(txt))
        if is_code and 300 <= len(txt) <= 512:
            code_chunks.append(row)
        elif not is_code and lang == "vi" and 350 <= len(txt) <= 512:
            vi_theory.append(row)
        elif not is_code and lang == "en" and 350 <= len(txt) <= 512:
            en_theory.append(row)

    def pick_evenly(pool: list[dict], count: int) -> list[dict]:
        step = max(1, len(pool) // count)
        return [pool[i * step] for i in range(count)]

    selected = pick_evenly(vi_theory, 10) + pick_evenly(en_theory, 10) + pick_evenly(code_chunks, 10)
    assert len(selected) == 30 and len({c["chunk_id"] for c in selected}) == 30
    return selected


def run_gpu_verification() -> dict:
    assert torch.cuda.is_available(), "CUDA must be available for GPU verification"
    cfg = RetrievalConfig.from_yaml(REPO / "configs" / "retrieval.mvp.yaml")
    query = "Giải thích thao tác quay cây trong Cây AVL và độ phức tạp thời gian tìm kiếm O(log n)?"
    candidates = select_representative_candidates(REPO / "data" / "corpus" / "dataset_clean" / "chunks_all.jsonl")

    # Measure exact query-chunk pair token lengths with cached BGE tokenizer
    tok = AutoTokenizer.from_pretrained(cfg.reranker_model, local_files_only=True)
    pair_lens = [
        len(tok(query, c["text"], add_special_tokens=True, truncation=False)["input_ids"])
        for c in candidates
    ]
    s_lens = sorted(pair_lens)
    token_stats = {
        "min": s_lens[0],
        "median": median(s_lens),
        "mean": round(mean(s_lens), 1),
        "p95": round(quantiles(s_lens, n=20, method="inclusive")[-1], 1),
        "max": s_lens[-1],
        "truncated_over_512": sum(1 for v in s_lens if v > cfg.reranker_max_length),
        "percentile_method": "statistics.quantiles(sorted_token_lens, n=20, method='inclusive')[-1]",
    }

    # Wrap constructors to count real model initializations
    counts = {"SentenceTransformer": 0, "CrossEncoder": 0}
    orig_st_init = SentenceTransformer.__init__
    orig_ce_init = CrossEncoder.__init__

    def counted_st_init(self, *args, **kwargs):
        counts["SentenceTransformer"] += 1
        return orig_st_init(self, *args, **kwargs)

    def counted_ce_init(self, *args, **kwargs):
        counts["CrossEncoder"] += 1
        return orig_ce_init(self, *args, **kwargs)

    SentenceTransformer.__init__ = counted_st_init
    CrossEncoder.__init__ = counted_ce_init

    try:
        embeddings._get_model.cache_clear()
        reranker._get_model.cache_clear()
        torch.cuda.reset_peak_memory_stats()

        mem_before = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1048576, 2),
        }

        # Warmup + Real Embedding Inference Verification
        emb_warmed = embeddings._get_model(cfg.embedding_model, cfg.embedding_device)
        sample_texts = [c["text"] for c in candidates[:6]]
        vecs_auto = embeddings.embed_texts(sample_texts, cfg.embedding_model, cfg.batch_size, device="auto")
        vecs_cuda = embeddings.embed_texts(sample_texts, cfg.embedding_model, cfg.batch_size, device="cuda")
        qvec_cuda = embeddings.embed_query(query, cfg.embedding_model, device="cuda")
        qvec_auto = embeddings.embed_query(query, cfg.embedding_model, device="auto")
        emb_cuda_ref = embeddings._get_model(cfg.embedding_model, "cuda")

        assert emb_warmed is emb_cuda_ref
        assert counts["SentenceTransformer"] == 1
        assert str(emb_warmed.device).startswith("cuda")
        assert len(vecs_auto) == 6 and len(vecs_auto[0]) == 384
        assert np.all(np.isfinite(vecs_auto)) and np.all(np.isfinite(qvec_cuda))
        norms = [float(np.linalg.norm(v)) for v in vecs_auto + [qvec_cuda]]
        assert all(abs(n - 1.0) < 1e-4 for n in norms)
        allclose_ok = bool(
            np.allclose(vecs_auto, vecs_cuda, atol=1e-5)
            and np.allclose(qvec_auto, qvec_cuda, atol=1e-5)
        )
        assert allclose_ok
        max_abs_diff = float(
            max(
                np.max(np.abs(np.asarray(vecs_auto) - np.asarray(vecs_cuda))),
                np.max(np.abs(np.asarray(qvec_auto) - np.asarray(qvec_cuda))),
            )
        )
        bitwise_equal = bool(vecs_auto == vecs_cuda and qvec_auto == qvec_cuda)
        emb_cache_info = embeddings._get_model.cache_info()._asdict()

        # Warmup Reranker with device="auto", then 10 sequential rerank() calls on 30 candidates
        rer_warmed = reranker._get_model(cfg.reranker_model, cfg.reranker_device, cfg.reranker_max_length)
        rer_device = str(next(rer_warmed.model.parameters()).device)
        assert rer_device.startswith("cuda")

        latencies_ms = []
        top_rankings = []
        for _ in range(10):
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            res = reranker.rerank(
                query,
                candidates,
                model_name=cfg.reranker_model,
                top_k=15,
                batch_size=cfg.reranker_batch_size,
                device=cfg.reranker_device,
                max_length=cfg.reranker_max_length,
            )
            torch.cuda.synchronize()
            dt_ms = (time.perf_counter_ns() - t0) / 1_000_000
            latencies_ms.append(round(dt_ms, 2))
            assert len(res) == 15
            assert all(math.isfinite(r["score"]) for r in res)
            assert [r["rank"] for r in res] == list(range(1, 16))
            assert all(r["retrieval_method"] == "hybrid_rerank" for r in res)
            keys = [(-r["score"], r["chunk_id"]) for r in res]
            assert keys == sorted(keys)
            top_rankings.append([r["chunk_id"] for r in res])

        rer_cuda_ref = reranker._get_model(cfg.reranker_model, "cuda", cfg.reranker_max_length)
        assert rer_cuda_ref is rer_warmed
        assert counts["CrossEncoder"] == 1
        assert all(rk == top_rankings[0] for rk in top_rankings)
        rer_cache_info = reranker._get_model.cache_info()._asdict()

        mem_after = {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1048576, 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1048576, 2),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1048576, 2),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1048576, 2),
        }

        s_lat = sorted(latencies_ms)
        return {
            "mode": "gpu",
            "environment": {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cached_model_revisions": {
                "embedding": inspect_cached_hf_revision(cfg.embedding_model),
                "reranker": inspect_cached_hf_revision(cfg.reranker_model),
            },
            "query": query,
            "config": {
                "embedding_model": cfg.embedding_model,
                "reranker_model": cfg.reranker_model,
                "embedding_device": cfg.embedding_device,
                "reranker_device": cfg.reranker_device,
                "reranker_batch_size": cfg.reranker_batch_size,
                "reranker_max_length": cfg.reranker_max_length,
                "candidate_count": len(candidates),
                "top_k": 15,
                "runs": 10,
            },
            "candidate_chunk_ids": [c["chunk_id"] for c in candidates],
            "pair_token_stats": token_stats,
            "constructor_counts": counts,
            "actual_devices": {
                "embedding": str(emb_warmed.device),
                "reranker": rer_device,
            },
            "embedding_verification": {
                "dimension": len(vecs_auto[0]),
                "norm_min": round(min(norms), 6),
                "norm_max": round(max(norms), 6),
                "auto_cuda_allclose_atol_1e5": allclose_ok,
                "auto_cuda_max_abs_diff": max_abs_diff,
                "auto_cuda_bitwise_equal": bitwise_equal,
                "cache_info": emb_cache_info,
            },
            "reranker_verification": {
                "top5_ids": top_rankings[0][:5],
                "top5_scores": [round(r["score"], 4) for r in res[:5]],
                "all_10_runs_identical_order": True,
                "cache_info": rer_cache_info,
            },
            "pytorch_cuda_memory_note": "PyTorch CUDA allocator metrics only; excludes CUDA context/driver/external process VRAM.",
            "pytorch_cuda_memory_before": mem_before,
            "pytorch_cuda_memory_after": mem_after,
            "smoke_latencies_ms": latencies_ms,
            "smoke_latency_summary_ms": {
                "min": s_lat[0],
                "mean": round(mean(s_lat), 2),
                "p50": round(median(s_lat), 2),
                "p95": round(quantiles(s_lat, n=20, method="inclusive")[-1], 2),
                "max": s_lat[-1],
                "percentile_method": "statistics.quantiles(sorted_latencies, n=20, method='inclusive')[-1]",
            },
            "oom_occurred": False,
        }
    finally:
        SentenceTransformer.__init__ = orig_st_init
        CrossEncoder.__init__ = orig_ce_init


def run_cpu_verification() -> dict:
    cuda_avail = torch.cuda.is_available()
    assert cuda_avail is False, "CUDA must be unavailable for isolated CPU verification"
    cfg = RetrievalConfig.from_yaml(REPO / "configs" / "retrieval.mvp.yaml")
    query = "Giải thích thao tác quay cây trong Cây AVL và độ phức tạp thời gian tìm kiếm O(log n)?"
    candidates = select_representative_candidates(REPO / "data" / "corpus" / "dataset_clean" / "chunks_all.jsonl")[:6]

    counts = {"SentenceTransformer": 0, "CrossEncoder": 0}
    orig_st_init = SentenceTransformer.__init__
    orig_ce_init = CrossEncoder.__init__

    def counted_st_init(self, *args, **kwargs):
        counts["SentenceTransformer"] += 1
        return orig_st_init(self, *args, **kwargs)

    def counted_ce_init(self, *args, **kwargs):
        counts["CrossEncoder"] += 1
        return orig_ce_init(self, *args, **kwargs)

    SentenceTransformer.__init__ = counted_st_init
    CrossEncoder.__init__ = counted_ce_init

    try:
        embeddings._get_model.cache_clear()
        reranker._get_model.cache_clear()

        resolved_emb = embeddings.resolve_device("auto")
        resolved_rer = reranker._resolve_device("auto")

        # Embeddings CPU + auto->cpu fallback verification
        emb_auto = embeddings._get_model(cfg.embedding_model, "auto")
        vecs_cpu = embeddings.embed_texts([c["text"] for c in candidates], cfg.embedding_model, device="cpu")
        qvec_auto = embeddings.embed_query(query, cfg.embedding_model, device="auto")
        emb_cpu = embeddings._get_model(cfg.embedding_model, "cpu")
        assert emb_auto is emb_cpu
        assert str(emb_auto.device) == "cpu"
        assert counts["SentenceTransformer"] == 1
        assert len(vecs_cpu) == len(candidates) and len(vecs_cpu[0]) == 384
        assert np.all(np.isfinite(vecs_cpu)) and np.all(np.isfinite(qvec_auto))
        emb_cache_info = embeddings._get_model.cache_info()._asdict()

        # Reranker CPU + auto->cpu fallback verification (6 representative chunks x 2 runs)
        rer_auto = reranker._get_model(cfg.reranker_model, "auto", cfg.reranker_max_length)
        rer_device = str(next(rer_auto.model.parameters()).device)
        assert rer_device == "cpu"

        t0 = time.perf_counter_ns()
        res_cpu = reranker.rerank(
            query, candidates, model_name=cfg.reranker_model,
            top_k=5, batch_size=cfg.reranker_batch_size, device="cpu", max_length=cfg.reranker_max_length,
        )
        dt_cpu_ms = round((time.perf_counter_ns() - t0) / 1_000_000, 2)

        t1 = time.perf_counter_ns()
        res_auto = reranker.rerank(
            query, candidates, model_name=cfg.reranker_model,
            top_k=5, batch_size=cfg.reranker_batch_size, device="auto", max_length=cfg.reranker_max_length,
        )
        dt_auto_ms = round((time.perf_counter_ns() - t1) / 1_000_000, 2)

        rer_cpu = reranker._get_model(cfg.reranker_model, "cpu", cfg.reranker_max_length)
        assert rer_auto is rer_cpu
        assert counts["CrossEncoder"] == 1
        assert len(res_cpu) == 5 and len(res_auto) == 5
        assert [r["chunk_id"] for r in res_cpu] == [r["chunk_id"] for r in res_auto]
        assert all(math.isfinite(r["score"]) for r in res_cpu)
        rer_cache_info = reranker._get_model.cache_info()._asdict()

        return {
            "mode": "cpu",
            "environment": {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "torch_version": torch.__version__,
            "torch_cuda_is_available": cuda_avail,
            "cached_model_revisions": {
                "embedding": inspect_cached_hf_revision(cfg.embedding_model),
                "reranker": inspect_cached_hf_revision(cfg.reranker_model),
            },
            "resolved_auto_devices": {"embedding": resolved_emb, "reranker": resolved_rer},
            "candidate_count": len(candidates),
            "candidate_chunk_ids": [c["chunk_id"] for c in candidates],
            "constructor_counts": counts,
            "actual_devices": {"embedding": str(emb_auto.device), "reranker": rer_device},
            "embedding_cache_info": emb_cache_info,
            "reranker_cache_info": rer_cache_info,
            "reranker_top5_ids": [r["chunk_id"] for r in res_cpu],
            "reranker_top5_scores": [round(r["score"], 4) for r in res_cpu],
            "cpu_vs_auto_identical_order": True,
            "smoke_latencies_ms": {"cpu_run": dt_cpu_ms, "auto_run": dt_auto_ms},
        }
    finally:
        SentenceTransformer.__init__ = orig_st_init
        CrossEncoder.__init__ = orig_ce_init


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("gpu", "cpu"), required=True)
    args = parser.parse_args()
    out = run_gpu_verification() if args.mode == "gpu" else run_cpu_verification()
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

---

## 10. Raw Runtime Evidence

> **Ghi chú minh bạch về log Round 01:** Stdout của lần chạy Round 01 trước đó không được lưu thành file riêng trong thư mục scratch (chỉ có trong log phiên hội thoại). Vì `backend/app/retrieval/reranker.py` đã được chỉnh sửa trong Round 02, toàn bộ kiểm chứng GPU và CPU đã được thực thi lại hoàn chỉnh ở Round 02, lưu ra file trong scratch (`e1_01_r2_gpu_output.json`, `e1_01_r2_cpu_output.json`) và đính kèm nguyên văn bên dưới. Không có bất kỳ log nào bị mô phỏng.

### 10.1. Exact Verification Commands & Environment Variables

```powershell
# GPU verification (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 set inside script before imports; CUDA_VISIBLE_DEVICES unset)
Set-Location D:\DSA-Document-Assistant-Using-Hybrid-RAG
.\.venv\Scripts\python.exe 'C:\Users\OS\.gemini\antigravity\brain\22590912-d102-441a-b738-13958f36d035\scratch\verify_e1_01_runtime.py' --mode gpu
# Actual exit code: GPU_EXIT=0

# Isolated CPU & Automatic Fallback verification (CUDA_VISIBLE_DEVICES="-1" scoped strictly to child PowerShell/Python process)
powershell -NoProfile -Command "$env:CUDA_VISIBLE_DEVICES='-1'; & 'D:\DSA-Document-Assistant-Using-Hybrid-RAG\.venv\Scripts\python.exe' 'C:\Users\OS\.gemini\antigravity\brain\22590912-d102-441a-b738-13958f36d035\scratch\verify_e1_01_runtime.py' --mode cpu"
# Actual exit code: CPU_EXIT=0
# Parent shell verification after child exited: PARENT_CUDA_VISIBLE_DEVICES=[]
```

### 10.2. Verbatim Raw GPU Verification Output (`GPU_EXIT=0`)

```json
{
  "mode": "gpu",
  "environment": {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "CUDA_VISIBLE_DEVICES": null
  },
  "gpu_name": "NVIDIA GeForce RTX 5060 Laptop GPU",
  "torch_version": "2.11.0+cu128",
  "cuda_runtime": "12.8",
  "cached_model_revisions": {
    "embedding": {
      "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      "cached_ref_main": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
      "local_snapshots": [
        "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
      ],
      "ref_matches_local_snapshot": true
    },
    "reranker": {
      "model_id": "BAAI/bge-reranker-v2-m3",
      "cached_ref_main": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
      "local_snapshots": [
        "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
      ],
      "ref_matches_local_snapshot": true
    }
  },
  "query": "Giải thích thao tác quay cây trong Cây AVL và độ phức tạp thời gian tìm kiếm O(log n)?",
  "config": {
    "embedding_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "reranker_model": "BAAI/bge-reranker-v2-m3",
    "embedding_device": "auto",
    "reranker_device": "auto",
    "reranker_batch_size": 4,
    "reranker_max_length": 512,
    "candidate_count": 30,
    "top_k": 15,
    "runs": 10
  },
  "candidate_chunk_ids": [
    "9a5778b6959843e9",
    "3d1ba83588a32ee4",
    "e88e7b109a45131a",
    "d65d9ddb0bcb5672",
    "ad92dee7d65e5998",
    "0839f89b382ac156",
    "976cc156ef0715f3",
    "9500c6ba0d20cb36",
    "d52b99313bfeba1a",
    "2c3428750be97a5b",
    "cadfe9f5bbd6b3ee",
    "fb8045aa8cefea19",
    "bc71e5d56e2b9593",
    "7ac62b020f32be38",
    "b5320539f21d8497",
    "b7b6cb135ebefc70",
    "df1f8a4019d4606c",
    "31f319af2e6e2358",
    "30fa847825acb2e0",
    "7fca8b88a6d6762f",
    "b90ed61c0e6840f0",
    "f44845be12e1f01b",
    "7c535cf6773a5459",
    "8d56363654455fbf",
    "af353dd60a55b45f",
    "7a8db12bc33de10e",
    "3f749344925a5a19",
    "03211e679cdf573d",
    "72df45c8b748824c",
    "849f519a3b9b28d3"
  ],
  "pair_token_stats": {
    "min": 111,
    "median": 172.0,
    "mean": 172.6,
    "p95": 226.6,
    "max": 268,
    "truncated_over_512": 0,
    "percentile_method": "statistics.quantiles(sorted_token_lens, n=20, method='inclusive')[-1]"
  },
  "constructor_counts": {
    "SentenceTransformer": 1,
    "CrossEncoder": 1
  },
  "actual_devices": {
    "embedding": "cuda:0",
    "reranker": "cuda:0"
  },
  "embedding_verification": {
    "dimension": 384,
    "norm_min": 1.0,
    "norm_max": 1.0,
    "auto_cuda_allclose_atol_1e5": true,
    "auto_cuda_max_abs_diff": 0.0,
    "auto_cuda_bitwise_equal": true,
    "cache_info": {
      "hits": 9,
      "misses": 1,
      "maxsize": 4,
      "currsize": 1
    }
  },
  "reranker_verification": {
    "top5_ids": [
      "7a8db12bc33de10e",
      "af353dd60a55b45f",
      "f44845be12e1f01b",
      "e88e7b109a45131a",
      "cadfe9f5bbd6b3ee"
    ],
    "top5_scores": [
      0.3381,
      0.2116,
      0.1126,
      0.0585,
      0.0179
    ],
    "all_10_runs_identical_order": true,
    "cache_info": {
      "hits": 11,
      "misses": 1,
      "maxsize": 4,
      "currsize": 1
    }
  },
  "pytorch_cuda_memory_note": "PyTorch CUDA allocator metrics only; excludes CUDA context/driver/external process VRAM.",
  "pytorch_cuda_memory_before": {
    "allocated_mb": 0.0,
    "reserved_mb": 0.0
  },
  "pytorch_cuda_memory_after": {
    "allocated_mb": 2623.89,
    "reserved_mb": 2718.0,
    "peak_allocated_mb": 2670.26,
    "peak_reserved_mb": 2718.0
  },
  "smoke_latencies_ms": [
    696.7,
    639.45,
    643.54,
    640.95,
    643.74,
    640.76,
    641.3,
    642.36,
    642.84,
    643.51
  ],
  "smoke_latency_summary_ms": {
    "min": 639.45,
    "mean": 647.51,
    "p50": 642.6,
    "p95": 672.87,
    "max": 696.7,
    "percentile_method": "statistics.quantiles(sorted_latencies, n=20, method='inclusive')[-1]"
  },
  "oom_occurred": false
}
```

### 10.3. Verbatim Raw CPU Verification Output (`CPU_EXIT=0`)

```json
{
  "mode": "cpu",
  "environment": {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "CUDA_VISIBLE_DEVICES": "-1"
  },
  "torch_version": "2.11.0+cu128",
  "torch_cuda_is_available": false,
  "cached_model_revisions": {
    "embedding": {
      "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
      "cached_ref_main": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
      "local_snapshots": [
        "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
      ],
      "ref_matches_local_snapshot": true
    },
    "reranker": {
      "model_id": "BAAI/bge-reranker-v2-m3",
      "cached_ref_main": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
      "local_snapshots": [
        "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
      ],
      "ref_matches_local_snapshot": true
    }
  },
  "resolved_auto_devices": {
    "embedding": "cpu",
    "reranker": "cpu"
  },
  "candidate_count": 6,
  "candidate_chunk_ids": [
    "9a5778b6959843e9",
    "3d1ba83588a32ee4",
    "e88e7b109a45131a",
    "d65d9ddb0bcb5672",
    "ad92dee7d65e5998",
    "0839f89b382ac156"
  ],
  "constructor_counts": {
    "SentenceTransformer": 1,
    "CrossEncoder": 1
  },
  "actual_devices": {
    "embedding": "cpu",
    "reranker": "cpu"
  },
  "embedding_cache_info": {
    "hits": 5,
    "misses": 1,
    "maxsize": 4,
    "currsize": 1
  },
  "reranker_cache_info": {
    "hits": 3,
    "misses": 1,
    "maxsize": 4,
    "currsize": 1
  },
  "reranker_top5_ids": [
    "e88e7b109a45131a",
    "ad92dee7d65e5998",
    "3d1ba83588a32ee4",
    "0839f89b382ac156",
    "d65d9ddb0bcb5672"
  ],
  "reranker_top5_scores": [
    0.0585,
    0.0007,
    0.0003,
    0.0003,
    0.0002
  ],
  "cpu_vs_auto_identical_order": true,
  "smoke_latencies_ms": {
    "cpu_run": 1546.54,
    "auto_run": 1508.24
  }
}
```

---

## 11. Corrected Reporting Statements (AC-18)

| # | Phát biểu cần hiệu chỉnh từ Round 01 | Phát biểu chính xác sau hiệu chỉnh (Round 02) |
|---|---|---|
| 1 | Gọi `np.allclose(..., atol=1e-5)` là "khớp tuyệt đối" (mặc định bitwise equality). | `np.allclose(..., atol=1e-5)` là **numerical equivalence within tolerance `1e-5`**. Trong Round 02, script đã kiểm tra tách biệt cả ba chỉ số thực đo trên GPU: `auto_cuda_allclose_atol_1e5 = true`, `auto_cuda_max_abs_diff = 0.0`, và `auto_cuda_bitwise_equal = true`. |
| 2 | Liệt kê `test_study_sessions.py` chung trong câu của targeted suite 53 tests. | Targeted suite (`pytest tests/unit/test_reranker.py tests/unit/test_embeddings.py tests/unit/test_optimizations.py -v`) gồm **55 tests** ở Round 02 (53 ở Round 01) và **không** bao gồm `tests/integration/test_study_sessions.py`. `test_study_sessions.py` chạy và PASS trong **full regression suite** (`pytest -q`, 282 passed). |
| 3 | Chưa phân tách rõ direct reranker runtime verification với live `/api/chat` verification. | Kiểm chứng runtime của E1-01 là **direct Python verification** gọi trực tiếp `embeddings.*` và `reranker.*` (không khởi động FastAPI server hay gọi HTTP `/api/chat`). |
| 4 | Có thể gây hiểu nhầm rằng toàn bộ lỗi HTTP 500 của `/api/chat` đã được xác minh khắc phục hoàn toàn. | E1-01 chỉ xác minh khắc phục nguyên nhân nạp trùng reranker (`EV-P0-01` / `DEF-P1-01`) ở tầng unit test và direct GPU/CPU loader; **không** tuyên bố mọi lỗi HTTP 500 có thể có của `/api/chat` đã được kiểm thử vì endpoint `/api/chat` nằm ngoài phạm vi E1-01. |
| 5 | Nêu `allocated/reserved` của PyTorch rồi suy ra `~5.4 GB VRAM còn trống`. | Các số `allocated_mb` (`2,623.89 MB`), `reserved_mb` (`2,718.00 MB`), `peak_allocated_mb` (`2,670.26 MB`), `peak_reserved_mb` (`2,718.00 MB`) là **PyTorch CUDA allocator metrics**, **không** đồng nhất với total GPU VRAM consumption (vốn còn bao gồm CUDA context, cuBLAS/cuDNN workspace, driver overhead và tiến trình khác trên máy). |
| 6 | Tránh diễn đạt gây hiểu nhầm full `pytest` đạt PASS toàn phần. | Full regression `pytest -q` trả về **`282 passed, 4 failed, 0 skipped, 1 warning`** (exit code **`1`**): có **0 regression mới** so với baseline `272 passed, 4 failed`, nhưng full suite vẫn còn 4 pre-existing failures tại `tests/unit/test_llm_reliability.py:43` (ngoài phạm vi E1-01). |
| 7 | Phân biệt rõ hơn smoke-test latency của reranker với toàn bộ đường ống Retrieval. | Latency `mean = 647.51 ms` (`p50 = 642.60 ms`, `p95 = 672.87 ms`) là **isolated reranker smoke-test latency** cho `reranker.rerank()` trên 30 candidates tĩnh — **không phải** total Hybrid Rerank Retrieval Latency (vốn cộng thêm query embedding, Qdrant dense search, BM25 search, RRF fusion) và **không phải** official E4 benchmark. |

---

## 12. Data Integrity

Kiểm tra SHA-256 trước khi chạy kiểm chứng runtime Round 02 và sau khi hoàn tất toàn bộ Round 02:

- **Công thức tái lập SHA-256 manifest của raw dataset (54 files):** duyệt `sorted(p for p in Path('data/corpus/dataset_raw').rglob('*') if p.is_file())`, cập nhật `sha256` tuần tự bằng `f.relative_to(raw_dir).as_posix().encode('utf-8')` và `hashlib.sha256(f.read_bytes()).digest()`.

| Artifact | Pre-Round-02 SHA-256 | Post-Round-02 SHA-256 | Bytes / Points | Kết quả |
|---|---|---|---|---|
| Raw Dataset (54 files in `data/corpus/dataset_raw/`) | `c35b29e6d3437d0dd55300642528759d700b8a23803788715c7d363a5beeb8c0` | `c35b29e6d3437d0dd55300642528759d700b8a23803788715c7d363a5beeb8c0` | `54` files | **IDENTICAL** |
| Processed Corpus (`data/corpus/dataset_clean/chunks_all.jsonl`) | `dccdecdcf7ef8fd81860f44860e553e1026ee19a4e56e39d2dfe95b38b872c90` *(khớp Round 01)* | `dccdecdcf7ef8fd81860f44860e553e1026ee19a4e56e39d2dfe95b38b872c90` | `4,100,882` bytes | **IDENTICAL** |
| Qdrant Local SQLite (`data/runtime/qdrant_local/collection/dsa_chunks_v1/storage.sqlite`) | `165bd70ef44a7c9f5ceb262bb75e20e9169437a93b9170c99d33d13c1f3c3f80` *(khớp Round 01)* | `165bd70ef44a7c9f5ceb262bb75e20e9169437a93b9170c99d33d13c1f3c3f80` | `20,480` bytes; `dsa_chunks_v1` `points=0` | **IDENTICAL** |

---

## 13. Git Integrity

| Kiểm tra Git | Kết quả thực tế sau Round 02 |
|---|---|
| `git rev-parse --show-toplevel` | `D:/DSA-Document-Assistant-Using-Hybrid-RAG` |
| `git branch --show-current` | `feat/final-readiness` |
| `git rev-parse HEAD` | `8fe5458adfbff6d291615d64b11dcfa1a213434c` (bất biến) |
| `git rev-parse refs/heads/main` | `b84f09450f213b03ad57ef8b388f482ee0f0eaa2` (bất biến) |
| `git diff --check` | Exit code `0` (không lỗi whitespace) |
| `git diff --cached --stat` | Rỗng (`0` file staged) |
| Modified files (`git status --short`) | Đúng 4 file allowlist: `backend/app/retrieval/embeddings.py`, `backend/app/retrieval/reranker.py`, `tests/unit/test_embeddings.py`, `tests/unit/test_reranker.py` (`4 files changed, 314 insertions(+), 24 deletions(-)` so với `8fe5458`) |
| Untracked files | `EVALUATION_MASTER_AUDIT_REPORT.md`, `PHASE_E1_01_IMPLEMENTATION_REPORT.md`, `PHASE_E1_01_ROUND_02_VERIFICATION_REPORT.md`, `PROJECT_AUDIT_FINAL_READINESS.md` |

Không có thao tác `add`, `commit`, `push`, `pull`, `merge`, `rebase`, `reset`, `stash`, `clean` hay tạo branch mới nào được thực hiện.

---

## 14. Updated Acceptance Criteria Matrix (AC-01 – AC-24)

*(Giữ nguyên 21 AC đã được nghiệm thu PASS ở Round 01 sau khi tái kiểm chứng không phát sinh regression; cập nhật đầy đủ bằng chứng Round 02 cho AC-18, AC-21, AC-22.)*

| AC ID | Requirement | Trạng thái Round 01 | Trạng thái đề xuất Round 02 | Bằng chứng xác minh |
|---|---|---|---|---|
| AC-01 | Workspace và Git baseline chính xác | PASS | **PASS** | `feat/final-readiness`, `HEAD=8fe5458`, `main=b84f094` |
| AC-02 | Root cause của reranker cache mismatch được xác minh | PASS | **PASS** | Giữ nguyên kết quả Round 01 (`reranker.py`, `main.py`) |
| AC-03 | Embedding cache risk được phân loại đúng | PASS | **PASS** | Giữ nguyên kết quả Round 01 (`EV-P1-07` latent risk; `embeddings.py` không sửa thêm) |
| AC-04 | Proposed fix giữ nguyên model configuration và scoring semantics | PASS | **PASS** | Diff Mục 4.1 chỉ chuẩn hóa cache key & khôi phục legacy `max_length` semantics |
| AC-05 | Cache normalization đúng với cùng effective device | PASS | **PASS** | `55/55` targeted PASS + GPU (`auto`↔`cuda`) + CPU (`auto`↔`cpu`) |
| AC-06 | Different effective configurations không dùng chung sai model instance | PASS | **PASS** | `@lru_cache(maxsize=4, typed=True)` + `test_forwarded_max_length_preserves_type_and_isolates_cache` |
| AC-07 | Existing callers và cache interfaces vẫn tương thích | PASS | **PASS** | `55/55` targeted PASS; `test_study_sessions.py` PASS trong full suite (`282 passed`) |
| AC-08 | Cache regression tests PASS | PASS | **PASS** | `55 passed in 3.70s` (`exit 0`) |
| AC-09 | Real GPU warmup và inference chỉ khởi tạo 1 reranker instance | PASS | **PASS** | Round 02 GPU reverification: `CrossEncoder.__init__ == 1`, `hits=11, misses=1` |
| AC-10 | Real GPU smoke workload 30 candidates × 10 lượt không OOM | PASS | **PASS** | Round 02 GPU reverification: `oom_occurred=false`, `peak_allocated=2,670.26 MB` |
| AC-11 | Output schema và scores hợp lệ | PASS | **PASS** | 15 kết quả/lượt, finite scores, strict `(-score, chunk_id)` order, 10/10 runs identical |
| AC-12 | CPU execution path và cache reuse được kiểm chứng | PASS | **PASS** | Round 02 CPU reverification: `CPU_EXIT=0`, `actual_devices=cpu`, constructor `1/1` |
| AC-13 | GPU memory và smoke latency evidence đầy đủ | PASS | **PASS** | Mục 6 & Mục 10.2 (PyTorch allocator stats + 10 raw synced latencies + percentile formula) |
| AC-14 | Full regression không có lỗi mới | PASS | **PASS** | `282 passed` (+10 vs baseline, +2 vs R1), `0` new failures (`4` pre-existing `test_llm_reliability.py:43`) |
| AC-15 | Không thay đổi Retrieval algorithms hoặc production defaults | PASS | **PASS** | `configs/retrieval.mvp.yaml`, scoring/ranking bất biến |
| AC-16 | Không thay đổi dataset, corpus, index hay module ngoài allowlist | PASS | **PASS** | Chỉ đúng 4 file allowlist modified (Round 02 chỉ sửa thêm `reranker.py` & `test_reranker.py`) |
| AC-17 | Git integrity được bảo toàn | PASS | **PASS** | Mục 13 (`feat/final-readiness`, `main` bất biến, 0 staged/committed) |
| **AC-18** | **Implementation report có đầy đủ bằng chứng chính xác** | UNVERIFIED | **PASS** | Mục 11 hiệu chỉnh đủ 7 phát biểu kỹ thuật + toàn bộ bằng chứng thực tế trong báo cáo này |
| AC-19 | Actual GPU embedding inference hoạt động và tái sử dụng đúng 1 instance | PASS | **PASS** | `SentenceTransformer.__init__ == 1`, `dim=384`, `norm=1.0`, `allclose=true`, `max_abs_diff=0.0`, `bitwise_equal=true` |
| AC-20 | CPU verification trong tiến trình cách ly với CUDA availability xác minh | PASS | **PASS** | `CUDA_VISIBLE_DEVICES="-1"` → `torch.cuda.is_available() == False`; `PARENT_CUDA_VISIBLE_DEVICES=[]` |
| **AC-21** | **Cache normalization không thay đổi semantics của valid/invalid args** | FAIL | **PASS** | Khôi phục legacy `if max_length and max_length > 0:` + `hash()` + `typed=True`; kiểm chứng toàn bộ ma trận ở Mục 3 & `55/55` PASS |
| **AC-22** | **Verification scripts, candidate selection & runtime evidence đủ tái lập** | UNVERIFIED | **PASS** | Mục 9 (toàn bộ script nguyên văn) + Mục 10 (raw GPU/CPU JSON, exit codes, commands, env vars, cached HF revisions, 30 chunk IDs, latency samples, percentile formula) |
| AC-23 | Raw dataset, processed corpus & Qdrant index integrity được xác minh | PASS | **PASS** | Mục 12 (SHA-256 pre/post khớp 100%, Qdrant `points=0`) |
| AC-24 | Phân biệt smoke-test latency với official benchmark latency | PASS | **PASS** | Phân tách rõ trong Mục 6, Mục 11 (#7), và Mục 15 |

**Tổng hợp Acceptance Criteria (đề xuất sau Round 02):** **24 / 24 PASS** (`0` FAIL, `0` BLOCKED, `0` UNVERIFIED).

---

## 15. Known Limitations

1. **Pre-existing IEEE-754 clock flake ngoài phạm vi E1-01:** 4 parametrized cases của `tests/unit/test_llm_reliability.py::test_timeout_per_task_and_remaining_budget` tiếp tục fail với `0.8000000000001819 <= 0.8` trên Windows uptime cao (giống baseline trước E1-01). Được giữ nguyên theo chỉ đạo không sửa/skip/xfail.
2. **Pre-existing PyMuPDF `fitz` deprecation warning:** 1 warning có sẵn ngoài phạm vi E1-01.
3. **Constructor forwarding vs real `CrossEncoder` acceptance:** Wrapper `_get_model()` bảo toàn chính xác hành vi chuyển tiếp đối số của checkpoint `8fe5458` (đã xác minh bằng mocked `CrossEncoder` đối với `True`, `512.0`, `512.5`), nhưng điều này không đảm bảo thư viện `sentence_transformers.CrossEncoder` thật chấp nhận mọi kiểu dữ liệu không chuẩn này.
4. **Phạm vi kiểm chứng runtime:** E1-01 kiểm chứng trực tiếp ở tầng loader/retrieval module (`embeddings.*`, `reranker.*`), không kiểm thử live HTTP `/api/chat` và không phải benchmark E4 đo tổng Hybrid Rerank Retrieval Latency.
5. **PyTorch CUDA allocator metrics:** Số đo bộ nhớ GPU chỉ phản ánh PyTorch CUDA allocator, không đo toàn bộ VRAM tiêu thụ bởi driver/CUDA context/tiến trình hệ thống.

---

## 16. Final Agent Self-Assessment

- **Trạng thái đề xuất:** **READY FOR REVIEW** *(không tự tuyên bố nghiệm thu chính thức)*.
- Cả 3 tiêu chí cần xử lý ở Round 02 (**AC-21**, **AC-22**, **AC-18**) đã được khắc phục tối thiểu, xác minh đầy đủ bằng unit tests cách ly (`55/55 PASS`), kiểm chứng lại GPU thực tế (`GPU_EXIT=0`, constructor count `1/1`, `0` OOM), kiểm chứng CPU cách ly (`CPU_EXIT=0`, `torch.cuda.is_available() == False`, không rò biến môi trường), full regression (`282 passed, 4 pre-existing failed, 0 new regressions`), `pip check` sạch (`exit 0`), SHA-256 dữ liệu bất biến và Git integrity sạch hoàn toàn.
- Dừng triển khai theo đúng Stop Condition (không stage, không commit, không push, không merge, không bắt đầu `E1-02`, không sửa `test_llm_reliability.py`) và chờ người quản lý dự án review & nghiệm thu chính thức.

---

## 17. Final Acceptance Check & Official Acceptance

### A. Official Acceptance

- **Acceptance Criteria (`AC-01`–`AC-24`):** **`24 / 24 PASS`** (`0` FAIL, `0` BLOCKED, `0` UNVERIFIED).
- **Overall Status:** **`ACCEPTED`**.
- **Closed Issues:**
  - `EV-P0-01` / `DEF-P1-01` (Reranker cache key mismatch): **`CLOSED`**.
  - `EV-P1-07` (Embedding cache key latent risk): **`CLOSED`**.
- **Round 03:** Không cần Round 03; đã được người quản lý phê duyệt tạo Git Checkpoint chính thức cho `E1-01`.

### B. Raw Dataset Integrity Reconciliation

Kiểm tra read-only đối chiếu sự khác biệt giữa aggregate SHA-256 của `data/corpus/dataset_raw/` ở Round 01 và Round 02:

- **Công thức Round 01 (từ transcript step 412, `2026-11-27T13:49:32Z`):**
  - Đường dẫn repo-relative (`p.as_posix()`, gồm tiền tố `data/corpus/dataset_raw/...`).
  - Hexadecimal SHA-256 cho từng file.
  - Phân tách bằng dấu `:` giữa path và hex digest, nối giữa các file bằng `\n` (không có `\n` cuối):
    `hashlib.sha256("\n".join(f"{p.as_posix()}:{sha256_file(p)}" for p in files).encode("utf-8")).hexdigest()`
  - **Reproduced Round 01 hash:** `afe41adad8ef53bea6533f50ce8ea164e0cb691ac076e4391c39a209da630253` (`MATCHES_R1 = True`).
- **Công thức Round 02:**
  - Đường dẫn dataset-relative (`p.relative_to(raw_dir).as_posix()`, bỏ tiền tố `data/corpus/dataset_raw/`).
  - Binary 32-byte SHA-256 digests (`hashlib.sha256(p.read_bytes()).digest()`).
  - Cập nhật nối tiếp tuần tự (`h.update(path_bytes); h.update(raw_digest)`) không có dấu phân tách `:` hay `\n`.
  - **Reproduced Round 02 hash:** `c35b29e6d3437d0dd55300642528759d700b8a23803788715c7d363a5beeb8c0` (`MATCHES_R2 = True`).
- **Lệnh tái lập read-only & kết quả thực tế:**
  ```powershell
  Set-Location D:\DSA-Document-Assistant-Using-Hybrid-RAG
  @'
  import hashlib
  from pathlib import Path

  raw_dir = Path("data/corpus/dataset_raw")
  files = sorted(p for p in raw_dir.rglob("*") if p.is_file())

  def sha256_file(p: Path) -> str:
      h = hashlib.sha256()
      with p.open("rb") as f:
          for b in iter(lambda: f.read(65536), b""):
              h.update(b)
      return h.hexdigest()

  r1_hash = hashlib.sha256("\n".join(f"{p.as_posix()}:{sha256_file(p)}" for p in files).encode("utf-8")).hexdigest()
  h2 = hashlib.sha256()
  for p in files:
      h2.update(p.relative_to(raw_dir).as_posix().encode("utf-8"))
      h2.update(hashlib.sha256(p.read_bytes()).digest())
  r2_hash = h2.hexdigest()
  print(f"FILE_COUNT={len(files)} TOTAL_BYTES={sum(p.stat().st_size for p in files)}")
  print(f"REPRODUCED_R1={r1_hash} MATCHES_R1={r1_hash == 'afe41adad8ef53bea6533f50ce8ea164e0cb691ac076e4391c39a209da630253'}")
  print(f"REPRODUCED_R2={r2_hash} MATCHES_R2={r2_hash == 'c35b29e6d3437d0dd55300642528759d700b8a23803788715c7d363a5beeb8c0'}")
  '@ | .\.venv\Scripts\python.exe -
  ```
  - Output thực tế (exit code `0`):
    - `FILE_COUNT=54`, `TOTAL_BYTES=80294690` (`76.57 MiB`), `UNIQUE_SHA256=54`
    - `REPRODUCED_R1=afe41adad8ef53bea6533f50ce8ea164e0cb691ac076e4391c39a209da630253` (`MATCHES_R1=True`)
    - `REPRODUCED_R2=c35b29e6d3437d0dd55300642528759d700b8a23803788715c7d363a5beeb8c0` (`MATCHES_R2=True`)
- **Verified Inventory & Git Tree Proof:**
  - `54` raw files (đều được Git theo dõi: `git ls-files data/corpus/dataset_raw` = `54`).
  - `80,294,690` bytes (`76.57 MiB`), `54` SHA-256 duy nhất (`0` exact duplicates).
  - Raw dataset Git tree khớp tuyệt đối giữa local `main` (`b84f09450f213b03ad57ef8b388f482ee0f0eaa2`) và Phase 01 checkpoint (`8fe5458adfbff6d291615d64b11dcfa1a213434c`): **`a2ae3efd0760c023356eeefbcb440737eb2352c3`**, với `git diff` so với `main` và `HEAD` đều rỗng (`0` bytes).
- **Giải thích:** Sự khác biệt giữa hai aggregate hashes hoàn toàn do cách serialize manifest khác nhau giữa lệnh Round 01 và Round 02, **không** phải do thay đổi nội dung raw dataset.
- **`AC-23`: `PASS`**.

### C. Reporting Clarifications

1. Phép so sánh Python list equality (`vecs_auto == vecs_cuda and qvec_auto == qvec_cuda`) cho trường `auto_cuda_bitwise_equal` chứng minh **exact numerical equality** (`max_abs_diff = 0.0`) trên các giá trị float được so sánh, **không** phải strict byte/bit-level representation equality (ví dụ `-0.0 == 0.0` vẫn `True`).
2. `np.allclose(..., atol=1e-5)` vẫn áp dụng relative tolerance mặc định của NumPy (`rtol=1e-5`) nếu không truyền tường minh `rtol=0`.
3. Các chỉ số PyTorch CUDA allocator chia cho `1,048,576` (`1024²`) bytes có đơn vị chính xác là **MiB** (`allocated = 2,623.89 MiB`, `reserved = 2,718.00 MiB`, `peak_allocated = 2,670.26 MiB`, `peak_reserved = 2,718.00 MiB`), không phải decimal MB.
4. Câu lệnh PowerShell chuẩn để tái lập tiến trình CPU cách ly (bọc `-Command '...'` trong dấu nháy đơn để `$env:CUDA_VISIBLE_DEVICES` chỉ được thiết lập bên trong tiến trình con, không bị shell cha nội suy trước):
   ```powershell
   powershell -NoProfile -Command '$env:CUDA_VISIBLE_DEVICES = "-1"; & "D:\DSA-Document-Assistant-Using-Hybrid-RAG\.venv\Scripts\python.exe" "C:\Users\OS\.gemini\antigravity\brain\22590912-d102-441a-b738-13958f36d035\scratch\verify_e1_01_runtime.py" --mode cpu'
   ```
5. Full regression: **`282 passed, 4 pre-existing failed, 0 skipped, 1 warning`** (`PYTEST_EXIT=1`).
6. Targeted regression: **`55/55 passed`** (`TARGETED_EXIT=0`).
7. **Zero new regression.**
8. Reranker smoke-test latency không phải official Retrieval benchmark latency (và không phải total Hybrid Rerank Retrieval Latency).
9. Actual `/api/chat` chưa được kiểm thử lại trong phạm vi `E1-01`.

### D. Known Limitations & Handover

- Bốn pre-existing failures tại `tests/unit/test_llm_reliability.py::test_timeout_per_task_and_remaining_budget` tiếp tục được giữ nguyên ngoài phạm vi `E1-01`.
- Pre-existing PyMuPDF `fitz` deprecation warning được giữ nguyên.
- Không tuyên bố full `pytest` suite đạt 100% PASS.
- Không tuyên bố đã đo mức cải thiện official Retrieval latency.
- Production integration (bao gồm kiểm thử `/api/chat` end-to-end) và Answer Evaluation thuộc các nhiệm vụ tiếp theo.
- `E1-02` (`Structure-aware Adaptive Chunking Specification & Ownership`) chưa bắt đầu.
