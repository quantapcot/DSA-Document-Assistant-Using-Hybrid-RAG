"""
Pipeline: Raw Dataset → Clean → Chunk → DocumentChunk (JSON)
Chạy từ thư mục gốc dự án:
  python data/corpus/run_pipeline.py

Output: data/corpus/dataset_clean/chunks_all.jsonl
        data/corpus/dataset_clean/pipeline_report.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# --- Thêm backend vào sys.path để import các module ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.document_processing.document_parser import parse_document
from app.document_processing.text_cleaner import clean_text, is_valid_chunk_text
from app.document_processing.chunker import chunk_text
from app.document_processing.metadata import build_chunk

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR = ROOT / "data" / "corpus" / "dataset_raw"
OUT_DIR = ROOT / "data" / "corpus" / "dataset_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 512          # characters
CHUNK_OVERLAP = 100       # characters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUT_DIR / "pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Language detection (optional — graceful fallback)
# ---------------------------------------------------------------------------
def _detect_lang(text: str, hint: str = "unknown") -> str:
    """Detect language từ 500 ký tự đầu. Dùng hint nếu langdetect fail."""
    try:
        from langdetect import detect
        return detect(text[:500])
    except Exception:
        return hint


def _lang_hint_from_path(path: Path) -> str:
    """Suy ngôn ngữ từ tên thư mục: vie_* → vi, eng_* / end_* → en."""
    folder = path.parent.name.lower()
    if folder.startswith("vie"):
        return "vi"
    if folder.startswith("eng") or folder.startswith("end"):
        return "en"
    return "unknown"


# ---------------------------------------------------------------------------
# Pipeline cho 1 file
# ---------------------------------------------------------------------------
def process_file(file_path: Path) -> list[dict]:
    lang_hint = _lang_hint_from_path(file_path)
    filename = file_path.name

    log.info(f"  → Parsing   : {filename}")
    pages = parse_document(file_path)

    if not pages:
        log.warning(f"  ✗ Không lấy được text: {filename}")
        return []

    all_chunks: list[dict] = []
    chunk_global_idx = 0

    for page in pages:
        # 1. Detect language (dùng hint từ folder nếu không detect được)
        lang = _detect_lang(page.raw_text, hint=lang_hint)

        # 2. Clean
        clean = clean_text(page.raw_text, language=lang)
        if not clean:
            continue

        # 3. Chunk
        chunks = chunk_text(clean, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        for ch in chunks:
            text = ch["text"]
            section = ch["section"]

            # 4. QA filter
            if not is_valid_chunk_text(text, min_words=15):
                continue

            # 5. Build DocumentChunk
            doc_chunk = build_chunk(
                filename=filename,
                page=page.page,
                text=text,
                chunk_index=chunk_global_idx,
                section=section,
                language=lang,
                extra_metadata={
                    "file_type": page.file_type,
                    "char_count": len(text),
                },
            )
            all_chunks.append(doc_chunk.model_dump())
            chunk_global_idx += 1

    return all_chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()

    # Collect tất cả file PDF + DOCX
    all_files = sorted(
        list(RAW_DIR.rglob("*.pdf")) + list(RAW_DIR.rglob("*.docx"))
    )
    log.info(f"Tìm thấy {len(all_files)} file trong {RAW_DIR}")

    out_jsonl = OUT_DIR / "chunks_all.jsonl"
    report: dict = {
        "total_files": len(all_files),
        "processed": 0,
        "failed": 0,
        "total_chunks": 0,
        "files": [],
    }

    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for file_path in all_files:
            log.info(f"[{file_path.parent.name}] {file_path.name}")
            try:
                chunks = process_file(file_path)
                for c in chunks:
                    fout.write(json.dumps(c, ensure_ascii=False) + "\n")

                report["processed"] += 1
                report["total_chunks"] += len(chunks)
                report["files"].append({
                    "file": file_path.name,
                    "folder": file_path.parent.name,
                    "chunks": len(chunks),
                    "status": "ok",
                })
                log.info(f"  ✓ {len(chunks)} chunks")

            except Exception as e:
                log.error(f"  ✗ FAILED: {e}")
                report["failed"] += 1
                report["files"].append({
                    "file": file_path.name,
                    "folder": file_path.parent.name,
                    "chunks": 0,
                    "status": f"error: {e}",
                })

    report["elapsed_sec"] = round(time.time() - t0, 2)

    report_path = OUT_DIR / "pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log.info("=" * 60)
    log.info(f"DONE  |  Files: {report['processed']}/{report['total_files']}")
    log.info(f"       |  Chunks: {report['total_chunks']}")
    log.info(f"       |  Failed: {report['failed']}")
    log.info(f"       |  Time  : {report['elapsed_sec']}s")
    log.info(f"Output : {out_jsonl}")
    log.info(f"Report : {report_path}")


if __name__ == "__main__":
    main()

