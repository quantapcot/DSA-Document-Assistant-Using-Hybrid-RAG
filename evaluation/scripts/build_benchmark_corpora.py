"""CLI entry point to build dual benchmark corpora, inspect diagnostics, and index into qdrant_eval."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from evaluation.corpus.benchmark_builder import build_dual_corpora, index_benchmark_corpora


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-determinism", action="store_true", help="Run twice from raw files and verify exact determinism")
    parser.add_argument("--index", action="store_true", help="Index built corpora into isolated data/runtime/qdrant_eval")
    parser.add_argument("--no-rebuild", action="store_true", help="Do not pass rebuild=True during indexing")
    args = parser.parse_args(argv)

    summary = build_dual_corpora(verify_determinism=args.verify_determinism)
    if args.index:
        summary["indexing"] = index_benchmark_corpora(rebuild=not args.no_rebuild)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
