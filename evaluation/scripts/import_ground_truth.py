"""CLI: Import human-completed Stage 03 XLSX workbook into Profile A & Profile B Ground Truth JSONLs.

Run ONLY after human review & annotation (40/40 APPROVED + 100% graded 0/1/2) is complete.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.stage03.importer import import_automated_ground_truth, import_ground_truth_from_workbook


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("human", "automated"),
        default="human",
        help="Strict human-reviewed XLSX mode (default) or explicit automated LLM-as-a-Judge mode",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=ROOT / "evaluation" / "datasets" / "ground_truth_annotation.xlsx",
    )
    parser.add_argument(
        "--final-annotations",
        type=Path,
        default=ROOT / "evaluation" / "artifacts" / "stage03" / "ai_annotations_final.jsonl",
    )
    parser.add_argument(
        "--quality-audit",
        type=Path,
        default=ROOT / "evaluation" / "artifacts" / "stage03" / "automated_quality_audit.json",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "evaluation" / "datasets" / "questions_40.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "evaluation" / "datasets" / "dataset_manifest.json",
    )
    parser.add_argument(
        "--pool-a",
        type=Path,
        default=ROOT / "evaluation" / "artifacts" / "stage03" / "candidate_pool_recursive.jsonl",
    )
    parser.add_argument(
        "--pool-b",
        type=Path,
        default=ROOT / "evaluation" / "artifacts" / "stage03" / "candidate_pool_adaptive.jsonl",
    )
    parser.add_argument(
        "--corpus-a",
        type=Path,
        default=ROOT / "evaluation" / "artifacts" / "recursive_512_100" / "chunks.jsonl",
    )
    parser.add_argument(
        "--corpus-b",
        type=Path,
        default=ROOT / "evaluation" / "artifacts" / "structure_adaptive" / "chunks.jsonl",
    )
    parser.add_argument(
        "--output-a",
        type=Path,
        default=ROOT / "evaluation" / "datasets" / "ground_truth_recursive.jsonl",
    )
    parser.add_argument(
        "--output-b",
        type=Path,
        default=ROOT / "evaluation" / "datasets" / "ground_truth_adaptive.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing JSONL outputs")
    args = parser.parse_args(argv)

    if args.mode == "automated":
        result = import_automated_ground_truth(
            questions_path=args.questions,
            manifest_path=args.manifest,
            pool_a_path=args.pool_a,
            pool_b_path=args.pool_b,
            corpus_a_path=args.corpus_a,
            corpus_b_path=args.corpus_b,
            final_annotations_path=args.final_annotations,
            quality_audit_path=args.quality_audit,
            output_a_path=None if args.dry_run else args.output_a,
            output_b_path=None if args.dry_run else args.output_b,
        )
    else:
        result = import_ground_truth_from_workbook(
            args.workbook,
            questions_path=args.questions,
            manifest_path=args.manifest,
            pool_a_path=args.pool_a,
            pool_b_path=args.pool_b,
            corpus_a_path=args.corpus_a,
            corpus_b_path=args.corpus_b,
            output_a_path=None if args.dry_run else args.output_a,
            output_b_path=None if args.dry_run else args.output_b,
        )
    print(json.dumps({
        "mode": args.mode,
        "dataset_version": result["dataset_version"],
        "question_count": result["question_count"],
        "human_validated": result.get("human_validated", args.mode == "human"),
        "grade_counts": result["grade_counts"],
        "zero_positive_queries": result["zero_positive_queries"],
        "written_sha256": result["written_sha256"],
        "dry_run": args.dry_run,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
