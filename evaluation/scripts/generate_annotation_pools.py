"""CLI: Execute 320 official Stage 03 pooling calls, deduplicate pools, and generate XLSX workbook."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from evaluation.stage03.pooling import run_official_pooling
from evaluation.stage03.workbook import generate_annotation_workbook


def main() -> int:
    datasets_dir = ROOT / "evaluation" / "datasets"
    q_path = datasets_dir / "questions_40.jsonl"
    ev_path = datasets_dir / "source_evidence.jsonl"
    xlsx_path = datasets_dir / "ground_truth_annotation.xlsx"

    questions = [json.loads(ln) for ln in q_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    evidences = [json.loads(ln) for ln in ev_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    pooled = run_official_pooling(questions=questions, evidences=evidences, root=ROOT)
    wb_verify = generate_annotation_workbook(
        output_path=xlsx_path,
        questions=questions,
        evidences=evidences,
        pool_recursive=pooled["pool_recursive"],
        pool_adaptive=pooled["pool_adaptive"],
        overwrite=True,
    )

    summary = {
        "artifacts_dir": pooled["artifacts_dir"],
        "totals_by_profile": pooled["diagnostics"]["totals_by_profile"],
        "totals_combined": pooled["diagnostics"]["totals_combined"],
        "execution_environment": pooled["diagnostics"]["execution_environment"],
        "artifact_sha256": pooled["diagnostics"]["artifact_sha256"],
        "workbook_verification": wb_verify,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
