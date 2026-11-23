"""Export blinded Retrieval candidates from an existing query list."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from qdrant_client import models

from app.retrieval.config import RetrievalConfig
from app.retrieval.hybrid_search import RetrievalEngine
from app.retrieval.vector_store import scope_filter
from evaluation.retrieval_results import DEFAULT_CANDIDATE_POOL, METHODS, RETRIEVE_TOP_K
from evaluation.scripts.run_benchmark import evaluation_config, load_queries

RESULTS = ROOT / "evaluation" / "results" / "k15"
HEADERS = ("query_id", "query", "chunk_id", "filename", "page", "chunk_text", "relevance_grade", "notes")


def require_scope(engine, corpus_id, workspace_id):
    scope = scope_filter(corpus_id, workspace_id)
    collection = engine.config.collection
    try:
        exists = engine.client.collection_exists(collection)
        count = engine.client.count(collection, count_filter=scope, exact=True).count if exists else 0
    except Exception as exc:
        raise RuntimeError(f"Cannot read Qdrant collection {collection!r} ({type(exc).__name__}).") from None
    if not exists:
        raise ValueError(f"Qdrant collection {collection!r} does not exist.")
    if count == 0:
        raise ValueError(f"Qdrant scope is empty: workspace={workspace_id!r}, corpus={corpus_id!r}.")
    return scope


def read_candidate_payloads(engine, scope, ids):
    """Read the canonical source fields from Qdrant within the requested scope."""
    payloads = {}
    expected = set(ids)
    ordered = sorted(expected)
    for start in range(0, len(ordered), 100):
        batch = ordered[start:start + 100]
        selector = models.Filter(must=[
            *scope.must, models.FieldCondition(key="chunk_id", match=models.MatchAny(any=batch)),
        ])
        offset = None
        while True:
            try:
                points, offset = engine.client.scroll(
                    engine.config.collection, scroll_filter=selector, offset=offset, limit=100,
                    with_payload=["chunk_id", "filename", "page", "text"], with_vectors=False,
                )
            except Exception as exc:
                raise RuntimeError(f"Cannot verify candidate IDs in Qdrant ({type(exc).__name__}).") from None
            for point in points:
                payload = point.payload or {}
                chunk_id = payload.get("chunk_id")
                if chunk_id not in batch or chunk_id in payloads:
                    raise ValueError("Qdrant returned a duplicate or unexpected chunk_id.")
                filename, page, body = (payload.get(key) for key in ("filename", "page", "text"))
                if (not isinstance(filename, str) or not filename.strip() or type(page) is not int
                        or not isinstance(body, str) or not body.strip()):
                    raise ValueError(f"Qdrant chunk {chunk_id!r} is missing a filename, page or text.")
                payloads[chunk_id] = (filename, page, body)
            if offset is None:
                break
    if missing := expected - payloads.keys():
        example = ", ".join(repr(chunk_id) for chunk_id in sorted(missing)[:3])
        raise ValueError(f"Cannot verify {len(missing)} chunk IDs in the requested Qdrant scope: {example}.")
    return payloads


def collect_candidates(engine, queries, corpus_id, workspace_id):
    if engine.config.candidate_top_k != DEFAULT_CANDIDATE_POOL:
        raise ValueError(f"Annotation candidates require a fixed candidate pool of {DEFAULT_CANDIDATE_POOL}.")
    scope = require_scope(engine, corpus_id, workspace_id)
    pairs = []
    for item in queries:
        qid, query = item["query_id"], item["query"]
        ids = set()
        for method in METHODS:
            try:
                hits = engine.retrieve(query, method, top_k=RETRIEVE_TOP_K,
                                       corpus_id=corpus_id, workspace_id=workspace_id)
            except Exception as exc:
                raise RuntimeError(f"Retrieval failed for {qid!r} with {method} ({type(exc).__name__}). "
                                   "Check Qdrant and the configured models.") from None
            for hit in hits:
                chunk_id = hit.get("chunk_id")
                if not isinstance(chunk_id, str) or not chunk_id.strip():
                    raise ValueError(f"Retrieval returned a missing or invalid chunk_id for {qid!r}.")
                if hit.get("workspace_id", workspace_id) != workspace_id or hit.get("corpus_id", corpus_id) != corpus_id:
                    raise ValueError(f"Retrieval returned a candidate outside the requested scope for {qid!r}.")
                ids.add(chunk_id)
        if not ids:
            print(f"Warning: no candidates returned for query {qid!r}.", file=sys.stderr)
        # Stable ordering independent of method, rank and the order of the four calls.
        pairs.extend((qid, query, chunk_id) for chunk_id in sorted(
            ids, key=lambda key: hashlib.sha256(f"{qid}\0{key}".encode()).digest(),
        ))
    if not pairs:
        raise ValueError("No candidates could be retrieved for any query in this Qdrant scope.")
    payloads = read_candidate_payloads(engine, scope, {chunk_id for _, _, chunk_id in pairs})
    return [(qid, query, chunk_id, *payloads[chunk_id], None, None) for qid, query, chunk_id in pairs]


def artifact_runtime(path):
    default = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node"
    runtime = Path(path or os.environ.get("ANNOTATION_ARTIFACT_RUNTIME") or default).resolve()
    node = runtime / "bin" / ("node.exe" if sys.platform == "win32" else "node")
    if not node.is_file() or not (runtime / "node_modules" / "@oai" / "artifact-tool").is_dir():
        raise ValueError("The spreadsheet runtime is unavailable. Set --artifact-runtime to its Node directory.")
    return runtime, node


def write_workbook(output, rows, runtime, node, *, overwrite=False):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".annotation-") as temporary:
        staged = Path(temporary) / "candidates.xlsx"
        data = json.dumps([HEADERS, *rows], ensure_ascii=False).encode("utf-8")
        subprocess.run(
            [str(node), str(Path(__file__).with_name("annotation_workbook.mjs")), str(runtime), str(staged)],
            input=data, check=True,
        )
        if output.exists() and not overwrite:
            raise FileExistsError(f"Workbook already exists: {output}. Use --overwrite only to replace it.")
        staged.replace(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=RESULTS / "dsa_queries_candidate_20.jsonl")
    parser.add_argument("--output", type=Path, default=RESULTS / "annotation_candidates.xlsx")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "retrieval.mvp.yaml")
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--artifact-runtime", help="Node runtime directory containing @oai/artifact-tool")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace an existing workbook")
    args = parser.parse_args(argv)
    try:
        if args.output.suffix.lower() != ".xlsx":
            raise ValueError("Output must be an .xlsx file.")
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"Workbook already exists: {args.output}. Use --overwrite only to replace it.")
        queries = load_queries(args.queries, graded=False)
        runtime, node = artifact_runtime(args.artifact_runtime)
        config = evaluation_config(RetrievalConfig.from_yaml(args.config))
        if not config.reranker_model:
            raise ValueError("Hybrid Rerank requires a configured reranker model.")
        try:
            engine = RetrievalEngine(config=config)
        except Exception as exc:
            raise RuntimeError(f"Cannot open configured Qdrant ({type(exc).__name__}). "
                               "If using local storage, stop other processes that have it open.") from None
        try:
            rows = collect_candidates(engine, queries, args.corpus_id, args.workspace_id)
        finally:
            engine.client.close()
        write_workbook(args.output, rows, runtime, node, overwrite=args.overwrite)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(f"Exported {len(rows)} candidates for {len({row[0] for row in rows})}/{len(queries)} queries to {args.output}")


if __name__ == "__main__":
    main()
