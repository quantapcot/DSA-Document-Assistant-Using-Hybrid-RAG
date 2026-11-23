"""Annotation workbook handoff tests with a scoped Retrieval fixture."""
from collections import Counter
from unittest.mock import Mock
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pytest
from qdrant_client import QdrantClient

from app.retrieval.config import RetrievalConfig
from app.retrieval.hybrid_search import RetrievalEngine
from app.retrieval.indexing import build_index
from evaluation.retrieval_results import METHODS
from evaluation.scripts.export_annotation_candidates import (
    HEADERS, artifact_runtime, collect_candidates, write_workbook,
)
from evaluation.scripts.run_benchmark import evaluation_config, load_queries


def queries():
    return [{"query_id": "001", "query": "binary tree"}, {"query_id": "002", "query": "tree traversal"}]


@pytest.fixture
def engine():
    client = QdrantClient(":memory:")
    config = evaluation_config(RetrievalConfig(qdrant_mode="memory", embedding_model="fixture"))
    try:
        for workspace, chunk_id in [("student", "stored-id"), ("other", "private-id")]:
            build_index([{
                "chunk_id": chunk_id, "document_id": "document", "filename": "tree.pdf",
                "page": 4, "text": "The binary tree supports traversal algorithms.",
            }], client=client, config=config, corpus_id="course", workspace_id=workspace,
                embedder=lambda texts: [[1., 0.] for _ in texts], dimension=2)
        yield RetrievalEngine(
            client=client, config=config, query_embedder=lambda _: [1., 0.],
            rerank_scorer=lambda pairs: [.9] * len(pairs),
        )
    finally:
        client.close()


def test_collects_real_ids_from_all_methods_and_deduplicates_per_query(engine):
    engine.retrieve = Mock(wraps=engine.retrieve)
    rows = collect_candidates(engine, queries(), "course", "student")
    assert rows == [
        (q["query_id"], q["query"], "stored-id", "tree.pdf", 4,
         "The binary tree supports traversal algorithms.", None, None)
        for q in queries()
    ]
    assert Counter(call.args[1] for call in engine.retrieve.call_args_list) == dict.fromkeys(METHODS, 2)
    assert all(call.kwargs == {"top_k": 15, "workspace_id": "student", "corpus_id": "course"}
               for call in engine.retrieve.call_args_list)


def test_empty_scope_and_cross_scope_id_are_rejected(engine):
    with pytest.raises(ValueError, match="scope is empty"):
        collect_candidates(engine, queries(), "course", "empty")
    engine.retrieve = Mock(return_value=[{"chunk_id": "private-id"}])
    with pytest.raises(ValueError, match="Cannot verify.*private-id"):
        collect_candidates(engine, queries(), "course", "student")


def test_retrieval_failure_names_method_and_query_without_exposing_provider_details(engine):
    engine.retrieve = Mock(side_effect=RuntimeError("private-provider-detail"))
    with pytest.raises(RuntimeError, match="'001'.*bm25") as exc:
        collect_candidates(engine, queries(), "course", "student")
    assert "private-provider-detail" not in str(exc.value)


def test_query_loader_requires_unique_ids_without_requiring_labels(tmp_path):
    path = tmp_path / "queries.jsonl"
    line = '{"query_id":"001","query":"tree"}\n'
    path.write_text(line, encoding="utf-8-sig")
    assert load_queries(path, graded=False) == [{"query_id": "001", "query": "tree"}]
    path.write_text(line * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_queries(path, graded=False)


def test_workbook_has_only_requested_fields_and_blank_editable_cells(tmp_path):
    try:
        runtime, node = artifact_runtime(None)
    except ValueError:
        pytest.skip("annotation spreadsheet runtime unavailable")
    path = tmp_path / "candidates.xlsx"
    write_workbook(path, [("001", "=A1", "000123", "tree.pdf", 4, "Cây nhị phân", None, None)],
                   runtime, node)
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with ZipFile(path) as workbook:
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        strings = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
        shared = ["".join(node.itertext()) for node in strings.findall(f"{ns}si")]
        cells = {cell.attrib["r"]: cell for cell in sheet.iter(f"{ns}c")}

        def value(address):
            cell = cells.get(address)
            item = cell.findtext(f"{ns}v") if cell is not None else None
            return shared[int(item)] if item is not None and cell.attrib.get("t") == "s" else item

        assert tuple(value(f"{col}1") for col in "ABCDEFGH") == HEADERS
        assert value("A2") == "001" and value("C2") == "000123" and value("E2") == "4"
        assert value("F2") == "Cây nhị phân"
        assert value("B2") == "=A1" and not list(sheet.iter(f"{ns}f"))
        assert value("G2") is None and value("H2") is None
        assert len(ET.fromstring(workbook.read("xl/workbook.xml")).findall(f".//{ns}sheet")) == 1
