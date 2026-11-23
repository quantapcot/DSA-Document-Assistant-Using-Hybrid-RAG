"""Annotation workbook handoff tests with a scoped Retrieval fixture."""
from collections import Counter
from unittest.mock import Mock
from zipfile import ZIP_STORED, ZipFile, is_zipfile

import pytest
from openpyxl import Workbook, load_workbook
from qdrant_client import QdrantClient

from app.retrieval.config import RetrievalConfig
from app.retrieval.hybrid_search import RetrievalEngine
from app.retrieval.indexing import build_index
from evaluation.retrieval_results import METHODS
from evaluation.scripts.export_annotation_candidates import (
    HEADERS, collect_candidates, write_workbook,
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
    path = tmp_path / "candidates.xlsx"
    row = ("001", "=A1", "000123", "tree.pdf", 4, "Cây nhị phân\nDòng thứ hai", None, None)
    write_workbook(path, [row])
    assert is_zipfile(path)
    with ZipFile(path) as archive:
        assert archive.testzip() is None
    workbook = load_workbook(path)
    try:
        assert workbook.sheetnames == ["Annotation"]
        sheet = workbook["Annotation"]
        assert list(sheet.values) == [HEADERS, row]
        assert sheet["B2"].data_type == "s" and sheet["E2"].data_type == "n"
        assert sheet.freeze_panes == "A2" and sheet.auto_filter.ref == "A1:H2"
        assert sheet["F2"].alignment.wrap_text and sheet.row_dimensions[2].height >= 24
        [validation] = sheet.data_validations.dataValidation
        assert validation.type == "list" and validation.formula1 == '"0,1,2"'
        assert "G2" in validation and "H2" not in validation
        assert validation.allow_blank and validation.showDropDown is False
    finally:
        workbook.close()


@pytest.mark.parametrize(("failure", "message"), [
    ("not_zip", "not an XLSX ZIP"),
    ("bad_crc", "corrupt ZIP entry"),
    ("not_xlsx", "Cannot reopen"),
])
def test_export_does_not_replace_existing_workbook_with_invalid_file(tmp_path, monkeypatch, failure, message):
    path = tmp_path / "candidates.xlsx"
    row = ("q1", "tree", "a", "tree.pdf", 1, "Binary tree", None, None)
    write_workbook(path, [row])
    original = path.read_bytes()

    def save_invalid(self, target):
        if failure == "not_zip":
            target.write_bytes(b"not a ZIP archive")
            return
        with ZipFile(target, "w", compression=ZIP_STORED) as archive:
            archive.writestr("fixture.txt", b"fixture-marker")
        if failure == "bad_crc":
            target.write_bytes(target.read_bytes().replace(b"fixture-marker", b"fixture-damage", 1))
        assert is_zipfile(target)

    monkeypatch.setattr(Workbook, "save", save_invalid)
    with pytest.raises(ValueError, match=message):
        write_workbook(path, [row], overwrite=True)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
