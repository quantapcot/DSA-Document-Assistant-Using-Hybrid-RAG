"""Developer-only snapshots of the chunks sent to indexing."""
import json
import logging
from pathlib import Path, PureWindowsPath
import re
import shutil
from tempfile import NamedTemporaryFile

from app.schemas.contracts import DocumentChunk

logger = logging.getLogger(__name__)
_FIELDS = {"chunk_id", "document_id", "filename", "page", "section", "text"}
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _component(value: str) -> str:
    if not value or value.endswith((" ", ".")) or _UNSAFE.search(value) or PureWindowsPath(value).is_reserved():
        raise ValueError("Invalid component of a debug chunks path")
    return value


def _workspace_dir(runtime_root: Path, workspace_id: str) -> Path:
    base = runtime_root.resolve() / "debug_chunks"
    directory = base / _component(workspace_id)
    if base.is_symlink() or directory.is_symlink() or directory.resolve() != directory:
        raise ValueError("Debug chunks directory must be inside the runtime directory")
    return directory


def _document_path(runtime_root: Path, workspace_id: str, document_id: str) -> Path:
    return _workspace_dir(runtime_root, workspace_id) / f"{_component(document_id)}_chunks.jsonl"


def export_debug_chunks(
    runtime_root: Path, workspace_id: str, document_id: str, chunks: list[DocumentChunk],
) -> Path:
    """Publish one complete UTF-8 JSONL snapshot, replacing the previous version."""
    path = _document_path(runtime_root, workspace_id, document_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=".debug-chunks-", suffix=".tmp",
            mode="w", encoding="utf-8", newline="\n", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for chunk in chunks:
                row = chunk.model_dump(include=_FIELDS)
                row["chunk_index"] = chunk.metadata["chunk_index"]
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def remove_debug_document(runtime_root: Path, workspace_id: str, document_id: str) -> None:
    """Best-effort cleanup that does not interfere with normal document operations."""
    try:
        _document_path(runtime_root, workspace_id, document_id).unlink(missing_ok=True)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Debug chunks document cleanup failed: %s", type(exc).__name__)


def remove_debug_workspace(runtime_root: Path, workspace_id: str) -> None:
    """Remove a session's debug directory, allowing the session janitor to retry."""
    directory = _workspace_dir(runtime_root, workspace_id)
    if directory.exists():
        shutil.rmtree(directory)
