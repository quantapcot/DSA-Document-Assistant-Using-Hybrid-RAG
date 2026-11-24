"""Persistent leases for anonymous, isolated study workspaces."""
from contextlib import closing, contextmanager
from hashlib import sha256
from pathlib import Path
import logging
import re
import secrets
import shutil
import sqlite3
import time
from uuid import uuid4

from fastapi import HTTPException
from qdrant_client import models

from app.retrieval.hybrid_search import invalidate_all_bm25_caches
from app.retrieval.indexing import scope_mutation_lock
from app.retrieval.vector_store import scope_filter
from .debug_chunks import remove_debug_workspace

SESSION_CORPUS = "study-sessions"
_WORKSPACE = re.compile(r"session-[0-9a-f]{32}\Z")
_TOKEN = re.compile(r"[A-Za-z0-9_-]{43}\Z")
logger = logging.getLogger(__name__)


def _denied():
    return HTTPException(401, {
        "code": "invalid_session", "message": "Phiên học không hợp lệ hoặc đã hết hạn.",
    }, headers={"WWW-Authenticate": "Bearer"})


class SessionRegistry:
    def __init__(self, runtime_root, ttl_seconds, *, clock=time.time):
        self.root = Path(runtime_root)
        self.ttl_seconds = ttl_seconds
        self.clock = clock

    @contextmanager
    def database(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.root / "study_sessions.sqlite3")) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS study_sessions (
                    workspace TEXT PRIMARY KEY,
                    token_hash TEXT NOT NULL UNIQUE,
                    expires REAL NOT NULL,
                    deleting INTEGER NOT NULL DEFAULT 0
                )
            """)
            yield connection

    def create(self):
        workspace = f"session-{uuid4().hex}"
        token = secrets.token_urlsafe(32)
        expires = self.clock() + self.ttl_seconds
        with self.database() as db:
            db.execute("INSERT INTO study_sessions (workspace, token_hash, expires) VALUES (?, ?, ?)",
                       (workspace, sha256(token.encode("ascii")).hexdigest(), expires))
        return {"token": token, "workspace_id": workspace, "corpus_id": SESSION_CORPUS, "expires_at": expires}

    def authenticate(self, token):
        if not _TOKEN.fullmatch(token):
            raise _denied()
        with self.database() as db:
            row = db.execute(
                "SELECT workspace FROM study_sessions WHERE token_hash = ? AND expires > ? AND deleting = 0",
                (sha256(token.encode("ascii")).hexdigest(), self.clock()),
            ).fetchone()
        if row is None or not _WORKSPACE.fullmatch(row[0]):
            raise _denied()
        return row[0]

    def ensure_active(self, workspace):
        with self.database() as db:
            row = db.execute("SELECT 1 FROM study_sessions WHERE workspace = ? AND expires > ? AND deleting = 0",
                             (workspace, self.clock())).fetchone()
        if row is None:
            raise _denied()

    def mark_deleted(self, workspace):
        with self.database() as db:
            db.execute("UPDATE study_sessions SET deleting = 1 WHERE workspace = ?", (workspace,))

    def pending_cleanup(self):
        with self.database() as db:
            db.execute("UPDATE study_sessions SET deleting = 1 WHERE expires <= ?", (self.clock(),))
            return [row[0] for row in db.execute("SELECT workspace FROM study_sessions WHERE deleting = 1")]

    def directory(self, workspace):
        if not _WORKSPACE.fullmatch(workspace):
            raise ValueError("Invalid session workspace")
        base = self.root.resolve() / "sessions"
        directory = base / workspace
        if base.is_symlink() or directory.is_symlink() or directory.resolve().parent != base:
            raise ValueError("Invalid session directory")
        return directory

    def purge(self, workspace, engine):
        directory = self.directory(workspace)
        client, collection = engine.client, engine.config.collection
        with scope_mutation_lock(collection, SESSION_CORPUS, workspace):
            with self.database() as db:
                if not db.execute("SELECT 1 FROM study_sessions WHERE workspace = ? AND deleting = 1",
                                  (workspace,)).fetchone():
                    return
            if client.collection_exists(collection):
                client.delete(
                    collection_name=collection,
                    points_selector=models.FilterSelector(filter=scope_filter(SESSION_CORPUS, workspace)),
                    wait=True,
                )
            invalidate_all_bm25_caches(collection, SESSION_CORPUS, workspace)
            remove_debug_workspace(self.root, workspace)
            if directory.exists():
                shutil.rmtree(directory)
            with self.database() as db:
                db.execute("DELETE FROM study_sessions WHERE workspace = ? AND deleting = 1", (workspace,))

    def cleanup(self, engine):
        for workspace in self.pending_cleanup():
            try:
                self.purge(workspace, engine)
            except Exception as exc:
                logger.warning("Session cleanup will retry error=%s", type(exc).__name__)


def resolve_scope(request, workspace=None, corpus=None, *, require_session=False):
    """Use bearer identity for session scopes and preserve the legacy scope API."""
    workspaces = [str(value).strip() for value in (workspace, request.headers.get("X-Workspace-Id"))
                  if value is not None]
    corpora = [str(value).strip() for value in (corpus, request.headers.get("X-Corpus-Id")) if value is not None]
    auth = request.headers.get("Authorization")
    if auth is not None:
        scheme, _, token = auth.partition(" ")
        if scheme.casefold() != "bearer":
            raise _denied()
        actual = request.app.state.study_sessions.authenticate(token)
        if any(value != actual for value in workspaces) or any(value != SESSION_CORPUS for value in corpora):
            raise HTTPException(403, "Yêu cầu không thuộc phiên học hiện tại.")
        return actual, SESSION_CORPUS
    if require_session or any(value.casefold().startswith("session-") for value in workspaces) or any(
        value.casefold() == SESSION_CORPUS for value in corpora
    ):
        raise _denied()
    if "" in workspaces or "" in corpora:
        raise HTTPException(400, "Workspace và corpus không được để trống.")
    return next(iter(workspaces), "default-workspace"), next(iter(corpora), "user-uploads")
