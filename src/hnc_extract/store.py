"""Content-addressed cache of extraction calls (SQLite, one file per run dir).

The key of a call hashes everything that determines its output: document text,
field spec (incl. shared instructions), model config, sample index, and the
parent context (target instance + dependency results). A run's results are
therefore a pure function of (documents, specs, model) looked up in this
store: re-running after an edit recomputes only the affected calls.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

from hnc_extract.extract import CallRecord

# Records with these statuses are final; others ("error") are retried.
FINAL_STATUSES = {"ok", "flagged", "invalid"}


def call_key(
    *,
    doc_sha256: str,
    spec_fingerprint: str,
    model_fingerprint: str,
    context: dict,
    sample: int = 0,
    prompt_format: int = 0,
) -> str:
    payload = {
        "prompt_format": prompt_format,
        "doc": doc_sha256,
        "spec": spec_fingerprint,
        "model": model_fingerprint,
        "context": context,
        "sample": sample,
    }
    blob = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Store:
    def __init__(self, path: Path, readonly: bool = False):
        if readonly:
            # A finished run, e.g. published on read-only shared storage: no locks,
            # no journal, never written.
            self.db = sqlite3.connect(
                f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True
            )
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS calls (
                key TEXT PRIMARY KEY,
                field TEXT NOT NULL,
                doc_sha256 TEXT NOT NULL,
                status TEXT NOT NULL,
                record TEXT NOT NULL,
                created_utc TEXT NOT NULL
            )""")
        self.db.commit()

    @staticmethod
    def open_for_reading(path: Path) -> Store:
        """Read-only (immutable) if the run dir isn't writable, e.g. a published run."""
        if not path.exists():
            sys.exit(f"{path} not found")
        return Store(path, readonly=not os.access(path.parent, os.W_OK))

    def finalize(self) -> None:
        """Fold the WAL back into the file, so a copy of it alone is complete."""
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.commit()

    def get(self, key: str) -> CallRecord | None:
        row = self.db.execute(
            "SELECT record FROM calls WHERE key = ?", (key,)
        ).fetchone()
        return CallRecord.from_json(row[0]) if row else None

    def get_final(self, key: str) -> CallRecord | None:
        rec = self.get(key)
        return rec if rec is not None and rec.status in FINAL_STATUSES else None

    def put(self, rec: CallRecord) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO calls (key, field, doc_sha256, status, record, created_utc) VALUES (?,?,?,?,?,?)",
            (
                rec.key,
                rec.field,
                rec.doc_sha256,
                rec.status,
                rec.to_json(),
                rec.created_utc,
            ),
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()
