"""Canonical document format: the contract between ingest adapters and the pipeline.

Each source (synthetic TSVs now, real EHR exports later) gets its own adapter
that emits `Document` records as JSONL. Everything downstream reads only this
format, so source-specific quirks never leak into extraction code.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

DocType = Literal["progress_note", "pathology_report", "radiology_report"]
DOC_TYPES: tuple[DocType, ...] = (
    "progress_note",
    "pathology_report",
    "radiology_report",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class Document:
    doc_id: str  # stable unique id from the source system
    doc_type: DocType
    patient_id: str
    # Reference datetime for the document (ISO 8601, naive local time). Used to
    # resolve relative time expressions ("3 months ago") and to order notes.
    anchor_dttm: str | None
    anchor_source: str | None  # which source column anchor_dttm came from
    text: str
    text_sha256: str  # extraction cache key; identical texts are extracted once
    source: dict  # provenance: adapter, dataset, file, row index
    meta: dict  # all remaining source columns, untouched
    ingest_ops: list[str]  # transformations applied to text, in order

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> Document:
        return cls(**json.loads(line))


def write_jsonl(docs: Iterable[Document], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(d.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[Document]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Document.from_json(line)
