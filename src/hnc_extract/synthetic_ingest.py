"""Ingest adapter for the SYNTHETIC note TSVs in data/<dataset>/. Synthetic only.

Applies minimal fixes that make the synthetic text look more like real EHR text,
then writes canonical `Document` JSONL. Original files are only ever read.

Fixes applied (recorded per document in `ingest_ops`):
  * html_unescape: '&amp;' -> '&', '&lt;' -> '<', ... repeated until stable
    (at least one note is double-escaped).
  * Empty-text rows are dropped and listed in the manifest.

Deliberately NOT done (to keep ingest minimal and lossless):
  * No whitespace normalization, no reconstruction of line breaks.
  * No de-duplication. Identical texts stay separate documents (e.g. one
    radiology dictation attached to several orders). Downstream caching is
    keyed on text_sha256, so duplicates are extracted once.
  * No merging of pathology rows. Order_proc_id is unreliable in the synthetic
    data (rows sharing it have different accessions months apart).
  * No attempt to undo de-identification masks (XX/XX/XXXX, XX-XXXX).

Usage:
  python -m hnc_extract.synthetic_ingest data/dev-test50
  -> data/ingested/dev-test50/{progress_note,pathology_report,radiology_report}.jsonl
     data/ingested/dev-test50/manifest.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import sys
from collections import Counter
from pathlib import Path

from hnc_extract.documents import DOC_TYPES, DocType, Document, sha256_text, write_jsonl

ADAPTER = "synthetic_tsv_v1"

TEXT_COLUMN: dict[DocType, str] = {
    "progress_note": "Note_Narr",
    "pathology_report": "path_note",
    "radiology_report": "Rad_note",
}

# First non-empty column wins. Chosen as "when the content applies":
#   progress note  -> date of service
#   radiology      -> exam time, then report time
#   pathology      -> report result/sign-out time (the time the text was written)
ANCHOR_COLUMNS: dict[DocType, tuple[str, ...]] = {
    "progress_note": ("DATE_OF_SERVICE_DTTM", "NOTE_File_DTTM"),
    "radiology_report": ("TAKEN_DTTM", "NOTE_DATE", "RESULT_DTTM"),
    "pathology_report": ("result_dttm", "order_dttm", "enc_date"),
}

ID_COLUMN = "id"
PATIENT_COLUMN = "subject.submitter_id"


def unescape_html(text: str, max_rounds: int = 5) -> str:
    for _ in range(max_rounds):
        new = html.unescape(text)
        if new == text:
            return new
        text = new
    return text


def parse_dttm(value: str) -> str | None:
    value = value.strip()
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value).isoformat()
    except ValueError:
        return None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_file(
    path: Path, doc_type: DocType, dataset: str
) -> tuple[list[Document], dict]:
    csv.field_size_limit(sys.maxsize)
    text_col = TEXT_COLUMN[doc_type]
    docs: list[Document] = []
    dropped_empty: list[str] = []
    bad_anchor: list[str] = []
    n_rows = 0

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = {text_col, ID_COLUMN, PATIENT_COLUMN} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}")
        for i, row in enumerate(reader):
            n_rows += 1
            raw = row[text_col] or ""
            if not raw.strip():
                dropped_empty.append(row[ID_COLUMN])
                continue

            ops: list[str] = []
            text = unescape_html(raw)
            if text != raw:
                ops.append("html_unescape")

            anchor, anchor_src = None, None
            for col in ANCHOR_COLUMNS[doc_type]:
                anchor = parse_dttm(row.get(col) or "")
                if anchor:
                    anchor_src = col
                    break
            if anchor is None:
                bad_anchor.append(row[ID_COLUMN])

            docs.append(
                Document(
                    doc_id=row[ID_COLUMN],
                    doc_type=doc_type,
                    patient_id=row[PATIENT_COLUMN],
                    anchor_dttm=anchor,
                    anchor_source=anchor_src,
                    text=text,
                    text_sha256=sha256_text(text),
                    source={
                        "adapter": ADAPTER,
                        "dataset": dataset,
                        "file": path.name,
                        "row_index": i,
                    },
                    meta={k: v for k, v in row.items() if k != text_col},
                    ingest_ops=ops,
                )
            )

    ids = Counter(d.doc_id for d in docs)
    dup_ids = [k for k, v in ids.items() if v > 1]
    if dup_ids:
        raise ValueError(f"{path}: duplicate doc ids {dup_ids[:5]}")

    docs.sort(key=lambda d: (d.patient_id, d.anchor_dttm or "", d.doc_id))
    text_groups = Counter((d.patient_id, d.text_sha256) for d in docs)
    stats = {
        "input_file": str(path),
        "input_sha256": _file_sha256(path),
        "rows": n_rows,
        "documents": len(docs),
        "patients": len({d.patient_id for d in docs}),
        "dropped_empty_text": dropped_empty,
        "html_unescaped": sum("html_unescape" in d.ingest_ops for d in docs),
        "missing_anchor_dttm": bad_anchor,
        "anchor_sources": dict(Counter(d.anchor_source for d in docs)),
        "duplicate_text_within_patient": {
            "groups": sum(1 for v in text_groups.values() if v > 1),
            "extra_docs": sum(v - 1 for v in text_groups.values() if v > 1),
        },
    }
    return docs, stats


def ingest_dataset(in_dir: Path, out_dir: Path, overwrite: bool = False) -> dict:
    in_dir, out_dir = in_dir.resolve(), out_dir.resolve()
    if out_dir == in_dir:
        raise ValueError("Output directory must differ from input directory.")
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} is not empty; pass --overwrite to replace.")

    manifest = {
        "adapter": ADAPTER,
        "dataset": in_dir.name,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "files": {},
    }
    for doc_type in DOC_TYPES:
        src = in_dir / f"{doc_type}.tsv"
        if not src.exists():
            continue
        docs, stats = ingest_file(src, doc_type, in_dir.name)
        dst = out_dir / f"{doc_type}.jsonl"
        if dst.resolve() == src.resolve():
            raise ValueError(f"Refusing to write over input file {src}")
        write_jsonl(docs, dst)
        stats["output_file"] = str(dst)
        manifest["files"][doc_type] = stats

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("in_dir", type=Path, help="e.g. data/dev-test50")
    p.add_argument(
        "--out", type=Path, default=None, help="default: data/ingested/<dataset>"
    )
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    out = a.out or a.in_dir.resolve().parent / "ingested" / a.in_dir.resolve().name
    m = ingest_dataset(a.in_dir, out, overwrite=a.overwrite)
    for t, s in m["files"].items():
        print(
            f"{t:18s} rows={s['rows']:5d} docs={s['documents']:5d} patients={s['patients']:3d} "
            f"dropped_empty={len(s['dropped_empty_text'])} unescaped={s['html_unescaped']} "
            f"no_anchor={len(s['missing_anchor_dttm'])} dup_text_extra={s['duplicate_text_within_patient']['extra_docs']}"
        )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
