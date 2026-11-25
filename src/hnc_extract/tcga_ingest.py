"""Ingest adapter for TCGA pathology reports (TCGA_Reports.csv, Kefeli et al. 2024).

The CSV holds OCR text of the TCGA pathology report PDFs, one row per PDF, columns
patient_filename ("TCGA-CR-7390.<GDC file uuid>") and text. This adapter keeps the
TCGA-HNSC rows (configs/tcga/hnsc_cases.tsv) and writes canonical `Document` JSONL.

Report kinds, decided by tissue source site (configs/tcga/tss.tsv), because each
site sent one kind of document:
  clinical          the hospital's surgical pathology report. Kept.
  biobank_abstract  F7 (Asterand): the biobank's re-typed summary and checklist
                    form, not the hospital report. Dropped.
  procurement_form  T3 (Molecular Response): tissue procurement form, no
                    diagnosis. Dropped.
Reports of D6 (Poland), WA (Kiel) and UF (Barretos) are English translations of the
hospital report. Kept, with meta.translated = true so they can be scored apart.

Text is kept verbatim. The OCR dataset already replaced line breaks with ". ", so
sentences can be split mid-phrase ("SQUAMOUS CELL. CARCINOMA"); that can't be undone.
The reports carry no dates, so anchor_dttm is None.

meta.gdc holds the case's GDC primary site and disease type: chart-abstracted
labels for stratifying and cross-checks, never shown to a model.

Usage:
  python -m hnc_extract.tcga_ingest /opt/gpudata/cancer/tcga/reports/TCGA_Reports.csv
  -> data/ingested/tcga-hnsc/{pathology_report.jsonl,manifest.json}
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

from hnc_extract import paths
from hnc_extract.documents import Document, sha256_text, write_jsonl
from hnc_extract.hashing import sha256_file
from hnc_extract.tcga_refs import read_table

ADAPTER = "tcga_reports_v1"
DATASET = "tcga-hnsc"
REFS_DIR = paths.CONFIGS_DIR / "tcga"

DROPPED_KIND_BY_TSS = {"F7": "biobank_abstract", "T3": "procurement_form"}
TRANSLATED_TSS = frozenset({"D6", "WA", "UF"})


def split_filename(name: str) -> tuple[str, str]:
    """'TCGA-CR-7390.<uuid>' -> ('TCGA-CR-7390', '<uuid>')."""
    patient, sep, uuid = name.partition(".")
    if not sep or patient.count("-") != 2 or not uuid:
        raise ValueError(f"unexpected patient_filename {name!r}")
    return patient, uuid


def ingest_csv(
    csv_path: Path,
    out_dir: Path,
    refs_dir: Path | None = None,
    overwrite: bool = False,
) -> dict:
    out_dir, refs_dir = out_dir.resolve(), refs_dir or REFS_DIR
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} is not empty; pass --overwrite to replace.")
    cases_path, tss_path = refs_dir / "hnsc_cases.tsv", refs_dir / "tss.tsv"
    cases = {r["submitter_id"]: r for r in read_table(cases_path)}
    sites = {r["tss"]: r["source_site"] for r in read_table(tss_path)}

    csv.field_size_limit(sys.maxsize)
    docs: list[Document] = []
    dropped: dict[str, list[str]] = {k: [] for k in DROPPED_KIND_BY_TSS.values()}
    dropped["empty_text"] = []
    n_rows = 0
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"patient_filename", "text"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path}: missing required columns {sorted(missing)}")
        for i, row in enumerate(reader):
            n_rows += 1
            patient, uuid = split_filename(row["patient_filename"])
            if patient not in cases:
                continue
            doc_id, tss = row["patient_filename"], patient.split("-")[1]
            if tss not in sites:
                raise ValueError(f"{doc_id}: TSS {tss} not in {tss_path}")
            if tss in DROPPED_KIND_BY_TSS:
                dropped[DROPPED_KIND_BY_TSS[tss]].append(doc_id)
                continue
            text = row["text"]
            if not text.strip():
                dropped["empty_text"].append(doc_id)
                continue
            case = cases[patient]
            docs.append(
                Document(
                    doc_id=doc_id,
                    doc_type="pathology_report",
                    patient_id=patient,
                    anchor_dttm=None,
                    anchor_source=None,
                    text=text,
                    text_sha256=sha256_text(text),
                    source={
                        "adapter": ADAPTER,
                        "dataset": DATASET,
                        "file": csv_path.name,
                        "row_index": i,
                    },
                    meta={
                        "tss": tss,
                        "source_site": sites[tss],
                        "gdc_file_id": uuid,
                        "report_kind": "clinical",
                        "translated": tss in TRANSLATED_TSS,
                        "gdc": {
                            "case_id": case["case_id"],
                            "primary_site": case["primary_site"],
                            "disease_type": case["disease_type"],
                        },
                    },
                    ingest_ops=[],
                )
            )

    if dup := [k for k, v in Counter(d.doc_id for d in docs).items() if v > 1]:
        raise ValueError(f"{csv_path}: duplicate doc ids {dup[:5]}")
    docs.sort(key=lambda d: (d.patient_id, d.doc_id))
    dst = out_dir / "pathology_report.jsonl"
    write_jsonl(docs, dst)

    with_report = {d.patient_id for d in docs} | {
        split_filename(i)[0] for ids in dropped.values() for i in ids
    }
    stats = {
        "input_file": str(csv_path),
        "input_sha256": sha256_file(csv_path),
        "output_file": str(dst),
        "rows": n_rows,
        "hnsc_rows": len(docs) + sum(len(v) for v in dropped.values()),
        "documents": len(docs),
        "patients": len({d.patient_id for d in docs}),
        "translated": sum(d.meta["translated"] for d in docs),
        "dropped": dropped,
        "cases_without_report": sorted(set(cases) - with_report),
        "documents_by_tss": dict(Counter(d.meta["tss"] for d in docs).most_common()),
        "duplicate_text": sum(
            v - 1 for v in Counter(d.text_sha256 for d in docs).values() if v > 1
        ),
    }
    manifest = {
        "adapter": ADAPTER,
        "dataset": DATASET,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "references": {p.name: sha256_file(p) for p in (cases_path, tss_path)},
        "files": {"pathology_report": stats},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("csv", type=Path, help="TCGA_Reports.csv")
    p.add_argument(
        "--out", type=Path, default=None, help=f"default: data/ingested/{DATASET}"
    )
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    out = a.out or paths.REPO_ROOT / "data" / "ingested" / DATASET
    m = ingest_csv(a.csv, out, overwrite=a.overwrite)
    s = m["files"]["pathology_report"]
    print(
        f"rows={s['rows']} hnsc={s['hnsc_rows']} docs={s['documents']} "
        f"translated={s['translated']} "
        + " ".join(f"{k}={len(v)}" for k, v in s["dropped"].items())
        + f" cases_without_report={len(s['cases_without_report'])}"
    )
    print(f"-> {out}")


if __name__ == "__main__":
    main()
