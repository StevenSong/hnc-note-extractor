"""TCGA report adapter: HNSC filter, kinds by site, provenance, publish idempotence."""

import csv
import json
from pathlib import Path

import pytest

from hnc_extract import paths, tcga_ingest
from hnc_extract.cli import main
from hnc_extract.documents import read_jsonl
from hnc_extract.tcga_refs import CASE_COLUMNS, TSS_COLUMNS, read_table, write_table

REPO = Path(__file__).resolve().parents[1]
CSV_ON_NODES = Path("/opt/gpudata/cancer/tcga/reports/TCGA_Reports.csv")

CASES = ["TCGA-CV-0001", "TCGA-F7-0002", "TCGA-D6-0003", "TCGA-T3-0004", "TCGA-CV-0005"]
SITES = {"CV": "MD Anderson", "F7": "Asterand", "D6": "Poland", "T3": "MR"}


@pytest.fixture
def refs(tmp_path):
    d = tmp_path / "refs"
    write_table(
        d / "hnsc_cases.tsv",
        CASE_COLUMNS,
        [
            {
                "submitter_id": c,
                "case_id": f"id-{c}",
                "primary_site": "Larynx",
                "disease_type": "SCC",
            }
            for c in CASES
        ],
        "test",
    )
    write_table(
        d / "tss.tsv",
        TSS_COLUMNS,
        [
            {"tss": k, "source_site": v, "study": "HNSC", "bcr": "IGC"}
            for k, v in SITES.items()
        ],
        "test",
    )
    return d


def _csv(path: Path, rows: list[tuple[str, str]]) -> Path:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_filename", "text"])
        w.writerows(rows)
    return path


ROWS = [
    ("TCGA-CV-0001.u1", "DIAGNOSIS. LARYNX: SQUAMOUS CELL. CARCINOMA."),
    ("TCGA-F7-0002.u2", "LARYNX TISSUE CHECKLIST. Specimen type: Laryngectomy."),
    ("TCGA-D6-0003.u3", "Examination result No. Material: larynx."),
    ("TCGA-T3-0004.u4", "Sample Collection Date: Procured by:"),
    ("TCGA-AA-9999.u9", "colon adenocarcinoma"),  # another project
]


def test_keeps_hnsc_clinical_reports_and_tags_translated(tmp_path, refs):
    src = _csv(tmp_path / "TCGA_Reports.csv", ROWS)
    m = tcga_ingest.ingest_csv(src, tmp_path / "out", refs_dir=refs)
    docs = {d.doc_id: d for d in read_jsonl(tmp_path / "out/pathology_report.jsonl")}
    assert sorted(docs) == ["TCGA-CV-0001.u1", "TCGA-D6-0003.u3"]
    cv = docs["TCGA-CV-0001.u1"]
    assert cv.patient_id == "TCGA-CV-0001" and cv.anchor_dttm is None
    assert cv.text == ROWS[0][1] and cv.ingest_ops == []  # verbatim
    assert cv.meta["source_site"] == "MD Anderson" and not cv.meta["translated"]
    assert (
        cv.meta["gdc_file_id"] == "u1"
        and cv.meta["gdc"]["case_id"] == "id-TCGA-CV-0001"
    )
    assert docs["TCGA-D6-0003.u3"].meta["translated"]
    s = m["files"]["pathology_report"]
    assert s["rows"] == 5 and s["hnsc_rows"] == 4
    assert s["dropped"]["biobank_abstract"] == ["TCGA-F7-0002.u2"]
    assert s["dropped"]["procurement_form"] == ["TCGA-T3-0004.u4"]
    assert s["cases_without_report"] == ["TCGA-CV-0005"]
    assert set(m["references"]) == {"hnsc_cases.tsv", "tss.tsv"}


def test_unknown_site_code_is_an_error(tmp_path, refs):
    rows = [("TCGA-ZZ-0001.u1", "text")]
    cases = read_table(refs / "hnsc_cases.tsv") + [
        {
            "submitter_id": "TCGA-ZZ-0001",
            "case_id": "x",
            "primary_site": "",
            "disease_type": "",
        }
    ]
    write_table(refs / "hnsc_cases.tsv", CASE_COLUMNS, cases, "test")
    with pytest.raises(ValueError, match="TSS ZZ"):
        tcga_ingest.ingest_csv(
            _csv(tmp_path / "r.csv", rows), tmp_path / "out", refs_dir=refs
        )


def test_ingest_publish_tcga_is_idempotent(tmp_path, refs, monkeypatch):
    src = _csv(tmp_path / "TCGA_Reports.csv", ROWS)
    reg = tmp_path / "datasets.yaml"
    reg.write_text(f"tcga-hnsc: {{adapter: tcga_reports, source: {src}}}\n")
    monkeypatch.setattr(paths, "DATASETS_FILE", reg)
    monkeypatch.setattr(paths, "SHARED_ROOT", tmp_path / "shared")
    monkeypatch.setattr(paths, "RUN_ROOT", tmp_path / "rr")
    monkeypatch.setattr(tcga_ingest, "REFS_DIR", refs)
    main(["ingest-publish", "tcga-hnsc"])
    dest = tmp_path / "shared/ingested/tcga-hnsc"
    assert len(list(read_jsonl(dest / "pathology_report.jsonl"))) == 2
    main(["ingest-publish", "tcga-hnsc"])  # same inputs: nothing to do
    # A reference table change is an input change, like a CSV change.
    sites = read_table(refs / "tss.tsv")
    sites[0]["source_site"] = "renamed"
    write_table(refs / "tss.tsv", TSS_COLUMNS, sites, "edited")
    with pytest.raises(SystemExit, match="other inputs"):
        main(["ingest-publish", "tcga-hnsc"])


def test_unknown_dataset_is_an_error():
    with pytest.raises(SystemExit, match="unknown dataset"):
        main(["ingest-publish", "no-such-dataset"])


def test_committed_refs_cover_the_real_csv(tmp_path):
    """On a node with the CSV: every HNSC report's site is in the committed table and
    the kind rules leave the counts the dev set was planned on."""
    if not CSV_ON_NODES.is_file():
        pytest.skip("TCGA_Reports.csv not on this host")
    m = tcga_ingest.ingest_csv(CSV_ON_NODES, tmp_path / "out")
    s = m["files"]["pathology_report"]
    assert (s["hnsc_rows"], s["documents"], s["translated"]) == (520, 505, 39)
    assert {k: len(v) for k, v in s["dropped"].items()} == {
        "biobank_abstract": 13,
        "procurement_form": 2,
        "empty_text": 0,
    }
