import csv
import json

import pytest

from hnc_extract.documents import read_jsonl
from hnc_extract.synthetic_ingest import ingest_dataset, unescape_html

PN_COLS = [
    "type",
    "id",
    "DATE_OF_SERVICE_DTTM",
    "NOTE_File_DTTM",
    "Note_Narr",
    "subject.submitter_id",
]


def _write_tsv(path, cols, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


@pytest.fixture
def dataset(tmp_path):
    d = tmp_path / "data" / "toy"
    d.mkdir(parents=True)
    _write_tsv(
        d / "progress_note.tsv",
        PN_COLS,
        [
            {
                "type": "progress_note",
                "id": "b",
                "DATE_OF_SERVICE_DTTM": "2020-01-02 10:00:00",
                "NOTE_File_DTTM": "",
                "Note_Narr": "T2N1 SCC &amp; p16 &gt; 70%",
                "subject.submitter_id": "p1",
            },
            {
                "type": "progress_note",
                "id": "a",
                "DATE_OF_SERVICE_DTTM": "",
                "NOTE_File_DTTM": "2019-05-01 08:00:00",
                "Note_Narr": "seen today",
                "subject.submitter_id": "p1",
            },
            {
                "type": "progress_note",
                "id": "c",
                "DATE_OF_SERVICE_DTTM": "2020-01-03 10:00:00",
                "NOTE_File_DTTM": "",
                "Note_Narr": "   ",
                "subject.submitter_id": "p1",
            },
            {
                "type": "progress_note",
                "id": "d",
                "DATE_OF_SERVICE_DTTM": "2020-01-04 10:00:00",
                "NOTE_File_DTTM": "",
                "Note_Narr": "seen today",
                "subject.submitter_id": "p1",
            },
        ],
    )
    return d


def test_unescape_double_escaped():
    assert unescape_html("CULTURE &amp;amp; STAIN") == "CULTURE & STAIN"
    assert unescape_html("no entities") == "no entities"


def test_ingest_roundtrip(dataset, tmp_path):
    before = (dataset / "progress_note.tsv").read_bytes()
    out = tmp_path / "out"
    m = ingest_dataset(dataset, out)

    assert (dataset / "progress_note.tsv").read_bytes() == before  # input untouched
    docs = list(read_jsonl(out / "progress_note.jsonl"))
    assert [d.doc_id for d in docs] == [
        "a",
        "b",
        "d",
    ]  # sorted by anchor, empty dropped

    a, b, d = docs
    assert b.text == "T2N1 SCC & p16 > 70%" and b.ingest_ops == ["html_unescape"]
    assert (
        a.anchor_source == "NOTE_File_DTTM" and a.anchor_dttm == "2019-05-01T08:00:00"
    )
    assert a.text_sha256 == d.text_sha256  # duplicates kept, share a cache key
    assert "Note_Narr" not in b.meta

    s = m["files"]["progress_note"]
    assert s["dropped_empty_text"] == ["c"]
    assert s["duplicate_text_within_patient"] == {"groups": 1, "extra_docs": 1}
    assert (
        json.loads((out / "manifest.json").read_text())["adapter"] == "synthetic_tsv_v1"
    )


def test_refuses_to_clobber(dataset, tmp_path):
    with pytest.raises(ValueError):
        ingest_dataset(dataset, dataset)
    out = tmp_path / "out"
    ingest_dataset(dataset, out)
    with pytest.raises(FileExistsError):
        ingest_dataset(dataset, out)
    ingest_dataset(dataset, out, overwrite=True)
