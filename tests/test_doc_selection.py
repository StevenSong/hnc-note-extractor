import argparse
from pathlib import Path

import pytest
from conftest import make_doc

from hnc_extract.cli import _doc_filter, _select_docs, read_doc_ids_file
from hnc_extract.documents import write_jsonl

ROOT = Path(__file__).resolve().parents[1]
SMOKE_IDS = ROOT / "configs" / "eval" / "path-smoke.txt"
DEV50 = ROOT / "data" / "ingested" / "dev-test50" / "pathology_report.jsonl"


def _write_docs(path: Path, docs) -> Path:
    write_jsonl(docs, path)
    return path


def test_read_doc_ids_file_strips_comments(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("# header\n\nd1  # why\n  d2\n#d3\n")
    assert read_doc_ids_file(p) == ["d1", "d2"]


def test_read_doc_ids_file_rejects_duplicates_and_empty(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("d1\nd1 # again\n")
    with pytest.raises(SystemExit, match="duplicate"):
        read_doc_ids_file(p)
    p.write_text("# nothing\n")
    with pytest.raises(SystemExit, match="no doc ids"):
        read_doc_ids_file(p)


def test_select_docs_fails_on_unknown_id(tmp_path):
    docs = _write_docs(tmp_path / "d.jsonl", [make_doc("d1"), make_doc("d2")])
    got = _select_docs({"docs": str(docs), "doc_ids": ["d2"]})
    assert [d.doc_id for d in got] == ["d2"]
    with pytest.raises(SystemExit, match="not in"):
        _select_docs({"docs": str(docs), "doc_ids": ["d2", "nope"]})


def test_smoke_set_resolves_without_duplicate_texts():
    ids = read_doc_ids_file(SMOKE_IDS)
    if not DEV50.exists():
        pytest.skip("dev-test50 not ingested")
    docs = _select_docs({"docs": str(DEV50), "doc_ids": ids})
    assert len(docs) == len(ids)
    assert len({d.text_sha256 for d in docs}) == len(docs), "exact-duplicate texts"


def _args(**kw):
    base = {"all_docs": False, "limit": None, "doc_ids": None, "doc_ids_file": None}
    return argparse.Namespace(**{**base, **kw})


def test_doc_filter_overrides_and_clears_saved(tmp_path):
    p = tmp_path / "ids.txt"
    p.write_text("d1 # a\nd2\n")
    saved = _doc_filter(_args(doc_ids_file=p), {})
    assert saved == {"limit": None, "doc_ids": ["d1", "d2"], "doc_ids_file": str(p)}
    assert _doc_filter(_args(), saved) == saved  # resume keeps the filter
    assert _doc_filter(_args(doc_ids="d3"), saved)["doc_ids_file"] is None
    assert _doc_filter(_args(all_docs=True), saved) == {
        "limit": None,
        "doc_ids": None,
        "doc_ids_file": None,
    }
