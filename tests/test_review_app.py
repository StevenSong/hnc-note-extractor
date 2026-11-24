import json
import threading
import urllib.error
import urllib.request

import pytest
from conftest import scripted_pathology_responder
from test_gold import _b_positive, _model_run

from hnc_extract import cli, gold, review_app

META = {"created_utc": "2026-01-01T00:00:00+00:00", "seed": 0, "spot_check_rate": 0.1}


@pytest.fixture
def draft(specs, tmp_path):
    """A gold-draft dir from three mock models (m3 calls part B positive)."""
    views = {}
    for name, resp in (
        ("m1", scripted_pathology_responder),
        ("m2", scripted_pathology_responder),
        ("m3", _b_positive),
    ):
        doc, views[name], _ = _model_run(tmp_path, specs, name, resp)
    meta = {**META, "doc_ids": [doc.doc_id], "runs": {}, "specs": "x"}
    d = tmp_path / "gold"
    d.mkdir()
    gold.write_jsonl(d / "draft.jsonl", gold.draft(views, [doc.doc_id]))
    (d / "draft_meta.json").write_text(json.dumps(meta))
    records = gold.read_jsonl(d / "draft.jsonl")
    bundle = review_app.build_bundle(
        meta, records, [doc], views, gold.field_options(specs)
    )
    review_app.write_bundle(d, bundle)
    return d, doc, records, bundle, meta


def test_field_options_come_from_specs(specs):
    opts = gold.field_options(specs)
    assert opts["carcinoma_status"] == [
        "positive",
        "suspicious",
        "negative",
        "non_diagnostic",
    ]
    assert "not_stated" not in opts["invasion"]  # canon() drops it
    assert "not_stated" in opts["material"]  # a real answer there
    assert gold.validate_value("invasion", ["invasive", "bogus"], opts)
    assert gold.validate_value("carcinoma_status", ["positive", "negative"], opts)
    assert gold.validate_value("present", None, opts)
    assert gold.validate_value("carcinoma_status", None, opts) is None
    assert gold.validate_value("site", "anything", opts) is None


def test_bundle_has_per_field_evidence_and_review_reasons(draft):
    d, doc, records, bundle, meta = draft
    s2 = next(r for r in bundle["records"] if r["key"] == "S02")
    carc = next(v for v in s2["votes"] if v["field"] == "carcinoma_status")
    assert carc["reason"] == "disagreement"
    assert carc["answers"] == {"m1": "negative", "m2": "negative", "m3": "positive"}
    ev = s2["evidence"]["m3"]["carcinoma_status"]
    s, e = ev[0]["spans"][0]
    assert doc.text[s:e] == "negative for carcinoma (0/4)"
    assert "category=positive" in ev[0]["said"] and ev[0]["why"] == "j"
    # m1 never ran histology on its negative B: gated, not "found nothing"
    assert s2["evidence"]["m1"]["histology"] is None
    # same review rows as the workbook
    rows = gold.review_rows(records, seed=0, rate=0.1)
    marked = {
        f"{r['doc_id']}|{r['key']}|{v['field']}"
        for r in bundle["records"]
        for v in r["votes"]
        if v["reason"]
    }
    assert marked == {r["row_id"] for r in rows}


def test_state_to_gold_with_override_absent_item_and_added(draft):
    d, doc, records, bundle, meta = draft
    state = review_app.empty_state(meta)
    did = doc.doc_id
    state["fields"][f"{did}|S02|carcinoma_status"] = {
        "value": "positive",
        "reviewed": True,
    }
    state["fields"][f"{did}|S01|present"] = {"value": "no", "reviewed": True}
    state["added"].append(
        {
            "doc_id": did,
            "kind": "specimen",
            "key": "AS01",
            "spans": [[0, 10]],
            "values": {"label": "Z", "carcinoma_status": "negative", "invasion": []},
            "comment": "",
        }
    )
    assert review_app.validate_state(state, records, bundle["options"]) == []
    verdicts, added = review_app.state_verdicts(state)
    g = gold.build_gold(records, verdicts, added)[did]
    assert [(s["key"], s["carcinoma_status"]) for s in g["specimens"]] == [
        ("S02", "positive"),
        ("AS01", "negative"),
    ]
    assert g["specimens"][1]["histology"] == []  # unset list field
    assert g["report"]["report_carcinoma_status"] == "positive"


def test_validate_state_catches_bad_values_and_unmatchable_items(draft):
    d, doc, records, bundle, meta = draft
    state = review_app.empty_state(meta)
    did = doc.doc_id
    state["fields"][f"{did}|S02|carcinoma_status"] = {"value": "postive"}
    state["fields"][f"{did}|S99|site"] = {"value": "x"}
    state["added"].append(
        {"doc_id": did, "kind": "specimen", "key": "AS01", "spans": [], "values": {}}
    )
    errs = review_app.validate_state(state, records, bundle["options"])
    assert len(errs) == 3
    assert "postive" in errs[0] and "not in the draft" in errs[1]
    assert "part label or selected evidence" in errs[2]


def test_progress_counts_if_present_rows_only_for_present_items(draft):
    d, doc, records, bundle, meta = draft
    reasons = {
        f"{r['doc_id']}|{r['key']}|{v['field']}": v["reason"]
        for r in bundle["records"]
        for v in r["votes"]
        if v["reason"]
    }
    state = review_app.empty_state(meta)
    p0 = review_app.progress(records, reasons, state)
    assert p0["done"] == 0 and p0["required"] == len(reasons)
    did = doc.doc_id
    rid = f"{did}|S02|carcinoma_status"
    state["fields"][rid] = {"value": "positive", "reviewed": True}
    assert review_app.progress(records, reasons, state)["done"] == 1
    # S02 was found by every model, so its present row needs no review; once it
    # is answered "no", S02's disagreement rows no longer count either
    s02 = [k for k in reasons if k.startswith(f"{did}|S02|")]
    assert f"{did}|S02|present" not in reasons and len(s02) == 4
    state["fields"][f"{did}|S02|present"] = {"value": "no", "reviewed": True}
    p = review_app.progress(records, reasons, state)
    assert p == {"required": len(reasons) - 4, "done": 0}


def _req(url, data=None, headers=None):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_server_roundtrip(draft):
    d, doc, records, bundle, meta = draft
    srv = review_app.make_server(d, bundle, records, meta, port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        code, body = _req(base + "/")
        assert code == 200 and b"Gold review" in body
        code, body = _req(base + "/api/bundle")
        assert json.loads(body)["records"][0]["key"] == "S01"
        code, body = _req(base + "/api/state", headers={"Host": "evil.example"})
        assert code == 403
        state = json.loads(_req(base + "/api/state")[1])
        did = doc.doc_id
        state["fields"][f"{did}|S02|carcinoma_status"] = {
            "value": "positive",
            "reviewed": True,
        }
        js = {"Content-Type": "application/json"}
        code, body = _req(base + "/api/state", json.dumps(state).encode(), js)
        out = json.loads(body)
        assert code == 200 and out["errors"] == []
        assert out["reports"][did]["report_carcinoma_status"] == "positive"
        assert json.loads((d / "verdicts.json").read_text()) == state
        # a second save keeps the first as .bak
        _req(base + "/api/state", json.dumps(state).encode(), js)
        assert (d / "verdicts.json.bak").exists()
        # state for another draft, or not JSON: refused, file untouched
        bad = {**state, "draft_created_utc": "other"}
        assert _req(base + "/api/state", json.dumps(bad).encode(), js)[0] == 400
        assert _req(base + "/api/state", b"x", {"Content-Type": "text/plain"})[0] == 415
        assert json.loads((d / "verdicts.json").read_text()) == state
    finally:
        srv.shutdown()
        srv.server_close()


def test_gold_import_from_app_state(draft, capsys):
    d, doc, records, bundle, meta = draft
    state = review_app.empty_state(meta)
    state["fields"][f"{doc.doc_id}|S02|carcinoma_status"] = {
        "value": "positive",
        "reviewed": True,
    }
    review_app.write_state(d, state)
    cli.main(["gold-import", "--dir", str(d)])
    out = capsys.readouterr().out
    assert "verdicts.json" in out and "required rows not reviewed" in out
    g = gold.read_jsonl(d / "gold.jsonl")[0]
    assert [s["carcinoma_status"] for s in g["specimens"]] == ["positive", "positive"]

    state["fields"][f"{doc.doc_id}|S02|carcinoma_status"]["value"] = "postive"
    review_app.write_state(d, state)
    with pytest.raises(SystemExit, match="1 invalid verdict"):
        cli.main(["gold-import", "--dir", str(d)])
