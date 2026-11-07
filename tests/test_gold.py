import json

from conftest import make_doc, scripted_pathology_responder
from openpyxl import load_workbook

from hnc_extract import gold
from hnc_extract.dag import run, traverse
from hnc_extract.llm import MockClient, ModelConfig
from hnc_extract.rollup import pathology_rollup
from hnc_extract.store import Store


def test_norm_label():
    assert gold.norm_label("A.") == "A"
    assert gold.norm_label(" b) ") == "B"
    assert gold.norm_label("X") == "X"  # part X is real
    assert gold.norm_label("XX-XXXX") is None  # de-identification mask
    assert gold.norm_label(None) is None


def test_equivalence_rules():
    c = gold.canon
    assert gold.equivalent("site", c("site", "Fundal"), c("site", "Fundal masses"))
    assert not gold.equivalent("site", c("site", "Left neck"), c("site", "Right neck"))
    h = "histology"
    assert gold.equivalent(
        h,
        c(h, ["Diffuse large B-cell lymphoma, not otherwise specified"]),
        c(h, ["Diffuse large B-cell lymphoma, not otherwise specified - Non-GCB"]),
    )
    assert not gold.equivalent(h, c(h, ["Carcinoma"]), c(h, []))
    assert c("invasion", ["not_stated"]) == gold.NONE
    assert gold.equivalent(
        "specimen_label", c("specimen_label", "B2"), c("specimen_label", "B")
    )
    assert not gold.equivalent(
        "specimen_label", c("specimen_label", "AB"), c("specimen_label", "A")
    )
    assert c("pdl1_scores", ["TPS 15%"]) == c("pdl1_scores", ["TPS 15"])


def test_vote_needs_min_agree_models():
    v = gold.vote(
        "carcinoma_status", {"a": "positive", "b": "positive", "c": gold.ABSENT}
    )
    assert v.proposed == "positive" and not v.agreed  # only 2 models found it
    v = gold.vote("carcinoma_status", {"a": "x", "b": "x", "c": "x", "d": "y"})
    assert (v.proposed, v.n_majority, v.agreed) == ("x", 3, False)
    v = gold.vote("carcinoma_status", {"a": "x", "b": "x", "c": "x", "d": gold.ABSENT})
    assert v.agreed


def _sp(label, spans):
    return {"label": label, "spans": spans}


def test_align_by_label_then_span():
    items = {
        "m1": [_sp("A", [(0, 30)]), _sp(None, [(100, 140)])],
        "m2": [_sp(None, [(105, 140)]), _sp("A", [(200, 230)])],  # A by label
        "m3": [_sp("B", [(100, 140)])],  # different label from nothing labeled B
    }
    cl = gold.align(items, gold.specimen_score)
    assert {"m1": 0, "m2": 1} in cl
    assert {"m1": 1, "m2": 0, "m3": 0} in cl
    # different non-null labels never match, even on the same text
    assert gold.specimen_score(_sp("A", [(0, 50)]), _sp("B", [(0, 50)])) == 0


def test_biomarkers_in_one_sentence_stay_apart():
    her2 = {"test": "HER2", "test_type": "biomarker_ihc", "spans": [(0, 60)]}
    ar = {"test": "AR", "test_type": "biomarker_ihc", "spans": [(0, 60)]}
    cl = gold.align({"m1": [her2, ar], "m2": [ar, her2]}, gold.test_score)
    assert sorted(cl, key=lambda c: c["m1"]) == [{"m1": 0, "m2": 1}, {"m1": 1, "m2": 0}]


def _model_run(tmp_path, specs, name, responder):
    model = ModelConfig(name=name, provider="mock")
    store = Store(tmp_path / f"{name}.sqlite")
    doc = make_doc()
    run(
        [doc],
        specs,
        model,
        MockClient(responder),
        store,
        concurrency=2,
        log=lambda m: None,
    )
    nodes = {doc.doc_id: traverse(doc, specs, store, model.fingerprint())}
    rollups = {d: pathology_rollup(n) for d, n in nodes.items()}
    return doc, gold.run_views(nodes, rollups), rollups


def _b_positive(messages, schema, meta):
    """Calls part B positive (the scripted responder says negative)."""
    chain = meta["context"]["chain"]
    if meta["field"] == "carcinoma_status" and chain[-1]["item"]["label"] == "B":
        return json.dumps(
            {
                "extractions": [
                    {
                        "evidence": ["negative for carcinoma (0/4)"],
                        "justification": "j",
                        "assertion": "affirmed",
                        "value": "carcinoma",
                        "category": "positive",
                        "basis": None,
                    }
                ],
                "summary": "s",
            }
        )
    return scripted_pathology_responder(messages, schema, meta)


def test_draft_adjudicate_import_score(specs, tmp_path):
    views, rollups = {}, {}
    for name, resp in (
        ("m1", scripted_pathology_responder),
        ("m2", scripted_pathology_responder),
        ("m3", _b_positive),
    ):
        doc, views[name], rollups[name] = _model_run(tmp_path, specs, name, resp)
    records = gold.draft(views, [doc.doc_id])
    specimens = [r for r in records if r["kind"] == "specimen"]
    assert [r["members"] for r in specimens] == [
        {"m1": "specimen[0]", "m2": "specimen[0]", "m3": "specimen[0]"},
        {"m1": "specimen[1]", "m2": "specimen[1]", "m3": "specimen[1]"},
    ]
    rows = gold.review_rows(records, seed=0)
    dis = [r for r in rows if r["reason"] == "disagreement"]
    # m3's positive B also ran histology/keratinization (gated out for m1, m2)
    assert [(r["key"], r["field"], r["proposed"]) for r in dis] == [
        ("S02", "carcinoma_status", "negative"),
        ("S02", "histology", gold.NONE),
        ("S02", "invasion", gold.NONE),
        ("S02", "keratinization", gold.NONE),
    ]

    # adjudicate in the workbook: overrule the majority on S02
    xlsx = tmp_path / "adjudicate.xlsx"
    gold.write_adjudication(xlsx, records, rows, [doc], views, {"t": 1})
    wb = load_workbook(xlsx)
    ws = wb["adjudicate"]
    header = [c.value for c in ws[1]]
    for row in ws.iter_rows(min_row=2):
        if row[header.index("row_id")].value == dis[0]["row_id"]:
            row[header.index(gold.GOLD_COL)].value = "positive"
    wb.save(xlsx)
    verdicts = gold.read_verdicts(xlsx)
    g = gold.build_gold(records, verdicts)[doc.doc_id]
    assert [s["carcinoma_status"] for s in g["specimens"]] == ["positive", "positive"]
    assert g["report"]["report_carcinoma_status"] == "positive"

    # m3 now matches gold on that field; m1 does not
    gd = {doc.doc_id: json.loads(json.dumps(g))}
    s1 = gold.score_run(gd, views["m1"], rollups["m1"])
    s3 = gold.score_run(gd, views["m3"], rollups["m3"])
    assert s1["items"]["specimens"] == {"matched": 2, "missed": 0, "extra": 0}
    assert s1["fields"]["specimens.carcinoma_status"] == {"agree": 1, "disagree": 1}
    assert s3["fields"]["specimens.carcinoma_status"] == {"agree": 2}
    assert [m["field"] for m in s1["mismatches"]] == ["carcinoma_status"]
    assert s3["fields"]["specimens.histology"] == {"agree": 1, "disagree": 1}


def test_presence_no_drops_item():
    rec = {
        "doc_id": "d",
        "kind": "specimen",
        "key": "S01",
        "members": {"m1": "specimen[0]"},
        "spans": [(0, 5)],
        "votes": [
            {
                "field": "present",
                "proposed": "no",
                "answers": {},
                "n_majority": 1,
                "agreed": False,
            },
            {
                "field": "carcinoma_status",
                "proposed": "positive",
                "answers": {},
                "n_majority": 1,
                "agreed": False,
            },
        ],
    }
    assert gold.build_gold([rec], {})["d"]["specimens"] == []
    kept = gold.build_gold([rec], {"d|S01|present": "yes"})["d"]["specimens"]
    assert kept[0]["carcinoma_status"] == "positive"
