import shutil

import yaml
from conftest import SPECS, make_doc, scripted_pathology_responder

from hnc_extract.dag import run, traverse
from hnc_extract.llm import MockClient, ModelConfig
from hnc_extract.rollup import pathology_rollup
from hnc_extract.spec import load_specs
from hnc_extract.store import Store

MODEL = ModelConfig(name="t", provider="mock")


def _run(specs, store, docs, responder=scripted_pathology_responder, **kw):
    client = MockClient(responder)
    run(docs, specs, MODEL, client, store, concurrency=4, log=lambda m: None, **kw)
    return client


def test_full_dag_gating_and_rollup(specs, tmp_path):
    store = Store(tmp_path / "c.sqlite")
    doc = make_doc()
    _run(specs, store, [doc])
    nodes = traverse(doc, specs, store, MODEL.fingerprint())
    st = {k: n.status for k, n in nodes.items()}
    assert st[("specimen", "")] == "done"
    assert st[("histology", "specimen[0]")] == "done"
    assert st[("histology", "specimen[1]")] == "gated_out"  # B is negative
    assert (
        st[("dysplasia_grade", "specimen[0]")] == "gated_out"
    )  # dysplasia not mentioned
    assert st[("pdl1_score", "ancillary_test[0]")] == "gated_out"  # p16, not PD-L1
    assert st[("test_result", "ancillary_test[0]")] == "done"
    assert all(s in ("done", "gated_out") for s in st.values())

    ru = pathology_rollup(nodes)
    assert ru["report_carcinoma_status"] == "positive"
    assert ru["report_invasive_carcinoma"] is True
    assert ru["report_dysplasia_status"] == "not_mentioned"
    assert [s["carcinoma_status"] for s in ru["specimens"]] == ["positive", "negative"]
    assert ru["tests"][0]["result"] == ["negative"]
    assert ru["incomplete_fields"] == []


def test_child_prompt_gets_target_and_siblings(specs, tmp_path):
    client = _run(specs, Store(tmp_path / "c.sqlite"), [make_doc()])
    calls = [c for c in client.calls if c["meta"]["field"] == "carcinoma_status"]
    by_label = {c["meta"]["context"]["chain"][0]["item"]["label"]: c for c in calls}
    ctx_b = by_label["B"]["meta"]["context"]
    assert ctx_b["chain"][0]["siblings"] == [
        '"Left tongue; excision" (label=A, site=Left tongue)'
    ]
    hist = [c for c in client.calls if c["meta"]["field"] == "histology"]
    assert (
        len(hist) == 1
        and hist[0]["meta"]["context"]["deps"][0]["items"][0]["category"] == "positive"
    )


def test_resume_makes_no_calls_and_dedupes_identical_texts(specs, tmp_path):
    store = Store(tmp_path / "c.sqlite")
    docs = [make_doc("d1"), make_doc("d2", patient="p2")]  # same text
    first = _run(specs, store, docs)
    n_first = len(first.calls)
    single = _run(specs, Store(tmp_path / "other.sqlite"), docs[:1])
    assert n_first == len(single.calls)  # duplicate text extracted once
    again = _run(specs, store, docs)
    assert again.calls == []


def test_spec_edit_reruns_only_field_and_descendants(tmp_path):
    d = tmp_path / "specs"
    shutil.copytree(SPECS, d)
    specs = load_specs(d, "pathology_report")
    store = Store(tmp_path / "c.sqlite")
    _run(specs, store, [make_doc()])

    p = d / "pathology_report" / "carcinoma_status.yaml"
    raw = yaml.safe_load(p.read_text())
    raw["instruction"] += "\n- extra rule"
    p.write_text(yaml.safe_dump(raw))
    specs2 = load_specs(d, "pathology_report")
    client = _run(specs2, store, [make_doc()])
    fields = sorted({c["meta"]["field"] for c in client.calls})
    # histology/keratinization see identical parent outputs -> cached
    assert fields == ["carcinoma_status"]


def test_upstream_change_invalidates_children(specs, tmp_path):
    store = Store(tmp_path / "c.sqlite")
    _run(specs, store, [make_doc()])

    def changed(messages, schema, meta):
        out = scripted_pathology_responder(messages, schema, meta)
        if meta["field"] == "specimen":
            out = out.replace('"site": "Left tongue"', '"site": "Left lateral tongue"')
        return out

    # same spec + model fingerprint but a different store entry would be needed to
    # change the specimen answer; simulate by deleting the specimen call
    store.db.execute("DELETE FROM calls WHERE field = 'specimen'")
    store.db.commit()
    client = _run(specs, store, [make_doc()], responder=changed)
    fields = {c["meta"]["field"] for c in client.calls}
    assert (
        "specimen" in fields and "carcinoma_status" in fields
    )  # children re-run on new target
    assert "ancillary_test" not in fields  # unrelated subtree stays cached


def test_failed_parent_blocks_children(specs, tmp_path):
    store = Store(tmp_path / "c.sqlite")

    def broken(messages, schema, meta):
        if meta["field"] == "specimen":
            return "garbage"
        return scripted_pathology_responder(messages, schema, meta)

    _run(specs, store, [make_doc()], responder=broken)
    nodes = traverse(make_doc(), specs, store, MODEL.fingerprint())
    assert nodes[("specimen", "")].status == "failed"
    assert nodes[("carcinoma_status", "/?")].status == "blocked"
    assert nodes[("test_result", "ancillary_test[0]")].status == "done"
    assert "carcinoma_status" in pathology_rollup(nodes)["incomplete_fields"]


def test_fields_filter_runs_only_requested(specs, tmp_path):
    store = Store(tmp_path / "c.sqlite")
    client = _run(specs, store, [make_doc()], fields={"specimen", "procedure_type"})
    assert {c["meta"]["field"] for c in client.calls} == {"specimen", "procedure_type"}
    nodes = traverse(make_doc(), specs, store, MODEL.fingerprint())
    assert nodes[("histology", "specimen[0]")].status == "blocked"
