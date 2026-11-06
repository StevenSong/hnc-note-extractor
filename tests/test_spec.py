import shutil
from pathlib import Path

import pytest
import yaml

from hnc_extract.spec import load_specs

SPECS = Path(__file__).resolve().parents[1] / "specs"


def test_pathology_specs_load_and_order():
    s = load_specs(SPECS, "pathology_report")
    order = s.order
    assert order[0] == "specimen"
    assert order.index("carcinoma_status") < order.index("histology")
    assert order.index("ancillary_test") < order.index("pdl1_score")
    for spec in s:
        schema = spec.json_schema()
        assert schema["additionalProperties"] is False
        assert spec.fingerprint


def test_output_model_enforces_categories_and_max_items():
    s = load_specs(SPECS, "pathology_report")
    m = s["procedure_type"].output_model()
    item = {
        "evidence": ["x"],
        "justification": "j",
        "assertion": "affirmed",
        "value": "biopsy",
        "category": "biopsy",
    }
    m.model_validate({"extractions": [item], "summary": "s"})
    with pytest.raises(Exception):
        m.model_validate({"extractions": [item, item], "summary": "s"})  # max_items 1
    with pytest.raises(Exception):
        m.model_validate(
            {"extractions": [{**item, "category": "biopsy-ish"}], "summary": "s"}
        )
    with pytest.raises(Exception):
        m.model_validate({"extractions": [{**item, "evidence": []}], "summary": "s"})
    with pytest.raises(Exception):
        m.model_validate({"extractions": [{**item, "extra": 1}], "summary": "s"})


def _copy_specs(tmp_path: Path) -> Path:
    dst = tmp_path / "specs"
    shutil.copytree(SPECS, dst)
    return dst


def _edit(path: Path, **changes):
    raw = yaml.safe_load(path.read_text())
    raw.update(changes)
    path.write_text(yaml.safe_dump(raw))


@pytest.mark.parametrize(
    "field,changes,msg",
    [
        ("histology", {"scope": "carcinoma_status"}, "not an entity"),
        (
            "histology",
            {"depends_on": [{"field": "carcinoma_status", "categories": ["maybe"]}]},
            "no categories",
        ),
        ("histology", {"depends_on": [{"field": "ancillary_test"}]}, "share scope"),
        ("pdl1_score", {"scope_filter": {"test_type": ["pdl2"]}}, "scope_filter"),
        ("specimen", {"target_instruction": None}, "target_instruction"),
    ],
)
def test_invalid_specs_rejected(tmp_path, field, changes, msg):
    d = _copy_specs(tmp_path)
    _edit(d / "pathology_report" / f"{field}.yaml", **changes)
    with pytest.raises(ValueError, match=msg):
        load_specs(d, "pathology_report")


def test_bad_example_rejected(tmp_path):
    d = _copy_specs(tmp_path)
    p = d / "pathology_report" / "keratinization.yaml"
    raw = yaml.safe_load(p.read_text())
    raw["examples"][0]["output"]["extractions"][0]["category"] = "keratinised"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="example 0"):
        load_specs(d, "pathology_report")


def test_fingerprint_changes_with_instruction_but_not_order(tmp_path):
    d = _copy_specs(tmp_path)
    before = {s.name: s.fingerprint for s in load_specs(d, "pathology_report")}
    _edit(d / "pathology_report" / "histology.yaml", order=999)
    mid = {s.name: s.fingerprint for s in load_specs(d, "pathology_report")}
    assert mid == before
    _edit(d / "pathology_report" / "histology.yaml", instruction="changed")
    after = {s.name: s.fingerprint for s in load_specs(d, "pathology_report")}
    assert [n for n in before if before[n] != after[n]] == ["histology"]
