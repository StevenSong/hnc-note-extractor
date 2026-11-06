import json
from pathlib import Path

import pytest

from hnc_extract.documents import Document, sha256_text
from hnc_extract.spec import load_specs

SPECS = Path(__file__).resolve().parents[1] / "specs"

PATH_TEXT = (
    "Surgical Pathology Report  A.  Left tongue; excision:  - Invasive squamous cell carcinoma, keratinizing.  "
    "B.  Left neck; dissection:  - Four lymph nodes, negative for carcinoma (0/4).  "
    "Addendum: On A, p16 immunostain is negative."
)


def make_doc(doc_id="d1", text=PATH_TEXT, patient="p1") -> Document:
    return Document(
        doc_id=doc_id,
        doc_type="pathology_report",
        patient_id=patient,
        anchor_dttm="2024-01-02T03:04:05",
        anchor_source="test",
        text=text,
        text_sha256=sha256_text(text),
        source={},
        meta={},
        ingest_ops=[],
    )


@pytest.fixture
def specs():
    return load_specs(SPECS, "pathology_report")


def item(evidence, value, **kw):
    return {
        "evidence": evidence,
        "justification": "j",
        "assertion": "affirmed",
        "value": value,
        **kw,
    }


def answer(*items, summary="s"):
    return json.dumps({"extractions": list(items), "summary": summary})


def scripted_pathology_responder(messages, schema, meta):
    """Correct answers for PATH_TEXT, keyed on field and target label."""
    f = meta["field"]
    chain = meta["context"]["chain"]
    tgt = chain[-1]["item"] if chain else {}
    label = tgt.get("label") or tgt.get("specimen_label")
    if f == "specimen":
        return answer(
            item(
                ["A.  Left tongue; excision:"],
                "Left tongue; excision",
                label="A",
                site="Left tongue",
                site_source="designation",
                material="submitted_tissue",
            ),
            item(
                ["B.  Left neck; dissection:"],
                "Left neck; dissection",
                label="B",
                site="Left neck",
                site_source="designation",
                material="submitted_tissue",
            ),
        )
    if f == "procedure_type":
        return answer(
            item(["excision"], "excision", category="excision")
            if label == "A"
            else item(["dissection"], "dissection", category="lymph_node_dissection")
        )
    if f == "carcinoma_status":
        if label == "A":
            return answer(
                item(
                    ["Invasive squamous cell carcinoma"],
                    "Invasive squamous cell carcinoma",
                    category="positive",
                    basis=None,
                )
            )
        return answer(
            item(
                ["negative for carcinoma (0/4)"],
                "negative for carcinoma",
                category="negative",
                basis="explicit_negative",
            )
        )
    if f == "histology":
        return answer(
            item(
                ["Invasive squamous cell carcinoma"],
                "Invasive squamous cell carcinoma",
                invasion="invasive",
                differentiation=None,
                described_as_metastatic="no",
            )
        )
    if f == "keratinization":
        return answer(item(["keratinizing"], "keratinizing", category="keratinizing"))
    if f in ("dysplasia_status", "pdl1_score"):
        return answer()
    if f == "ancillary_test":
        return answer(
            item(
                ["On A, p16 immunostain is negative."],
                "p16 immunostain",
                test_type="p16_ihc",
                specimen_label="A",
            )
        )
    if f == "test_result":
        return answer(
            item(
                ["p16 immunostain is negative"],
                "negative",
                category="negative",
                percent_cells=None,
                intensity=None,
                pattern=None,
            )
        )
    raise AssertionError(f"unexpected field {f}")
