import json

from conftest import answer, item, make_doc

from hnc_extract.extract import parse_json_object, run_call
from hnc_extract.llm import Completion, ModelConfig, ScriptedClient

MODEL = ModelConfig(name="t", provider="mock", max_attempts=3)
GOOD = answer(item(["keratinizing"], "keratinizing", category="keratinizing"))
CTX = {"chain": [], "deps": []}


def _run(client, specs, field="keratinization", model=MODEL):
    return run_call(
        client=client,
        model=model,
        spec=specs[field],
        instructions=specs.instructions,
        doc=make_doc(),
        context=CTX,
        key="ab" * 32,
    )


def test_ok_first_try(specs):
    rec = _run(ScriptedClient([GOOD]), specs)
    assert rec.status == "ok" and len(rec.attempts) == 1
    assert rec.evidence[0][0]["found"]


def test_invalid_json_then_fixed_sends_feedback(specs):
    c = ScriptedClient(["not json", GOOD])
    rec = _run(c, specs)
    assert rec.status == "ok" and len(rec.attempts) == 2
    second = c.calls[1]["messages"]
    assert second[2] == {"role": "assistant", "content": "not json"}
    assert "does not conform" in second[3]["content"]


def test_schema_violation_retried(specs):
    bad = answer(item(["keratinizing"], "keratinizing", category="very keratinizing"))
    rec = _run(ScriptedClient([bad, GOOD]), specs)
    assert rec.status == "ok" and "invalid" in rec.attempts[0]["error"]


def test_unverified_evidence_retried_then_flagged(specs):
    fake = answer(
        item(
            ["keratinizing squamous carcinoma, invasive"],
            "keratinizing",
            category="keratinizing",
        )
    )
    c = ScriptedClient([fake, fake, fake])
    rec = _run(c, specs)
    assert rec.status == "flagged" and len(rec.attempts) == 3
    assert rec.output is not None and not rec.evidence[0][0]["found"]
    assert "do not occur verbatim" in c.calls[1]["messages"][3]["content"]


def test_unverified_then_fixed(specs):
    fake = answer(item(["keratinising"], "keratinizing", category="keratinizing"))
    rec = _run(ScriptedClient([fake, GOOD]), specs)
    assert rec.status == "ok"


def test_all_invalid(specs):
    rec = _run(ScriptedClient(["x", "y", "z"]), specs)
    assert rec.status == "invalid" and rec.output is None


def test_truncation_retried_fresh(specs):
    class C(ScriptedClient):
        def chat(self, messages, schema, seed, meta):
            comp = super().chat(messages, schema, seed, meta)
            if len(self.calls) == 1:
                comp.finish_reason = "length"
            return comp

    c = C(['{"extr', GOOD])
    rec = _run(c, specs)
    assert rec.status == "ok"
    assert len(c.calls[1]["messages"]) == 2  # no correction turn after truncation


def test_transport_error_is_error_status(specs):
    class Boom:
        def chat(self, *a, **k):
            raise ConnectionError("down")

    rec = _run(Boom(), specs)
    assert rec.status == "error" and "down" in rec.attempts[0]["error"]


def test_seed_derived_from_key_and_incremented(specs):
    c = ScriptedClient(["x", GOOD])
    _run(c, specs)
    s0, s1 = c.calls[0]["seed"], c.calls[1]["seed"]
    assert s0 == int("ab" * 4, 16) and s1 == s0 + 1


def test_parse_json_object_variants():
    obj = {"extractions": [], "summary": "s"}
    assert parse_json_object(json.dumps(obj)) == obj
    assert parse_json_object("```json\n" + json.dumps(obj) + "\n```") == obj
    assert parse_json_object("<think>hmm {</think>" + json.dumps(obj)) == obj
    assert parse_json_object("Here you go: " + json.dumps(obj) + " done") == obj


def test_prompt_contains_doc_field_and_target(specs):
    c = ScriptedClient([GOOD])
    ctx = {
        "chain": [
            {
                "field": "specimen",
                "title": "Specimen",
                "target_instruction": "Answer ONLY about this specimen.",
                "item": item(
                    ["A.  Left tongue; excision:"],
                    "Left tongue; excision",
                    label="A",
                    site="Left tongue",
                ),
                "siblings": ['"Left neck; dissection" (label=B)'],
            }
        ],
        "deps": [
            {
                "field": "carcinoma_status",
                "items": [item(["Invasive"], "Invasive", category="positive")],
            }
        ],
    }
    run_call(
        client=c,
        model=MODEL,
        spec=specs["keratinization"],
        instructions=specs.instructions,
        doc=make_doc(),
        context=ctx,
        key="0" * 64,
    )
    sys_msg, user = c.calls[0]["messages"]
    assert "faithfulness" in sys_msg["content"].lower()
    assert make_doc().text in user["content"]
    assert "Field: keratinization" in user["content"]
    assert "This request targets ONE specimen" in user["content"]
    assert "OUT OF SCOPE" in user["content"] and "label=B" in user["content"]
    assert "carcinoma_status: positive" in user["content"]
    assert "{DOC_TEXT}" not in user["content"] and "{FIELD_NAME}" not in user["content"]


def test_system_prompt_suffix(specs):
    c = ScriptedClient([GOOD])
    m = ModelConfig.model_validate(
        {
            **MODEL.model_dump(),
            "reasoning": {"system_prompt_suffix": "Reasoning strength: high"},
        }
    )
    _run(c, specs, model=m)
    assert c.calls[0]["messages"][0]["content"].endswith("Reasoning strength: high")


def test_completion_defaults():
    assert Completion("x").usage == {}


SPEC_A = item(
    ["A.  Left tongue; excision:"],
    "Left tongue; excision",
    label="A",
    site="Left tongue",
    site_source="designation",
    material="submitted_tissue",
)
SPEC_B_GOOD = item(
    ["B.  Left neck; dissection:"],
    "Left neck; dissection",
    label="B",
    site="Left neck",
    site_source="designation",
    material="submitted_tissue",
)
SPEC_B_BAD = {**SPEC_B_GOOD, "evidence": ["Specimens: A) tongue B) neck"]}


def test_correction_that_drops_an_extraction_keeps_earlier_answer(specs):
    # Seen with Nemotron: an unverifiable quote led the model to delete a real specimen.
    c = ScriptedClient([answer(SPEC_A, SPEC_B_BAD), answer(SPEC_A)])
    rec = _run(c, specs, field="specimen")
    assert rec.status == "flagged" and len(rec.attempts) == 2
    assert [e["label"] for e in rec.output["extractions"]] == ["A", "B"]
    assert not rec.evidence[1][0]["found"]
    assert "dropped 1 of 2" in rec.attempts[1]["error"]
    assert "Fix the quotes, not the answer" in c.calls[1]["messages"][3]["content"]


def test_correction_that_fixes_the_quote_is_ok(specs):
    c = ScriptedClient([answer(SPEC_A, SPEC_B_BAD), answer(SPEC_A, SPEC_B_GOOD)])
    rec = _run(c, specs, field="specimen")
    assert rec.status == "ok" and len(rec.output["extractions"]) == 2


def test_truncation_during_correction_restarts_fresh(specs):
    class C(ScriptedClient):
        def chat(self, messages, schema, seed, meta):
            comp = super().chat(messages, schema, seed, meta)
            if len(self.calls) == 2:
                comp.finish_reason = "length"
            return comp

    fake = answer(item(["keratinising"], "keratinizing", category="keratinizing"))
    c = C([fake, '{"extr', GOOD])
    rec = _run(c, specs)
    assert rec.status == "ok" and len(rec.attempts) == 3
    assert len(c.calls[1]["messages"]) == 4  # correction turn
    assert len(c.calls[2]["messages"]) == 2  # fresh after truncation
