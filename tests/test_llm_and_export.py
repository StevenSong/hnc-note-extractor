import openpyxl
from conftest import make_doc, scripted_pathology_responder

from hnc_extract.dag import run, traverse
from hnc_extract.export import write_workbook
from hnc_extract.llm import MockClient, ModelConfig, OpenAICompatClient
from hnc_extract.store import Store


def test_request_body_structured_and_reasoning():
    cfg = ModelConfig(
        name="q",
        served_model_name="qwen",
        top_p=0.95,
        top_k=20,
        reasoning={
            "reasoning_effort": "xhigh",
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    body = OpenAICompatClient(cfg).build_body(
        [{"role": "user", "content": "x"}], {"type": "object"}, 7, "f"
    )
    assert body["model"] == "qwen" and body["seed"] == 7 and body["top_k"] == 20
    assert body["reasoning_effort"] == "xhigh"
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert "min_p" not in body

    none = cfg.model_copy(update={"structured_output": "none"})
    assert "response_format" not in OpenAICompatClient(none).build_body(
        [], {"type": "object"}, None, "f"
    )


def test_fingerprint_ignores_url_but_not_sampling():
    a = ModelConfig(name="a", base_url="http://x/v1")
    assert (
        a.fingerprint() == ModelConfig(name="b", base_url="http://y/v1").fingerprint()
    )
    assert a.fingerprint() != ModelConfig(name="a", temperature=0.5).fingerprint()


def test_workbook_export(specs, tmp_path):
    model = ModelConfig(name="t", provider="mock")
    store = Store(tmp_path / "c.sqlite")
    docs = [make_doc()]
    run(
        docs,
        specs,
        model,
        MockClient(scripted_pathology_responder),
        store,
        log=lambda m: None,
    )
    nodes = {d.doc_id: traverse(d, specs, store, model.fingerprint()) for d in docs}
    res = write_workbook(tmp_path / "r.xlsx", docs, nodes, specs.order, {"run": "test"})
    wb = openpyxl.load_workbook(tmp_path / "r.xlsx", rich_text=True)
    assert wb.sheetnames == ["README", "review", "reports", "documents"]
    ws = wb["review"]
    hdr = [c.value for c in ws[1]]
    rows = [dict(zip(hdr, [c.value for c in r])) for r in ws.iter_rows(min_row=2)]
    assert len(rows) == res["review_rows"]
    assert {"review_verdict", "review_value", "review_comment", "context"} <= set(hdr)
    kera = [r for r in rows if r["field"] == "keratinization"]
    assert len(kera) == 1 and kera[0]["category"] == "keratinizing"
    # evidence is highlighted as a rich-text run
    ctx = kera[0]["context"]
    assert any(getattr(b, "text", None) == "keratinizing" for b in ctx)
    abst = [r for r in rows if r["status"] == "abstained"]
    assert {r["field"] for r in abst} == {"dysplasia_status"}
    assert wb["reports"]["D2"].value == "positive"


def _fake_server(responses):
    """httpx transport that replays (status, json) pairs and records request bodies."""
    import json as _json

    import httpx

    seen = []

    def handler(request):
        seen.append(_json.loads(request.content))
        status, payload = responses.pop(0)
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler), seen


def _ok(content, reasoning="thinking..."):
    return (
        200,
        {
            "choices": [
                {
                    "message": {"content": content, "reasoning": reasoning},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def test_http_client_end_to_end_with_retry(specs):
    from conftest import answer, item

    from hnc_extract.extract import run_call

    good = answer(item(["keratinizing"], "keratinizing", category="keratinizing"))
    transport, seen = _fake_server([(503, {"error": "loading"}), _ok(good)])
    cfg = ModelConfig(name="q", served_model_name="qwen", base_url="http://llm:8000/v1")
    client = OpenAICompatClient(cfg, transport=transport, backoff_s=0)
    rec = run_call(
        client=client,
        model=cfg,
        spec=specs["keratinization"],
        instructions=specs.instructions,
        doc=make_doc(),
        context={"chain": [], "deps": []},
        key="1" * 64,
    )
    assert rec.status == "ok"
    assert rec.attempts[0]["reasoning"] == "thinking..."
    assert rec.attempts[0]["usage"]["completion_tokens"] == 5
    assert len(seen) == 2 and seen[1]["response_format"]["type"] == "json_schema"
    assert seen[1]["messages"][0]["role"] == "system"


def test_http_client_4xx_not_retried(specs):
    import pytest

    transport, seen = _fake_server([(400, {"error": "too long"})])
    client = OpenAICompatClient(ModelConfig(name="q"), transport=transport, backoff_s=0)
    with pytest.raises(RuntimeError, match="HTTP 400"):
        client.chat([{"role": "user", "content": "x"}], None, None, {})
    assert len(seen) == 1
