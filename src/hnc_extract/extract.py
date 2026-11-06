"""One extraction call: prompt -> LLM -> parse -> validate -> verify evidence -> retry.

Outcomes (`CallRecord.status`):
  ok       schema-valid output; every evidence quote found in the document
  flagged  schema-valid output, but some quote could not be verified after all
           attempts (kept for review; never silently trusted)
  invalid  no schema-valid output after all attempts (children are blocked)
  error    transport/server error (retried on the next run)
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import re
import traceback

from pydantic import ValidationError

from hnc_extract.documents import Document
from hnc_extract.evidence import EvidenceIndex
from hnc_extract.llm import LLMClient, ModelConfig
from hnc_extract.prompting import render_messages
from hnc_extract.spec import FieldSpec

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


@dataclasses.dataclass
class CallRecord:
    key: str
    doc_sha256: str
    field: str
    spec_version: int
    spec_fingerprint: str
    model_name: str
    model_fingerprint: str
    status: str
    output: dict | None
    evidence: list[list[dict]] | None  # per extraction, per quote: SpanMatch dict
    attempts: list[dict]
    created_utc: str

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> CallRecord:
        return cls(**json.loads(s))


def parse_json_object(content: str) -> dict:
    s = _THINK.sub("", content).strip()
    s = _FENCE.sub("", s).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        i, j = s.find("{"), s.rfind("}")
        if i == -1 or j <= i:
            raise
        obj = json.loads(s[i : j + 1])
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON value is not an object")
    return obj


def _feedback_for_validation(e: Exception) -> str:
    return (
        "Your previous answer could not be used because it does not conform to the required "
        f"JSON schema:\n{e}\n\nReturn a corrected JSON object only."
    )


def _feedback_for_evidence(bad: list[tuple[int, str]]) -> str:
    lines = [
        "Your previous answer was parsed, but these evidence quotes do not occur verbatim in the "
        "document (checked automatically, ignoring only whitespace and case):"
    ]
    lines += [
        f"  - extraction {i}: {json.dumps(q, ensure_ascii=False)}" for i, q in bad
    ]
    lines.append(
        "Copy each quote character-for-character from the document, including masking characters "
        "and typos, without ellipses; split non-contiguous text into separate quotes. If the value "
        "has no verbatim support, remove that extraction. Return the full corrected JSON object only."
    )
    return "\n".join(lines)


def _target_anchor(idx: EvidenceIndex, context: dict) -> int | None:
    """Offset of the innermost target instance's first evidence quote, if any."""
    for level in reversed(context.get("chain") or []):
        for q in level["item"].get("evidence", []):
            m = idx.match(q)
            if m.found:
                return m.start
    return None


def run_call(
    *,
    client: LLMClient,
    model: ModelConfig,
    spec: FieldSpec,
    instructions: dict,
    doc: Document,
    context: dict,
    key: str,
    evidence_index: EvidenceIndex | None = None,
) -> CallRecord:
    out_model = spec.output_model()
    schema = spec.json_schema()
    messages = render_messages(
        spec,
        instructions,
        doc,
        context,
        system_suffix=model.reasoning.system_prompt_suffix,
    )
    idx = evidence_index or EvidenceIndex(doc.text)
    near = _target_anchor(idx, context)
    seed = int(key[:8], 16) if model.seed == "auto" else model.seed
    meta = {"field": spec.name, "doc": doc, "context": context, "spec": spec}

    attempts: list[dict] = []
    best: tuple[dict, list[list[dict]]] | None = None
    status = "invalid"
    for n in range(model.max_attempts):
        att: dict = {"n": n}
        attempts.append(att)
        try:
            comp = client.chat(
                messages, schema, None if seed is None else seed + n, meta
            )
        except Exception as e:  # noqa: BLE001
            att["error"] = f"{type(e).__name__}: {e}"
            att["traceback"] = traceback.format_exc(limit=3)
            status = "error"
            break
        att.update(
            content=comp.content,
            reasoning=comp.reasoning,
            finish_reason=comp.finish_reason,
            usage=comp.usage,
            latency_s=round(comp.latency_s, 3),
        )
        if comp.finish_reason == "length":
            att["error"] = "truncated (finish_reason=length)"
            continue  # fresh retry; a truncated answer is not worth correcting

        try:
            parsed = out_model.model_validate(
                parse_json_object(comp.content)
            ).model_dump(mode="json")
        except (ValueError, ValidationError) as e:
            att["error"] = f"invalid: {e}"
            messages = messages[:2] + [
                {"role": "assistant", "content": comp.content},
                {"role": "user", "content": _feedback_for_validation(e)},
            ]
            continue

        checks = [
            [idx.match(q, near).to_dict() for q in item["evidence"]]
            for item in parsed["extractions"]
        ]
        bad = [
            (i, c["quote"])
            for i, item in enumerate(checks)
            for c in item
            if not c["found"]
        ]
        best = (parsed, checks)
        if not bad:
            status = "ok"
            break
        status = "flagged"
        att["error"] = f"unverified evidence: {len(bad)} quote(s)"
        messages = messages[:2] + [
            {"role": "assistant", "content": comp.content},
            {"role": "user", "content": _feedback_for_evidence(bad)},
        ]

    if status == "error" and best is not None:
        status = "flagged"
    if status == "invalid" and best is not None:
        status = "flagged"
    output, checks = best if best else (None, None)
    return CallRecord(
        key=key,
        doc_sha256=doc.text_sha256,
        field=spec.name,
        spec_version=spec.version,
        spec_fingerprint=spec.fingerprint,
        model_name=model.name,
        model_fingerprint=model.fingerprint(),
        status=status if output is not None or status == "error" else "invalid",
        output=output,
        evidence=checks,
        attempts=attempts,
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    )
