"""Render chat messages for one (field, document, target) extraction call.

Message layout is chosen for vLLM prefix caching and long-context quality:
  system = shared rules (identical for every call)
  user   = document, then the field definition, then the target context, then the ask
"""

from __future__ import annotations

import json

from hnc_extract.documents import Document
from hnc_extract.spec import FieldSpec

# Bump whenever this module changes what a rendered prompt looks like: it is part
# of every cache key, so old cached calls are not reused for the new prompts.
PROMPT_FORMAT_VERSION = 1

DOC_LABELS = {
    "progress_note": "Progress note",
    "pathology_report": "Pathology report",
    "radiology_report": "Radiology report",
}


def _fill(template: str, values: dict[str, str]) -> str:
    out = template
    for k, v in values.items():
        out = out.replace("{" + k + "}", v)
    return out


def _bullets(d: dict[str, str], indent: str = "  ") -> str:
    return "\n".join(f"{indent}- {k}: {v.strip()}" for k, v in d.items())


def render_field_section(spec: FieldSpec) -> str:
    parts = [
        f"Field: {spec.name} ({spec.title})",
        "",
        "Definition and guidance:",
        spec.instruction.strip(),
        "",
    ]
    parts.append(f"`value`: {spec.value_description.strip()}")
    if spec.categories:
        parts += [
            "",
            "`category` (choose exactly one per extraction):",
            _bullets(spec.categories),
        ]
    if spec.attributes:
        parts += ["", "Attributes (one key per extraction):"]
        for name, a in spec.attributes.items():
            kind = "one of: " + " | ".join(a.values) if a.type == "enum" else "string"
            if a.nullable:
                kind += ", or null"
            parts.append(f"  - {name} ({kind}): {a.description.strip()}")
            if a.type == "enum":
                parts.append(_bullets(a.values, indent="      "))
    parts += [
        "",
        "Temporality: "
        + (
            "REQUIRED for each extraction (rule 5)."
            if spec.temporality
            else "not used."
        ),
    ]
    if spec.max_items is not None:
        parts.append(f"At most {spec.max_items} extraction(s).")
    if spec.examples:
        parts += [
            "",
            "Worked examples (illustrative excerpts, NOT from this document):",
        ]
        for i, ex in enumerate(spec.examples, 1):
            parts.append(f"Example {i}:")
            parts.append(f'Excerpt: """{ex.excerpt.strip()}"""')
            if ex.target:
                parts.append(f"Target: {ex.target.strip()}")
            parts.append("Answer: " + json.dumps(ex.output, ensure_ascii=False))
    return "\n".join(parts)


def describe_item(item: dict) -> str:
    """One-line description of an extracted item (used for targets and siblings)."""
    attrs = {
        k: v
        for k, v in item.items()
        if k
        not in (
            "evidence",
            "justification",
            "assertion",
            "value",
            "category",
            "temporality",
        )
        and v is not None
    }
    s = json.dumps(item.get("value", ""), ensure_ascii=False)
    if attrs:
        s += " (" + ", ".join(f"{k}={v}" for k, v in attrs.items()) + ")"
    if item.get("assertion") == "uncertain":
        s += " [uncertain]"
    return s


def render_target_section(context: dict) -> str:
    chain = context.get("chain", [])
    deps = context.get("deps", [])
    if not chain and not deps:
        return ""
    lines = ["", "====================", "TARGET", "===================="]
    if not chain:
        lines.append("Scope: the whole document.")
    for level in chain:
        title = level["title"]
        lines.append(
            f"This request targets ONE {title.lower()} of the document: {describe_item(level['item'])}"
        )
        lines.append(
            "Its identifying evidence: "
            + json.dumps(level["item"].get("evidence", []), ensure_ascii=False)
        )
        lines.append(level["target_instruction"].strip())
        if level["siblings"]:
            lines.append(
                f"Other {title.lower()} entries in this document (OUT OF SCOPE for this request):"
            )
            lines += [f"  - {s}" for s in level["siblings"]]
        lines.append("")
    if deps:
        lines.append(
            "Already extracted for this target (context only; do not re-extract these fields):"
        )
        for d in deps:
            if not d["items"]:
                lines.append(f"  - {d['field']}: (none)")
            for it in d["items"]:
                cat = f"{it['category']}: " if "category" in it else ""
                lines.append(
                    f"  - {d['field']}: {cat}{describe_item(it)}; evidence: "
                    + json.dumps(it.get("evidence", []), ensure_ascii=False)
                )
    return "\n".join(lines) + "\n"


def render_messages(
    spec: FieldSpec,
    instructions: dict,
    doc: Document,
    context: dict,
    system_suffix: str | None = None,
) -> list[dict]:
    system = instructions["system"].rstrip()
    if system_suffix:
        system += "\n\n" + system_suffix.strip()
    user = _fill(
        instructions["user"],
        {
            "DOC_LABEL": DOC_LABELS.get(doc.doc_type, doc.doc_type),
            "ANCHOR_DATE": (doc.anchor_dttm or "unknown")[:10],
            "FIELD_SECTION": render_field_section(spec),
            "TARGET_SECTION": render_target_section(context),
            "FIELD_NAME": spec.name,
        },
    )
    # The document goes in last so that no template placeholder can match inside it.
    user = user.replace("{DOC_TEXT}", doc.text)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
