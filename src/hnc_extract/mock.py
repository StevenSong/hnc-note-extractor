"""Regex-based stand-in for an LLM, for OFFLINE pipeline tests only.

It produces schema-valid answers with verbatim evidence for the pathology
fields so the DAG, caching, rollup and review export can be exercised end to
end without a GPU. Its extraction quality is deliberately naive; never use its
output as data.
"""

from __future__ import annotations

import json
import re

from hnc_extract.spec import FieldSpec

_DESIG = re.compile(
    r"(?:(?<=\s)|^)([A-Z])[.)]\s+([A-Z][^:;]{2,80}?);\s*([A-Za-z][A-Za-z ,\-]{2,60}?):"
)
_DIAG_LINE = re.compile(r"-\s+([A-Z][^.]{3,200}\.)")


def _item(
    spec: FieldSpec,
    evidence: list[str],
    value: str,
    category: str | None = None,
    **attrs,
) -> dict:
    it: dict = {
        "evidence": evidence,
        "justification": "mock heuristic",
        "assertion": "affirmed",
        "value": value,
    }
    if spec.categories:
        it["category"] = (
            category if category in spec.categories else next(iter(spec.categories))
        )
    for name, a in spec.attributes.items():
        if name in attrs:
            it[name] = attrs[name]
        elif a.nullable:
            it[name] = None
        else:
            vals = list(a.values or {})
            it[name] = next(
                (v for v in ("not_stated", "no") if v in vals), vals[0] if vals else ""
            )
    if spec.temporality:
        it["temporality"] = {"frame": "unknown", "expression": None}
    return it


def _out(items: list[dict], summary: str = "mock") -> str:
    return json.dumps({"extractions": items, "summary": summary})


def _window(text: str, context: dict, size: int = 400) -> str:
    chain = context.get("chain") or []
    if not chain:
        return text
    ev = chain[-1]["item"]["evidence"][0]
    i = text.find(ev)
    if i == -1:
        return text[:size]
    end = i + len(ev)
    nxt = _DESIG.search(text, end)
    stop = min(nxt.start() if nxt else len(text), end + size)
    return text[i:stop]


def _specimens(spec: FieldSpec, text: str) -> list[dict]:
    seen, items = set(), []
    for m in _DESIG.finditer(text):
        label = m.group(1)
        if label in seen:
            continue
        seen.add(label)
        items.append(
            _item(
                spec,
                [m.group(0).strip()],
                f"{m.group(2).strip()}; {m.group(3).strip()}",
                label=label,
                site=m.group(2).strip(),
            )
        )
    if not items and (m := _DIAG_LINE.search(text)):
        items.append(_item(spec, [m.group(0).strip()], m.group(1)))
    return items


_PROC = [
    ("fine_needle_aspiration", r"\bFNA\b|aspirat"),
    ("lymph_node_dissection", r"dissection|lymphadenectomy"),
    ("resection", r"ectomy|resection"),
    ("excision", r"excision"),
    ("biopsy", r"biops"),
]


def _first(pattern: str, text: str, flags=re.I) -> re.Match | None:
    return re.search(pattern, text, flags)


def heuristic_responder(messages: list[dict], schema: dict | None, meta: dict) -> str:
    spec: FieldSpec = meta["spec"]
    text: str = meta["doc"].text
    ctx: dict = meta["context"]
    win = _window(text, ctx)
    f = spec.name

    if f == "specimen":
        return _out(_specimens(spec, text))
    if f == "procedure_type":
        for cat, pat in _PROC:
            if m := _first(pat, ctx["chain"][-1]["item"]["evidence"][0]):
                return _out([_item(spec, [m.group(0)], m.group(0), cat)])
        return _out([])
    if f == "carcinoma_status":
        if m := _first(
            r"negative for (?:malignancy|carcinoma)[^.]*|no (?:malignancy|carcinoma|tumor)[^.]*",
            win,
        ):
            return _out(
                [
                    _item(
                        spec,
                        [m.group(0)],
                        m.group(0),
                        "negative",
                        basis="explicit_negative",
                    )
                ]
            )
        if m := _first(r"suspicious for [^.]*", win):
            return _out([_item(spec, [m.group(0)], m.group(0), "suspicious")])
        if m := _first(
            r"[A-Za-z ,\-]*\b(?:carcinoma|lymphoma|melanoma|sarcoma)\b[^.]*", win
        ):
            return _out(
                [_item(spec, [m.group(0).strip()], m.group(0).strip(), "positive")]
            )
        if m := _first(r"benign[^.]*", win):
            return _out(
                [
                    _item(
                        spec,
                        [m.group(0)],
                        m.group(0),
                        "negative",
                        basis="benign_diagnosis",
                    )
                ]
            )
        return _out([])
    if f == "histology":
        m = _first(
            r"(?:invasive |metastatic )?(?:[a-z\-]+ ){0,3}(?:carcinoma|lymphoma|melanoma|sarcoma)(?: in[- ]situ)?",
            win,
        )
        if not m:
            return _out([])
        v = m.group(0)
        inv = (
            "in_situ"
            if "situ" in v.lower()
            else (
                "invasive"
                if re.search(r"invasive|metastatic", v, re.I)
                else "not_stated"
            )
        )
        meta_ = "yes" if "metastatic" in v.lower() else "no"
        return _out([_item(spec, [v], v, invasion=inv, described_as_metastatic=meta_)])
    if f == "keratinization":
        if m := _first(r"\bnon-?keratinizing\b|\bkeratinizing\b", win):
            cat = (
                "non_keratinizing"
                if m.group(0).lower().startswith("non")
                else "keratinizing"
            )
            return _out([_item(spec, [m.group(0)], m.group(0), cat)])
        return _out([])
    if f == "dysplasia_status":
        if m := _first(r"no dysplasia|negative for dysplasia", win):
            return _out([_item(spec, [m.group(0)], m.group(0), "negative")])
        if m := _first(
            r"(?:mild|moderate|severe|high-grade|low-grade)[^.]{0,20}dysplasia", win
        ):
            return _out([_item(spec, [m.group(0)], m.group(0), "positive")])
        return _out([])
    if f == "dysplasia_grade":
        if m := _first(r"\b(mild|moderate|severe|high-grade|low-grade)\b", win):
            g = m.group(1).lower().replace("-", "_")
            return _out([_item(spec, [m.group(0)], m.group(0), g)])
        return _out([])
    if f == "ancillary_test":
        items, seen = [], set()
        for pat, tt in [
            (
                r"(?:On ([A-Z]), )?p16 [^.]{0,80}?(?:negative|positive|pending|positivity)",
                "p16_ihc",
            ),
            (r"HPV[^.]{0,40}?\bISH\b[^.]{0,60}", "hpv_ish"),
            (r"(?:On ([A-Z]), )?[^.]{0,30}\(22C3\)[^.]{0,40}CPS[^.]{0,20}", "pdl1_ihc"),
            (r"OncoPlus[^.]{0,60}", "molecular_panel"),
        ]:
            for m in re.finditer(pat, text):
                label = m.group(1) if m.lastindex else None
                if (tt, label) in seen:
                    continue
                seen.add((tt, label))
                items.append(
                    _item(
                        spec,
                        [m.group(0).strip()],
                        m.group(0).strip(),
                        test_type=tt,
                        specimen_label=label,
                    )
                )
        return _out(items)
    if f == "test_result":
        ev = ctx["chain"][-1]["item"]["evidence"][0]
        for cat, pat in [
            ("negative", r"negative"),
            ("positive", r"positive\b"),
            ("pending", r"pending"),
        ]:
            if m := _first(pat, ev):
                return _out([_item(spec, [ev], m.group(0), cat)])
        return _out([_item(spec, [ev], ev, "result_without_interpretation")])
    if f == "pdl1_score":
        if m := _first(r"CPS (?:of )?([<>]?\s?\d+)", win):
            return _out(
                [
                    _item(
                        spec,
                        [m.group(0)],
                        m.group(0),
                        "cps",
                        score=m.group(1),
                        clone="22C3" if "22C3" in text else None,
                    )
                ]
            )
        return _out([])
    return _out([], summary="mock: field not handled; abstaining")
