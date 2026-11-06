"""Deterministic report-level rollups over note-level nodes (pathology).

No LLM involvement: these are rules over the per-specimen / per-test results,
so they are auditable and can change without re-running extraction.
"""

from __future__ import annotations

from hnc_extract.dag import Node

CARCINOMA_PRECEDENCE = ["positive", "suspicious", "negative", "non_diagnostic"]
# Ordinal for the "worst grade". low/high-grade are two-tier terms placed between
# the three-tier grades; this is a convention for sorting only, not a mapping.
GRADE_ORDER = [
    "mild",
    "low_grade",
    "moderate",
    "high_grade",
    "severe",
    "carcinoma_in_situ",
]


def _by(nodes: dict[tuple[str, str], Node], field: str) -> dict[str, Node]:
    return {path: n for (f, path), n in nodes.items() if f == field}


def _top(categories: list[str], order: list[str]) -> str | None:
    ranked = [c for c in order if c in categories]
    return ranked[0] if ranked else None


def _specimen_flags(nodes: dict[tuple[str, str], Node], path: str) -> list[str]:
    """Structural review flags for one specimen (no clinical inference)."""
    flags = []
    carc = nodes.get(("carcinoma_status", path))
    if carc is not None and carc.status == "done" and not carc.items:
        flags.append("no_diagnosis_stated")
    return flags


def _final_results(items: list[dict]) -> list[dict]:
    """Prefer resulted over pending statements of the same test."""
    resulted = [i for i in items if i.get("category") != "pending"]
    return resulted or items


def pathology_rollup(nodes: dict[tuple[str, str], Node]) -> dict:
    incomplete = sorted(
        {
            n.spec.name
            for n in nodes.values()
            if n.status in ("pending", "blocked", "failed")
        }
    )
    flagged = sorted(
        {
            n.spec.name
            for n in nodes.values()
            if n.record is not None and n.record.status == "flagged"
        }
    )

    spec_nodes = _by(nodes, "specimen")
    root = spec_nodes.get("")
    specimens = []
    if root is not None and root.status == "done":
        for i, item in enumerate(root.items):
            path = f"specimen[{i}]"

            def items(field: str, path: str = path) -> list[dict]:
                n = nodes.get((field, path))
                return n.items if n is not None else []

            carc = [it["category"] for it in items("carcinoma_status")]
            specimens.append(
                {
                    "path": path,
                    "label": item.get("label"),
                    "site": item.get("site"),
                    "site_source": item.get("site_source"),
                    "material": item.get("material"),
                    "procedure": [it["category"] for it in items("procedure_type")],
                    "carcinoma_status": _top(carc, CARCINOMA_PRECEDENCE),
                    "carcinoma_status_conflict": len(set(carc)) > 1,
                    "histology": [it["value"] for it in items("histology")],
                    "invasion": sorted({it["invasion"] for it in items("histology")}),
                    "keratinization": sorted(
                        {it["category"] for it in items("keratinization")}
                    ),
                    "dysplasia_status": _top(
                        [it["category"] for it in items("dysplasia_status")],
                        ["positive", "negative"],
                    ),
                    "dysplasia_grades": sorted(
                        {it["category"] for it in items("dysplasia_grade")}
                    ),
                    "review_flags": _specimen_flags(nodes, path),
                }
            )

    tests = []
    test_root = _by(nodes, "ancillary_test").get("")
    if test_root is not None and test_root.status == "done":
        for i, item in enumerate(test_root.items):
            path = f"ancillary_test[{i}]"
            res = nodes.get(("test_result", path))
            pdl1 = nodes.get(("pdl1_score", path))
            final = _final_results(res.items) if res is not None else []
            tests.append(
                {
                    "path": path,
                    "test": item["value"],
                    "test_type": item["test_type"],
                    "specimen_label": item.get("specimen_label"),
                    "result": sorted({r["category"] for r in final}),
                    "result_text": [r["value"] for r in final],
                    "pdl1_scores": [
                        f"{s['category'].upper()} {s['score']}"
                        for s in (pdl1.items if pdl1 else [])
                    ],
                }
            )

    carc_all = [s["carcinoma_status"] for s in specimens if s["carcinoma_status"]]
    dys_all = [s["dysplasia_status"] for s in specimens if s["dysplasia_status"]]
    grades = sorted({g for s in specimens for g in s["dysplasia_grades"]})
    return {
        "report_carcinoma_status": _top(carc_all, CARCINOMA_PRECEDENCE)
        or "not_assessed",
        "report_invasive_carcinoma": any(
            "invasive" in s["invasion"] for s in specimens
        ),
        "report_dysplasia_status": _top(dys_all, ["positive", "negative"])
        or "not_mentioned",
        "worst_dysplasia_grade": _top(grades, GRADE_ORDER[::-1]),
        "specimens": specimens,
        "tests": tests,
        "incomplete_fields": incomplete,
        "flagged_fields": flagged,
    }
