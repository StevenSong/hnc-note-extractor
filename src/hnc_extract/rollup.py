"""Deterministic report-level rollups over note-level nodes (pathology).

No LLM involvement: these are rules over the per-specimen / per-test results,
so they are auditable and can change without re-running extraction.
"""

from __future__ import annotations

from hnc_extract.dag import Node

CARCINOMA_PRECEDENCE = ["positive", "suspicious", "negative", "non_diagnostic"]
# Ordinal for the "worst grade". low/high-grade are two-tier terms placed between
# the three-tier grades; this is a convention for sorting only, not a mapping.
# Histology invasion values that make a carcinoma invasive (microinvasive /
# superficially invasive carcinoma is invasive carcinoma).
INVASIVE = {"invasive", "microinvasive"}
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


def _report_flags(specimens: list[dict]) -> list[str]:
    """Structural review flags for the whole report (no clinical inference).

    These catch specimen lists that did not reconcile with the diagnoses, as in
    scrambled outside consults whose part letters and site-headed diagnoses do
    not line up. The rollup does not pick an interpretation; it asks for review.
    """
    flags = []
    labels = [s["label"] for s in specimens if s["label"]]
    if len(labels) != len(set(labels)):
        flags.append("duplicate_specimen_labels")
    diagnosed = [s["carcinoma_status"] is not None for s in specimens]
    if any(s["label"] and not d for s, d in zip(specimens, diagnosed)) and any(
        not s["label"] and d for s, d in zip(specimens, diagnosed)
    ):
        flags.append("specimen_list_unreconciled")
    if any(diagnosed) and not all(diagnosed):
        flags.append("some_specimens_undiagnosed")
    return flags


def _final_results(items: list[dict]) -> list[dict]:
    """Prefer resulted over pending statements of the same test."""
    resulted = [i for i in items if i.get("category") != "pending"]
    return resulted or items


def report_summary(specimens: list[dict]) -> dict:
    """Report-level fields from specimen dicts (model rollups and gold alike).

    Each specimen needs carcinoma_status, dysplasia_status, dysplasia_grades,
    invasion (affirmed) and invasion_uncertain.
    """
    carc_all = [s["carcinoma_status"] for s in specimens if s["carcinoma_status"]]
    dys_all = [s["dysplasia_status"] for s in specimens if s["dysplasia_status"]]
    grades = sorted({g for s in specimens for g in s["dysplasia_grades"]})
    invasive = any(INVASIVE & set(s["invasion"]) for s in specimens)
    return {
        "report_carcinoma_status": _top(carc_all, CARCINOMA_PRECEDENCE)
        or "not_assessed",
        "report_invasive_carcinoma": invasive,
        # uncertain invasion somewhere, without affirmed invasion in the report
        "report_invasion_suspected": not invasive
        and any(INVASIVE & set(s["invasion_uncertain"]) for s in specimens),
        "report_dysplasia_status": _top(dys_all, ["positive", "negative"])
        or "not_mentioned",
        "worst_dysplasia_grade": _top(grades, GRADE_ORDER[::-1]),
    }


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
            hist = items("histology")
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
                    "histology": [it["value"] for it in hist],
                    # invasion of affirmed histology only; uncertain ones
                    # ("suspicious for superficial invasion") are kept apart
                    "invasion": sorted(
                        {
                            it["invasion"]
                            for it in hist
                            if it.get("assertion", "affirmed") == "affirmed"
                        }
                    ),
                    "invasion_uncertain": sorted(
                        {
                            it["invasion"]
                            for it in hist
                            if it.get("assertion") == "uncertain"
                        }
                    ),
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

    return {
        **report_summary(specimens),
        "specimens": specimens,
        "tests": tests,
        "review_flags": _report_flags(specimens),
        "incomplete_fields": incomplete,
        "flagged_fields": flagged,
    }
