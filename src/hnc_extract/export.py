"""Export a run: long-format JSONL plus a review workbook for manual annotation.

Workbook sheets:
  README     run metadata, node counts, how to review
  review     one row per extracted value (and one per abstention), with the
             evidence highlighted in context and empty reviewer columns
  reports    deterministic report-level rollup, one row per document
  documents  full document text, for reading around the evidence
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from openpyxl import Workbook
from openpyxl.cell.rich_text import CellRichText, TextBlock
from openpyxl.cell.text import InlineFont
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils.exceptions import IllegalCharacterError
from openpyxl.worksheet.datavalidation import DataValidation

try:
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
except ImportError:  # pragma: no cover
    from openpyxl.utils.cell import ILLEGAL_CHARACTERS_RE  # type: ignore

from hnc_extract.dag import Node
from hnc_extract.documents import Document
from hnc_extract.prompting import describe_item
from hnc_extract.rollup import pathology_rollup

VERDICTS = {
    "correct": "Value, category and evidence are right.",
    "wrong_value": "The document supports a different value/category (put it in review_value).",
    "should_abstain": "The document does not state this; nothing should have been extracted.",
    "missed": "(abstention rows) The document does state it; put the value in review_value.",
    "wrong_target": "True for a different specimen/test than the target.",
    "partially_correct": "Mostly right; explain in review_comment.",
    "unsure": "Cannot decide from the document.",
}
EXCEL_CELL_MAX = 32767
CONTEXT_PAD = 160
HILITE = InlineFont(b=True, color="C00000")
REVIEW_COLS = ["review_verdict", "review_value", "review_comment"]


def _clean(s: str) -> str:
    return ILLEGAL_CHARACTERS_RE.sub("", s)


def _target_desc(node: Node) -> str:
    parts = []
    for level in node.instance.chain():
        item = level.item or {}
        label = item.get("label") or item.get("specimen_label")
        head = f"{level.field}[{level.index}]" + (f" {label}" if label else "")
        parts.append(f"{head}: {describe_item(item)}")
    return " > ".join(parts) or "(document)"


def _context_rich(text: str, checks: list[dict]) -> CellRichText | str:
    spans = sorted((c["start"], c["end"]) for c in checks if c.get("found"))
    if not spans:
        return ""
    # merge spans into windows
    windows: list[list[int]] = []
    for s, e in spans:
        if windows and s <= windows[-1][1] + 2 * CONTEXT_PAD:
            windows[-1][1] = max(windows[-1][1], e)
        else:
            windows.append([s, e])
    rt = CellRichText()
    for wi, (ws, we) in enumerate(windows):
        lo, hi = max(0, ws - CONTEXT_PAD), min(len(text), we + CONTEXT_PAD)
        if wi:
            rt.append("\n…\n")
        if lo > 0:
            rt.append("…")
        cur = lo
        for s, e in spans:
            if s < ws or e > we:
                continue
            s = max(s, cur)
            if s >= e:
                continue
            if s > cur:
                rt.append(_clean(text[cur:s]))
            rt.append(TextBlock(HILITE, _clean(text[s:e])))
            cur = e
        if hi > cur:
            rt.append(_clean(text[cur:hi]))
        if hi < len(text):
            rt.append("…")
    return rt


def node_rows(node: Node) -> list[dict]:
    rec = node.record
    base = {
        "patient_id": node.doc.patient_id,
        "doc_id": node.doc.doc_id,
        "doc_date": (node.doc.anchor_dttm or "")[:10],
        "field": node.spec.name,
        "target": _target_desc(node),
        "instance_path": node.instance.path,
        "model": rec.model_name if rec else "",
        "spec_version": node.spec.version,
        "call_key": node.key or "",
    }
    if node.status != "done":
        return [
            {
                **base,
                "row_id": f"{node.doc.doc_id}|{node.spec.name}|{node.instance.path}|-",
                "status": node.status,
                "summary": node.reason or "",
            }
        ]
    items = node.items
    summary = rec.output.get("summary", "") if rec and rec.output else ""
    if not items:
        return [
            {
                **base,
                "row_id": f"{node.doc.doc_id}|{node.spec.name}|{node.instance.path}|-",
                "status": "abstained",
                "value": "(not stated)",
                "summary": summary,
            }
        ]
    cats = {it.get("category") for it in items}
    rows = []
    for i, it in enumerate(items):
        checks = rec.evidence[i] if rec and rec.evidence else []
        unverified = [c for c in checks if not c["found"]]
        hints = []
        if unverified:
            hints.append(f"evidence unverified ({len(unverified)}/{len(checks)})")
        if it.get("assertion") == "uncertain":
            hints.append("uncertain")
        if node.spec.kind == "attribute" and len(cats) > 1:
            hints.append("multiple categories")
        if len(items) > 1 and node.spec.kind == "attribute":
            hints.append(f"{len(items)} values")
        attrs = {k: v for k, v in it.items() if k in node.spec.attributes}
        if "temporality" in it:
            attrs["temporality"] = (
                f"{it['temporality']['frame']} ({it['temporality']['expression']})"
            )
        methods = sorted({c["method"] for c in checks if c["found"]})
        multi = max((c["n_occurrences"] for c in checks if c["found"]), default=0)
        if multi > 1:
            hints.append(f"quote occurs {multi}x; highlight = nearest to target")
        rows.append(
            {
                **base,
                "row_id": f"{node.doc.doc_id}|{node.spec.name}|{node.instance.path}|{i}",
                "status": "flagged" if unverified else "ok",
                "category": it.get("category", ""),
                "value": it["value"],
                "assertion": it["assertion"],
                "attributes": "; ".join(f"{k}={v}" for k, v in attrs.items()),
                "evidence": " | ".join(it["evidence"]),
                "evidence_match": ",".join(methods)
                + (" + UNVERIFIED" if unverified else ""),
                "review_hints": "; ".join(hints),
                "justification": it.get("justification", ""),
                "summary": summary,
                "_checks": checks,
            }
        )
    return rows


REVIEW_HEADER = [
    ("row_id", 18),
    ("patient_id", 14),
    ("doc_date", 11),
    ("doc_id", 14),
    ("field", 16),
    ("target", 36),
    ("status", 10),
    ("category", 16),
    ("value", 36),
    ("assertion", 10),
    ("attributes", 28),
    ("context", 80),
    ("review_hints", 20),
    ("review_verdict", 16),
    ("review_value", 24),
    ("review_comment", 30),
    ("justification", 40),
    ("summary", 40),
    ("evidence", 40),
    ("evidence_match", 12),
    ("model", 14),
    ("spec_version", 6),
    ("call_key", 12),
]


def _sheet_header(
    ws, cols: list[tuple[str, int]], review: set[str] = frozenset()
) -> None:
    bold = Font(bold=True)
    for j, (name, width) in enumerate(cols, 1):
        c = ws.cell(row=1, column=j, value=name)
        c.font = bold
        if name in review:
            c.fill = PatternFill("solid", fgColor="FFF2CC")
        ws.column_dimensions[c.column_letter].width = width
    ws.freeze_panes = "A2"


def _safe(v):
    if isinstance(v, str):
        v = _clean(v)
        return (
            v[: EXCEL_CELL_MAX - 20] + " [TRUNCATED]" if len(v) > EXCEL_CELL_MAX else v
        )
    if isinstance(v, (list, dict)):
        return _safe(json.dumps(v, ensure_ascii=False))
    return v


def write_workbook(
    path: Path,
    docs: list[Document],
    nodes_by_doc: dict[str, dict[tuple[str, str], Node]],
    field_order: list[str],
    run_info: dict,
) -> dict:
    wb = Workbook()
    readme = wb.active
    readme.title = "README"
    ws = wb.create_sheet("review")
    reports = wb.create_sheet("reports")
    docsheet = wb.create_sheet("documents")

    # ---- review sheet
    _sheet_header(ws, REVIEW_HEADER, review=set(REVIEW_COLS))
    col = {name: j for j, (name, _) in enumerate(REVIEW_HEADER, 1)}
    wrap = Alignment(wrap_text=True, vertical="top")
    order = {f: i for i, f in enumerate(field_order)}
    counts: Counter = Counter()
    r = 1
    for doc in docs:
        nodes = sorted(
            nodes_by_doc[doc.doc_id].values(),
            key=lambda n: (order[n.spec.name], n.instance.path),
        )
        for node in nodes:
            counts[(node.spec.name, node.status)] += 1
            if node.status == "gated_out":
                continue
            for row in node_rows(node):
                r += 1
                for name, j in col.items():
                    if name == "context":
                        val = _context_rich(doc.text, row.get("_checks", []))
                    else:
                        val = _safe(row.get(name, ""))
                    try:
                        cell = ws.cell(row=r, column=j, value=val)
                    except IllegalCharacterError:
                        cell = ws.cell(row=r, column=j, value=_safe(str(val)))
                    cell.alignment = wrap
                if row["status"] in ("flagged", "failed", "pending", "blocked"):
                    ws.cell(row=r, column=col["status"]).fill = PatternFill(
                        "solid", fgColor="F4CCCC"
                    )
    n_rows = r - 1
    ws.auto_filter.ref = (
        f"A1:{ws.cell(row=1, column=len(REVIEW_HEADER)).column_letter}{max(r, 2)}"
    )
    dv = DataValidation(
        type="list", formula1='"' + ",".join(VERDICTS) + '"', allow_blank=True
    )
    ws.add_data_validation(dv)
    vcol = ws.cell(row=1, column=col["review_verdict"]).column_letter
    dv.add(f"{vcol}2:{vcol}{max(r, 2) + 1000}")

    # ---- reports sheet
    rep_cols = [
        ("patient_id", 14),
        ("doc_date", 11),
        ("doc_id", 14),
        ("report_carcinoma_status", 14),
        ("report_invasive_carcinoma", 10),
        ("report_dysplasia_status", 14),
        ("worst_dysplasia_grade", 14),
        ("specimens", 70),
        ("tests", 60),
        ("incomplete_fields", 20),
        ("flagged_fields", 20),
    ]
    _sheet_header(reports, rep_cols)
    rollups = {}
    for i, doc in enumerate(docs, 2):
        ru = (
            pathology_rollup(nodes_by_doc[doc.doc_id])
            if doc.doc_type == "pathology_report"
            else {}
        )
        rollups[doc.doc_id] = ru
        spec_txt = "\n".join(
            f"{s['label'] or '-'} {s['site'] or ''}: carcinoma={s['carcinoma_status']}"
            + (" (CONFLICT)" if s["carcinoma_status_conflict"] else "")
            + (f"; histology={'; '.join(s['histology'])}" if s["histology"] else "")
            + (
                f"; keratinization={','.join(s['keratinization'])}"
                if s["keratinization"]
                else ""
            )
            + f"; dysplasia={s['dysplasia_status']}"
            + (f" ({','.join(s['dysplasia_grades'])})" if s["dysplasia_grades"] else "")
            + (f"; procedure={','.join(s['procedure'])}" if s["procedure"] else "")
            for s in ru.get("specimens", [])
        )
        test_txt = "\n".join(
            f"{t['test_type']}[{t['specimen_label'] or '-'}]: {','.join(t['result']) or '?'}"
            + (f" {'; '.join(t['pdl1_scores'])}" if t["pdl1_scores"] else "")
            for t in ru.get("tests", [])
        )
        vals = {
            "patient_id": doc.patient_id,
            "doc_date": (doc.anchor_dttm or "")[:10],
            "doc_id": doc.doc_id,
            "specimens": spec_txt,
            "tests": test_txt,
            **{k: v for k, v in ru.items() if k not in ("specimens", "tests")},
        }
        for j, (name, _) in enumerate(rep_cols, 1):
            reports.cell(row=i, column=j, value=_safe(vals.get(name, ""))).alignment = (
                wrap
            )

    # ---- documents sheet
    _sheet_header(
        docsheet, [("doc_id", 14), ("patient_id", 14), ("doc_date", 11), ("text", 160)]
    )
    for i, doc in enumerate(docs, 2):
        for j, v in enumerate(
            [doc.doc_id, doc.patient_id, (doc.anchor_dttm or "")[:10], doc.text], 1
        ):
            docsheet.cell(row=i, column=j, value=_safe(v)).alignment = wrap

    # ---- README
    lines = [
        ("HNC extraction review workbook", None),
        ("", None),
        *[(f"{k}: {v}", None) for k, v in run_info.items()],
        ("", None),
        ("HOW TO REVIEW (sheet 'review')", True),
        (
            "Each row is one extracted value, or one '(not stated)' abstention for a field/target.",
            None,
        ),
        (
            "'context' shows the document around the evidence; quoted evidence is bold red.",
            None,
        ),
        ("Fill the yellow columns. review_verdict options:", None),
        *[(f"  {k}: {v}", None) for k, v in VERDICTS.items()],
        ("Rows with status flagged/failed/pending/blocked are shaded red.", None),
        (
            "Gated-out nodes (e.g. histology for a carcinoma-negative specimen) are not listed.",
            None,
        ),
        (
            "Keep row_id intact: verdicts are joined back to the run by row_id and call_key.",
            None,
        ),
        ("", None),
        ("NODE COUNTS (field / status)", True),
    ]
    for k, v in sorted(counts.items(), key=lambda kv: (order[kv[0][0]], kv[0][1])):
        lines.append((f"  {k[0]:20s} {k[1]:10s} {v}", None))
    for i, (t, bold) in enumerate(lines, 1):
        c = readme.cell(row=i, column=1, value=_safe(t))
        if bold:
            c.font = Font(bold=True)
    readme.column_dimensions["A"].width = 120

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return {
        "review_rows": n_rows,
        "counts": {f"{a}/{b}": v for (a, b), v in counts.items()},
        "rollups": rollups,
    }


def node_record(node: Node) -> dict:
    rec = node.record
    return {
        "patient_id": node.doc.patient_id,
        "doc_id": node.doc.doc_id,
        "doc_type": node.doc.doc_type,
        "anchor_dttm": node.doc.anchor_dttm,
        "field": node.spec.name,
        "spec_version": node.spec.version,
        "instance_path": node.instance.path,
        "target": _target_desc(node) if node.instance.path else None,
        "status": node.status,
        "reason": node.reason,
        "call_key": node.key,
        "call_status": rec.status if rec else None,
        "model": rec.model_name if rec else None,
        "extractions": node.items,
        "evidence_checks": rec.evidence if rec else None,
        "summary": rec.output.get("summary") if rec and rec.output else None,
    }
