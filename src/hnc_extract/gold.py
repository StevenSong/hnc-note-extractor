"""Gold key for pathology runs: draft from multi-model consensus, adjudicate, score.

Workflow
  1. draft     align each report's specimens and tests across several runs (one
               model each). A field whose answers agree across >= MIN_AGREE models
               (all models that found the item) is accepted. Everything else, plus
               a seeded random sample of accepted fields, goes to an adjudication
               workbook with the majority answer pre-filled.
  2. import    read the adjudicated workbook back and write gold.jsonl.
  3. score     align a run against gold.jsonl with the same matcher and report
               per-field agreement, item recall/precision and report-level rollups.

Alignment is deterministic. Two items (from different models, or a model and
gold) are the same specimen/test when their part labels match, or when the text
spans they cite overlap. Specimens with different non-null labels never match.

Answers are compared after light normalization (case, punctuation, and for
free-text fields token containment, so "Fundal" ~ "Fundal masses"). The goal is
to send a human only the disagreements that could matter.
"""

from __future__ import annotations

import dataclasses
import json
import random
import re
from collections import Counter
from pathlib import Path

from hnc_extract.rollup import report_summary

MIN_AGREE = 3  # a field is accepted only if at least this many models found the item
SPOT_CHECK_RATE = 0.1
MIN_OVERLAP_CHARS = 12
MIN_OVERLAP_RATIO = 0.5
ABSENT = "(absent)"
NONE = "(none)"
LIST_SEP = " | "

SPECIMEN_FIELDS = [
    "label",
    "site",
    "material",
    "procedure",
    "carcinoma_status",
    "histology",
    "invasion",
    "invasion_uncertain",
    "keratinization",
    "dysplasia_status",
    "dysplasia_grades",
]
TEST_FIELDS = ["test", "test_type", "specimen_label", "result", "pdl1_scores"]
# compared by token containment instead of exact equality
TEXT_FIELDS = {"site", "histology", "test"}
LIST_FIELDS = {
    "procedure",
    "histology",
    "invasion",
    "invasion_uncertain",
    "keratinization",
    "dysplasia_grades",
    "result",
    "pdl1_scores",
}


# ---------------------------------------------------------------- normalization


def norm_label(label: str | None) -> str | None:
    """Part label for matching: upper-case alphanumerics; masked labels are None."""
    if not label:
        return None
    s = re.sub(r"[^0-9A-Za-z]", "", label).upper()
    if not s or (len(s) > 1 and set(s) == {"X"}):  # "XX-XXXX" mask; "X" is a part
        return None
    return s


def _tokens(s: str) -> frozenset[str]:
    return frozenset(re.findall(r"[0-9a-z]+", s.lower().replace("%", "")))


def _text_equiv(a: str, b: str) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return ta == tb
    return ta <= tb or tb <= ta


def canon(field: str, value):
    """Hashable comparison form of a field value."""
    if value is None or value == [] or value == "":
        return NONE
    if field in LIST_FIELDS:
        vals = value if isinstance(value, list) else [value]
        if field == "pdl1_scores":
            vals = [
                re.sub(r"\s+", " ", v.replace("%", "")).strip().upper() for v in vals
            ]
        elif field not in TEXT_FIELDS:
            vals = [str(v) for v in vals]
        if field in ("invasion", "invasion_uncertain"):
            # "not_stated" carries no information beyond "no invasion stated"
            vals = [v for v in vals if v != "not_stated"]
            if not vals:
                return NONE
        return tuple(sorted(set(vals)))
    if field in ("label", "specimen_label"):
        return norm_label(value) or NONE
    return str(value)


def equivalent(field: str, a, b) -> bool:
    """Do two canonical answers agree?"""
    if a == b:
        return True
    if ABSENT in (a, b) or NONE in (a, b):
        return False
    if field == "specimen_label":
        # a block ("B2") is on its part ("B")
        short, long_ = sorted((a, b), key=len)
        return long_.startswith(short) and long_[len(short) :].isdigit()
    if field in TEXT_FIELDS:
        if isinstance(a, tuple):
            return all(any(_text_equiv(x, y) for y in b) for x in a) and all(
                any(_text_equiv(y, x) for x in a) for y in b
            )
        return _text_equiv(a, b)
    return False


def render(value) -> str:
    if value is None or value == () or value == []:
        return NONE
    if isinstance(value, (list, tuple)):
        return LIST_SEP.join(str(v) for v in value)
    return str(value)


def parse(field: str, text) -> object:
    """Inverse of render for an adjudicated cell."""
    text = "" if text is None else str(text).strip()
    if text in ("", NONE):
        return [] if field in LIST_FIELDS else None
    if field in LIST_FIELDS:
        return [p.strip() for p in text.split(LIST_SEP.strip()) if p.strip()]
    return text


# ---------------------------------------------------------------- views of a run


def _spans(checks) -> list[tuple[int, int]]:
    out = []
    for c in checks or []:
        if isinstance(c, list):
            out.extend(_spans(c))
        elif c.get("found") and c.get("start") is not None:
            out.append((c["start"], c["end"]))
    return out


def run_views(nodes_by_doc: dict, rollups: dict) -> dict[str, dict]:
    """Per doc: specimens and tests from the rollup, each with cited text spans."""
    views = {}
    for doc_id, nodes in nodes_by_doc.items():
        ru = rollups[doc_id]

        def node_spans(field: str, path: str, index: int | None = None):
            n = nodes.get((field, path))
            if n is None or n.record is None or not n.record.evidence:
                return []
            ev = n.record.evidence
            if index is not None:
                return _spans(ev[index]) if index < len(ev) else []
            return _spans(ev)

        specimens = []
        for i, s in enumerate(ru.get("specimens", [])):
            spans = node_spans("specimen", "", i)
            for f in ("carcinoma_status", "histology", "dysplasia_status"):
                spans += node_spans(f, s["path"])
            specimens.append({**s, "spans": spans})
        tests = []
        for i, t in enumerate(ru.get("tests", [])):
            spans = node_spans("ancillary_test", "", i) + node_spans(
                "test_result", t["path"]
            )
            tests.append({**t, "spans": spans})
        views[doc_id] = {"specimens": specimens, "tests": tests}
    return views


# ---------------------------------------------------------------- alignment


def _overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> float:
    ca = {i for s, e in a for i in range(s, e)}
    cb = {i for s, e in b for i in range(s, e)}
    if not ca or not cb:
        return 0.0
    inter = len(ca & cb)
    if inter < MIN_OVERLAP_CHARS:
        return 0.0
    return inter / min(len(ca), len(cb))


def specimen_score(a: dict, b: dict) -> float:
    la, lb = norm_label(a.get("label")), norm_label(b.get("label"))
    if la and lb:
        return 2.0 if la == lb else 0.0
    ov = _overlap(a["spans"], b["spans"])
    return ov if ov >= MIN_OVERLAP_RATIO else 0.0


def _test_key(t: dict) -> frozenset[str]:
    return _tokens(t.get("test") or "") - {"xx", "xxxx", "immunostain", "stain"}


def test_score(a: dict, b: dict) -> float:
    ov = _overlap(a["spans"], b["spans"])
    same_type = a.get("test_type") == b.get("test_type")
    names = bool(_test_key(a) & _test_key(b))
    if a.get("test_type") == "biomarker_ihc" or b.get("test_type") == "biomarker_ihc":
        # several markers often share one sentence: the marker name must match
        if not names:
            return 0.0
        return 1.0 + ov
    sa, sb = norm_label(a.get("specimen_label")), norm_label(b.get("specimen_label"))
    # "B2" (a block) is compatible with part "B"
    if sa and sb and not (sa.startswith(sb) or sb.startswith(sa)) and same_type:
        return 0.0  # same test on two different parts
    if ov >= MIN_OVERLAP_RATIO:
        return ov + (0.5 if same_type else 0.0)
    if same_type and names:
        return 0.6
    return 0.0


def align(items_by_source: dict[str, list[dict]], score) -> list[dict[str, int]]:
    """Group items from several sources into clusters (one item per source each).

    Sources are processed in order; each source's items are assigned greedily,
    best score first, to the cluster whose best-matching member scores highest.
    Returns clusters as {source: item index}.
    """
    clusters: list[dict[str, int]] = []
    for src, items in items_by_source.items():
        cand = []
        for i, it in enumerate(items):
            for c, cl in enumerate(clusters):
                sc = max(score(it, items_by_source[s][j]) for s, j in cl.items())
                if sc > 0:
                    cand.append((sc, i, c))
        used_i, used_c = set(), set()
        for sc, i, c in sorted(cand, key=lambda x: (-x[0], x[1], x[2])):
            if i in used_i or c in used_c:
                continue
            clusters[c][src] = i
            used_i.add(i)
            used_c.add(c)
        for i in range(len(items)):
            if i not in used_i:
                clusters.append({src: i})
    return clusters


def _cluster_sort_key(cl: dict[str, int], items_by_source) -> tuple:
    members = [items_by_source[s][i] for s, i in cl.items()]
    starts = [min(s for s, _ in m["spans"]) for m in members if m["spans"]]
    return (min(starts) if starts else 10**9,)


# ---------------------------------------------------------------- consensus


@dataclasses.dataclass
class FieldVote:
    field: str
    answers: dict[str, object]  # source -> canonical answer (ABSENT if not found)
    proposed: object
    n_majority: int
    agreed: bool


def vote(field: str, answers: dict[str, object]) -> FieldVote:
    present = {s: a for s, a in answers.items() if a != ABSENT}
    classes: list[list[str]] = []
    for s, a in present.items():
        for cl in classes:
            if equivalent(field, present[cl[0]], a):
                cl.append(s)
                break
        else:
            classes.append([s])
    if not classes:
        return FieldVote(field, answers, None, 0, False)
    best = max(classes, key=len)
    # representative: most common exact answer in the class, ties -> longest render
    counts = Counter(present[s] for s in best)
    proposed = max(counts, key=lambda a: (counts[a], len(render(a))))
    agreed = len(classes) == 1 and len(present) >= MIN_AGREE
    return FieldVote(field, answers, proposed, len(best), agreed)


def _presence_vote(cl: dict[str, int], sources: list[str]) -> FieldVote:
    answers = {s: ("yes" if s in cl else "no") for s in sources}
    n = sum(a == "yes" for a in answers.values())
    agreed = n == len(sources)
    return FieldVote(
        "present", answers, "yes" if n * 2 >= len(sources) else "no", n, agreed
    )


def draft(
    views_by_source: dict[str, dict[str, dict]], doc_ids: list[str]
) -> list[dict]:
    """Cluster records for every doc: per-field votes for specimens and tests."""
    sources = list(views_by_source)
    records = []
    for doc_id in doc_ids:
        for kind, fields, score in (
            ("specimen", SPECIMEN_FIELDS, specimen_score),
            ("test", TEST_FIELDS, test_score),
        ):
            key = "specimens" if kind == "specimen" else "tests"
            items = {s: views_by_source[s][doc_id][key] for s in sources}
            clusters = align(items, score)
            clusters.sort(key=lambda cl: _cluster_sort_key(cl, items))
            for n, cl in enumerate(clusters, 1):
                members = {s: items[s][i] for s, i in cl.items()}
                votes = [_presence_vote(cl, sources)]
                for f in fields:
                    ans = {
                        s: canon(f, members[s].get(f)) if s in members else ABSENT
                        for s in sources
                    }
                    votes.append(vote(f, ans))
                spans = sorted({sp for m in members.values() for sp in m["spans"]})
                records.append(
                    {
                        "doc_id": doc_id,
                        "kind": kind,
                        "key": f"{'S' if kind == 'specimen' else 'T'}{n:02d}",
                        "members": {s: members[s].get("path") for s in members},
                        "spans": spans,
                        "votes": [dataclasses.asdict(v) for v in votes],
                    }
                )
    return records


def review_rows(
    records: list[dict], seed: int = 0, rate: float = SPOT_CHECK_RATE
) -> list[dict]:
    """One row per field that needs a human: disagreements and a spot-check sample."""
    rng = random.Random(seed)
    rows = []
    for r in records:
        presence = r["votes"][0]
        for v in r["votes"]:
            if v["agreed"]:
                reason = "spot_check" if rng.random() < rate else None
            elif v["field"] != "present" and not presence["agreed"]:
                # not every model found the item: decide its 'present' row
                # first; fill these only if the answer there is yes
                reason = "if_present"
            else:
                reason = "disagreement"
            if reason is None:
                continue
            rows.append(
                {
                    "row_id": f"{r['doc_id']}|{r['key']}|{v['field']}",
                    "doc_id": r["doc_id"],
                    "kind": r["kind"],
                    "key": r["key"],
                    "field": v["field"],
                    "reason": reason,
                    "answers": {s: render(a) for s, a in v["answers"].items()},
                    "n_majority": v["n_majority"],
                    "proposed": render(v["proposed"]),
                }
            )
    return rows


# ---------------------------------------------------------------- gold file


def build_gold(records: list[dict], verdicts: dict[str, str]) -> dict[str, dict]:
    """Gold per doc from draft records plus adjudicated cells {row_id: gold text}.

    A row without a verdict keeps the proposed (majority/consensus) answer.
    """
    gold: dict[str, dict] = {}
    for r in records:
        doc = gold.setdefault(
            r["doc_id"], {"doc_id": r["doc_id"], "specimens": [], "tests": []}
        )
        vals = {}
        for v in r["votes"]:
            rid = f"{r['doc_id']}|{r['key']}|{v['field']}"
            if rid in verdicts:
                vals[v["field"]] = parse(v["field"], verdicts[rid])
            elif v["field"] == "present":
                vals["present"] = v["proposed"]
            else:
                p = v["proposed"]
                vals[v["field"]] = (
                    None
                    if p in (None, NONE)
                    else list(p) if isinstance(p, tuple) else p
                )
                if vals[v["field"]] is None and v["field"] in LIST_FIELDS:
                    vals[v["field"]] = []
        if str(vals.pop("present")).strip().lower() not in ("yes", "y", "true"):
            continue
        item = {"key": r["key"], "spans": [list(s) for s in r["spans"]], **vals}
        doc["specimens" if r["kind"] == "specimen" else "tests"].append(item)
    for doc in gold.values():
        doc["report"] = report_summary(
            [
                {
                    "carcinoma_status": s.get("carcinoma_status"),
                    "dysplasia_status": s.get("dysplasia_status"),
                    "dysplasia_grades": s.get("dysplasia_grades") or [],
                    "invasion": s.get("invasion") or [],
                    "invasion_uncertain": s.get("invasion_uncertain") or [],
                }
                for s in doc["specimens"]
            ]
        )
    return gold


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------- scoring

REPORT_FIELDS = [
    "report_carcinoma_status",
    "report_invasive_carcinoma",
    "report_invasion_suspected",
    "report_dysplasia_status",
    "worst_dysplasia_grade",
]


def score_run(
    gold: dict[str, dict], views: dict[str, dict], rollups: dict[str, dict]
) -> dict:
    """Agreement of one run with gold: item matching, per-field, report level."""
    out = {"items": {}, "fields": {}, "report": {}, "mismatches": []}
    for kind, fields, score in (
        ("specimens", SPECIMEN_FIELDS, specimen_score),
        ("tests", TEST_FIELDS, test_score),
    ):
        tp = fn = fp = 0
        for doc_id, g in gold.items():
            gi = [{**s, "spans": [tuple(x) for x in s["spans"]]} for s in g[kind]]
            mi = views.get(doc_id, {}).get(kind, [])
            for cl in align({"gold": gi, "model": mi}, score):
                if "gold" in cl and "model" in cl:
                    tp += 1
                    gs, ms = gi[cl["gold"]], mi[cl["model"]]
                    for f in fields:
                        a, b = canon(f, gs.get(f)), canon(f, ms.get(f))
                        ok = equivalent(f, a, b)
                        c = out["fields"].setdefault(f"{kind}.{f}", Counter())
                        c["agree" if ok else "disagree"] += 1
                        if not ok:
                            out["mismatches"].append(
                                {
                                    "doc_id": doc_id,
                                    "kind": kind,
                                    "gold_key": gs["key"],
                                    "field": f,
                                    "gold": render(a),
                                    "model": render(b),
                                }
                            )
                elif "gold" in cl:
                    fn += 1
                    out["mismatches"].append(
                        {
                            "doc_id": doc_id,
                            "kind": kind,
                            "gold_key": gi[cl["gold"]]["key"],
                            "field": "present",
                            "gold": "yes",
                            "model": "no",
                        }
                    )
                else:
                    fp += 1
                    m = mi[cl["model"]]
                    out["mismatches"].append(
                        {
                            "doc_id": doc_id,
                            "kind": kind,
                            "gold_key": None,
                            "field": "present",
                            "gold": "no",
                            "model": f"yes ({m.get('label') or m.get('test') or '-'})",
                        }
                    )
        out["items"][kind] = {"matched": tp, "missed": fn, "extra": fp}
    for doc_id, g in gold.items():
        ru = rollups.get(doc_id, {})
        for f in REPORT_FIELDS:
            ok = g["report"].get(f) == ru.get(f)
            out["report"].setdefault(f, Counter())["agree" if ok else "disagree"] += 1
            if not ok:
                out["mismatches"].append(
                    {
                        "doc_id": doc_id,
                        "kind": "report",
                        "gold_key": None,
                        "field": f,
                        "gold": render(g["report"].get(f)),
                        "model": render(ru.get(f)),
                    }
                )
    return out


# ---------------------------------------------------------------- workbook

GOLD_COL = "gold"
COMMENT_COL = "comment"


def _item_desc(r: dict, src: str, views_by_source: dict) -> str:
    path = r["members"].get(src)
    if path is None:
        return "-"
    key = "specimens" if r["kind"] == "specimen" else "tests"
    it = next(x for x in views_by_source[src][r["doc_id"]][key] if x["path"] == path)
    if r["kind"] == "specimen":
        return f"{it.get('label') or '?'}: {it.get('site') or ''}"
    return f"{it.get('test')} [{it.get('specimen_label') or '-'}]"


def write_adjudication(
    path: Path,
    records: list[dict],
    rows: list[dict],
    docs: list,
    views_by_source: dict,
    info: dict,
) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, PatternFill

    from hnc_extract.export import _context_rich, _safe, _sheet_header

    sources = list(views_by_source)
    text = {d.doc_id: d.text for d in docs}
    rec = {(r["doc_id"], r["key"]): r for r in records}
    wrap = Alignment(wrap_text=True, vertical="top")
    yellow = PatternFill("solid", fgColor="FFF2CC")

    wb = Workbook()
    readme = wb.active
    readme.title = "README"
    lines = [
        "Gold-key adjudication workbook",
        "",
        *[f"{k}: {v}" for k, v in info.items()],
        "",
        "Sheet 'adjudicate': one row per field a person must decide.",
        "  reason=disagreement  the models disagree, or fewer than "
        f"{MIN_AGREE} models found the item.",
        "  reason=if_present    not every model found this item. Decide its 'present' row "
        "first; fill these only if you answer yes there.",
        f"  reason=spot_check    all models agreed; a {info.get('spot_check_rate', SPOT_CHECK_RATE):.0%} random sample "
        "to estimate errors the models share.",
        f"Edit the yellow '{GOLD_COL}' column (pre-filled with the majority answer). "
        f"Lists are separated by '{LIST_SEP.strip()}'; '{NONE}' means not stated / empty.",
        "For 'present' rows answer yes or no. Keep row_id intact.",
        "Fields not listed were accepted by consensus and go to gold as is.",
        "Sheet 'items': every aligned specimen/test with what each model called it.",
    ]
    for i, line in enumerate(lines, 1):
        readme.cell(row=i, column=1, value=line)
    readme.column_dimensions["A"].width = 120

    ws = wb.create_sheet("adjudicate")
    cols = (
        [
            ("row_id", 16),
            ("doc_id", 12),
            ("key", 6),
            ("kind", 9),
            ("field", 16),
            ("reason", 12),
        ]
        + [(f"item: {s}", 22) for s in sources]
        + [(s, 24) for s in sources]
        + [("n_majority", 6), (GOLD_COL, 28), (COMMENT_COL, 24), ("context", 90)]
    )
    _sheet_header(ws, cols, review={GOLD_COL, COMMENT_COL})
    for i, row in enumerate(rows, 2):
        r = rec[(row["doc_id"], row["key"])]
        vals = (
            [
                row["row_id"],
                row["doc_id"][:8],
                row["key"],
                row["kind"],
                row["field"],
                row["reason"],
            ]
            + [_item_desc(r, s, views_by_source) for s in sources]
            + [row["answers"][s] for s in sources]
            + [row["n_majority"], row["proposed"], ""]
        )
        for j, v in enumerate(vals, 1):
            ws.cell(row=i, column=j, value=_safe(v)).alignment = wrap
        ws.cell(row=i, column=len(vals) - 1).fill = yellow
        checks = [{"found": True, "start": s, "end": e} for s, e in r["spans"]]
        ws.cell(
            row=i, column=len(vals) + 1, value=_context_rich(text[r["doc_id"]], checks)
        ).alignment = wrap
    ws.auto_filter.ref = (
        f"A1:{ws.cell(row=1, column=len(cols)).column_letter}{max(len(rows) + 1, 2)}"
    )

    items = wb.create_sheet("items")
    icols = [("doc_id", 12), ("key", 6), ("kind", 9), ("n_models", 8)] + [
        (s, 34) for s in sources
    ]
    _sheet_header(items, icols)
    for i, r in enumerate(records, 2):
        vals = [r["doc_id"][:8], r["key"], r["kind"], len(r["members"])] + [
            _item_desc(r, s, views_by_source) for s in sources
        ]
        for j, v in enumerate(vals, 1):
            items.cell(row=i, column=j, value=_safe(v)).alignment = wrap

    ds = wb.create_sheet("documents")
    _sheet_header(ds, [("doc_id", 40), ("text", 160)])
    for i, d in enumerate(docs, 2):
        ds.cell(row=i, column=1, value=d.doc_id)
        ds.cell(row=i, column=2, value=_safe(d.text)).alignment = wrap
    wb.save(path)


def read_verdicts(path: Path) -> dict[str, str]:
    """{row_id: gold text} from an adjudicated workbook."""
    from openpyxl import load_workbook

    ws = load_workbook(path, read_only=True)["adjudicate"]
    rows = ws.iter_rows(values_only=True)
    header = list(next(rows))
    ri, gi = header.index("row_id"), header.index(GOLD_COL)
    return {r[ri]: ("" if r[gi] is None else str(r[gi])) for r in rows if r[ri]}
