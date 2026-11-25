"""Regenerate the committed TCGA-HNSC reference tables in configs/tcga/ from the GDC.

A dev-time tool (needs internet); jobs only read the committed tables. Both sources are
open access.

  hnsc_cases.tsv  every TCGA-HNSC case: submitter id, GDC case id, primary site,
                  disease type (GDC API /cases). Defines which reports are HNSC.
  tss.tsv         tissue source site codes of the HNSC study (GDC code table page):
                  code, source site name, study, BCR.

Usage:
  python -m hnc_extract.tcga_refs            # -> configs/tcga/
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

from hnc_extract import paths

GDC_CASES = "https://api.gdc.cancer.gov/cases"
GDC_TSS = (
    "https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/"
    "tissue-source-site-codes"
)
PROJECT = "TCGA-HNSC"
STUDY = "Head and Neck squamous cell carcinoma"
OUT_DIR = paths.CONFIGS_DIR / "tcga"

CASE_COLUMNS = ("submitter_id", "case_id", "primary_site", "disease_type")
TSS_COLUMNS = ("tss", "source_site", "study", "bcr")


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "hnc-extract"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_cases() -> list[dict]:
    query = {
        "filters": json.dumps(
            {"op": "=", "content": {"field": "project.project_id", "value": PROJECT}}
        ),
        "fields": "submitter_id,primary_site,disease_type",
        "size": "2000",
        "format": "json",
    }
    data = json.loads(_get(f"{GDC_CASES}?{urllib.parse.urlencode(query)}"))["data"]
    if data["pagination"]["total"] != len(data["hits"]):
        raise RuntimeError("GDC returned a partial case list; raise size")
    rows = [
        {
            "submitter_id": h["submitter_id"],
            "case_id": h["id"],
            "primary_site": h.get("primary_site") or "",
            "disease_type": h.get("disease_type") or "",
        }
        for h in data["hits"]
    ]
    return sorted(rows, key=lambda r: r["submitter_id"])


def parse_tss_table(page: str) -> list[dict]:
    rows = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
        cells = [
            html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
            for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)
        ]
        if len(cells) == 4 and cells[2] == STUDY:
            rows.append(dict(zip(TSS_COLUMNS, cells)))
    if not rows:
        raise RuntimeError("no HNSC rows in the TSS page; its layout changed")
    return sorted(rows, key=lambda r: r["tss"])


def write_table(path: Path, columns: tuple[str, ...], rows: list[dict], source: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    day = dt.date.today().isoformat()
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(
            f"# source: {source} (retrieved {day}; python -m hnc_extract.tcga_refs)\n"
        )
        w = csv.DictWriter(f, columns, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_table(path: Path) -> list[dict]:
    """Rows of a table written by write_table ('#' lines are provenance comments)."""
    with path.open(encoding="utf-8", newline="") as f:
        lines = [line for line in f if not line.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", type=Path, default=OUT_DIR)
    a = p.parse_args(argv)
    cases = fetch_cases()
    write_table(a.out / "hnsc_cases.tsv", CASE_COLUMNS, cases, f"{GDC_CASES} {PROJECT}")
    tss = parse_tss_table(_get(GDC_TSS).decode("utf-8", "replace"))
    write_table(a.out / "tss.tsv", TSS_COLUMNS, tss, GDC_TSS)
    print(f"{len(cases)} cases, {len(tss)} TSS codes -> {a.out}")


if __name__ == "__main__":
    main()
