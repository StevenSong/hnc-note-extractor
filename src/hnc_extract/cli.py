"""Command line entry point: python -m hnc_extract <command> ...

  ingest-synthetic  synthetic TSVs -> canonical document JSONL
  run               run (or resume) the field DAG for one doc type with one model
  status            node counts per field/status for a run dir
  export            JSONL + review workbook for a run dir
  probe             check a model server: reachability, JSON-schema decoding, reasoning field

A run dir holds one model's results for one document set:
  <run-dir>/run.json        settings (docs, specs, model config, doc filter)
  <run-dir>/cache.sqlite    every call, keyed by content (resumable)
  <run-dir>/export/         outputs of `export`
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

from hnc_extract.dag import run, traverse
from hnc_extract.documents import Document, read_jsonl
from hnc_extract.llm import ModelConfig, make_client
from hnc_extract.spec import load_specs
from hnc_extract.store import Store


def _load_run(run_dir: Path) -> dict:
    p = run_dir / "run.json"
    if not p.exists():
        sys.exit(
            f"{p} not found; start the run with `run --docs ... --model ...` first"
        )
    return json.loads(p.read_text())


def _select_docs(settings: dict) -> list[Document]:
    docs = list(read_jsonl(Path(settings["docs"])))
    ids = settings.get("doc_ids")
    if ids:
        wanted = set(ids)
        docs = [d for d in docs if d.doc_id in wanted]
    if settings.get("limit"):
        docs = docs[: settings["limit"]]
    return docs


def _doc_type(docs: list[Document]) -> str:
    types = {d.doc_type for d in docs}
    if len(types) != 1:
        sys.exit(f"expected one doc type per run, got {sorted(types)}")
    return types.pop()


def cmd_run(a: argparse.Namespace) -> None:
    run_dir: Path = a.run_dir
    p = run_dir / "run.json"
    saved = json.loads(p.read_text()) if p.exists() else None
    if saved is None and (a.docs is None or a.model is None):
        sys.exit("new run dir: --docs and --model are required")

    model = (
        ModelConfig.load(a.model)
        if a.model
        else ModelConfig.model_validate(saved["model"])
    )
    if a.base_url:
        model = model.model_copy(update={"base_url": a.base_url})
    settings = {
        "docs": str(a.docs) if a.docs else saved["docs"],
        "specs": str(a.specs) if a.specs else (saved or {}).get("specs", "specs"),
        "limit": a.limit if a.limit is not None else (saved or {}).get("limit"),
        "doc_ids": (
            a.doc_ids.split(",") if a.doc_ids else (saved or {}).get("doc_ids")
        ),
        "model": model.model_dump(mode="json"),
        "model_fingerprint": model.fingerprint(),
    }
    if (
        saved
        and saved["model_fingerprint"] != settings["model_fingerprint"]
        and not a.force
    ):
        sys.exit(
            f"{run_dir} holds results of model config {saved['model']['name']!r} with different settings; "
            "use a new run dir per model config (or --force to switch; old calls stay cached but unused)"
        )
    settings["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(
        timespec="seconds"
    )
    settings["created_utc"] = (
        saved.get("created_utc") if saved else settings["updated_utc"]
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(settings, indent=2) + "\n")

    docs = _select_docs(settings)
    specs = load_specs(Path(settings["specs"]), _doc_type(docs))
    fields = set(a.fields.split(",")) if a.fields else None
    if fields and (unknown := fields - set(specs.by_name)):
        sys.exit(f"unknown fields: {sorted(unknown)}")
    client = make_client(model)
    store = Store(run_dir / "cache.sqlite")
    print(
        f"run {run_dir}: {len(docs)} docs, model {model.name} ({model.served_model_name} @ {model.base_url})",
        file=sys.stderr,
    )
    summary = run(
        docs, specs, model, client, store, fields=fields, concurrency=a.concurrency
    )
    store.close()
    for f, c in summary.items():
        print(f"{f:18s} {dict(c)}")


def _traverse_all(run_dir: Path):
    settings = _load_run(run_dir)
    model = ModelConfig.model_validate(settings["model"])
    docs = _select_docs(settings)
    specs = load_specs(Path(settings["specs"]), _doc_type(docs))
    store = Store(run_dir / "cache.sqlite")
    nodes = {d.doc_id: traverse(d, specs, store, model.fingerprint()) for d in docs}
    store.close()
    return settings, model, docs, specs, nodes


def cmd_status(a: argparse.Namespace) -> None:
    _, _, docs, specs, nodes = _traverse_all(a.run_dir)
    c = Counter((n.spec.name, n.status) for nd in nodes.values() for n in nd.values())
    calls = Counter(
        (n.spec.name, n.record.status)
        for nd in nodes.values()
        for n in nd.values()
        if n.record
    )
    print(f"{len(docs)} docs")
    for f in specs.order:
        st = {s: v for (ff, s), v in c.items() if ff == f}
        cs = {s: v for (ff, s), v in calls.items() if ff == f}
        print(f"{f:18s} nodes {st}  calls {cs}")


def cmd_export(a: argparse.Namespace) -> None:
    from hnc_extract.export import node_record, write_workbook

    settings, model, docs, specs, nodes = _traverse_all(a.run_dir)
    out = a.out or a.run_dir / "export"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "note_extractions.jsonl").open("w", encoding="utf-8") as f:
        for d in docs:
            for n in nodes[d.doc_id].values():
                f.write(json.dumps(node_record(n), ensure_ascii=False) + "\n")
    info = {
        "run_dir": str(a.run_dir),
        "documents": settings["docs"],
        "n_documents": len(docs),
        "model": f"{model.name} ({model.served_model_name})",
        "exported_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    res = write_workbook(out / "review.xlsx", docs, nodes, specs.order, info)
    with (out / "report_rollup.jsonl").open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(
                json.dumps(
                    {
                        "doc_id": d.doc_id,
                        "patient_id": d.patient_id,
                        **res["rollups"][d.doc_id],
                    }
                )
                + "\n"
            )
    print(f"{res['review_rows']} review rows -> {out / 'review.xlsx'}")
    print(f"JSONL -> {out / 'note_extractions.jsonl'}, {out / 'report_rollup.jsonl'}")


def cmd_probe(a: argparse.Namespace) -> None:
    """Send one small real extraction (carcinoma_status) in both decoding modes."""
    from hnc_extract.documents import sha256_text
    from hnc_extract.extract import run_call
    from hnc_extract.llm import OpenAICompatClient

    model = ModelConfig.load(a.model)
    if a.base_url:
        model = model.model_copy(update={"base_url": a.base_url})
    client = OpenAICompatClient(model, transport_retries=0)
    try:
        print("served models:", client.list_models())
    except Exception as e:  # noqa: BLE001
        sys.exit(f"cannot reach {model.base_url}: {e}")
    specs = load_specs(a.specs, "pathology_report")
    spec = specs["carcinoma_status"]
    text = (
        "Surgical Pathology Report  A.  Left lateral tongue; excision:  - Invasive squamous cell carcinoma, "
        "keratinizing.  B.  Left neck level II; dissection:  - Four lymph nodes, negative for carcinoma (0/4)."
    )
    doc = Document(
        "probe",
        "pathology_report",
        "probe",
        "2024-01-01T00:00:00",
        None,
        text,
        sha256_text(text),
        {},
        {},
        [],
    )
    target = {
        "evidence": ["B.  Left neck level II; dissection:"],
        "assertion": "affirmed",
        "value": "Left neck level II; dissection",
        "label": "B",
        "site": "Left neck level II",
    }
    ctx = {
        "chain": [
            {
                "field": "specimen",
                "title": "Specimen",
                "target_instruction": specs["specimen"].target_instruction,
                "item": target,
                "siblings": [
                    '"Left lateral tongue; excision" (label=A, site=Left lateral tongue)'
                ],
            }
        ],
        "deps": [],
    }
    for mode in ("json_schema", "none"):
        m = model.model_copy(update={"structured_output": mode, "max_attempts": 1})
        rec = run_call(
            client=client,
            model=m,
            spec=spec,
            instructions=specs.instructions,
            doc=doc,
            context=ctx,
            key="0" * 64,
        )
        att = rec.attempts[-1]
        print(f"\n=== structured_output={mode}: status={rec.status}")
        print(
            f"finish_reason={att.get('finish_reason')} latency={att.get('latency_s')}s usage={att.get('usage')}"
        )
        print(
            f"reasoning returned: {bool(att.get('reasoning'))} ({len(att.get('reasoning') or '')} chars)"
        )
        if att.get("error"):
            print("error:", att["error"][:500])
        print("content:", (att.get("content") or "")[:800])
        print("expected: one extraction, category=negative, basis=explicit_negative")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="hnc_extract",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("ingest-synthetic", help="see hnc_extract.synthetic_ingest")
    sp.add_argument("rest", nargs=argparse.REMAINDER)

    sp = sub.add_parser("run")
    sp.add_argument("--run-dir", type=Path, required=True)
    sp.add_argument("--docs", type=Path, help="canonical document JSONL (one doc type)")
    sp.add_argument("--model", type=Path, help="model config YAML")
    sp.add_argument("--specs", type=Path, help="spec dir (default: specs)")
    sp.add_argument("--base-url", help="override the model server URL")
    sp.add_argument(
        "--fields",
        help="comma-separated subset of fields to run (upstream must be done)",
    )
    sp.add_argument("--limit", type=int, help="only the first N documents")
    sp.add_argument("--doc-ids", help="comma-separated doc ids")
    sp.add_argument(
        "--concurrency", type=int, default=16, help="parallel requests to the server"
    )
    sp.add_argument("--force", action="store_true")

    for name in ("status", "export"):
        sp = sub.add_parser(name)
        sp.add_argument("--run-dir", type=Path, required=True)
        if name == "export":
            sp.add_argument("--out", type=Path)

    sp = sub.add_parser("probe")
    sp.add_argument("--model", type=Path, required=True)
    sp.add_argument("--specs", type=Path, default=Path("specs"))
    sp.add_argument("--base-url")

    a = p.parse_args(argv)
    if a.cmd == "ingest-synthetic":
        from hnc_extract.synthetic_ingest import main as ingest_main

        ingest_main(a.rest)
    else:
        {
            "run": cmd_run,
            "status": cmd_status,
            "export": cmd_export,
            "probe": cmd_probe,
        }[a.cmd](a)


if __name__ == "__main__":
    main()
