"""Command line entry point: python -m hnc_extract <command> ...

  ingest-synthetic  synthetic TSVs -> canonical document JSONL
  run               run (or resume) the field DAG for one doc type with one model
  status            node counts per field/status for a run dir
  export            JSONL + review workbook for a run dir
  probe             check a model server: reachability, JSON-schema decoding, reasoning field
  gold-draft        align several runs (one model each) into a consensus gold draft + adjudication workbook
  gold-review       local web app to adjudicate a gold draft (saves verdicts.json)
  gold-import       adjudicated verdicts (review app or workbook) -> gold.jsonl
  score             compare a run against gold.jsonl

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


def read_doc_ids_file(path: Path) -> list[str]:
    """One doc id per line; text after '#' and blank lines are ignored."""
    ids = []
    for line in path.read_text().splitlines():
        s = line.split("#", 1)[0].strip()
        if s:
            ids.append(s)
    if not ids:
        sys.exit(f"{path}: no doc ids")
    if len(set(ids)) != len(ids):
        dup = sorted(i for i, n in Counter(ids).items() if n > 1)
        sys.exit(f"{path}: duplicate doc ids {dup}")
    return ids


def _select_docs(settings: dict) -> list[Document]:
    docs = list(read_jsonl(Path(settings["docs"])))
    ids = settings.get("doc_ids")
    if ids:
        wanted = set(ids)
        docs = [d for d in docs if d.doc_id in wanted]
        missing = wanted - {d.doc_id for d in docs}
        if missing:
            sys.exit(
                f"{len(missing)} requested doc ids not in {settings['docs']}: {sorted(missing)}"
            )
    if settings.get("limit"):
        docs = docs[: settings["limit"]]
    return docs


def _doc_type(docs: list[Document]) -> str:
    types = {d.doc_type for d in docs}
    if len(types) != 1:
        sys.exit(f"expected one doc type per run, got {sorted(types)}")
    return types.pop()


def _doc_filter(a: argparse.Namespace, prev: dict) -> dict:
    """Document filter settings: new flags override the saved run.json values."""
    if a.all_docs:
        if a.limit:
            sys.exit("--all-docs and --limit are exclusive")
        return {"limit": None, "doc_ids": None, "doc_ids_file": None}
    out = {"limit": a.limit if a.limit is not None else prev.get("limit")}
    if a.doc_ids:
        out.update(doc_ids=a.doc_ids.split(","), doc_ids_file=None)
    elif a.doc_ids_file:
        # The resolved ids are saved, so later edits to the file don't change this run.
        out.update(
            doc_ids=read_doc_ids_file(a.doc_ids_file),
            doc_ids_file=str(a.doc_ids_file),
        )
    else:
        out.update(doc_ids=prev.get("doc_ids"), doc_ids_file=prev.get("doc_ids_file"))
    return out


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
        **_doc_filter(a, saved or {}),
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


def _traverse_all(run_dir: Path, specs_dir: Path | None = None):
    """Rebuild a run's nodes from its cache.

    specs_dir overrides the run's spec dir, e.g. a checkout of the specs the run
    was made with: after a spec edit the cached calls no longer match.
    """
    settings = _load_run(run_dir)
    model = ModelConfig.model_validate(settings["model"])
    docs = _select_docs(settings)
    specs = load_specs(specs_dir or Path(settings["specs"]), _doc_type(docs))
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


def _views(run_dir: Path, specs_dir: Path | None = None):
    from hnc_extract.gold import run_views
    from hnc_extract.rollup import pathology_rollup

    settings, model, docs, specs, nodes = _traverse_all(run_dir, specs_dir)
    pending = sum(
        n.status in ("pending", "blocked") for nd in nodes.values() for n in nd.values()
    )
    if pending:
        sys.exit(
            f"{run_dir}: {pending} nodes have no cached result under these specs. "
            "If the specs changed since the run, pass --specs with the run's spec dir "
            "(e.g. `git archive <commit> specs | tar -x -C <dir>`)."
        )
    if specs.doc_type != "pathology_report":
        sys.exit(f"{run_dir}: gold/score support pathology_report only")
    rollups = {d: pathology_rollup(n) for d, n in nodes.items()}
    return model, docs, run_views(nodes, rollups), rollups, specs


def cmd_gold_draft(a: argparse.Namespace) -> None:
    from hnc_extract import gold, review_app

    views, docs, info = {}, None, {"runs": []}
    for rd in a.run_dir:
        model, d, v, _, specs = _views(rd, a.specs)
        if model.name in views:
            sys.exit(f"two runs of model {model.name!r}; pass one run per model")
        ids = [x.doc_id for x in d]
        if docs is not None and ids != [x.doc_id for x in docs]:
            sys.exit(f"{rd}: different documents from {a.run_dir[0]}")
        docs = d
        views[model.name] = v
        info["runs"].append(f"{model.name}: {rd}")
    doc_ids = [d.doc_id for d in docs]
    records = gold.draft(views, doc_ids)
    rows = gold.review_rows(records, seed=a.seed, rate=a.spot_check_rate)
    a.out.mkdir(parents=True, exist_ok=True)
    gold.write_jsonl(a.out / "draft.jsonl", records)
    meta = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "runs": {m: str(rd) for m, rd in zip(views, a.run_dir)},
        "specs": str(a.specs) if a.specs else "from each run.json",
        "doc_ids": doc_ids,
        "min_agree": gold.MIN_AGREE,
        "spot_check_rate": a.spot_check_rate,
        "seed": a.seed,
    }
    (a.out / "draft_meta.json").write_text(json.dumps(meta, indent=2))
    info = {
        **{k: v for k, v in meta.items() if k != "doc_ids"},
        "n_documents": len(doc_ids),
    }
    gold.write_adjudication(a.out / "adjudicate.xlsx", records, rows, docs, views, info)
    # round-trip through JSON so the bundle sees exactly what draft.jsonl holds
    review_app.write_bundle(
        a.out,
        review_app.build_bundle(
            meta,
            gold.read_jsonl(a.out / "draft.jsonl"),
            docs,
            views,
            gold.field_options(specs),
        ),
    )
    n_fields = sum(len(r["votes"]) for r in records)
    reasons = Counter(r["reason"] for r in rows)
    kinds = Counter(r["kind"] for r in records)
    print(
        f"{len(doc_ids)} docs, {kinds['specimen']} specimens, {kinds['test']} tests, "
        f"{n_fields} field decisions; to review: {dict(reasons)}"
    )
    print(f"-> {a.out / 'adjudicate.xlsx'}")
    print(f"review in the browser: python -m hnc_extract gold-review --dir {a.out}")


def _rebuild_bundle(directory: Path) -> dict:
    """Review bundle from the runs named in draft_meta.json (checks they still match)."""
    from hnc_extract import gold, review_app

    meta = json.loads((directory / "draft_meta.json").read_text())
    records = gold.read_jsonl(directory / "draft.jsonl")
    specs_dir = Path(meta["specs"]) if Path(meta["specs"]).is_dir() else None
    views, docs, specs = {}, None, None
    for name, rd in meta["runs"].items():
        model, docs, views[name], _, specs = _views(Path(rd), specs_dir)
        if model.name != name:
            sys.exit(f"{rd}: model {model.name!r}, draft_meta.json says {name!r}")
    if [d.doc_id for d in docs] != meta["doc_ids"]:
        sys.exit("the runs' documents differ from draft_meta.json")
    redraft = json.loads(json.dumps(gold.draft(views, meta["doc_ids"])))
    if redraft != records:
        sys.exit(
            "the runs no longer reproduce draft.jsonl (cache or specs changed); "
            "re-run gold-draft"
        )
    return review_app.build_bundle(
        meta, records, docs, views, gold.field_options(specs)
    )


def cmd_gold_review(a: argparse.Namespace) -> None:
    from hnc_extract import gold, review_app

    meta = json.loads((a.dir / "draft_meta.json").read_text())
    if a.rebuild or not (a.dir / review_app.BUNDLE).exists():
        print("building the review bundle from the runs ...", flush=True)
        review_app.write_bundle(a.dir, _rebuild_bundle(a.dir))
    bundle = review_app.read_bundle(a.dir)
    records = gold.read_jsonl(a.dir / "draft.jsonl")
    try:
        srv = review_app.make_server(a.dir, bundle, records, meta, a.host, a.port)
    except ValueError as e:
        sys.exit(f"{a.dir / review_app.STATE}: {e}")
    host = "localhost" if a.host in ("127.0.0.1", "::1") else a.host
    print(f"gold review: http://{host}:{srv.server_address[1]}/  (Ctrl-C to stop)")
    print(f"answers are saved to {a.dir / review_app.STATE} after every change")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


def _gold_options(directory: Path, meta: dict) -> dict | None:
    from hnc_extract import gold, review_app

    if (directory / review_app.BUNDLE).exists():
        return review_app.read_bundle(directory)["options"]
    if Path(meta.get("specs", "")).is_dir():
        return gold.field_options(load_specs(Path(meta["specs"]), "pathology_report"))
    return None


def cmd_gold_import(a: argparse.Namespace) -> None:
    from hnc_extract import gold, review_app

    records = gold.read_jsonl(a.dir / "draft.jsonl")
    meta = json.loads((a.dir / "draft_meta.json").read_text())
    known = {
        f"{r['doc_id']}|{r['key']}|{v['field']}": v["field"]
        for r in records
        for v in r["votes"]
    }
    source = a.source
    if source == "auto":
        source = "app" if (a.dir / review_app.STATE).exists() else "xlsx"
    options = _gold_options(a.dir, meta)
    if options is None:
        print("warning: no review bundle or spec dir; coded values are not checked")
        options = {}
    added: list[dict] = []
    if source == "app":
        try:
            state = review_app.read_state(a.dir, meta)
        except ValueError as e:
            sys.exit(f"{a.dir / review_app.STATE}: {e}")
        errors = review_app.validate_state(state, records, options)
        verdicts, added = review_app.state_verdicts(state)
        bundle = review_app.read_bundle(a.dir)
        reasons = {
            f"{r['doc_id']}|{r['key']}|{v['field']}": v["reason"]
            for r in bundle["records"]
            for v in r["votes"]
            if v["reason"]
        }
        prog = review_app.progress(records, reasons, state)
        print(
            f"source: {a.dir / review_app.STATE}  ({prog['done']}/{prog['required']} required rows reviewed)"
        )
        if prog["done"] < prog["required"]:
            print(
                f"warning: {prog['required'] - prog['done']} required rows not reviewed "
                "yet; they keep the proposed answer"
            )
    else:
        verdicts = gold.read_verdicts(a.dir / "adjudicate.xlsx")
        print(f"source: {a.dir / 'adjudicate.xlsx'}")
        unknown = sorted(set(verdicts) - set(known))
        if unknown:
            sys.exit(
                f"{len(unknown)} row_ids in the workbook are not in draft.jsonl, e.g. {unknown[:3]}"
            )
        errors = [
            f"{rid}: {msg}"
            for rid, text in verdicts.items()
            if (
                msg := gold.validate_value(
                    known[rid], gold.parse(known[rid], text), options
                )
            )
        ]
    if errors:
        print("\n".join(errors[:30]))
        sys.exit(f"{len(errors)} invalid verdict(s); fix them and re-run")
    g = gold.build_gold(records, verdicts, added)
    gold.write_jsonl(a.dir / "gold.jsonl", g.values())
    n_s = sum(len(d["specimens"]) for d in g.values())
    n_t = sum(len(d["tests"]) for d in g.values())
    print(f"{len(g)} docs, {n_s} specimens, {n_t} tests -> {a.dir / 'gold.jsonl'}")


def cmd_score(a: argparse.Namespace) -> None:
    from hnc_extract import gold

    g = {d["doc_id"]: d for d in gold.read_jsonl(a.gold)}
    model, docs, views, rollups, _ = _views(a.run_dir, a.specs)
    missing = set(g) - set(views)
    if missing:
        sys.exit(f"{len(missing)} gold docs are not in run {a.run_dir}")
    res = gold.score_run(g, views, rollups)
    print(f"{model.name} vs {a.gold} ({len(g)} docs)")
    for kind, c in res["items"].items():
        m, n, e = c["matched"], c["missed"], c["extra"]
        rec = m / (m + n) if m + n else float("nan")
        prec = m / (m + e) if m + e else float("nan")
        print(
            f"  {kind:10s} matched {m:4d}  missed {n:3d}  extra {e:3d}  recall {rec:.3f}  precision {prec:.3f}"
        )
    for section in ("fields", "report"):
        for f, c in res[section].items():
            tot = c["agree"] + c["disagree"]
            print(f"  {f:32s} {c['agree']:4d}/{tot:<4d} {c['agree'] / tot:.3f}")
    if a.out:
        a.out.write_text(json.dumps(res, indent=2, default=dict))
        print(f"-> {a.out}")


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
    ids = sp.add_mutually_exclusive_group()
    ids.add_argument("--doc-ids", help="comma-separated doc ids")
    ids.add_argument(
        "--doc-ids-file",
        type=Path,
        help="file with one doc id per line ('#' comments allowed), e.g. configs/eval/path-smoke.txt",
    )
    ids.add_argument(
        "--all-docs",
        action="store_true",
        help="clear a saved --limit/--doc-ids filter and run every document",
    )
    sp.add_argument(
        "--concurrency", type=int, default=16, help="parallel requests to the server"
    )
    sp.add_argument("--force", action="store_true")

    for name in ("status", "export"):
        sp = sub.add_parser(name)
        sp.add_argument("--run-dir", type=Path, required=True)
        if name == "export":
            sp.add_argument("--out", type=Path)

    sp = sub.add_parser("gold-draft")
    sp.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        help="repeat: one run dir per model",
    )
    sp.add_argument("--out", type=Path, required=True)
    sp.add_argument(
        "--seed", type=int, default=0, help="seed for the spot-check sample"
    )
    sp.add_argument(
        "--spot-check-rate",
        type=float,
        default=0.1,
        help="fraction of consensus fields sent for review anyway (shared-error check)",
    )
    sp.add_argument(
        "--specs",
        type=Path,
        help="spec dir the run(s) were made with (default: from run.json)",
    )

    sp = sub.add_parser("gold-review")
    sp.add_argument("--dir", type=Path, required=True, help="gold-draft output dir")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address; keep the default unless you know the network is private",
    )
    sp.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild review_bundle.json from the runs in draft_meta.json",
    )

    sp = sub.add_parser("gold-import")
    sp.add_argument("--dir", type=Path, required=True, help="gold-draft output dir")
    sp.add_argument(
        "--source",
        choices=["auto", "app", "xlsx"],
        default="auto",
        help="auto: verdicts.json if it exists, else adjudicate.xlsx",
    )

    sp = sub.add_parser("score")
    sp.add_argument("--run-dir", type=Path, required=True)
    sp.add_argument("--gold", type=Path, required=True)
    sp.add_argument(
        "--out", type=Path, help="write the full result (with mismatches) as JSON"
    )
    sp.add_argument(
        "--specs",
        type=Path,
        help="spec dir the run(s) were made with (default: from run.json)",
    )

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
            "gold-draft": cmd_gold_draft,
            "gold-review": cmd_gold_review,
            "gold-import": cmd_gold_import,
            "score": cmd_score,
        }[a.cmd](a)


if __name__ == "__main__":
    main()
