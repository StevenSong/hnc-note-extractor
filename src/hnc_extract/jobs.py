"""Cluster job commands: publish ingested documents, run one model end to end.

Both build on node-local disk (paths.RUN_ROOT/work/) and publish to shared storage
(paths.SHARED_ROOT) once, at the end (publish.py).

  ingest-publish <dataset>   DATA_ROOT/<dataset>/*.tsv -> SHARED_ROOT/ingested/<dataset>/
  job --model <name> ...     serve the model (serving.py) on the job's GPU, then
                             probe, run, status, export -> SHARED_ROOT/runs/<run-name>/

A published run dir is an ordinary run dir (run.json, cache.sqlite, export/) plus
logs/job.json (commit, node, GPU, weights sha256 + revisions, image digest, vLLM
version, steps) and logs/server.log. Its run.json points at the published documents,
so every reader of the shared storage can open it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

from hnc_extract import paths
from hnc_extract.hashing import sha256_file
from hnc_extract.publish import publish_dir

SIGTERM_EXIT = 128 + signal.SIGTERM


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _job_tag() -> str:
    return os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"


def _work_dir() -> Path:
    d = paths.RUN_ROOT / "work" / _job_tag()
    d.mkdir(parents=True, exist_ok=True)
    return d


def git_commit() -> dict:
    """The code version: the job's pinned commit, or the dev checkout's HEAD."""
    if sha := os.environ.get("HNC_COMMIT"):
        return {"sha": sha, "dirty": False}  # slurm/submit.sh refuses dirty trees
    git = ["git", "-C", str(paths.REPO_ROOT)]
    sha = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True)
    st = subprocess.run([*git, "status", "--porcelain"], capture_output=True, text=True)
    return {"sha": sha.stdout.strip(), "dirty": bool(st.stdout.strip())}


def shared_path(p: Path) -> Path:
    """Relative paths of shared inputs are relative to SHARED_ROOT."""
    return p if p.is_absolute() else paths.SHARED_ROOT / p


# ---- ingest-publish ----------------------------------------------------------------------


def cmd_ingest_publish(a: argparse.Namespace) -> None:
    from hnc_extract.synthetic_ingest import ingest_dataset

    src = paths.DATA_ROOT / a.dataset
    dest = paths.SHARED_ROOT / "ingested" / a.dataset
    if not src.is_dir():
        sys.exit(f"{src} not found (HNC_DATA_ROOT={paths.DATA_ROOT})")
    out = _work_dir() / "ingested" / a.dataset
    shutil.rmtree(out, ignore_errors=True)
    m = ingest_dataset(src, out)
    inputs = {t: s["input_sha256"] for t, s in m["files"].items()}
    old = dest / "manifest.json"
    if old.is_file() and not a.overwrite:
        prev = json.loads(old.read_text())
        if {t: s["input_sha256"] for t, s in prev["files"].items()} == inputs:
            print(f"{dest} already holds these inputs; nothing to do")
            shutil.rmtree(out)
            return
        sys.exit(f"{dest} holds other inputs; runs point at it. Pass --overwrite.")
    # Recorded paths name the published copy, not the job's work dir.
    for t, s in m["files"].items():
        s["output_file"] = str(dest / Path(s["output_file"]).name)
    m["commit"] = git_commit()
    (out / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    publish_dir(out, dest, overwrite=a.overwrite)
    shutil.rmtree(out)
    for t, s in m["files"].items():
        print(f"{t:18s} docs={s['documents']:5d} patients={s['patients']:3d}")
    print(f"-> {dest}")


# ---- job ---------------------------------------------------------------------------------


class Steps:
    """Run CLI steps in order; record exit code and seconds of each. A failed step is
    recorded and the next steps still run; SIGTERM (scancel) ends the job."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    @property
    def failed(self) -> bool:
        return any(r["exit"] != 0 for r in self.rows)

    def __call__(self, name: str, argv: list[str]) -> int:
        from hnc_extract.cli import main

        print(f"[{_now()}] == {name}: hnc_extract {' '.join(argv)}", flush=True)
        t0 = time.monotonic()
        try:
            main(argv)
            rc = 0
        except SystemExit as e:
            if e.code == SIGTERM_EXIT:
                raise
            rc = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
            if e.code not in (None, 0) and not isinstance(e.code, int):
                print(e.code, file=sys.stderr)
        except Exception:  # noqa: BLE001  (recorded; the job goes on to publish)
            traceback.print_exc()
            rc = 1
        secs = round(time.monotonic() - t0)
        self.rows.append({"step": name, "exit": rc, "seconds": secs})
        print(f"[{_now()}] {name}: exit {rc} after {secs} s", flush=True)
        return rc


def cmd_job(a: argparse.Namespace) -> None:
    from hnc_extract import serving, weights
    from hnc_extract.llm import ModelConfig
    from hnc_extract.store import Store

    model_cfg = paths.CONFIGS_DIR / "models" / f"{a.model}.yaml"
    if not model_cfg.is_file():
        sys.exit(f"no model config {model_cfg}")
    serve = serving.load_config("serving").get(a.model)
    if serve is None:
        sys.exit(f"no entry {a.model!r} in configs/serving.yaml")
    model = ModelConfig.load(model_cfg)
    docs = shared_path(a.docs)
    if not docs.is_file():
        sys.exit(f"{docs} not found; publish it first (hnc_extract ingest-publish)")
    name = a.run_name.format(model=a.model, job=_job_tag())
    dest = paths.SHARED_ROOT / "runs" / name
    if dest.exists() and not a.resume:
        sys.exit(f"{dest} exists; pick another --run-name, or --resume to continue it")
    if a.resume and not dest.exists():
        sys.exit(f"--resume: {dest} does not exist")

    # Fail before the server starts: pinned image on this node, weights sha256.
    image = serving.image_record(serve["container"])
    model_dir, weights_rec = weights.resolve(serve["weights"])

    run_dir = _work_dir() / "runs" / name
    shutil.rmtree(run_dir, ignore_errors=True)
    if a.resume:
        shutil.copytree(dest, run_dir)  # the cache makes the rerun skip finished calls
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    meta: dict = {
        "started_utc": _now(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "commit": git_commit(),
        "model": a.model,
        "model_config_sha256": sha256_file(model_cfg),
        "serve": serve,
        "image": image,
        "weights": weights_rec,
        "docs": {"path": str(docs), "sha256": sha256_file(docs)},
        "doc_ids_file": (
            {"path": str(a.doc_ids_file), "sha256": sha256_file(a.doc_ids_file)}
            if a.doc_ids_file
            else None
        ),
        "concurrency": a.concurrency,
        "resumed_from": str(dest) if a.resume else None,
    }
    if a.resume and (prev := logs / "job.json").is_file():
        meta["previous_jobs"] = [json.loads(prev.read_text())]
    select = (
        ["--limit", str(a.limit)]
        if a.limit
        else ["--doc-ids-file", str(a.doc_ids_file)] if a.doc_ids_file else []
    )
    steps = Steps()
    try:
        with serving.vllm_server(
            model.served_model_name,
            model_dir,
            image,
            serve["serve_args"],
            logs / "server.log",
        ) as srv:
            meta["server"] = srv
            url = ["--base-url", srv["base_url"]]
            if not a.skip_probe:
                steps("probe", ["probe", "--model", str(model_cfg), *url])
            run_argv = ["run", "--run-dir", str(run_dir), *url]
            if not a.resume:
                run_argv += ["--docs", str(docs), "--model", str(model_cfg)]
            if a.fields:
                run_argv += ["--fields", a.fields]
            steps("run", run_argv + select + ["--concurrency", str(a.concurrency)])
        steps("status", ["status", "--run-dir", str(run_dir)])
        steps("export", ["export", "--run-dir", str(run_dir)])
    finally:
        meta["steps"] = steps.rows
        meta["finished_utc"] = _now()
        meta["ok"] = bool(steps.rows) and not steps.failed
        cache = run_dir / "cache.sqlite"
        if cache.exists():
            st = Store(cache)
            st.finalize()
            st.close()
        (logs / "job.json").write_text(json.dumps(meta, indent=2) + "\n")
        # Published even when a step failed: the cache lets --resume continue it.
        publish_dir(run_dir, dest, overwrite=a.resume)
        shutil.rmtree(run_dir)
        print(f"-> {dest} (ok={meta['ok']})", flush=True)
    for r in steps.rows:
        print(f"  {r['step']:8s} exit {r['exit']}  {r['seconds']} s")
    if steps.failed:
        sys.exit(1)


def cmd_weights_pin(a: argparse.Namespace) -> None:
    from hnc_extract import weights

    print(weights.pin(a.name))


def add_parsers(sub: argparse._SubParsersAction) -> dict:
    sp = sub.add_parser(
        "ingest-publish",
        help="ingest a synthetic dataset from HNC_DATA_ROOT and publish it to shared storage",
    )
    sp.add_argument("dataset", help="dir name under HNC_DATA_ROOT, e.g. dev-test50")
    sp.add_argument("--overwrite", action="store_true")

    sp = sub.add_parser(
        "job", help="serve one model and run it end to end (a Slurm job step)"
    )
    sp.add_argument(
        "--model",
        required=True,
        help="name in configs/models/ and configs/serving.yaml",
    )
    sp.add_argument(
        "--docs",
        type=Path,
        required=True,
        help="document JSONL; relative paths are under HNC_SHARED_ROOT "
        "(e.g. ingested/dev-test50/pathology_report.jsonl)",
    )
    sp.add_argument(
        "--run-name",
        default="{model}-j{job}",
        help="published as HNC_SHARED_ROOT/runs/<run-name>; {model} and {job} are filled in",
    )
    sp.add_argument(
        "--resume",
        action="store_true",
        help="continue the published run of that name (cached calls are reused) and replace it",
    )
    ids = sp.add_mutually_exclusive_group()
    ids.add_argument("--doc-ids-file", type=Path)
    ids.add_argument("--limit", type=int)
    sp.add_argument("--fields")
    sp.add_argument("--concurrency", type=int, default=16)
    sp.add_argument("--skip-probe", action="store_true")

    sp = sub.add_parser(
        "weights-pin",
        help="print the files: block (sha256 of every file) for configs/weights.yaml",
    )
    sp.add_argument("name")
    return {
        "ingest-publish": cmd_ingest_publish,
        "job": cmd_job,
        "weights-pin": cmd_weights_pin,
    }
