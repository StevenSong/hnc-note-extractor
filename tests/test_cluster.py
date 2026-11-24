"""Cluster plumbing: atomic publish, weight pins, read-only stores, the job command."""

import contextlib
import json
import os
import shutil
import stat
from pathlib import Path

import pytest
import yaml
from conftest import make_doc

from hnc_extract import jobs, paths, publish, serving, weights
from hnc_extract.cli import main
from hnc_extract.documents import write_jsonl
from hnc_extract.hashing import sha256_file
from hnc_extract.store import Store

REPO = Path(__file__).resolve().parents[1]


def test_publish_dir_refuses_existing_unless_overwrite(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("1")
    dest = tmp_path / "shared" / "runs" / "x"
    publish.publish_dir(src, dest)
    assert (dest / "a.txt").read_text() == "1"
    with pytest.raises(SystemExit):
        publish.publish_dir(src, dest)
    (src / "a.txt").write_text("2")
    publish.publish_dir(src, dest, overwrite=True)
    assert (dest / "a.txt").read_text() == "2"
    # no temporary or set-aside copies are left next to it
    assert sorted(p.name for p in dest.parent.iterdir()) == ["x"]


def _weights_cfg(tmp_path, monkeypatch, files=None):
    cfg = tmp_path / "configs"
    cfg.mkdir(exist_ok=True)
    entry = {"repo": "org/m"}
    if files is not None:
        entry["files"] = files
    (cfg / "weights.yaml").write_text(yaml.safe_dump({"m": entry}))
    monkeypatch.setattr(paths, "CONFIGS_DIR", cfg)
    model_dir = tmp_path / "models" / "org" / "m"
    (model_dir / ".cache" / "huggingface" / "download").mkdir(
        parents=True, exist_ok=True
    )
    (model_dir / "config.json").write_text("{}")
    (model_dir / ".cache/huggingface/download/config.json.metadata").write_text(
        "abc123\netag\n"
    )
    return model_dir


def test_weights_resolve_checks_pins(tmp_path, monkeypatch):
    model_dir = _weights_cfg(tmp_path, monkeypatch)
    root = tmp_path / "models"
    with pytest.raises(SystemExit, match="no files pinned"):
        weights.resolve("m", root)
    pinned = weights.pin("m", root)
    assert "config.json: " + sha256_file(model_dir / "config.json") in pinned
    assert ".metadata" not in pinned and "abc123" in pinned

    _weights_cfg(
        tmp_path, monkeypatch, {"config.json": sha256_file(model_dir / "config.json")}
    )
    d, rec = weights.resolve("m", root)
    assert d == model_dir and rec["files"]["config.json"]["revision"] == "abc123"

    _weights_cfg(tmp_path, monkeypatch, {"config.json": "0" * 64})
    with pytest.raises(SystemExit, match="sha256"):
        weights.resolve("m", root)


def test_store_reads_published_copy_in_readonly_dir(tmp_path):
    from hnc_extract.extract import CallRecord

    src = tmp_path / "run"
    st = Store(src / "cache.sqlite")
    rec = CallRecord(
        key="k", doc_sha256="d", field="f", spec_version=1, spec_fingerprint="s",
        model_name="m", model_fingerprint="mf", status="ok", output=None,
        evidence=None, attempts=[], created_utc="t",
    )  # fmt: skip
    st.put(rec)
    st.finalize()
    st.close()
    dest = tmp_path / "pub"
    shutil.copytree(src, dest)
    os.chmod(dest, stat.S_IRUSR | stat.S_IXUSR)
    try:
        ro = Store.open_for_reading(dest / "cache.sqlite")
        assert ro.get("k").status == "ok"
        ro.close()
        assert sorted(p.name for p in dest.iterdir()) == ["cache.sqlite"]
    finally:
        os.chmod(dest, stat.S_IRWXU)


def test_docker_command_uses_job_gpus_and_offline_weights(tmp_path):
    cmd = serving.docker_command(
        "hnc-1-5", ["2"], 5, tmp_path / "m", tmp_path / "c", "img@sha", "q", ["--x"]
    )
    s = " ".join(cmd)
    assert '"device=2"' in s and "--tensor-parallel-size 1" in s
    assert f"{tmp_path / 'm'}:{tmp_path / 'm'}:ro" in s and "HF_HUB_OFFLINE=1" in s
    assert cmd[-1] == "--x" and f"--user {os.getuid()}:{os.getgid()}" in s


def test_every_served_model_has_config_weights_and_container():
    serve = yaml.safe_load((REPO / "configs/serving.yaml").read_text())
    w = yaml.safe_load((REPO / "configs/weights.yaml").read_text())
    c = yaml.safe_load((REPO / "configs/containers.yaml").read_text())
    for name, s in serve.items():
        assert (REPO / "configs/models" / f"{name}.yaml").is_file()
        assert s["weights"] in w and s["container"] in c
        # serving.py sets these itself
        assert not any(
            a.split("=")[0]
            in ("--tensor-parallel-size", "--served-model-name", "--port")
            for a in s["serve_args"]
        )


def test_job_end_to_end_with_mock_server(tmp_path, monkeypatch):
    """A job with the mock model: publishes a run that opens read-only and resumes."""
    cfg = tmp_path / "configs"
    (cfg / "models").mkdir(parents=True)
    shutil.copy(REPO / "configs/models/mock.yaml", cfg / "models/mock.yaml")
    (cfg / "serving.yaml").write_text(
        yaml.safe_dump(
            {"mock": {"weights": "w", "container": "vllm", "serve_args": []}}
        )
    )
    shared, run_root = tmp_path / "shared", tmp_path / "run-root"
    monkeypatch.setattr(paths, "CONFIGS_DIR", cfg)
    monkeypatch.setattr(paths, "SHARED_ROOT", shared)
    monkeypatch.setattr(paths, "RUN_ROOT", run_root)
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    monkeypatch.chdir(REPO)  # the run's spec dir is the relative "specs"
    monkeypatch.setattr(serving, "image_record", lambda n: {"ref": "img@sha"})
    monkeypatch.setattr(weights, "resolve", lambda n: (tmp_path, {"repo": "w"}))

    @contextlib.contextmanager
    def fake_server(*a, **k):
        yield {"base_url": "http://127.0.0.1:1/v1"}

    monkeypatch.setattr(serving, "vllm_server", fake_server)
    docs = shared / "ingested/x/pathology_report.jsonl"
    write_jsonl([make_doc("d1"), make_doc("d2", patient="p2")], docs)

    argv = ["job", "--model", "mock", "--docs", "ingested/x/pathology_report.jsonl"]
    main(argv + ["--skip-probe", "--limit", "1", "--run-name", "smoke/{model}-j{job}"])
    pub = shared / "runs/smoke/mock-j77"
    meta = json.loads((pub / "logs/job.json").read_text())
    assert meta["ok"] and [s["step"] for s in meta["steps"]] == [
        "run",
        "status",
        "export",
    ]
    assert json.loads((pub / "run.json").read_text())["docs"] == str(docs)
    assert (pub / "export/review.xlsx").is_file()
    assert not (run_root / "work/77/runs/smoke/mock-j77").exists()
    with pytest.raises(SystemExit, match="exists"):
        main(argv + ["--skip-probe", "--run-name", "smoke/mock-j77"])

    os.chmod(pub, stat.S_IRUSR | stat.S_IXUSR)
    try:
        main(["status", "--run-dir", str(pub)])
    finally:
        os.chmod(pub, stat.S_IRWXU)

    # resume: all docs this time; the published copy is replaced
    main(argv + ["--skip-probe", "--run-name", "smoke/mock-j77", "--resume"])
    meta = json.loads((pub / "logs/job.json").read_text())
    assert meta["ok"] and meta["previous_jobs"][0]["steps"]
    rollup = (pub / "export/report_rollup.jsonl").read_text().splitlines()
    assert len(rollup) == 1  # the saved --limit 1 filter is kept on resume


def test_ingest_publish_is_idempotent(tmp_path, monkeypatch):
    data = tmp_path / "data" / "ds"
    data.mkdir(parents=True)
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(paths, "SHARED_ROOT", tmp_path / "shared")
    monkeypatch.setattr(paths, "RUN_ROOT", tmp_path / "rr")
    src = REPO / "data" / "dev-test" / "pathology_report.tsv"
    if not src.is_file():
        pytest.skip("no synthetic data in this checkout")
    shutil.copy(src, data / "pathology_report.tsv")
    main(["ingest-publish", "ds"])
    dest = tmp_path / "shared/ingested/ds"
    m = json.loads((dest / "manifest.json").read_text())
    assert m["files"]["pathology_report"]["output_file"] == str(
        dest / "pathology_report.jsonl"
    )
    main(["ingest-publish", "ds"])  # same inputs: nothing to do
    other = REPO / "data" / "dev-test50" / "pathology_report.tsv"
    if not other.is_file():
        return
    shutil.copy(other, data / "pathology_report.tsv")
    with pytest.raises(SystemExit, match="other inputs"):
        main(["ingest-publish", "ds"])
