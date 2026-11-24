"""`vllm serve` in Docker for the length of one job.

``vllm_server`` runs the pinned image (configs/containers.yaml) on the job's GPUs
(CUDA_VISIBLE_DEVICES), tensor-parallel across them, as the job's user with HOME on a
node-local cache dir (compile caches survive jobs), weights read-only, offline, on a
free localhost port, its log streamed to a file. Always stopped on exit.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import yaml

from hnc_extract import paths

STARTUP_TIMEOUT_S = 3600  # first start on a node compiles kernels
STOP_TIMEOUT_S = 30


def load_config(kind: str) -> dict:
    """One of configs/{serving,containers}.yaml."""
    return yaml.safe_load((paths.CONFIGS_DIR / f"{kind}.yaml").read_text())


def image_id(ref: str) -> str | None:
    """The local image id of ``ref``, None if this node doesn't have it. Any other
    docker failure (e.g. no access to the daemon socket) exits with docker's error."""
    if shutil.which("docker") is None:
        raise SystemExit(f"docker not found on {socket.gethostname()}")
    out = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", ref],
        capture_output=True,
        text=True,
    )
    if out.returncode == 0:
        return out.stdout.strip()
    # Only this error means a missing image; others aren't fixed by pulling.
    if "No such image" in out.stderr:
        return None
    raise SystemExit(
        f"docker image inspect {ref} failed on {socket.gethostname()}:\n"
        f"{out.stderr.strip()}"
    )


def image_record(name: str) -> dict:
    """The pinned image, after checking this node has it (jobs never pull)."""
    entry = load_config("containers")[name]
    ref = f"{entry['image']}@{entry['digest']}"
    pulled = image_id(ref)
    if pulled is None:
        raise SystemExit(
            f"container {name}: {ref} not on {socket.gethostname()}; pre-pull it with\n"
            f"  docker pull {entry['image']}:{entry['tag']}@{entry['digest']}"
        )
    return {**entry, "ref": ref, "image_id": pulled}


def gpu_devices() -> list[str]:
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not devices:
        raise SystemExit(
            "CUDA_VISIBLE_DEVICES is not set: set it to the host GPU indices to use "
            "(a Slurm job with --gres sets it)"
        )
    return devices.split(",")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read() or b"{}")
    except OSError:
        return None


def gpu_names(devices: list[str]) -> list[str | None] | None:
    """Host GPU model of each device; None without nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        return None
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    rows = [line.split(", ") for line in out.stdout.strip().splitlines()]
    by_id = {r[0]: r[1] for r in rows} | {r[2]: r[1] for r in rows}
    return [by_id.get(d) for d in devices]


def _running(name: str) -> bool:
    out = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", name],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip() == "true"


def raise_on_sigterm() -> None:
    """Turn Slurm's SIGTERM (scancel) into SystemExit, so cleanup runs."""

    def handler(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handler)


def docker_command(
    name: str,
    devices: list[str],
    port: int,
    model_dir: Path,
    cache: Path,
    image_ref: str,
    served_name: str,
    serve_args: list[str],
) -> list[str]:
    return [
        "docker", "run", "--rm", "--detach", "--name", name,
        "--gpus", f'"device={",".join(devices)}"',
        "--ipc=host", "--network=host",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/cache", "-e", "USER=hnc",
        "-e", "HF_HUB_OFFLINE=1", "-e", "VLLM_NO_USAGE_STATS=1", "-e", "DO_NOT_TRACK=1",
        # Inductor's autotuning otherwise picks kernels by timing noise per compile.
        "-e", "TORCHINDUCTOR_DETERMINISTIC=1",
        "-v", f"{model_dir}:{model_dir}:ro",
        "-v", f"{cache}:/cache",
        image_ref,
        str(model_dir),
        "--served-model-name", served_name,
        "--host", "127.0.0.1", "--port", str(port),
        "--tensor-parallel-size", str(len(devices)),
        *serve_args,
    ]  # fmt: skip


@contextlib.contextmanager
def vllm_server(
    served_name: str,
    model_dir: Path,
    image: dict,
    serve_args: list[str],
    log_path: Path,
) -> Iterator[dict]:
    """Yield the server's URL, vLLM version, devices and GPU models while it runs."""
    raise_on_sigterm()
    devices = gpu_devices()
    port = _free_port()
    name = f"hnc-{os.environ.get('SLURM_JOB_ID', 'local')}-{port}"
    cache = paths.RUN_ROOT / "vllm-cache"
    cache.mkdir(parents=True, exist_ok=True)
    cmd = docker_command(
        name, devices, port, model_dir, cache, image["ref"], served_name, serve_args
    )
    print("starting:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w")
    logs = subprocess.Popen(
        ["docker", "logs", "--follow", "--timestamps", name],
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        start = time.monotonic()
        while _get_json(f"{base}/health") is None:
            if not _running(name):
                raise SystemExit(
                    f"vLLM container {name} exited during startup; see {log_path}"
                )
            if time.monotonic() - start > STARTUP_TIMEOUT_S:
                raise SystemExit(f"vLLM not healthy after {STARTUP_TIMEOUT_S} s")
            time.sleep(5)
        version = (_get_json(f"{base}/version") or {}).get("version")
        print(
            f"vLLM {version} ready after {time.monotonic() - start:.0f} s", flush=True
        )
        yield {
            "base_url": f"{base}/v1",
            "version": version,
            "container": name,
            "command": cmd,
            "devices": devices,
            "gpus": gpu_names(devices),
            "startup_s": round(time.monotonic() - start),
        }
    finally:
        subprocess.run(
            ["docker", "stop", "--time", str(STOP_TIMEOUT_S), name],
            capture_output=True,
        )
        logs.terminate()
        log_f.close()
        print(f"stopped {name}", flush=True)
