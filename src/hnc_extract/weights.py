"""Locate and verify pre-pulled model weights (configs/weights.yaml)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from hnc_extract import paths
from hnc_extract.hashing import sha256_file

HASH_WORKERS = min(16, int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))


def load_config() -> dict:
    return yaml.safe_load((paths.CONFIGS_DIR / "weights.yaml").read_text())


def revision(model_dir: Path, filename: str) -> str | None:
    """The commit `hf download --local-dir` recorded for a file (None if not recorded)."""
    meta = model_dir / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
    return meta.read_text().splitlines()[0].strip() if meta.is_file() else None


def _hash_all(model_dir: Path, files: list[str]) -> dict[str, str]:
    # hashlib releases the GIL, so threads hash files in parallel.
    with ThreadPoolExecutor(max_workers=HASH_WORKERS) as pool:
        return dict(zip(files, pool.map(sha256_file, [model_dir / f for f in files])))


def resolve(name: str, models_root: Path | None = None) -> tuple[Path, dict]:
    """Check one entry of configs/weights.yaml and return (model dir, provenance record).

    Fails if the entry pins no files, or any pinned file is missing or has another
    sha256. The record (repo, dir, per file sha256 and revision) goes into run metadata.
    """
    entry = load_config()[name]
    model_dir = (models_root or paths.MODELS_ROOT) / entry["repo"]
    pinned = entry.get("files") or {}
    if not pinned:
        raise SystemExit(
            f"weights {name}: no files pinned in configs/weights.yaml; on a node run\n"
            f"  hnc_extract weights-pin {name}\nand commit its output"
        )
    for filename in pinned:
        if not (model_dir / filename).is_file():
            raise SystemExit(
                f"weights {name}: missing {model_dir / filename} "
                "(pre-pull it; jobs never download)"
            )
    hashes = _hash_all(model_dir, list(pinned))
    files = {}
    for filename, expected in pinned.items():
        if hashes[filename] != expected:
            raise SystemExit(
                f"weights {name}: sha256 of {model_dir / filename} is "
                f"{hashes[filename]}, expected {expected}"
            )
        files[filename] = {
            "sha256": expected,
            "revision": revision(model_dir, filename),
        }
    return model_dir, {"repo": entry["repo"], "dir": str(model_dir), "files": files}


def pin(name: str, models_root: Path | None = None) -> str:
    """YAML `files:` block for an entry: every file of its pre-pulled dir except the
    download metadata (vLLM may read any of them). Printed for review, never written."""
    entry = load_config()[name]
    model_dir = (models_root or paths.MODELS_ROOT) / entry["repo"]
    if not model_dir.is_dir():
        raise SystemExit(f"weights {name}: {model_dir} not found (pre-pull it first)")
    files = sorted(
        str(p.relative_to(model_dir))
        for p in model_dir.rglob("*")
        if p.is_file() and ".cache" not in p.relative_to(model_dir).parts
    )
    hashes = _hash_all(model_dir, files)
    revs = {revision(model_dir, f) for f in files} - {None}
    lines = [f"# {name}: {entry['repo']}, revision(s) {', '.join(sorted(revs)) or '?'}"]
    lines.append("  files:")
    lines += [f"    {f}: {hashes[f]}" for f in files]
    return "\n".join(lines)
