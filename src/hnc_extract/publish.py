"""Publish a finished directory to shared storage (paths.SHARED_ROOT) in one step.

Jobs build on node-local disk. The directory is copied next to its destination under
a hidden temporary name, then renamed into place, so readers never see a partial copy.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def publish_dir(src: Path, dest: Path, overwrite: bool = False) -> Path:
    """Copy ``src`` to ``dest`` atomically. An existing ``dest`` is refused unless
    ``overwrite``; then it is renamed aside first and removed after the swap."""
    if dest.exists() and not overwrite:
        raise SystemExit(f"{dest} already exists; pick another name")
    tag = os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.name}.tmp-{tag}"
    old = dest.parent / f".{dest.name}.old-{tag}"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    if dest.exists():
        dest.rename(old)
    tmp.rename(dest)
    shutil.rmtree(old, ignore_errors=True)
    return dest
