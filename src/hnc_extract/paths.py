"""Filesystem roots. Override per host with environment variables."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Committed configs (models, serving, weights, containers, eval sets).
CONFIGS_DIR = REPO_ROOT / "configs"

# Read-only dataset releases, one dir per dataset (e.g. dev-test50/ with the synthetic
# TSVs). Only source files are read; locally derived files next to them are never used.
DATA_ROOT = Path(os.environ.get("HNC_DATA_ROOT", "/opt/gpudata/hnc"))

# Model weights, pre-pulled on every node (configs/weights.yaml). Jobs never download.
MODELS_ROOT = Path(os.environ.get("HNC_MODELS_ROOT", "/opt/gpudata/models"))

# Shared NFS storage visible from every node: ingested documents, published runs and
# finished job logs. Jobs build on local disk and publish here once (publish.py).
SHARED_ROOT = Path(
    os.environ.get("HNC_SHARED_ROOT", "/shared/gpudata/songs1/hnc-shared")
)

# Node-local run root of slurm/job.sh (commit clones, uv cache, logs, job work dirs).
# Long-lived node-local caches (vLLM's compile cache) live here too, never on NFS.
RUN_ROOT = Path(os.environ.get("HNC_RUN_ROOT", Path.home() / "hnc-run"))
