#!/usr/bin/env bash
# Submit one smoke job per model (default: all four), each on one GPU, at HEAD.
#
#   slurm/smoke.sh                                  # all four
#   slurm/smoke.sh gemma-4-31b qwen3.8-27b          # a subset
#   DOC_IDS_FILE=configs/eval/tcga-dev.txt slurm/smoke.sh qwen3.8-27b
#   DOCS=ingested/dev-test50/pathology_report.jsonl DOC_IDS_FILE=configs/eval/path-smoke.txt slurm/smoke.sh
#
# Default: the 18 TCGA-HNSC smoke reports. Each job publishes
# HNC_SHARED_ROOT/runs/smoke/<model>-j<jobid>/. The documents must be published first:
# slurm/submit.sh --cpus-per-task=4 -- hnc_extract ingest-publish tcga-hnsc
set -euo pipefail
cd "$(dirname "$0")/.."

models=("$@")
((${#models[@]})) || models=(qwen3.8-27b muse-glimmer-30b gemma-4-31b nemotron-3.5-super-nvfp4)
docs=${DOCS:-ingested/tcga-hnsc/pathology_report.jsonl}
ids=${DOC_IDS_FILE:-configs/eval/tcga-smoke.txt}

for m in "${models[@]}"; do
  [[ -f configs/models/$m.yaml ]] || { echo "unknown model $m" >&2; exit 2; }
done
for m in "${models[@]}"; do
  slurm/submit.sh -J "hnc-smoke-$m" --gres=gpu:1 -- hnc_extract job --model "$m" \
    --docs "$docs" --doc-ids-file "$ids" --run-name "smoke/{model}-j{job}" \
    --concurrency "${CONCURRENCY:-16}"
done
