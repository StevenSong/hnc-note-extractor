#!/usr/bin/env bash
# Submit one smoke job per model (default: all four), each on its own GPU.
#   slurm/submit_smoke.sh                                 # all four
#   slurm/submit_smoke.sh gemma-4-31b muse-glimmer-30b    # a subset
#   CONCURRENCY=64 slurm/submit_smoke.sh qwen3.8-27b      # env vars pass through to the job
# Extra sbatch flags (e.g. partition) go in SBATCH_ARGS: SBATCH_ARGS="-p gpu" slurm/submit_smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."

models=("$@")
((${#models[@]})) || models=(qwen3.8-27b muse-glimmer-30b gemma-4-31b nemotron-3.5-super-nvfp4)

mkdir -p runs/smoke   # the --output dir must exist before sbatch
# Build the runner image once, so concurrent jobs find it cached.
docker compose build runner

for m in "${models[@]}"; do
    [[ -f "configs/models/$m.yaml" ]] || { echo "unknown model $m" >&2; exit 2; }
    # shellcheck disable=SC2086
    sbatch ${SBATCH_ARGS:-} -J "smoke-$m" slurm/smoke.sbatch "$m"
done
echo "logs: runs/smoke/slurm-smoke-<model>-<jobid>.out; outputs: runs/smoke/<model>-j<jobid>/"
