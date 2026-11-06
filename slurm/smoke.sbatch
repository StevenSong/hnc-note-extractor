#!/usr/bin/env bash
# Smoke test for one model on one SLURM-reserved GPU: serve with vLLM (docker compose),
# then probe, a 10-report run, status and export. Every output goes to files for review.
#
#   mkdir -p runs/smoke && sbatch -J smoke-qwen3.8-27b slurm/smoke.sbatch qwen3.8-27b
#   slurm/submit_smoke.sh            # all four models (creates runs/smoke itself)
#
# SLURM only reserves the GPU here. The containers are started by the docker daemon,
# outside the job's cgroup, so the script reads the reserved GPU's host index and passes
# it to compose. Each job uses its own compose project (network, alias `llm`, containers)
# and its own host port, so several jobs can run on the node at the same time.
#
# Outputs: runs/smoke/<model>-j<jobid>/
#   run.json, cache.sqlite, export/        the run (same layout as any run dir)
#   logs/meta.txt                          host, GPU, git commit, image digest, vLLM version, HF revision
#   logs/git.diff                          uncommitted changes at submit time
#   logs/server.log                        vLLM server log, from container start
#   logs/{probe,run,status,export}.log     runner output per step
#   logs/steps.tsv                         step, exit code, seconds
#
# Env overrides: CONCURRENCY (default 16, same as the first Qwen smoke), LIMIT (default 10),
# DOCS, HF_TOKEN, HF_CACHE, VLLM_IMAGE. Gemma and possibly Muse are gated on Hugging Face:
# HF_TOKEN must be set (shell or .env) for the first download.
#
#SBATCH --job-name=hnc-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=runs/smoke/slurm-%x-%j.out
#SBATCH --signal=B:TERM@120

set -Eeuo pipefail

M="${1:?usage: sbatch slurm/smoke.sbatch <model profile, e.g. qwen3.8-27b>}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

JOB="${SLURM_JOB_ID:-local$$}"
CONCURRENCY="${CONCURRENCY:-16}"
LIMIT="${LIMIT:-10}"
DOCS="${DOCS:-data/ingested/dev-test50/pathology_report.jsonl}"
CFG="configs/models/$M.yaml"
RUN_DIR="runs/smoke/$M-j$JOB"
LOG="$RUN_DIR/logs"

ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*"; }

# ---- preflight -------------------------------------------------------------------------
[[ -f "$CFG" ]] || { say "no model config $CFG"; exit 2; }
[[ -f "$DOCS" ]] || { say "no docs at $DOCS; run ingest-synthetic first (docs/RUNNING.md step 0)"; exit 2; }
grep -qE "^  $M:\$" docker-compose.yaml || { say "no compose service '$M'"; exit 2; }

# Host GPU index. SLURM_JOB_GPUS holds the host indices. CUDA_VISIBLE_DEVICES is the
# fallback, and it is the host index too while SLURM does not constrain devices.
GPU="${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}"
[[ -n "$GPU" ]] || { say "no GPU reserved (request --gres=gpu:1)"; exit 2; }
[[ "$GPU" != *,* ]] || { say "expected one GPU, got '$GPU'"; exit 2; }

mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES="$GPU"          # read by compose for device_ids
export COMPOSE_PROJECT_NAME="hnc-smoke-$JOB"
export VLLM_PORT=$((18000 + GPU))           # unique host port per GPU
DC=(docker compose --profile "$M")

say "job $JOB: model $M on host GPU $GPU, project $COMPOSE_PROJECT_NAME, port $VLLM_PORT"
say "outputs in $RUN_DIR"

# ---- cleanup (normal exit, scancel, time limit) ------------------------------------------
LOGS_PID=""
STEP_PID=""
cleanup() {
    local rc=$?
    trap - EXIT TERM INT
    [[ -n "$STEP_PID" ]] && kill "$STEP_PID" 2>/dev/null || true
    say "stopping containers (project $COMPOSE_PROJECT_NAME)"
    docker compose --profile "$M" --profile runner down --remove-orphans --timeout 20 \
        >>"$LOG/teardown.log" 2>&1 || say "compose down failed; see $LOG/teardown.log"
    [[ -n "$LOGS_PID" ]] && kill "$LOGS_PID" 2>/dev/null || true
    say "exit $rc"
    exit "$rc"
}
trap cleanup EXIT
trap 'say "signal received"; exit 143' TERM INT

# Run a step in the background and wait for it, so signals interrupt the wait and cleanup
# runs at once. Output goes to logs/<name>.log and to the SLURM log. A failed step is
# recorded and the job continues. The job's exit code is nonzero if any step failed.
FAILED=0
LAST_RC=0
printf 'step\texit\tseconds\n' >"$LOG/steps.tsv"
step() {
    local name=$1; shift
    local t0=$SECONDS rc=0
    say "== $name: $*"
    ( "$@" 2>&1 | tee "$LOG/$name.log" ) &
    STEP_PID=$!
    wait "$STEP_PID" || rc=$?
    STEP_PID=""
    LAST_RC=$rc
    printf '%s\t%s\t%s\n' "$name" "$rc" "$((SECONDS - t0))" >>"$LOG/steps.tsv"
    if ((rc != 0)); then FAILED=1; say "!! $name failed (exit $rc)"; fi
    return 0
}

# ---- provenance --------------------------------------------------------------------------
{
    echo "date:        $(date -Is)"
    echo "host:        $(hostname)"
    echo "slurm job:   $JOB  host GPU=$GPU (SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset})  port=$VLLM_PORT"
    echo "model:       $M  config=$CFG"
    echo "docs:        $DOCS  limit=$LIMIT  concurrency=$CONCURRENCY"
    echo "git commit:  $(git rev-parse HEAD 2>/dev/null || echo n/a)  branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
    echo "git status:"; git status --short 2>/dev/null | sed 's/^/    /'
    echo "docs sha256: $(sha256sum "$DOCS" | cut -d' ' -f1)"
    echo "--- nvidia-smi (GPU $GPU) at start"
    nvidia-smi -i "$GPU" --query-gpu=index,name,uuid,memory.used,memory.total,driver_version --format=csv
    nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv
    echo "--- model config"; cat "$CFG"
} >"$LOG/meta.txt" 2>&1 || true   # `|| true`: provenance must never fail the job
git diff HEAD >"$LOG/git.diff" 2>/dev/null || true

used_mib=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ') || used_mib=0
if ((used_mib > 2048)); then
    say "WARNING: GPU $GPU already has ${used_mib} MiB in use (SLURM does not isolate GPUs here)"
fi

# ---- runner image and model server -------------------------------------------------------
step build docker compose build runner

say "starting $M"
"${DC[@]}" up -d "$M"
"${DC[@]}" logs -f --no-color --timestamps "$M" >"$LOG/server.log" 2>&1 &
LOGS_PID=$!
# Health check: 120 s start period + up to 120 x 30 s retries (first start downloads weights).
step wait_healthy "${DC[@]}" up -d --wait --wait-timeout 5400 "$M"
if ((LAST_RC != 0)); then
    say "server never became healthy; last server log lines:"
    tail -n 40 "$LOG/server.log" || true
    exit 1
fi

{
    echo "--- server"
    cid=$("${DC[@]}" ps -q "$M")
    echo "container:   $cid"
    echo "image:       $(docker inspect --format '{{.Config.Image}}' "$cid")"
    echo "image id:    $(docker inspect --format '{{.Image}}' "$cid")"
    echo "repo digest: $(docker image inspect --format '{{join .RepoDigests " "}}' \
        "$(docker inspect --format '{{.Config.Image}}' "$cid")")"
    echo "vllm:        $("${DC[@]}" exec -T "$M" python3 -c 'import vllm; print(vllm.__version__)' 2>&1 | tail -n1)"
    echo "command:     $(docker inspect --format '{{join .Config.Cmd " "}}' "$cid")"
    models_json=$(curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" || true)
    echo "/v1/models:  $models_json"
    root=$(grep -o '"root":"[^"]*"' <<<"$models_json" | head -n1 | cut -d'"' -f4)
    hf_cache="${HF_CACHE:-$HOME/.cache/huggingface}"
    if [[ -n "$root" ]]; then
        ref="$hf_cache/hub/models--${root//\//--}/refs/main"
        echo "hf revision: $root @ $(cat "$ref" 2>/dev/null || echo "unknown ($ref not found)")"
    fi
} >>"$LOG/meta.txt" 2>&1 || true

# ---- probe, smoke run, status, export ----------------------------------------------------
R=(docker compose run --rm -T runner)
step probe  "${R[@]}" probe --model "$CFG"
step run    "${R[@]}" run --run-dir "$RUN_DIR" --docs "$DOCS" --model "$CFG" \
                --limit "$LIMIT" --concurrency "$CONCURRENCY"
step status "${R[@]}" status --run-dir "$RUN_DIR"
step export "${R[@]}" export --run-dir "$RUN_DIR"

echo "--- nvidia-smi (GPU $GPU) at end" >>"$LOG/meta.txt"
nvidia-smi -i "$GPU" --query-gpu=memory.used,memory.total --format=csv >>"$LOG/meta.txt" 2>&1 || true

say "steps:"; column -t -s $'\t' "$LOG/steps.tsv" || cat "$LOG/steps.tsv"
exit "$FAILED"
