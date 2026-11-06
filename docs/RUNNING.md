# Running the pathology slice

Everything below runs from the repo root on the GPU host. One model server runs at a
time, on one GPU, chosen by the host's `CUDA_VISIBLE_DEVICES`. It is required for the model
servers and has no default: if unset, the server fails to start with a device error naming
`ERROR_CUDA_VISIBLE_DEVICES_NOT_SET`. Runner commands (`build`, `run`) don't need it. The runner is CPU-only and reaches the server at
`http://llm:8000/v1` on the compose network.

## 0. Once: ingest (synthetic data) and build the runner

```bash
python -m hnc_extract ingest-synthetic data/dev-test50          # -> data/ingested/dev-test50/
docker compose build runner     # outputs under ./runs are world-writable (umask 000)
```

## 1. Per model: serve, probe, run, export

Model profiles: `qwen3.8-27b`, `muse-glimmer-30b`, `gemma-4-31b`, `nemotron-3.5-super-nvfp4`.

```bash
export CUDA_VISIBLE_DEVICES=2              # host GPU for the model server (required)
M=qwen3.8-27b
docker compose --profile $M up -d
docker compose --profile $M logs -f        # wait for "Application startup complete" (first start downloads weights)

# a) Probe: one tiny extraction, with and without JSON-schema decoding
docker compose run --rm runner probe --model configs/models/$M.yaml

# b) Smoke run on the pinned 20-report set, then check it
docker compose run --rm runner run --run-dir runs/path-$M \
    --docs data/ingested/dev-test50/pathology_report.jsonl \
    --model configs/models/$M.yaml --doc-ids-file configs/eval/path-smoke.txt
docker compose run --rm runner status --run-dir runs/path-$M

# c) Full run (all 220 reports). Resumable: rerunning skips finished calls.
docker compose run --rm runner run --run-dir runs/path-$M --all-docs
docker compose run --rm runner export --run-dir runs/path-$M
#   -> runs/path-$M/export/review.xlsx, note_extractions.jsonl, report_rollup.jsonl

docker compose --profile $M down
```

`--all-docs` clears the saved document filter (`--doc-ids-file`, `--doc-ids` or `--limit`).
Other run settings are also saved in `runs/<dir>/run.json`, so a bare `run --run-dir ...`
resumes with the same settings. A doc-ids file is resolved when the run starts and the ids
are saved, so editing the file later doesn't change an existing run.

The smoke set `configs/eval/path-smoke.txt` lists each report with what it exercises. When
you change it, read the reports you add. `tests/test_spec_examples.py` fails if a spec
example shares an 8-word run with any ingested report (wording found in 5 or more reports
counts as boilerplate), so the examples can't leak answers for the reports we score.

## 1b. Smoke runs via SLURM (all models in parallel, one GPU each)

SLURM only reserves the GPU. Each job starts its own compose project (its own network,
`llm` alias and containers) on the reserved host GPU, with host port `18000 + GPU`, so
the four jobs don't collide. Step 0 (ingest) must be done first.

```bash
slurm/submit_smoke.sh                                  # all four models
slurm/submit_smoke.sh gemma-4-31b                      # or a subset
SBATCH_ARGS="-p <partition>" slurm/submit_smoke.sh     # extra sbatch flags
squeue --me
```

Jobs run the pinned set `configs/eval/path-smoke.txt` (override with `DOC_IDS_FILE=...`, or
`LIMIT=N` for the first N reports). Each job writes to `runs/smoke/<model>-j<jobid>/`: the run dir itself, plus
`export/review.xlsx` and `logs/` (`meta.txt` provenance: git commit, image digest,
vLLM version, HF revision, GPU; `git.diff`; `server.log`; `probe.log`, `run.log`,
`status.log`; `steps.tsv`). SLURM's own log is `runs/smoke/slurm-smoke-<model>-<jobid>.out`.
Every job uses a fresh run dir, so reruns never reuse cached calls.

Gemma (and possibly Muse) is gated on Hugging Face: set `HF_TOKEN` (shell or `.env`) and
accept the license once before the first download. If a job is killed with SIGKILL, so
its cleanup can't run, remove its containers by hand: `docker compose -p hnc-smoke-<jobid> down`.

## What to send back (no PHI in the synthetic data, but check before sharing real data)

1. The output of `probe` for each model. It tells us whether `structured_output: json_schema`
   works with that model's reasoning parser. The defaults are `json_schema` for Qwen and Gemma,
   which is documented to work, and for Nemotron (probe passed); `none` (parse, validate,
   retry) for Muse.
2. The output of `status` after the smoke run.
3. Any server errors from `docker compose logs`.
4. Later: `review.xlsx` with your verdicts filled in.

## Knobs

- `--concurrency N`: parallel requests to the server (default 16). vLLM batches them.
- `--fields a,b`: run only these fields. Their upstream fields must already be done.
- Edit a spec under `specs/pathology_report/*.yaml` and rerun. Only calls whose spec, model,
  or upstream context changed are recomputed.
- One run dir per model config. Changing sampling or reasoning settings needs a new run dir.
- `HNC_LLM_BASE_URL` or `--base-url` point the runner at a server outside compose,
  e.g. `http://localhost:8000/v1`.

## Untested against real servers (could only be checked offline in the sandbox)

- The compose commands follow the vLLM recipes and model cards (checked 2026-11-06) but have
  not been launched. Qwen and Gemma have no H200-specific recipe row. Muse Glimmer is not on
  the recipe's verified-Hopper list. If an image fails, the recipes name specific tags
  (`vllm/vllm-openai:qwen38`, `vllm/vllm-openai:gemma4-0505-cu129`). Set them with
  `VLLM_IMAGE=...`.
- `max_tokens` is 32768 per call, with `--max-model-len 65536` for the dense models. If you
  see `truncated (finish_reason=length)` in the attempts, reasoning ran out of budget.
