# Running the pathology slice

Jobs run on the kg35 H200 cluster through Slurm. Each job runs a pushed commit, cloned
on its node with its own uv env (`slurm/job.sh`), so it never sees uncommitted work.
A model job starts vLLM in Docker on the job's GPU, then runs the pipeline against it,
and publishes the result to shared storage once, at the end.

| What | Where (env override) |
|---|---|
| Source datasets, read-only (name -> adapter + path) | `configs/datasets.yaml`: synthetic TSVs in `/opt/gpudata/hnc/<dataset>/`, TCGA reports in `/opt/gpudata/cancer/tcga/reports/TCGA_Reports.csv` |
| Model weights, pre-pulled, never downloaded | `/opt/gpudata/models/<org>/<repo>` (`HNC_MODELS_ROOT`) |
| Published documents, runs, job logs | `/shared/gpudata/songs1/hnc-shared/` (`HNC_SHARED_ROOT`) |
| Node-local run root: commit clones, uv cache, work dirs, vLLM compile cache | `~/hnc-run/` (`HNC_RUN_ROOT`) |

Pinned inputs: the vLLM image by digest (`configs/containers.yaml`), the weights by sha256
of every file (`configs/weights.yaml`), the Python env by `uv.lock`. A job refuses to start
if its node lacks the image or a weight file differs.

## 0. Once per cluster (Steven, on a host)

```bash
# a) Data and weights in place on every node (jobs only read them)
#    every source in configs/datasets.yaml, e.g. /opt/gpudata/cancer/tcga/reports/TCGA_Reports.csv
#    hf download <repo> --local-dir /opt/gpudata/models/<repo>   for each repo in configs/weights.yaml
docker pull vllm/vllm-openai:v0.32.0@sha256:befccf0e...   # full digest in configs/containers.yaml

# b) Pin the weights: prints a files: block per model; paste it into configs/weights.yaml, commit
slurm/submit.sh --cpus-per-task=16 -- hnc_extract weights-pin qwen3.8-27b

# c) Ingest and publish the documents -> $HNC_SHARED_ROOT/ingested/<dataset>/
slurm/submit.sh --cpus-per-task=4 -- hnc_extract ingest-publish tcga-hnsc
slurm/submit.sh --cpus-per-task=4 -- hnc_extract ingest-publish dev-test50
```

`ingest-publish` is idempotent: with unchanged inputs (source file hashes, reference
tables, adapter version) it does nothing, and with changed inputs it refuses (runs point
at the published files) unless `--overwrite`.

`tcga-hnsc` keeps the 505 TCGA-HNSC hospital pathology reports of TCGA_Reports.csv (one
per patient), 39 of them English translations (`meta.translated`). It drops the 13
Asterand (F7) biobank abstract forms and the 2 Molecular Response (T3) procurement forms
(listed in the manifest). The HNSC case list and site codes are committed in
`configs/tcga/`; `python -m hnc_extract.tcga_refs` regenerates them from the GDC (needs
internet, never run by jobs). The text is verbatim OCR in which line breaks became ". ",
and the reports carry no dates.

## 1. Model jobs

Model names: `qwen3.8-27b`, `muse-glimmer-30b`, `gemma-4-31b`, `nemotron-3.5-super-nvfp4`
(each has `configs/models/<name>.yaml` for sampling and `configs/serving.yaml` for vLLM).
Commit and push first: `slurm/submit.sh` refuses a dirty tree or an unpushed HEAD.

```bash
slurm/smoke.sh                          # all four on the 18 TCGA smoke reports, one GPU each
slurm/smoke.sh gemma-4-31b              # or a subset

# Any run, e.g. all 505 TCGA reports with one model:
slurm/submit.sh --gres=gpu:1 -- hnc_extract job --model qwen3.8-27b \
    --docs ingested/tcga-hnsc/pathology_report.jsonl --run-name full/{model}-j{job}
# Continue a published run (failed step, new fields): cached calls are reused
slurm/submit.sh --gres=gpu:1 -- hnc_extract job --model qwen3.8-27b \
    --docs ingested/tcga-hnsc/pathology_report.jsonl --run-name full/qwen3.8-27b-j123 --resume
squeue --me
```

A job: checks the image and the weights' sha256, starts `vllm serve` (as the job's
user, offline, weights read-only, on `CUDA_VISIBLE_DEVICES`), runs `probe`, `run`,
`status` and `export` (a failed step is recorded, the next ones still run), stops the
server, and publishes `$HNC_SHARED_ROOT/runs/<run-name>/`:

- `run.json`, `cache.sqlite`, `export/` (`review.xlsx`, `note_extractions.jsonl`, `report_rollup.jsonl`): an ordinary run dir; `run.json` points at the published documents;
- `logs/job.json`: commit, node, GPUs, image digest, weights sha256 and revisions, vLLM version and command, steps with exit codes and seconds;
- `logs/server.log`: the vLLM log.

The job's full log is `$HNC_SHARED_ROOT/logs/<jobid>.out` once it ends; while it runs, on
`~/hnc-run/logs/<jobid>.out` of its node (in the dev container only for jobs on nvl01:
add `-w kg35-nvl01` to watch one live). A run is published even when a step failed, so
`--resume` can continue it. Published runs are read-only: `status`, `gold-draft` and
`score` open them in place (the cache read-only, without locks).

`--doc-ids-file` / `--limit` select documents as for `run`. The eval sets in
`configs/eval/` list each report with what it exercises: `tcga-dev.txt` (50 real
TCGA-HNSC reports, stratified by site), its 18-report subset `tcga-smoke.txt` (the
`smoke.sh` default), and the synthetic `path-smoke.txt`. When you change one,
read the reports you add. `tests/test_spec_examples.py` fails if a spec example
shares an 8-word run with any ingested report (wording found in 5 or more reports counts
as boilerplate), so the examples can't leak answers for the reports we score.

What to look at after a smoke round: `probe` in the job log (does `structured_output:
json_schema` work with that model's reasoning parser), `status`, and server errors in
`logs/server.log`.

Docker leftovers (containers of killed jobs): `slurm/submit.sh -w <node> --cpus-per-task=1
-- slurm/clean-docker.sh` lists them; `--apply` removes them. It also removes the
compose-era `hnc-smoke-*` projects and the `hnc-runner:local` image.

## 1b. Locally, without a cluster (CPU, mock model)

```bash
uv sync                                  # dev env from uv.lock (pinned uv: envs/install-uv.sh)
uv run python -m hnc_extract ingest-synthetic data/dev-test50        # -> data/ingested/dev-test50/
uv run python -m hnc_extract ingest-tcga /opt/gpudata/cancer/tcga/reports/TCGA_Reports.csv  # -> data/ingested/tcga-hnsc/
uv run python -m hnc_extract run --run-dir runs/mock --docs data/ingested/dev-test50/pathology_report.jsonl \
    --model configs/models/mock.yaml --doc-ids-file configs/eval/path-smoke.txt
# TCGA: --docs data/ingested/tcga-hnsc/pathology_report.jsonl --doc-ids-file configs/eval/tcga-smoke.txt
uv run pytest
```

`run --base-url http://127.0.0.1:<port>/v1` points `run`/`probe` at any running server.
On a dev-GPU lease (`slurm/dev-gpu.sh`, then `slurm/with-gpu.sh <cmd>`), a command sees
only the leased GPU.

## 2. Gold key: draft, review in the browser, import, score

```bash
# a) Align several runs (one per model) into a consensus draft (gold/ is gitignored)
python -m hnc_extract gold-draft --run-dir $HNC_SHARED_ROOT/runs/smoke/<model-a> --run-dir $HNC_SHARED_ROOT/runs/smoke/<model-b> ... \
  --out gold/path-smoke [--specs <spec dir the runs were made with>]
# b) Review: a local web app. In VS Code, open the forwarded port (http://localhost:8765)
python -m hnc_extract gold-review --dir gold/path-smoke
# c) Answers -> gold.jsonl (uses verdicts.json if present, else adjudicate.xlsx)
python -m hnc_extract gold-import --dir gold/path-smoke
# d) Score any run on the same documents
python -m hnc_extract score --run-dir $HNC_SHARED_ROOT/runs/smoke/<model> --gold gold/path-smoke/gold.jsonl --out score.json
```

- The app shows the report on the left (each model's quoted text is highlighted when you hover
  over its answer) and one card per aligned specimen or test on the right. Coded fields are
  dropdowns or chips, so a typo can't get in. Items no model found can be added; select their
  text in the report first. Press `?` in the app for the walkthrough and keyboard keys.
- Answers save after every change to `<dir>/verdicts.json`; the previous version is kept as
  `verdicts.json.bak`. The state is tied to its draft: after a new `gold-draft` in the same
  dir, the app refuses the old file. Move it away first.
- The server binds to localhost only and serves the report text, so treat `--host` with care
  on real data. `gold-draft` writes `review_bundle.json`. For an older draft, `gold-review`
  rebuilds it from the runs in `draft_meta.json` and checks that they still reproduce the draft.
- `adjudicate.xlsx` still works: `gold-import --source xlsx`. Coded values are checked there too.

## Knobs

- `--concurrency N`: parallel requests to the server (default 16). vLLM batches them.
- `--fields a,b`: run only these fields. Their upstream fields must already be done.
- Edit a spec under `specs/pathology_report/*.yaml` and rerun (`--resume`). Only calls
  whose spec, model, or upstream context changed are recomputed.
- One run per model config. Changing sampling or reasoning settings needs a new run name.

## Not yet checked on the cluster

- The serve arguments follow the vLLM recipes and model cards (checked 2026-11-06); the
  2026-11 smoke runs used them through compose. Qwen and Gemma have no H200-specific recipe
  row. Muse Glimmer is not on the recipe's verified-Hopper list.
- `configs/weights.yaml` has no pins yet (step 0b), so model jobs refuse to start until then.
