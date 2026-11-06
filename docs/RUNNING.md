# Running the pathology slice

Everything below runs from the repo root on the GPU host. One model server runs at a
time, on one GPU (`GPU_ID`, default 0). The runner is CPU-only and reaches the server at
`http://llm:8000/v1` on the compose network.

## 0. Once: ingest (synthetic data) and build the runner

```bash
python -m hnc_extract ingest-synthetic data/dev-test50          # -> data/ingested/dev-test50/
docker compose build runner
mkdir -p runs
```

## 1. Per model: serve, probe, run, export

Model profiles: `qwen3.8-27b`, `muse-glimmer-30b`, `gemma-4-31b`, `nemotron-3.5-super-nvfp4`.

```bash
M=qwen3.8-27b
docker compose --profile $M up -d
docker compose --profile $M logs -f        # wait for "Application startup complete" (first start downloads weights)

# a) Probe: one tiny extraction, with and without JSON-schema decoding
docker compose run --rm runner probe --model configs/models/$M.yaml

# b) Smoke run on 10 reports, then check it
docker compose run --rm runner run --run-dir runs/path-$M \
    --docs data/ingested/dev-test50/pathology_report.jsonl \
    --model configs/models/$M.yaml --limit 10
docker compose run --rm runner status --run-dir runs/path-$M

# c) Full run (all 220 reports). Resumable: rerunning skips finished calls.
docker compose run --rm runner run --run-dir runs/path-$M --limit 0
docker compose run --rm runner export --run-dir runs/path-$M
#   -> runs/path-$M/export/review.xlsx, note_extractions.jsonl, report_rollup.jsonl

docker compose --profile $M down
```

`--limit 0` clears the saved 10-document limit. Other run settings are also saved in
`runs/<dir>/run.json`, so a bare `run --run-dir ...` resumes with the same settings.

## What to send back (no PHI in the synthetic data, but check before sharing real data)

1. The output of `probe` for each model. It tells us whether `structured_output: json_schema`
   works with that model's reasoning parser. The defaults are `json_schema` for Qwen and Gemma,
   which is documented to work, and `none` (parse, validate, retry) for Muse and Nemotron,
   which is undocumented.
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
