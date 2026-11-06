# HNC Note Extractor: Design (draft v0, for iteration)

Status: proposal. Implemented so far: the synthetic ingest adapter (§2 [0]) and the
**pathology vertical slice** (§9): field specs, DAG runner, LLM clients, evidence verifier,
rollup, review workbook, and compose profiles. It has been tested offline with a mock model
only. How to run it: docs/RUNNING.md.

## 0. Decisions log
- 2026-11-06: Extractions are kept **separate per note**. **Both note-level and patient-level** outputs are banked.
- 2026-11-06: Extract **all cancers**, not only HNC, tagged by site.
- 2026-11-06: Values stay **verbatim text** for now. Normalization (ICD-O-3, AJCC) is a later, separate step.
- 2026-11-06: Multiple models are fine, but they must run **sequentially, not concurrently**. The pipeline is stage-wise and batch: run one model over all docs for a stage, persist, swap the vLLM server, run the next stage.
- 2026-11-06: The target cluster is unknown, so **every model must fit on a single H200 (141 GB)**. The pipeline must run end-to-end on one GPU. Extra GPUs are only data-parallel replicas.
- 2026-11-06: The user (MD/PhD candidate) does manual review, so review tooling should be clinician-friendly.
- 2026-11-06: The current ingest is **synthetic-data-only**. Real data gets its own adapter that emits the same canonical `Document` JSONL.
 Earlier work in `src/` and `prompts/` is
treated as disposable reference material.

## 1. Principles

1. **A wrong value is worse than a missing one.** Abstain or flag rather than guess.
2. **Every value has provenance:** document id, verbatim evidence span(s) with character offsets,
   model, prompt/spec version, and run id.
3. **One field at a time, conditioned on its parents** (field DAG, §3).
4. **Note-level extraction is kept separate from patient-level reconciliation** (§5). Note-level
   output records what a document says. Patient-level output records what we conclude.
5. **Run once, cache everything.** Results are keyed by
   `(doc_text_hash, field_spec_version, model_id, decoding_params)`. Adding notes or editing one
   field re-runs only what changed.
6. **Deterministic checks wherever possible:** span verification, enum/schema validation, date
   parsing, unit normalization. The LLM does only what needs judgment.

## 2. Pipeline stages

```
[0] ingest/normalize  ->  [1] note-level field DAG  ->  [2] verify  ->  [3] patient-level reconcile  ->  [4] export
     (CPU)                    (LLM, vLLM)                (CPU+LLM)        (CPU + LLM adjudication)          (parquet/JSONL)
```

**[0] Ingest adapters (CPU, deterministic)** emit canonical `Document` JSONL
(`src/hnc_extract/documents.py`): doc_id, doc_type, patient_id, anchor_dttm, text, text_sha256, meta,
provenance.
- `synthetic_ingest.py` (synthetic only): HTML unescape (repeated until stable) and drop empty rows.
  No dedup and no row merging, because `Order_proc_id` is unreliable in the synthetic data. Output goes
  to `data/ingested/<dataset>/` with a manifest. The originals are only read.
- **Anchor date** per document (progress: date of service; radiology: exam time; pathology: result
  time). Relative-time expressions are resolved against it.
- Identical texts remain separate documents but share `text_sha256`, so the extraction cache runs
  them once.
- Optional cheap **relevance router** (regex plus a small LLM call): does the note contain oncologic
  content? Notes that fail the router still get a recorded "not relevant" result. They are not
  silently dropped.

**[1] Note-level field DAG (LLM).** See §3. One call per (field, parent instance). Uses
JSON-schema-constrained decoding through vLLM structured outputs, after the model's reasoning.

**[2] Verify**
- Deterministic: every evidence span must fuzzy-match the source text (whitespace-insensitive) and
  is stored with offsets. A failed match triggers a retry with feedback, then a flag.
- Enum/normalization validity (e.g. a T category must be in the AJCC value set).
- Optional **LLM verifier pass** (a different prompt, ideally a different model): "Given this
  evidence and this value, is the value supported? yes / no / partially." Disagreement sends the
  value to `needs_review`.
- Optional **self-consistency:** k samples per field. Accept on agreement, flag on split.
  Affordable because this runs once.
- Because models run sequentially, verification is its own batch stage. Model A finishes the whole
  extraction DAG for all docs. Then the server is swapped to model B, which verifies the persisted
  extractions. If B rejects a parent instance, its children are flagged.

**[3] Patient-level reconciliation.** See §5.

**[4] Export.** Tables: `note_extractions` (long format, one row per extracted value with
provenance), `patient_summary` (one row per patient per tumor), and `review_queue`.

## 3. Field hierarchy (note-level)

The core idea is **entity instances first, attributes second**. Child prompts receive the parent
instance (its value plus evidence) so the model knows *which* tumor, specimen, or lesion it is
describing. If a parent abstains, its children are skipped, which reduces hallucinated attributes.

### Progress note
```
note
├── cancer_mentions?  (gate: does the note discuss the patient's own malignancy?)
│   └── tumor[i]  (distinct primary cancers: site + histology as an identity key)
│       ├── primary_site            (+ laterality, subsite)
│       ├── histology               (+ differentiation)
│       ├── stage                   (TNM components, overall stage group, edition, clinical vs pathologic, timing: initial vs restaging)
│       ├── diagnosis_date          (verbatim expression + normalized date + precision)
│       ├── disease_status          (newly diagnosed | recurrent | metastatic | NED... as of this note)
│       ├── recurrence_events[j]    (type: local/regional/distant; site; date)
│       ├── hpv                     (status + method: p16 IHC | HPV ISH | PCR; per test)
│       ├── pdl1                    (status / CPS / TPS + assay if stated)
│       └── keratinization
└── treatment_history
    └── regimen[k]                  (modality, agents, dates, intent, line, associated tumor[i])
```

### Pathology report
```
report
├── specimen[i]           (part label A/B/C, site, procedure)
│   ├── procedure_type    (biopsy | FNA | excision | resection | ...)
│   ├── carcinoma_status  (positive | negative | suspicious | non-diagnostic)
│   │   └── if positive: histology, site (from specimen), keratinization, differentiation
│   └── dysplasia_status  (positive | negative | not mentioned)
│       └── if positive: grade (mild | moderate | severe | CIS)
└── ancillary_tests[j]    (HPV ISH/PCR, p16 IHC, PD-L1, molecular panels such as OncoPlus; per test: result, specimen link)
```
Report-level carcinoma status = rollup over specimens (deterministic: any positive -> positive;
otherwise any suspicious -> suspicious; ...).

### Radiology report
```
report
├── overall_assessment    (positive | negative | suspicious | equivocal for malignancy; plus response terms such as "decreased")
└── lesion[i]             (site, description, measurements[] with unit, prior measurement if stated, suspicion level)
```

## 4. Item schema (shared)

Carried over from the prior draft:
- `value`
- `evidence` (verbatim spans, now also with offsets)
- `assertion` (affirmed | uncertain | negated, where negation is meaningful for the field)
- `temporality` (frame: current | historical | unknown, plus a verbatim expression)
- `experiencer` (patient | family)
- field-specific `qualifiers` (e.g. staging edition)
- `justification`

Free-text values are paired with a **normalized** value where a vocabulary exists. Candidate
vocabularies: ICD-O-3 topography/morphology for site and histology, AJCC 7/8 for stage, and
ISO dates with a precision flag. Normalization can be a separate DAG node, so raw extraction
and coding are audited independently.

## 5. Longitudinal handling (OPEN, to discuss)

Problem: most progress-note fields are patient-level facts (original site, original diagnosis
date, recurrence dates, treatment history), but they are stated piecemeal, inconsistently, and
often copy-forwarded across 1 to 340 notes per patient. Pathology and radiology reports are
mostly document-level facts that also feed patient-level facts.

Options:

| | Approach | Pros | Cons |
|---|---|---|---|
| A | **Independent per-note extraction, then patient-level reconciliation** over the extracted table | Parallel, cacheable, auditable, order-independent, incremental | Reconciliation logic is nontrivial (entity resolution, conflicts) |
| B | Sequential "running state": process notes chronologically, update the patient record | Natural for timelines | Error propagation, order-dependent, not parallel, hard to audit, re-run from scratch on any change |
| C | Per-field retrieval over all patient notes, one big prompt | Model sees everything at once | 300 notes x 7k tokens doesn't fit; long-context reasoning is less reliable; weak provenance |

**Proposal: A, with a two-pass tumor registry.**
1. *Pass 1 (note-level):* extract tumor instances (and their attributes) from every note independently.
2. *Patient tumor registry:* cluster the tumor mentions across notes into canonical tumors
   (deterministic grouping on normalized site/histology, with LLM adjudication for ambiguous
   cases such as "laryngeal SCC" vs "supraglottic SCC", or a second primary vs a recurrence).
3. *Pass 2 (optional):* re-run tumor-specific fields (stage, recurrence events, regimen to tumor
   linkage) conditioned on the patient's canonical tumor list, so notes are interpreted against
   the known context (e.g. "recurrence" -> which tumor).
4. *Per-field reconciliation rules:*
   - **Original diagnosis date / original site / histology:** earliest-dated, most specific,
     source-precedence weighted (pathology > onc progress note > other services). Conflicts go to
     review.
   - **HPV / PD-L1 / keratinization:** pathology reports are authoritative. Progress-note mentions
     corroborate and fill gaps (e.g. outside-hospital testing).
   - **Stage:** keep a *staging timeline* (initial clinical, pathologic, restaging at recurrence),
     not a single value.
   - **Disease status / recurrence:** build an *event timeline*. Deduplicate events across notes
     by (type, site, date window). Status at a given time comes from the timeline.
   - **Treatment history:** deduplicate regimens across notes by (agents, modality, date overlap)
     into a registry-style list.
5. **Copy-forward awareness:** the same text block repeated in 50 notes is one piece of evidence,
   not 50 votes. Hash normalized paragraphs and track evidence-span provenance so support counts
   are over distinct sources.

## 6. Models and serving

- vLLM OpenAI-compatible server in docker compose. There is **one compose profile per model**, so
  only one model is served at a time. The pipeline runner is a separate container that talks HTTP
  to the server. The LLM client is abstracted, so tests use a mock/replay client.
- Each model gets a config: served name, vLLM args, recommended sampling, reasoning-effort
  control, and whether JSON-schema-constrained decoding works alongside its reasoning parser. If
  it doesn't, the client falls back to parsing the output, validating with pydantic, and retrying
  with the error message.
- **Shortlist (Nov 2026; each model must fit ONE H200 = 141 GB with KV headroom at ~32k context; Hopper has no native FP4):**

| Model | Arch / size | License | Serving notes |
|---|---|---|---|
| Nemotron 3.5 Super VL 120B-A12B **NVFP4** | Mamba-2/attention hybrid MoE, 12B active, ~75 GB | OpenMDW-1.1 | vLLM >=0.31; recipe documents 1x H200 TP1 at 114,688 ctx; runs W4A16 on Hopper; `--reasoning-parser nemotron_v3`; some benchmark loss vs BF16 (e.g. MMLU-ProX 73.8 vs 76.9) |
| Qwen3.8-27B | dense, ~56 GB BF16 | Apache-2.0 | TP1, `--reasoning-parser qwen3`, `reasoning_effort` low/medium/xhigh; VL model, so `--language-model-only` |
| Meta Muse Glimmer 30B | dense, 59.6 GB BF16 (32.8 GB FP8) | Apache-2.0 | vLLM >=0.28, `--reasoning-parser muse_glimmer`, effort set by system-prompt line; never greedy |
| Gemma 4 31B | dense, ~62 GB BF16 | Apache-2.0 | configurable thinking; toggle syntax to confirm |

  These come from four families (NVIDIA, Qwen, Meta, Google), so any one can verify another. Secondary options:
  Nemotron 3.5 Lightning-30B-A3B and Mistral Small 4 (unverified). **gpt-oss-120b is not used,
  even as a baseline (user preference: too old).**

  Excluded at FP8 because the weights alone nearly fill one H200: Qwen3.5-122B-A10B-FP8 (~127 GB) and
  Nemotron Super FP8 (the NVFP4 build above fits). 4-bit builds of other ~120B MoEs may also fit
  (e.g. nvidia/Qwen3.5-122B-A10B-NVFP4), but they haven't been checked. Excluded because they are multi-GPU only:
  DeepSeek-V4.1-Flash, Qwen3.8-Flash-Next, GLM-5.x, Kimi K3.
- With ~30B models, the correctness machinery matters more: fine-grained field decomposition,
  self-consistency votes, and cross-family verification, all run sequentially on the same GPU.
- **Selection is by a bake-off** on the pathology slice against the user's gold labels. The
  verifier should ideally come from a different model family than the extractor.

## 7. Evaluation

- Need a small **gold set**: double-annotated if possible, about 50 to 100 documents per type,
  stratified by service and note type, plus patient-level gold for about 10 patients.
- Metrics per field: precision/recall/F1 on values, abstention accuracy, evidence validity rate,
  and patient-level agreement with the registry if Cancer Registry data is available.
- Synthetic-data caveats: masked dates (date fields can't be scored), de-identification over-masking
  of terms like PD-L1 and HPV.

## 8. Open questions
See the conversation thread. Will be folded in here as they are resolved.


## 9. Implementation (pathology slice, 2026-11-06)

Code is in `src/hnc_extract/`:

| Module | Role |
|---|---|
| `spec.py` | Field spec YAML -> pydantic output model (used for JSON-schema decoding and validation). Validates the DAG and the worked examples. |
| `prompting.py` | Renders prompts. The system prompt is the shared rules (identical across calls, so it hits the prefix cache). The user message holds the document, then the field definition, then the target, then the ask. |
| `llm.py` | `ModelConfig`, the OpenAI-compatible vLLM client (transport retries), and mock/scripted clients. |
| `extract.py` | One call: parse -> validate -> verify evidence -> retry with feedback. Ends in status ok / flagged / invalid / error. |
| `evidence.py` | Matches quotes against the source: exact, then whitespace-insensitive, then case/quote-folded. Ellipses are rejected. Records offsets of the occurrence nearest the target. |
| `store.py` | SQLite cache. Key = (doc text hash, spec + instructions fingerprint, model fingerprint, prompt-format version, target context, sample index). |
| `dag.py` | `traverse()` rebuilds a document's node graph from the cache. `run()` runs it stage-wise: one field at a time across all docs, concurrently, persisting each call. |
| `rollup.py` | Deterministic report-level rollup. |
| `export.py` | Long JSONL plus `review.xlsx` (README / review / reports / documents). |
| `mock.py` | Regex stand-in model for offline tests. Its output is not data. |
| `cli.py` | Commands: `run`, `status`, `export`, `probe`, `ingest-synthetic`. |

The spec format (`specs/<doc_type>/<field>.yaml`) has these keys:
- `kind`: entity or attribute.
- `scope`: document, or an entity field.
- `depends_on`: gates on the categories of same-scope fields.
- `scope_filter`: gates on attributes of the scope instance itself.
- Output shape: `categories`, `attributes`, `temporality`, `max_items`.
- Prompt content: `instruction`, `value_description`, `examples`.
- `target_instruction` (entities only): what child prompts are told about the target.

Children get the target instance, its siblings (marked out of scope), and their dependency
results as context. The context is part of the cache key, so a change upstream re-runs exactly
the affected descendants. Editing a spec re-runs only that field, plus any descendants whose
inputs actually changed.

Pathology DAG (run/display order):
`specimen` -> {`procedure_type`, `carcinoma_status` -> {`histology`, `keratinization`}, `dysplasia_status` -> `dysplasia_grade`};
`ancillary_test` -> {`test_result`, `pdl1_score` (only if test_type = pdl1_ihc)}.

**Spec rules.** Items 2 and 4–9 were reviewed with the user on 2026-11-06 after the 4-model
smoke run; items 1 and 3 are still defaults awaiting review.
1. carcinoma_status counts carcinoma in situ and non-carcinoma malignancies (lymphoma,
   melanoma, sarcoma) as positive. histology records the type, and `invasion`
   (invasive / in_situ / microinvasive / not_stated) separates CIS from invasive tumour. CIS is
   also captured as dysplasia_grade = carcinoma_in_situ, per the user's field list.
2. negative has a `basis` attribute: explicit_negative ("negative for carcinoma") vs
   non_malignant_diagnosis (benign, reactive, treatment effect, benign neoplasm, or dysplasia
   short of CIS; absence implied, not stated). Ancillary-test text ("% of tumor cells") is not a
   diagnosis; a part whose only statement is a test result abstains, and the rollup flags it
   `no_diagnosis_stated` (a structural flag, with no inference about the part's content).
3. dysplasia_status: silence means abstain. Only an explicit "no dysplasia" is negative.
4. Grades stay in the report's own system. mild/moderate/severe/CIS and low/high-grade are
   separate categories and are never converted. A range produces one extraction per endpoint.
   A non-standard degree word ("markedly dysplastic") is `other`, with the word verbatim.
   Extent words ("focal") are not grades. Abstaining on grade leaves dysplasia_status intact.
5. Test results use the pathologist's stated call. Raw scores without a call ("rare positivity
   <1%", "CPS 50") are `result_without_interpretation`. Pending followed by a resulted addendum
   gives both extractions, and the rollup prefers the resulted one. Stain descriptors are
   copied verbatim into `percent_cells` / `intensity` / `pattern`. Guideline interpretation
   (e.g. p16 >=70% nuclear+cytoplasmic for oropharyngeal SCC) is NOT done by the LLM. It is
   planned as a deterministic, versioned rule layer at patient level, because it depends on
   the primary site and on which guideline version applies.
6. procedure_type categories: biopsy, FNA, other cytology, excision, resection,
   lymph_node_dissection, other. Neck dissection is its own category.
7. ancillary tests in scope: p16 IHC, HPV ISH/PCR/unspecified, EBV ISH (EBER), PD-L1 IHC,
   biomarker IHC (exactly: HER2, AR, MMR proteins, pan-TRK, ALK, ROS1, Ki-67, p53, NUT),
   molecular panels, and single-gene molecular. Other diagnostic IHC and special stains are
   excluded (cost: one entity plus one result call per stain; meaning is already in histology).
8. Specimens include consult parts (outside slides/blocks), with `material` recording what was
   received. `site_source` records where `site` came from. The clinical history is used only
   for a single-specimen report whose history names exactly one site.
9. Evidence correction must not silently drop items. If a corrected answer has fewer
   extractions than the unverified one, the earlier answer is kept with status flagged.
