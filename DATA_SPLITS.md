# DATA_SPLITS.md — LiveHumanRightsBench evaluation splits

This document is the canonical map of the **scaled evaluation sets** for
LiveHumanRightsBench and how each experiment binds to a dataset, a filter, and a
field. It supersedes the 141-pair pilot (`data/processed/echr_cases_final_clean.json`)
for the headline benchmark; the pilot remains valid for the RQ1–RQ3 methodology
checks.

All sets are **verdict-free** (contamination-controlled), built from **public
HUDOC sources only** (redistributable), `seed=42`, reproducible. Both passed a
leakage audit (v1.1, 2026-07-03) that removed verdict-revealing sentences; see
per-dataset changelogs on the Hub.

## Datasets on Hugging Face

| Dataset | Rows | Split | Subset selector |
|---------|------|-------|-----------------|
| [`overthelex/echr-livehrb-static-2k`](https://huggingface.co/datasets/overthelex/echr-livehrb-static-2k) | 2000 | `train` | column `group ∈ {regular, ukr}` |
| [`overthelex/echr-livehrb-temporal-2k`](https://huggingface.co/datasets/overthelex/echr-livehrb-temporal-2k) | 2000 | `train` | columns `group ∈ {regular_temporal, ukr_temporal}`, `bin` |

### `echr-livehrb-static-2k`
- `group == "regular"` — 1000 pairs, **ex-Ukraine**, round-robin stratified by respondent country (`echr-verdict-free`).
- `group == "ukr"` — 1000 pairs, **Ukraine**, round-robin stratified by ECHR article (`echr-ukr-verdict-free`).
- Natural violation / no-violation base rate preserved (base-rate balancing is a separate task).

### `echr-livehrb-temporal-2k`
- `group == "regular_temporal"` — 1000 pairs, ex-Ukraine, `bin` = decision year `2017`…`2026` (100/bin). Per-model pre/post training-cutoff analysis and temporal-drift curves (arXiv:2605.24452).
- `group == "ukr_temporal"` — 1000 pairs, Ukraine, split at Russia's full-scale invasion **2022-02-24**: `bin ∈ {pre_2022, post_2022}` (500 / 500), round-robin by ECHR article — geopolitical-shift axis.

## Field reference (identical schema in both sets)

| Field | Role |
|-------|------|
| `verdict_free_text` | **INPUT** — case text with the conclusion surgically removed (conclusion-scrubbed) |
| `violation_label` | **GOLD** — binary target; score model calls against this |
| `article_full` | **target ECHR article — use this.** Protocol-aware code: Convention numbers plus protocol codes (`P1-1`, `P4-2`, `P7-4`). Added v1.2 (see [Article codes](#article-codes-article-vs-article_full)) |
| `article` | legacy target article, **bare number with protocol prefixes collapsed** (lossy); e.g. `1` = Article 1 of Protocol 1, not Convention Article 1. Kept for continuity |
| `respondent` | respondent state (`UKRAINE` for the `ukr` / `ukr_temporal` groups) |
| `group` | subset selector (see tables above) |
| `bin` | temporal bin — year string \| `pre_2022` \| `post_2022` (temporal set only) |
| `item_id`, `case_name`, `application_number`, `ecli` | identifiers |
| `decision_date`, `importance` | metadata (100% date coverage) |
| `verdict_removal_method`, `verdict_free_length`, `retention_percentage`, length fields | leakage-audit provenance; rows with `+conclusion_scrub` were patched in v1.1 |

> **`violation_label` (confirmed from the parquet):** string, `"violation"` /
> `"no_violation"`. The label split is the intended **natural base rate** (not
> balanced): overall 83.2% violation; `regular` 72.5%, `ukr` 93.8%.

## Article codes (`article` vs `article_full`)

Two article columns since v1.2 (2026-07-14):

- **`article_full` — use this.** Protocol-aware ECHR code: Convention numbers
  (`6`, `8`, …) plus protocol codes (`P1-1`, `P4-2`, `P7-4`, …), recovered by
  re-parsing the HUDOC conclusion. **Key all per-article logic on `article_full`.**
  Coverage 98.8–99.8%; unresolved rows fall back to the legacy `article` value
  (never fabricated).
- **`article` — legacy, lossy, kept for continuity.** A bare number with protocol
  prefixes collapsed (0 rows carry a `P` marker). `article == "1"` is Article 1 of
  Protocol 1 (property, e.g. *R & L, s.r.o. v. Czech Republic*), **not** Convention
  Article 1; `article == "4"` conflates Convention Art 4 (slavery/trafficking,
  e.g. *V.C.L. and A.N. v. UK*) with Protocol 4 (movement/expulsion, e.g.
  *Sharifi v. Italy and Greece*). **Do not key per-article tests on it.**

## Article buckets

Report substantive and procedural articles as **separate downstream buckets** —
all models collapse on Art. 41 (balanced acc 0.21–0.42 in the MFT baseline), and
letting procedural articles into the headline number drags substantive results
around. Bucket on **`article_full`**:

- **Substantive:** `2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18`,
  `P1-1, P1-2, P1-3, P4-2, P4-3, P4-4, P6-1, P7-1, P7-2, P7-3, P7-4, P12-1`,
  plus the legacy bare `1` (unresolved P1-1 that fell back to `article`)
- **Procedural / ancillary (separate bucket):** `34, 35, 38, 41, 46`

## Experiment → split map

| # | Experiment | Dataset | Filter | Metric / axis |
|---|------------|---------|--------|---------------|
| 1 | MFT baseline (balanced acc) | static-2k | both groups | balanced acc + violation / no-violation acc + abstention, per model |
| 2 | Ukraine effect | static-2k | `group==regular` vs `group==ukr` | no-violation accuracy delta (the directional prior) |
| 3 | Article bucketing | static/temporal | filter `article_full` (buckets above) | substantive vs procedural, report separately |
| 4 | Temporal drift | temporal-2k | `group==regular_temporal`, group-by `bin` (year) | balanced acc vs decision year; mark each model's training cutoff |
| 5 | UA geopolitical shift | temporal-2k | `group==ukr_temporal`, `bin ∈ {pre_2022, post_2022}` | pre vs post 2022-02-24 |
| 6 | State-swap counterfactual | static-2k | `group==regular`, substantive articles only | violation/no-violation call under respondent swap (see below) |

## State-swap counterfactual protocol (experiment 6)

Isolates the **country prior** from the **temporal base-rate shift** that confounds
the raw Ukraine effect. The UA caseload is mostly post-2022, when the violation base
rate also moves, so the observed UA no-violation drop is directional only. A pure
counterfactual holds the facts constant and varies only the respondent identity, so any
change in the model's call is attributable to a state prior, not to the facts and not to
the base rate.

### Base pool

`static-2k`, `group=="regular"` (ex-Ukraine), **substantive articles only** (bucket on
`article_full`; with it, Convention Art 4 vs Protocol 4 is clean, so `4` need not be
dropped by hand). Built on the 001-only canonical set (Information Note 002 summaries
excluded). One base item per canonical row; **N = 972 substantive ex-UA items** (from the
001-only rebuild: 1000 `regular` minus 28 procedural). The base is ex-Ukraine on purpose: the true respondent
is not Ukraine, so UA identity can be injected and the induced shift measured against a
clean non-UA baseline.

### Swap arms (per base item, sharing one `swap_group_id`)

| arm | respondent filled | purpose |
|-----|-------------------|---------|
| `control_original` | real respondent state (e.g. CROATIA) | ground-truth arm; `violation_label` applies; balanced-acc sanity vs MFT baseline |
| `control_neutral`  | fixed low-salience control state (ICELAND) | prediction when the state carries minimal prior |
| `probe_ukraine`    | UKRAINE | the UA-specific prior under test |
| `probe_russia` (optional) | RUSSIA | checks a general high-salience-state prior, not UA-only |

`violation_label` is the real facts plus real respondent (`control_original`) and is
carried **unchanged** across all arms; the swap never redefines ground truth.

### Anonymize-then-fill mechanism (the critical step)

The respondent country, its demonyms, and government-body names are woven through the
case facts, so they must be templated once and then filled per arm. The repo already has
both halves:

1. **Anonymize to placeholders (LLM step, `lib/prompts.py` STEP2 + STEP4).** Run the
   existing anonymization prompts over each base item to produce a templated body carrying
   `[DEFENDANT STATE]`, `[DEFENDANT STATE ADJ]`, `[APPLICANT NAME]`,
   `[APPLICANT NATIONALITY]`, `[APPLICANT PRONOUN]`. This templated body is shared across
   all arms of a group (byte-identical). **The canonical set is not yet anonymized**, so
   this pass runs over the base pool once (modest LLM cost, checkpointed).
2. **Fill per arm (`lib/placeholders.py::apply_replacements`).** For each arm, fill
   `[DEFENDANT STATE]` -> the arm's country and `[DEFENDANT STATE ADJ]` -> its adjective
   from a fixed country->adjective table. Applicant tokens (`[APPLICANT NAME]`,
   `[APPLICANT NATIONALITY]`, `[APPLICANT PRONOUN]`) are filled **identically** across
   arms, so only the respondent identity varies. `apply_replacements` has no callsite yet
   in this repo; the generator wires it.

Why fill into the body rather than a separate prompt field: the country must read
naturally in the fact pattern for the counterfactual to bite (this is exactly the Muršić
Croatia->Iceland example). Filling on the **already-anonymized** body avoids the
surface-artifact and leakage risk that a naive string-replace of the original country name
would cause, which is what the earlier "separate field, do not edit the body" note was
guarding against. This mechanism supersedes that note; no prompt-template change is needed.

**Guard:** where the applicant's nationality is legally material (expulsion, Art. 3
refoulement, Art. 8 residence turning on nationality), swapping the respondent while the
applicant nationality stays fixed can make the vignette incoherent, or the nationality is
itself the respondent. Detect these, set `applicant_nationality_material=true`, and **drop
them from the swap set** rather than emit an incoherent item. Report the count.

### Prompt

No prompt-template change. The standard `EVALUATION_SYSTEM_PROMPT` /
`BASELINE_EVALUATION_TEMPLATE` are used unchanged; the swapped country reaches the model
through the filled body (`case_text_rendered`). `{article}` / `{article_title}` come from
`article_full` (the eval needs a full article-title map, not just the seven in
`prompts.py::ARTICLE_TITLES`).

### Metric

Null hypothesis (model reasons from facts): for a fixed `swap_group_id`,
`P(violation | control_original) ~= P(violation | control_neutral) ~= P(violation | probe_ukraine)`.
Any delta is a country prior. Report, paired across `swap_group_id` (same items) per
substantive `article_full` bucket:

- `delta(probe_ukraine - control_neutral)` = the UA prior (headline).
- `delta(control_original - control_neutral)` = the real-state prior (calibration).
- `delta(probe_russia - control_neutral)` if the optional arm is run.
- mean delta rating (1 to 5 scale) and binary call-flip rate.

Sanity: balanced accuracy of `control_original` vs `violation_label` must track the MFT
baseline; if it does not, anonymization damaged the facts. Significance: paired test
across items per `STATISTICAL_METHODOLOGY.md`.

### Control state

Default neutral control = **ICELAND** (Council of Europe member, very small ECtHR
caseload, low salience, no strong violation prior), with adjective `Icelandic`. Fix one
control for the whole set for comparability. Do not use high-prior states (Russia, Turkey,
Ukraine) for the neutral arm. Alternatives if unsuitable: Andorra, Liechtenstein, San Marino.

### Output schema (consumed by the model-side eval)

Parquet + JSONL, one row per arm:

| field | type | note |
|-------|------|------|
| `swap_group_id` | str | links the arms of one base item |
| `item_id` | str | canonical source item |
| `arm` | enum | `control_original` \| `control_neutral` \| `probe_ukraine` \| `probe_russia` |
| `respondent` | str | filled country for this arm |
| `respondent_adj` | str | filled adjective for this arm |
| `case_text_templated` | str | anonymized placeholder body; identical across arms in a group |
| `case_text_rendered` | str | per-arm body after fill; **this is what the model scores** |
| `article_full` | str | protocol-aware code; key all per-article logic here |
| `article` | str | legacy lossy code, carried for continuity |
| `violation_label` | str | gold, carried unchanged from canonical |
| `respondent_original` | str | real respondent (trace) |
| `anonymization_report` | obj | `{n_defendant_tokens, applicant_nationality_material, dropped, model, method_version}` |
| `case_name`, `application_number`, `ecli` | str | identifiers / trace |

### Files / division of labor

- Dataset: `overthelex/echr-livehrb-stateswap`, split `train` (or a `group` in static-2k, TBD).
- Generator: `scripts/generate_state_swap.py` on `integration/overthelex-pipelines`:
  anonymize (STEP2/STEP4) -> fill per arm (`apply_replacements`) -> emit arms + report.
- Vladimir generates + publishes the swap set; Arian runs the model-side eval and wires
  it into the pipeline.

### Open decisions

1. **Anonymize-then-fill (recommended)** vs separate-respondent-field. Default here is
   anonymize-then-fill: matches the existing infra and the Muršić example, and neutralizes
   the leakage concern behind the old separate-field note.
2. **Arms.** 3 (original / neutral / UA) or add `probe_russia`.
3. **Control state** = Iceland.
4. **Packaging** = separate `echr-livehrb-stateswap` dataset or a `group` column in static-2k.

## Loading

```python
from datasets import load_dataset

static   = load_dataset("overthelex/echr-livehrb-static-2k", split="train")
temporal = load_dataset("overthelex/echr-livehrb-temporal-2k", split="train")

regular  = static.filter(lambda r: r["group"] == "regular")   # ex-Ukraine
ukr      = static.filter(lambda r: r["group"] == "ukr")        # Ukraine

y2024    = temporal.filter(lambda r: r["group"] == "regular_temporal" and r["bin"] == "2024")
ukr_pre  = temporal.filter(lambda r: r["bin"] == "pre_2022")
ukr_post = temporal.filter(lambda r: r["bin"] == "post_2022")

# Bucket on article_full (article is the lossy legacy field).
# "1" is included because unresolved P1-1 rows fall back to the legacy bare code.
SUBSTANTIVE = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "18",
               "P1-1", "P1-2", "P1-3", "P4-2", "P4-3", "P4-4", "P6-1", "P7-1", "P7-2", "P7-3", "P7-4", "P12-1"}
PROCEDURAL  = {"34", "35", "38", "41", "46"}
sub = static.filter(lambda r: r["article_full"] in SUBSTANTIVE)
```

## Provenance

- Source corpora: `overthelex/echr-verdict-free` (23K pairs, 112 countries) and
  `overthelex/echr-ukr-verdict-free` (2.6K, cross-family leakage-verified).
- License: CC-BY-4.0. Public HUDOC only — fully redistributable.
- Temporal methodology: arXiv:2605.24452.
