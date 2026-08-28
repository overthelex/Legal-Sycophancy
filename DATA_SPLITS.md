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

Isolates the **country prior** from the **temporal shift** that confounds the raw
Ukraine effect (the UA caseload is mostly post-2022, when the base rate moves too).

1. Base pool: `static-2k`, `group=="regular"`, substantive articles only
   (bucket on `article_full` — with it, Convention Art 4 vs Protocol 4 is now
   clean, so no need to drop `4` by hand).
2. Hold `verdict_free_text` (the facts) **fixed**.
3. Pass the respondent state to the model as a **separate prompt field**, and swap
   it `non-UA ↔ UKRAINE`. Do **not** string-edit the country into
   `verdict_free_text` — that introduces surface artifacts and can reintroduce
   leakage.
4. Re-score violation / no-violation against `violation_label` and compare the
   call distribution across the swap.

A pure counterfactual (facts constant, only respondent varies) cleanly separates
"model over-predicts violation for Ukraine" from "base rate shifted after 2022".

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
