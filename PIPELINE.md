# PIPELINE.md — Live HUDOC ingestion and eval-set construction

The "Live" part of LiveHumanRightsBench: a self-refreshing, contamination-controlled
data pipeline that turns the ECtHR's steady output into evaluation material. As new
judgments are published on HUDOC, the pipeline ingests them, removes verdict leakage,
and extends the benchmark, so metrics can be recomputed on guaranteed post-cutoff cases
rather than a frozen snapshot.

All outputs are public and redistributable (HUDOC sources only, `seed=42`).

## Auto-refresh chain

```
scripts/hudoc_live_refresh.py       # orchestrator: detect latest date -> fetch new -> scrub -> append -> (optional) push
  ├── scripts/hudoc_scraper.py          # HUDOC bulk ingestion (search + full texts)
  └── scripts/verdict_leakage_removal.py# 3-stage verdict removal (pattern truncation -> dual-model verify -> repair)
        └── scripts/conclusion_scrub.py # mandatory conclusion-scrub stage (regex), marks rows with `+conclusion_scrub`
```

`hudoc_live_refresh.py`:
1. determines the latest `decision_date` in the current dataset (local JSON or an HF dataset),
2. downloads judgments published since that date from HUDOC,
3. runs verdict-leakage removal on the new judgments,
4. appends the clean rows to the dataset,
5. optionally pushes the updated dataset to Hugging Face.

Designed to run as a cron job:

```bash
# daily at 06:00, refresh the verdict-free corpus in place and push
0 6 * * * cd /path/to/repo && python scripts/hudoc_live_refresh.py \
  --hf-dataset overthelex/echr-verdict-free \
  --output data/processed/echr_live.json \
  --full-pipeline \
  --push-to-hf overthelex/echr-verdict-free \
  --quiet
```

Run `python scripts/hudoc_live_refresh.py --help` for the full flag set
(`--country`, `--since-override`, `--workers`, `--keep-temp`, etc.).

## Eval-set construction

Builders that turn the verdict-free corpus into the evaluation sets documented in
[`DATA_SPLITS.md`](DATA_SPLITS.md):

```
scripts/clean_respondent_names.py   # respondent normalization (fills UKRAINE, etc.)
scripts/backfill_decision_dates.py  # HUDOC metadata sweep: item_id -> decision_date / appno
scripts/backfill_article_full.py    # HUDOC sweep: item_id -> article_full (protocol-aware codes, e.g. P1-1); resumable, --push-to-hf
scripts/build_stratified_sample.py  # k-per-state stratified sampling
scripts/build_livehrb_static.py     # -> echr-livehrb-static-2k  (regular + ukr, 1K + 1K)
scripts/build_temporal_split.py     # -> echr-livehrb-temporal-2k (year bins; UA pre/post 2022-02-24)
scripts/build_cutoff_partitions.py  # per-model matched pre/post training-cutoff partitions (contamination control)
scripts/enrich_and_push.py          # attach HUDOC metadata + push to Hugging Face
```

Per-model training cutoffs used by `build_cutoff_partitions.py` live in
[`configs/model_cutoffs.json`](configs/model_cutoffs.json) (extend as new models are added).

## Outputs

| Dataset | Built by |
|---------|----------|
| [`overthelex/echr-verdict-free`](https://huggingface.co/datasets/overthelex/echr-verdict-free) (23K pairs, 112 states) | live refresh + leakage removal |
| [`overthelex/echr-ukr-verdict-free`](https://huggingface.co/datasets/overthelex/echr-ukr-verdict-free) (2.6K, cross-family verified) | live refresh + leakage removal |
| [`overthelex/echr-livehrb-static-2k`](https://huggingface.co/datasets/overthelex/echr-livehrb-static-2k) | `build_livehrb_static.py` |
| [`overthelex/echr-livehrb-temporal-2k`](https://huggingface.co/datasets/overthelex/echr-livehrb-temporal-2k) | `build_temporal_split.py` |

## Dependencies

Standard library plus `requests` (HUDOC), `datasets` + `huggingface_hub` (HF I/O),
and, for the LLM verification stage, `openai` and/or `boto3` (AWS Bedrock). See
`requirements.txt`. The LLM-backed verification imports are optional at import time
(guarded), so ingestion and the regex conclusion-scrub run without cloud credentials.

## Not yet wired

The domestic-to-Strasbourg linkage layer (matching national-court decisions in EDRSR
to their ECtHR follow-on) is planned but not part of this PR.
