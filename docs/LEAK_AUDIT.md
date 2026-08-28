# Verdict leak in the released evaluation sets

Audited 2026-08-25/26 against the published parquet exports. Reproduce with:

```bash
python scripts/leak_audit/audit_leak.py
```

## What was found

Verdict removal is driven by markers such as "The Court's assessment". Where no
marker fires, the Court's entire legal reasoning survives into the field the
model is scored on.

| dataset | rows | leaking | after re-cut |
|---|---:|---:|---:|
| `echr-verdict-free` (source) | 23,204 | **2,179 (9.4%)** | 9 false flags |
| `echr-ukr-verdict-free` | 2,619 | 3 (0.1%) | 0 |
| `echr-livehrb-static-2k` | 2,000 | **186 (9.3%)** | 0 |
| `echr-livehrb-temporal-2k` | 2,000 | **171 (8.6%)** | 0 |
| `echr-livehrb-stateswap` | 3,264 | **584 (17.9%)** | 0 |

`BALL v. ANDORRA` keeps 18,160 characters after `THE LAW`, including
"3. The Court's assessment" and the conclusion that the authorities "did not
fail to fulfil their positive obligations under Article 8" — against a
`no_violation` label. `STRAND LOBBEN AND OTHERS v. NORWAY` keeps 69,119.

The residual nine flags are four Grand Chamber cases whose only surviving
`THE LAW` is the table-of-contents entry; a second probe confirms zero
merits-conclusion phrases in the retained text. They are left flagged rather
than silenced by loosening the detector.

## Why the earlier leakage audit passed

It was lexical. A lexical probe fails in both directions: it fires on domestic
courts quoted in the facts ("the City Court finds no violation of Article 6",
San Marino procedural law) and misses the Court's own conclusion phrased as
"did not fail to fulfil its positive obligations". Of three hits on
`static-2k`, two were false positives and the real leak in `BALL` was missed.

`retention_percentage` is not a usable filter either: leaking cases appear at
49% and 66% retention.

## The skew matters more than the rate

A defect spread evenly is a constant. This one is concentrated in exactly the
dimensions the paper compares on.

| axis | spread | detail |
|---|---:|---|
| pool (`static-2k`) | 18.4pp | `regular` 18.5% vs `ukr` 0.1% |
| pool (`temporal-2k`) | 16.7pp | `regular_temporal` 16.9% vs `ukr_temporal` 0.2% |
| outcome label | 8-12pp | `no_violation` leaks 2.4-2.6x more often, in all three sets |
| importance | 16.4pp | importance-1 21.3% vs importance-4 4.9% |
| decision year | 19.2pp | 2018 19.2% vs 2016 0% |
| Convention article | 17-20pp | Art 10 20.0% vs P4-2 0% |
| respondent state | 36-52pp | Ireland 52.4% vs Montenegro 0% |
| **state-swap arm** | **0.0pp** | 146/816 in each of the four arms |

The single root cause explains the pattern: long, complex, landmark judgments
defeat marker-based truncation, and those are distributed unevenly across
countries, articles, years and outcomes.

The state-swap arms being perfectly balanced is the one piece of good news — the
four arms derive from one base case, so the paired contrast is not
*differentially* biased. That protects the design, not the numbers.

## How much it inflates accuracy

Measured directly: same judge, same prompt, same rows, only the text differs —
leaking version against re-cut version. 186 leaking rows plus 120 clean rows as
a control given identical treatment with nothing removed.

| cohort | class | n | with leak | after re-cut | inflation |
|---|---|---:|---:|---:|---:|
| leaking | violation | 123 | 0.943 | 0.943 | **0.000** |
| leaking | no_violation | 63 | 0.857 | **0.714** | **+0.143** |
| leaking | **balanced** | 186 | 0.900 | **0.829** | **+0.071** |
| control | all | 120 | 0.950 | 0.950 | 0.000 |

The control drifting exactly zero is what makes the rest trustworthy.

The leak buys **nothing** on violation cases and **14.3 points** on no-violation
ones. Models lean toward "violation" regardless, so they get those right
unaided; the retained reasoning helps precisely where they are weak. Of the 15
predictions that changed once the reasoning was removed, 12 ran
`no_violation` → `violation`.

Pooled accuracy across `static-2k` moves only ~0.45 points, so headline accuracy
is broadly safe. Balanced accuracy — the primary metric — inflates 7.1 points on
affected rows, and the leaking subset is twice as rich in `no_violation` as the
set overall (34% vs 16.4%), so **19% of the entire `no_violation` class was
judged with the answer visible**.

## What this means for analysis

Splitting results into leaking and clean subsets is **not** a valid repair: the
split is confounded with pool, country, year, article and outcome
simultaneously, so a "clean subset" is a biased sample on every axis at once.
Use it to size the error, then recompute on corrected data.

## Repair

```bash
python scripts/leak_audit/rebuild_sets.py --out-dir build/
```

Cuts at the last `THE LAW` following the last `THE FACTS`, ignoring
table-of-contents occurrences. Only leaking rows are touched; the rest are
byte-identical to the published version. Provenance fields are recomputed.
Row membership is preserved rather than re-sampled, so previously reported
numbers stay comparable except where the text itself changed.
