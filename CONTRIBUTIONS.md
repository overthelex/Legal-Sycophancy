# Author Contributions — LiveHumanRightsBench

This file records the authorship and contribution agreement for the
LiveHumanRightsBench paper (working title; ICLR target, end-of-August 2026),
as agreed by email between the co-first authors and confirmed by the PI.

## Author list

As posted by Terry J. C. Zhang (email, 2026-06-18) and confirmed by
Zhijing Jin ("fully supportive of the author list", email, 2026-06-18):

1. **Terry J. C. Zhang\*** (University of Toronto / ETH Zurich)
2. **Arian Khorasani\*** (University of Toronto / Vector Institute)
3. **Volodymyr Ovcharov\*** (LEX AI / legal.org.ua)

\* co-first authors, equal contribution — each may list their name first
on their own CV.

Middle authors (order to be finalized): Jerick Shi, Keenan,
Yu Fan / Jingwei Ni (ETH Zurich), Elliott Ash (ETH Zurich, to be
confirmed), Oxford advisors (TBC), Daniel, Ilias Chalkidis.
Senior/last: Zhijing Jin (PI, University of Toronto).

## CRediT roles

The author list above is confirmed (Terry J. C. Zhang and Zhijing Jin,
email, 2026-06-18; Terry publicly on Slack, "both co-first authors on
this paper"). The CRediT role split below was proposed by V. Ovcharov
(email, 2026-06-20) and is **pending co-author sign-off**:

| CRediT role | Contributor(s) |
|---|---|
| **Conceptualization** | Terry J. C. Zhang, Volodymyr Ovcharov, Zhijing Jin |
| **Data Curation** | **Volodymyr Ovcharov** — ECtHR/HUDOC corpora ingestion (23,204-pair `echr-verdict-free`, 2,619-pair `echr-ukr-verdict-free`), verdict-leakage removal pipeline (truncation + LLM verification + conclusion scrub), the `echr-livehrb-static-2k` and `echr-livehrb-temporal-2k` evaluation sets, leakage audits (v1.1, 2026-07-03) |
| **Methodology** | **Volodymyr Ovcharov** — live (self-refreshing) benchmark design, contamination control (verdict-free construction, per-model pre/post-cutoff partitions), temporal axes (year bins 2017–2026; Ukraine pre/post 2022-02-24); **Terry J. C. Zhang** — perturbation-stability framework (summarization / framing / reconsideration) |
| **Software** | **Volodymyr Ovcharov** (data & live-refresh pipeline, experiment runners), **Arian Khorasani** (perturbation experiments on top of the pipeline) |
| **Investigation / Experiments & Evaluation** | **Arian Khorasani** and **Volodymyr Ovcharov** (shared) |
| **Resources / Funding (compute & API)** | Terry J. C. Zhang (OpenRouter, Bedrock credits), Volodymyr Ovcharov (Bedrock, DeepSeek API, data infrastructure) |
| **Writing — original draft** | co-first authors (Terry J. C. Zhang lead on framing/baselines sections; Volodymyr Ovcharov lead on data/benchmark-construction sections) |
| **Writing — review & editing** | all authors |
| **Supervision** | Zhijing Jin |

## Data & code pointers

- Datasets (HF, public, cc-by-4.0): `overthelex/echr-verdict-free`,
  `overthelex/echr-ukr-verdict-free`, `overthelex/echr-livehrb-static-2k`,
  `overthelex/echr-livehrb-temporal-2k` — all v1.1
  (conclusion-scrub leakage fix, 2026-07-03; see dataset READMEs).
- Pipeline & experiment code: this repository.

## Change log

- 2026-07-03 — initial version, per the email record of 2026-06-18/20.
  Any changes to the author list or role split should be reflected here
  by pull request so the record stays reviewable.
