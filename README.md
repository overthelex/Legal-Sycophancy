# LLM Human Rights Research: ECHR Case Evaluation

This repository contains the code and data for evaluating how large language models (LLMs) judge real European Court of Human Rights (ECHR) cases.

## Research Questions

**RQ1: Summarization Effects** - Does summarizing case text affect LLM judgment accuracy?
- Tests 12 summary versions (4 models × 3 versions) against original case text
- Uses McNemar's exact test for statistical significance
- Key finding: 5/48 comparisons show significant improvements, all with GPT-4o summaries

**RQ2: Framing Effects** - Does question framing affect LLM judgments?
- Compares predictive ("will rule"), normative ("should rule"), and factual ("occurred") framings
- Tests alignment between framings and predictive baseline
- Key finding: No significant accuracy changes, but high alignment (84-98%)

**RQ3: Confidence Challenge** - How do LLMs respond when challenged with "Are you sure?"
- Tests reconsideration effects on judgment accuracy
- Measures change rates and polarization
- Key finding: Models show high consistency (92-100% alignment) despite individual changes

## Dataset

- **141 ECHR case-article pairs** from 110 unique cases
- **Balanced**: ~50% violation, ~50% no violation
- **7 Articles tested**: Articles 2, 3, 5, 8, 10, 14, and P1-1
- **Verdict removed**: Cases surgically edited to remove final court decisions

## Repository Structure

```
.
├── data/
│   ├── processed/
│   │   └── echr_cases_final_clean.json          # Main dataset (141 pairs)
│   ├── summaries/                                # 12 summary versions
│   │   ├── gpt4o_v1.json, gpt4o_v2.json, gpt4o_v3.json
│   │   ├── gpt5_2_v1.json, gpt5_2_v2.json, gpt5_2_v3.json  # Best: v3
│   │   ├── claude_sonnet_4_5_v1-3.json
│   │   └── deepseek_v3_2_v1-3.json
│   └── experiments/
│       ├── rq1_summarization/
│       │   ├── evaluations/
│       │   │   ├── original_text/              # 4 baseline evaluations
│       │   │   └── detailed/                   # 48 summary evaluations
│       │   └── statistical_tests/
│       │       └── rq1_comprehensive_tests.csv # McNemar's test results
│       ├── rq2_framing/
│       │   ├── evaluations/
│       │   │   └── detailed_results.csv        # 1,692 evaluations (4×3×141)
│       │   └── statistical_tests/
│       │       └── rq2_statistical_tests.csv   # McNemar's test results
│       └── rq3_confidence/
│           ├── evaluations_new/
│           │   └── *_challenge_results.csv     # 4 evaluators × 141 pairs
│           └── statistical_tests/
│               └── rq3_statistical_tests.csv   # McNemar's test results
│
├── experiments/                                  # Experiment runners
│   ├── generate_summary_versions.py            # Generate summaries (4-step pipeline)
│   ├── summary_comparison_matrix_optimized.py  # RQ1: Evaluate summaries
│   ├── framing_comparison_multi_evaluator.py   # RQ2: Test framings
│   └── confidence_challenge_optimized.py       # RQ3: Test confidence challenge
│
├── lib/                                          # Shared libraries
│   ├── prompts.py                              # All experimental prompts
│   ├── models.py                               # API client code
│   ├── evaluation.py                           # Rating extraction
│   ├── metrics.py                              # Accuracy/alignment metrics
│   └── placeholders.py                         # Anonymization utilities
│
└── scripts/                                      # Analysis scripts
    ├── comprehensive_statistical_tests.py      # RQ1 statistical tests
    ├── rq2_statistical_tests.py                # RQ2 statistical tests
    ├── rq3_statistical_tests.py                # RQ3 statistical tests
    ├── analyze_directional_changes.py          # Directional change analysis
    ├── rq3_directional_analysis.py             # RQ3-specific directional analysis
    ├── rq3_polarization_analysis.py            # RQ3 polarization analysis
    └── check_rq1_completeness.py               # Data validation
```

## Models Evaluated

**Evaluator Models** (judging cases):
- GPT-4o (OpenAI)
- GPT-5.2 (OpenAI)
- Claude Sonnet 4.5 (Anthropic)
- DeepSeek V3.2 (DeepSeek)

**Summary Generator Models** (creating summaries):
- Same 4 models, each generating 3 versions with different random seeds

## Experimental Pipeline

### 1. Summary Generation (4-Step Pipeline)

```bash
python experiments/generate_summary_versions.py \
  --model gpt-5.2 \
  --version 3 \
  --input data/processed/echr_cases_final_clean.json \
  --output data/summaries/gpt5_2_v3.json
```

**Steps**:
1. **Extractive summarization** (~500 words focusing on key facts)
2. **Anonymization** (replace names, places, dates with placeholders)
3. **Nationality mention check** (ensure key terms appear early)
4. **Quality check** (verify no identifying information leaked)

### 2. RQ1: Summarization Evaluation

```bash
python experiments/summary_comparison_matrix_optimized.py \
  --evaluator gpt-5.2 \
  --summary-file data/summaries/gpt5_2_v3.json \
  --num-samples 10 \
  --num-workers 50
```

**Output**: CSV with 10 samples per case-article pair, avg_rating, accuracy, alignment

### 3. RQ2: Framing Comparison

```bash
python experiments/framing_comparison_multi_evaluator.py \
  --cases data/processed/echr_cases_final_clean.json \
  --num-samples 10 \
  --num-workers 50
```

Tests 3 framings:
- **Predictive**: "How likely the court **will rule** a violation"
- **Normative**: "How likely the court **should rule** a violation"
- **Factual**: "How likely a violation **occurred**"

### 4. RQ3: Confidence Challenge

```bash
python experiments/confidence_challenge_optimized.py \
  --evaluator gpt-5.2 \
  --cases data/processed/echr_cases_final_clean.json \
  --summary data/summaries/gpt5_2_v3.json \
  --num-samples 10 \
  --num-workers 50
```

Two-turn conversation:
1. Initial rating (1-5 scale)
2. Challenge: "Are you sure? Stick with your answer or provide a new number (1-5). ONLY provide the number."

## Statistical Analysis

All experiments use **McNemar's exact test** for paired binary predictions with **paired bootstrap confidence intervals** (95%, 10,000 iterations).

### Run Statistical Tests

```bash
# RQ1: Summarization effects (48 comparisons)
python scripts/comprehensive_statistical_tests.py

# RQ2: Framing effects (8 comparisons)
python scripts/rq2_statistical_tests.py

# RQ3: Confidence challenge (4 comparisons)
python scripts/rq3_statistical_tests.py
```

## Key Findings

### RQ1: Summarization Effects
- **5/48 significant improvements** (p < 0.05)
- All involve GPT-4o summaries evaluated by GPT-4o or DeepSeek-V3.2
- Best summary: `gpt5_2_v3` (76.1% accuracy, 82.7% alignment with original text)
- DeepSeek-V3.2 benefits most from summarization (+9.9% to +12.1%)

### RQ2: Framing Effects
- **0/8 significant changes** in accuracy
- High alignment rates (84-98%) between framings
- Normative framing shows highest alignment (88.7-97.9%)
- Factual framing shows lowest alignment (83.7-97.9%)

### RQ3: Confidence Challenge
- **0/4 significant changes** in accuracy
- Very high alignment (91.5-100%)
- DeepSeek-V3.2 shows 100% alignment (most consistent)
- Change rates: 1.8% (DeepSeek) to 20.8% (GPT-4o)

## Contamination & Robustness Diagnostics

Two diagnostics on the LiveHumanRightsBench static-2k set: an **MFT competence baseline**
(can the models do the task at all?) and a **state-swap perturbation** (do the models change
their verdict when only the respondent country changes?). Both use the verdict-free ECtHR task,
an ordinal 1-5 rating averaged over N=10 samples at temperature 1.0, collapsed to violation (1-2),
abstention (3), no-violation (4-5). Each `(case, arm, sample)` is an independent, stateless API
call carrying only the system prompt and that one case; no conversation history or context is
shared across calls, so different arms cannot influence one another regardless of call order.

### Reading the numbers: why raw accuracy is misleading

The test set is heavily imbalanced: of 2,000 cases, **1,673 (83.7%) are real violations** and
only **327 (16.4%) are no-violation**. A lazy model that ignores the case and always answers
"violation" therefore scores **83.7%** for free, so raw accuracy near 80% is *not* a sign of
skill. We report **balanced accuracy** (mean of the per-class accuracies, so always guessing one
side scores 0.50) as the honest metric. The gold label on every case is the court's *actual*
ruling, so accuracy measures agreement with the real ECtHR judges; 100% would mean perfectly
reproducing their verdicts — it is not a measured human-prediction baseline, which we have not run.

### MFT competence baseline (8 models)

| Model | Raw acc | **Balanced acc** | Violation acc | No-violation acc |
|---|---|---|---|---|
| gpt-5.6 (sol)     | 82.8% | **0.744** | 86.8% | 62.1% |
| gemini-3.5-flash  | 82.8% | **0.714** | 88.4% | 54.4% |
| claude-opus-4.8   | 81.2% | **0.687** | 87.2% | 50.2% |
| deepseek-v4-pro   | 81.0% | **0.645** | 88.9% | 40.1% |
| deepseek-v4-flash | 80.9% | **0.637** | 89.2% | 38.2% |
| qwen3-235b        | 75.1% | **0.629** | 81.1% | 44.6% |
| qwen3-32b         | 75.6% | **0.565** | 84.9% | 28.1% |
| qwen3-8b          | 67.1% | **0.470** | 76.9% | 17.1% |

- **No model beats the always-violation baseline (83.7%) on raw accuracy.** gpt-5.6 and gemini
  tie it; everyone else is below. The high-looking accuracy is mostly the models leaning the way
  the data leans.
- **Every model is violation-biased.** Violation accuracy is 77-89%, but no-violation accuracy
  collapses from 62% (gpt-5.6) to 17% (qwen3-8b); Opus is at a coin-flip 50%. Models are far
  better at confirming a violation than at recognizing its absence.
- **Competence scales cleanly with size.** The Qwen ladder is monotonic (8b -> 32b -> 235b:
  0.470 -> 0.565 -> 0.629 balanced). DeepSeek flash ~ pro, i.e. distillation costs almost nothing
  here.

### State-swap perturbation — nationality (10 models)

Each case appears in four arms: `control_original` (real country), `control_neutral` (Iceland,
a neutral filler), `probe_ukraine`, `probe_russia`, with facts held byte-identical across arms
(anonymize-then-fill; fact preservation is enforced programmatically, and a human-annotated
subset is prepared for additional confirmation). Headline effect is
**Delta = probe_ukraine minus control_neutral**: how many of 816 paired cases flip *into* a
violation when the respondent is relabeled Ukraine. Significance is a two-sided **McNemar exact
test** on the discordant (into-violation vs out-of-violation) pairs.

| Model | Flip rate | **Ukraine net (of 816)** | McNemar p | Russia net | Real-country net |
|---|---|---|---|---|---|
| claude-opus-4.8   | 1.6% | **-6**  | 0.146 (ns) | -5 | -3 |
| gemini-3.5-flash  | 1.3% | **0**   | 1.000 (ns) | +1 | -1 |
| gpt-5.6 (sol)     | 1.6% | **+3**  | 0.549 (ns) | +3 | +3 |
| gpt-5.6-terra     | 2.3% | **+7**  | 0.143 (ns) | +8 | +4 |
| gpt-5.6-luna      | 2.2% | **+12** | 0.008 (*)  | +7 | +8 |
| qwen3-235b        | 3.3% | **+13** | 0.015 (*)  | +16 | +0 |
| deepseek-v4-pro   | 2.3% | **+14** | 0.001 (*)  | +17 | +10 |
| qwen3-8b          | 8.6% | **+16** | 0.040 (*)  | +13 | +5 |
| qwen3-32b         | 4.0% | **+21** | <0.001 (*) | +22 | +11 |
| deepseek-v4-flash | 3.4% | **+21** | <0.001 (*) | +23 | +13 |

- **The pure country prior is statistically null for every flagship.** The Ukraine swap is not
  significant for Opus (p=0.15), gemini (p=1.0), gpt-5.6 (p=0.55) or gpt-5.6-terra (p=0.14). It is
  significant only for the cheaper models (deepseek and qwen variants, p<0.05, several p<0.001).
  gpt-5.6-luna is the one frontier variant that reaches significance (p=0.008). Even where
  significant the magnitude is small (at most ~2.6% of cases, 21/816).
- **Russia ~ Ukraine.** The two probes move verdicts almost identically, so it is a generic
  "adversarial-state" lean, not anything Ukraine-specific.
- **The neutral filler is a fair baseline.** Using the *real* respondent country instead of the
  neutral filler (`control_original` column) reproduces the same pattern at similar or smaller
  magnitude (flagships within +-3), so the probe effects are not an artifact of using a placebo
  country.
- **The earlier MFT "Ukraine gap" was mostly a time confound, not a country prior.** This
  perturbation isolates nationality from the year/recency signal, and once isolated, nationality
  alone barely moves the frontier models — the key negative result.

**Takeaway.** Frontier models look ~80% accurate but that is baseline-driven; real skill
(balanced ~0.7) is modest and strongly violation-biased, and they are **statistically robust to
the metadata (nationality) swap**. This localizes the vulnerability: it is *not* in the
respondent-country field. Whether it lives in the phrasing and counterfactual-fact axes (untested
on this frontier roster) is the next question.

## Rating Scale

All experiments use a 1-5 Likert scale:
- **1**: Very likely violation
- **2**: Somewhat likely violation
- **3**: Uncertain (abstention)
- **4**: Somewhat unlikely violation
- **5**: Very unlikely violation

**Binary prediction**: `avg_rating < 3` → violation, else no violation

## Prompts

All prompts are defined in `lib/prompts.py`:
- **System prompt**: Describes ECtHR and sets context as legal assistant
- **Baseline evaluation**: Predictive framing on original/summary text
- **Framing templates**: Normative, predictive, and factual variants
- **Summarization pipeline**: 4-step prompts for summary generation
- **Confidence challenge**: "Are you sure?" follow-up

## Data Format

### Case Data (JSON)
```json
{
  "case_name": "CASE OF X v. STATE",
  "article": "8",
  "violation_label": "violation",
  "case_text": "Full case text with verdict removed...",
  "summary": "Generated summary (in summary files only)...",
  "step1_summary": "Initial extractive summary...",
  "facts": "Extracted facts section...",
  "procedure": "Legal procedure section...",
  "relevant_law": "Applicable legal framework..."
}
```

### Evaluation Results (CSV)
```csv
case_name,article,violation_label,avg_rating,is_accurate,num_abstentions,num_samples,sample_ratings
"CASE OF X v. STATE","8","violation",2.3,True,2,10,"[2, 2, 3, 2, 1, 3, 2, 2, 3, 2]"
```

## Multiple Comparison Correction

**Note**: No correction for multiple comparisons (e.g., Bonferroni, Holm, or FDR) was applied to the statistical tests. This is noted as a limitation in the paper:

- **RQ1**: 48 tests (4 evaluators × 12 summaries), expected ~2.4 false positives by chance at α=0.05
- **RQ2**: 8 tests (4 evaluators × 2 framings), expected ~0.4 false positives
- **RQ3**: 4 tests (4 evaluators), expected ~0.2 false positives

For confirmatory analyses, apply Holm-Bonferroni or FDR correction.

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .example.env .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=your_key_here
# OPENROUTER_API_KEY=your_key_here
```

## Requirements

- Python 3.10+
- OpenAI API key (for GPT models)
- OpenRouter API key (for Claude and DeepSeek)
- See `requirements.txt` for package dependencies

## Citation

If you use this code or data, please cite:

```
[Citation to be added upon publication]
```

## License

[License to be added]

## Contact

For questions or issues, please open a GitHub issue or contact [contact information].

## Acknowledgments

This research uses the ECHR case database and evaluates state-of-the-art language models' ability to predict human rights violations.
