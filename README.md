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

#### Summaries for the perturbation runners

The `run_perturbation_*.py` runners take their summaries from a separate build, for
the same reason this pipeline is a separate step: a model must not be scored on a
summary it wrote itself, and the same judgment must not be re-summarised once per
judge. Build once, then pass the file to every runner:

```bash
python scripts/build_summaries.py \
  --cases data/processed/livehrb_1k.json \
  --summarizer x-ai/grok-4.6 \
  --api-key-env OPENROUTER_API_KEY \
  --out data/processed/summaries_grok46.json
```

The build deduplicates by judgment, checkpoints per (judgment, version), and records
the summariser inside the file, so results carry their own provenance. Runners fail
with a message rather than summarising for themselves when `--summaries` is missing.

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
