# Statistical Testing Methodology

## Overview

We employ paired statistical tests to evaluate whether experimental manipulations (summarization, framing, confidence challenges) significantly affect model accuracy in judging ECHR human rights violations.

## Test Selection Rationale

### Why Paired Tests?

Our experimental design involves **paired predictions**: each model evaluates the same case-article pairs under different conditions (e.g., original text vs. summary). This repeated-measures structure violates the independence assumption of unpaired tests, making paired tests the appropriate choice.

### McNemar's Exact Test

**Purpose**: Test whether the proportion of correct predictions differs significantly between two conditions.

**Method**:
- Construct a 2×2 contingency table of paired predictions
- Count discordant pairs:
  - `n01`: Cases where condition B correct, condition A wrong (improvement)
  - `n10`: Cases where condition A correct, condition B wrong (decline)
- Apply exact binomial test: `p-value = 2 × min(P(X ≤ k), 1 - P(X ≤ k - 1))` where `X ~ Binomial(n01 + n10, 0.5)` and `k = min(n01, n10)`

**Advantages**:
- Exact test (no large-sample approximation required)
- Appropriate for binary outcomes (correct/incorrect)
- Robust with moderate sample sizes (n=141 case-article pairs)

### Paired Bootstrap Confidence Intervals

**Purpose**: Estimate the uncertainty in accuracy differences between conditions.

**Method**:
1. Calculate paired differences: `diff[i] = correct_B[i] - correct_A[i]` for each case-article pair
2. Resample with replacement from the paired differences (10,000 iterations)
3. Calculate mean difference for each bootstrap sample
4. Extract 2.5th and 97.5th percentiles as 95% confidence interval bounds

**Advantages**:
- Non-parametric (no distributional assumptions)
- Accounts for paired structure of data
- Provides interpretable effect size estimates

**Implementation**:
```python
diffs = comparison_correct - baseline_correct  # Element-wise
bootstrap_means = []
for _ in range(10000):
    sample_diffs = np.random.choice(diffs, size=n, replace=True)
    bootstrap_means.append(np.mean(sample_diffs))
ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])
```

## Research Questions

### RQ1: Summarization Effect

**Comparison**: Original case text vs. case summary for each evaluator model

**Null Hypothesis (H0)**: Summarization does not affect accuracy

**Tests Conducted**: 48 comparisons (4 evaluators × 12 summary versions)

**Significance Threshold**: α = 0.05

### RQ2: Framing Effect

**Comparison**: Predictive framing ("will rule") vs. normative ("should rule") and factual ("occurred") framings for each evaluator

**Null Hypothesis (H0)**: Framing does not affect accuracy

**Tests Conducted**: 8 comparisons (4 evaluators × 2 alternative framings)

**Significance Threshold**: α = 0.05

### RQ3: Confidence Challenge Effect

**Comparison**: Original rating vs. rating after challenge ("Are you sure?") for each evaluator

**Null Hypothesis (H0)**: Confidence challenges do not affect ratings

**Analysis**: Directional change analysis (not significance testing, as changes are rare)

## Accuracy Calculation

For each case-article pair:
- **Ground truth**: Actual ECHR Court verdict (violation/no violation)
- **Model prediction**: Determined by average rating across 10 samples
  - Rating > 2.5 → "violation"
  - Rating ≤ 2.5 → "no violation"
- **Accuracy**: Proportion of predictions matching ground truth

## Alignment Calculation

Alignment measures consistency of a model's predictions across conditions:

```
Alignment = (# of case-article pairs with same prediction) / (total pairs)
```

For example, if a model predicts "violation" for Case X + Article 8 when evaluating both the original text and a summary, this contributes to alignment.

**Key distinction**: Accuracy measures correctness against ground truth; alignment measures self-consistency across conditions.

## Data Structure

- **Sample size**: 141 case-article pairs (100 unique cases, some with multiple articles)
- **Predictions per condition**: 10 independent samples per case-article pair
- **Averaging**: Ratings averaged across 10 samples before thresholding to binary prediction
- **Pairing key**: `(case_name, article)` tuple ensures proper alignment of paired observations

## Implementation Details

**Software**: Python 3.10
- `scipy.stats.binom` for exact McNemar's test
- `numpy` for bootstrap resampling (seed=42 for reproducibility)
- `pandas` for data manipulation

**Reproducibility**: All statistical test scripts are available in `scripts/`:
- `scripts/comprehensive_statistical_tests.py` (RQ1)
- `scripts/statistical_tests.py` (RQ2)
- `scripts/analyze_directional_changes.py` (RQ3)

## Interpretation Guidelines

### P-values
- **p < 0.05**: Statistically significant difference (reject H0)
- **p ≥ 0.05**: No significant difference (fail to reject H0)

### Confidence Intervals
- **95% CI excludes 0**: Effect is statistically significant
- **95% CI includes 0**: Effect is not statistically significant
- **Width of CI**: Indicates precision of estimate (wider = more uncertainty)

### Effect Sizes
- **Accuracy Δ**: Difference in proportion correct (range: -1 to +1)
  - |Δ| < 0.05: Small effect
  - 0.05 ≤ |Δ| < 0.10: Moderate effect
  - |Δ| ≥ 0.10: Large effect

## Limitations

1. **Sample Size**: With 141 case-article pairs, we have ~80% power to detect effects of |Δ| ≥ 0.15 at α = 0.05. Smaller effects may not reach significance.

2. **Multiple Comparisons**: We conduct 48 tests for RQ1. While we report raw p-values, researchers should be cautious about inflated Type I error rates. A Bonferroni correction (α = 0.05/48 ≈ 0.001) would make significance harder to achieve.

3. **Binary Threshold**: Converting 5-point ratings to binary predictions (at 2.5 threshold) may lose information about rating magnitudes.

4. **Independence of Samples**: The 10 samples per case-article pair are drawn independently, but temperature sampling may introduce slight dependencies.

## Reporting Standards

All results tables include:
- Model name
- Condition comparison
- Accuracy difference (Δ) with sign and 3 decimal places
- McNemar's p-value (4 decimal places)
- 95% bootstrap confidence interval [lower, upper]
- Significance marker (*** for p < 0.05, ns otherwise)

This ensures transparent reporting of both statistical significance and practical effect sizes.
