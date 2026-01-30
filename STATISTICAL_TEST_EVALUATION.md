# Statistical Test Evaluation and Alternatives

## Current Approach: McNemar's Test + Paired Bootstrap CIs

### Why This Is Appropriate

**Data Structure**:
- Paired binary outcomes (correct/incorrect for same case-article pairs)
- Two conditions being compared (e.g., original vs. summary)
- Each observation is paired by `(case_name, article)`

**McNemar's Test Strengths**:
✅ **Designed for paired binary data** - Perfect match for our structure
✅ **Exact test available** - No large-sample approximation needed (we have n=141)
✅ **Tests marginal homogeneity** - Directly answers "do the conditions differ in accuracy?"
✅ **Non-parametric** - No distributional assumptions
✅ **Widely accepted** - Standard test in medical/clinical research for paired proportions

**Paired Bootstrap CI Strengths**:
✅ **Complements hypothesis test** - Provides effect size estimate with uncertainty
✅ **Non-parametric** - No assumptions about distribution of differences
✅ **Respects pairing** - Samples paired differences, not unpaired observations
✅ **Interpretable** - Directly estimates accuracy difference in percentage points

### Potential Concerns

**Issue 1: Discarding Concordant Pairs**
- McNemar's test only uses discordant pairs (n01 + n10)
- Ignores cases where both conditions are correct or both are wrong
- **Why this is OK**: Those cases provide no information about *difference* between conditions

**Issue 2: Loss of Rating Information**
- We convert 5-point ratings → binary predictions at 2.5 threshold
- Loses information about rating magnitudes and uncertainty
- **Impact**: May reduce statistical power

**Issue 3: Multiple Samples Averaged**
- We sample 10 times per case-article pair and average
- This averaging reduces variance but may mask within-case uncertainty
- **Why this is OK**: We're interested in expected behavior, not single-sample variability

**Issue 4: Multiple Comparisons**
- RQ1: 48 tests (4 evaluators × 12 summaries)
- Inflates family-wise Type I error rate
- **Current approach**: Report raw p-values without correction
- **Risk**: ~95% chance of at least one false positive if all null hypotheses are true

## Alternative Approaches to Consider

### 1. Ordinal Analysis (Preserving Rating Scale)

**Wilcoxon Signed-Rank Test**
- Compares paired ratings on 1-5 scale (not binary)
- Preserves ordinal information about rating magnitude
- More powerful if differences are consistent in direction and magnitude

**Implementation**:
```python
from scipy.stats import wilcoxon

baseline_ratings = [...]  # 1-5 scale
comparison_ratings = [...]  # 1-5 scale

statistic, p_value = wilcoxon(baseline_ratings, comparison_ratings)
```

**Advantages**:
- Uses more information (rating magnitude)
- Can detect shifts in ratings even if binary accuracy unchanged
- Standard non-parametric test for ordinal paired data

**Disadvantages**:
- Tests ratings, not accuracy (less directly interpretable)
- Assumes ordinal differences are meaningful

**Recommendation**: ⭐ **Yes, do this as supplementary analysis**

---

### 2. Mixed-Effects Logistic Regression

**Model**: Predict correctness with fixed effects for condition and random effects for cases

```python
import statsmodels.formula.api as smf

# Data structure: one row per (case, article, condition, sample)
model = smf.mixedlm(
    "correct ~ condition",  # Fixed effect for condition
    data=df,
    groups=df["case_article"],  # Random intercept per case-article
    re_formula="~1"
)
result = model.fit()
```

**Advantages**:
- ✅ Models correlations within case-article pairs
- ✅ Can include covariates (e.g., case complexity, article type)
- ✅ Provides odds ratios (interpretable effect sizes)
- ✅ Accounts for within-case variance from 10 samples

**Disadvantages**:
- ❌ More complex, harder to explain to reviewers
- ❌ Requires larger sample sizes for stable estimates
- ❌ Parametric assumptions (though robust to violations)

**Recommendation**: ⚠️ **Optional - if reviewers request**

---

### 3. Permutation Test

**Approach**: Randomly permute condition labels within each case-article pair

```python
def permutation_test(baseline_correct, comparison_correct, n_perm=10000):
    observed_diff = np.mean(comparison_correct) - np.mean(baseline_correct)

    perm_diffs = []
    for _ in range(n_perm):
        # Randomly swap within each pair
        perm_comparison = [c if np.random.rand() < 0.5 else b
                          for b, c in zip(baseline_correct, comparison_correct)]
        perm_baseline = [b if perm_comparison[i] == c else c
                        for i, (b, c) in enumerate(zip(baseline_correct, comparison_correct))]
        perm_diffs.append(np.mean(perm_comparison) - np.mean(perm_baseline))

    p_value = np.mean([abs(d) >= abs(observed_diff) for d in perm_diffs])
    return p_value
```

**Advantages**:
- ✅ Exact test under the null (no assumptions)
- ✅ Can be used for any test statistic
- ✅ Intuitive interpretation

**Disadvantages**:
- ❌ Computationally intensive
- ❌ McNemar's test is already exact, so this adds no benefit

**Recommendation**: ❌ **Not needed - McNemar's is already exact**

---

### 4. Bayesian Analysis

**Approach**: Model differences with posterior distributions

```python
import pymc as pm

with pm.Model() as model:
    # Prior on difference in accuracy
    delta = pm.Normal('delta', mu=0, sigma=0.1)

    # Likelihood for paired differences
    diffs = comparison_correct - baseline_correct
    obs = pm.Normal('obs', mu=delta, sigma=0.5, observed=diffs)

    # Sample posterior
    trace = pm.sample(2000)

# Posterior probability that delta > 0
prob_improvement = (trace['delta'] > 0).mean()
```

**Advantages**:
- ✅ Provides probability of improvement (not just p-value)
- ✅ Incorporates prior knowledge
- ✅ Naturally handles multiple comparisons via hierarchical models
- ✅ Can model uncertainty at multiple levels (samples, cases, evaluators)

**Disadvantages**:
- ❌ Requires specifying priors
- ❌ More complex, less familiar to most reviewers
- ❌ Computationally intensive

**Recommendation**: ⚠️ **Advanced option - probably overkill**

---

### 5. Multiple Comparisons Correction

**Problem**: 48 tests for RQ1 → inflated Type I error rate

**Options**:

**a) Bonferroni Correction**
- Adjusted α = 0.05 / 48 ≈ 0.001
- **Result**: Only 3 of our 7 significant results would remain significant
- **Issue**: Very conservative, reduces power

**b) Holm-Bonferroni (Step-Down)**
- Less conservative than Bonferroni
- Sequentially tests ordered p-values with adjusted α

**c) False Discovery Rate (FDR) Control (Benjamini-Hochberg)**
- Controls expected proportion of false positives among rejections
- Less conservative than family-wise error rate control

```python
from statsmodels.stats.multitest import multipletests

p_values = [...]  # All 48 p-values
reject, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
```

**d) No Correction (Current Approach)**
- Report raw p-values
- Justify exploratory nature of analysis
- **Rationale**: We're not claiming a single definitive comparison, but exploring patterns across multiple summaries

**Recommendation**: ⭐ **Add FDR-corrected results as sensitivity analysis**

**Implementation**:
```python
# After computing all p-values
from statsmodels.stats.multitest import multipletests

# RQ1: 48 comparisons
rq1_pvalues = df['mcnemar_p'].values
reject_fdr, padj_fdr, _, _ = multipletests(rq1_pvalues, alpha=0.05, method='fdr_bh')

# Add to results
df['mcnemar_p_fdr'] = padj_fdr
df['significant_fdr'] = reject_fdr
```

---

### 6. Stratified Analysis

**Approach**: Separate analyses by case characteristics

**Stratify by**:
- **Verdict**: violation vs. no violation cases
- **Article**: Common articles (3, 5, 6, 8) vs. others
- **Case complexity**: Length of text, number of facts

**Purpose**:
- Check if summarization/framing effects differ by case type
- Identify when interventions help vs. hurt

**Example**:
```python
violation_cases = df[df['violation_label'] == 'violation']
no_violation_cases = df[df['violation_label'] == 'no_violation']

# Separate tests
test_violation = mcnemar_test(violation_cases)
test_no_violation = mcnemar_test(no_violation_cases)
```

**Recommendation**: ⭐ **Yes, valuable exploratory analysis**

---

### 7. Effect Size Measures

**Beyond p-values, report**:

**Cohen's h (for proportions)**:
```python
import numpy as np

def cohens_h(p1, p2):
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi1 - phi2

h = cohens_h(baseline_acc, comparison_acc)
# Interpretation: |h| < 0.2 = small, 0.5 = medium, 0.8 = large
```

**Number Needed to Treat (NNT)**:
```python
# If positive effect
nnt = 1 / (comparison_acc - baseline_acc)
# Interpretation: Need to use summary for NNT cases to get 1 additional correct prediction
```

**Recommendation**: ⭐ **Yes, add Cohen's h to results table**

---

## Recommended Statistical Testing Strategy

### Essential (Keep Current Approach)
✅ **McNemar's exact test** - Core hypothesis test
✅ **Paired bootstrap CIs** - Effect size with uncertainty
✅ **Current reporting** - Transparent tables with all results

### High Priority Additions
⭐ **Wilcoxon signed-rank test** - Analyze rating magnitudes (not just binary)
⭐ **FDR correction** - Sensitivity analysis for multiple comparisons
⭐ **Cohen's h** - Standardized effect sizes
⭐ **Stratified analysis** - Test effects separately for violation vs. no-violation cases

### Optional (If Reviewers Request)
⚠️ **Mixed-effects models** - If asked about modeling case-level variance
⚠️ **Bayesian analysis** - If asked about probability of improvement

### Not Recommended
❌ **Permutation tests** - McNemar's is already exact
❌ **Unpaired tests** - Ignores paired structure
❌ **T-tests on proportions** - Incorrect for binary outcomes

---

## Specific Concerns Addressed

### "Are we throwing away information by binarizing ratings?"

**Yes, somewhat.** Consider:

1. **Add Wilcoxon signed-rank test on raw ratings**
   - Tests if median rating shifts
   - Example: "Are models rating violations more/less severely with summaries?"

2. **Report mean rating differences**
   - Even without hypothesis test, descriptive statistics are valuable
   - "Summaries increased mean rating by 0.3 points (from 2.1 → 2.4)"

### "Should we correct for multiple comparisons?"

**Depends on research goals:**

- **Exploratory analysis**: No correction needed, but acknowledge inflated Type I error
- **Confirmatory claims**: Apply FDR or Bonferroni correction

**Recommendation**: Report both raw and FDR-adjusted p-values

### "Are our bootstrap CIs valid?"

**Yes, with caveats:**

✅ **Correct approach** for paired differences
✅ **10,000 iterations** is sufficient for stable estimates
✅ **Percentile method** is appropriate for symmetric distributions

⚠️ **Consider BCa (bias-corrected accelerated) bootstrap** for skewed differences
- More accurate for small samples or skewed data
- Implemented in `scipy.stats.bootstrap` with `method='BCa'`

### "What about the 10 samples per case-article pair?"

**Current approach**: Average 10 samples → single prediction per condition

**Alternative**: Treat as hierarchical data
- Case-article pairs nested within evaluator-summary combinations
- Account for within-pair sampling variance

**Recommendation**: Current approach is fine for main analysis. Hierarchical model is overkill.

---

## Bottom Line

**Your current statistical approach is appropriate and defensible.**

**Key strengths**:
- McNemar's test is the standard for paired binary data
- Bootstrap CIs provide interpretable effect sizes
- Transparent reporting of all comparisons

**Recommended additions** (in priority order):
1. ⭐ **Wilcoxon signed-rank test** on rating magnitudes
2. ⭐ **FDR correction** for multiple comparisons
3. ⭐ **Cohen's h** effect sizes
4. ⭐ **Stratified analysis** by violation vs. no-violation

**For reviewers**: These tests are standard in medical/clinical research for paired proportions. McNemar's test appears in thousands of published papers analyzing diagnostic test comparisons, treatment effects, and model evaluations with paired binary outcomes.
