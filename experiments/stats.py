"""Paired tests and multiplicity control for the perturbation arms.

The run produces a family of McNemar tests: every (model, arm, variant) compared
against that model's baseline. Reporting the raw p-values would be the third time
this project drew a conclusion that a correction removes -- on the 14 Aug review,
BH kept all of RQ3 but dropped RQ1 Opus (q=0.096) and RQ2 gpt-5.6 factual (q=0.084).

Accuracy is reported alongside balanced accuracy because the corpus leans one way:
84% of ECtHR verdicts in the static set were violations, so a model that always
answers "violation" scores 84% without reading anything.
"""

from math import comb


def mcnemar_exact(n01, n10):
    """Two-sided exact McNemar test on the discordant pairs.

    n01 and n10 are the counts that changed in each direction; concordant pairs
    carry no information about the change and are excluded by construction.
    Returns (n_discordant, p_value).
    """
    n = n01 + n10
    if n == 0:
        return 0, 1.0
    tail = sum(comb(n, k) for k in range(min(n01, n10) + 1))
    return n, min(2.0 * tail / (2 ** n), 1.0)


def benjamini_hochberg(pvalues):
    """BH-adjusted q-values, returned in the order the p-values were given.

    Step-up over the sorted p-values with the running minimum enforced, so the
    q-values stay monotone in p.
    """
    m = len(pvalues)
    if m == 0:
        return []
    ascending = sorted(range(m), key=lambda i: pvalues[i])
    q = [1.0] * m
    running = 1.0
    for rank, i in enumerate(reversed(ascending), start=1):
        position = m - rank + 1
        running = min(running, pvalues[i] * m / position)
        q[i] = running
    return q


def balanced_accuracy(rows, label_key="violation_label", pred_key="prediction"):
    """Mean of the per-class recalls, or None when a class is absent.

    Rows whose prediction is None are counted as wrong rather than dropped: a
    failed call is not a free pass, and dropping them would quietly shrink the
    denominator of whichever class failed more often.
    """
    classes = {r[label_key] for r in rows}
    if len(classes) < 2:
        return None
    recalls = []
    for cls in sorted(classes):
        of_class = [r for r in rows if r[label_key] == cls]
        recalls.append(sum(1 for r in of_class if r[pred_key] == cls) / len(of_class))
    return sum(recalls) / len(recalls)


def flip_direction(reference, prediction):
    """How a judgment moved against its reference, or None when it did not.

    Recorded per row rather than derived later because the aggregate flip rate
    hides the asymmetry: the same rate can mean drift toward "violation" or away
    from it, and those are different findings.
    """
    if reference is None or prediction is None or reference == prediction:
        return None
    return f"{reference}->{prediction}"
