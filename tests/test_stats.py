import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))

from stats import (balanced_accuracy, benjamini_hochberg,   # noqa: E402
                   flip_direction, mcnemar_exact)


def test_mcnemar_matches_the_exact_binomial():
    # 1 change one way, 9 the other: 2 * (C(10,0) + C(10,1)) / 2^10
    n, p = mcnemar_exact(1, 9)
    assert n == 10
    assert abs(p - 2 * 11 / 1024) < 1e-12


def test_mcnemar_on_a_perfect_split_cannot_exceed_one():
    n, p = mcnemar_exact(5, 5)
    assert n == 10 and p == 1.0


def test_mcnemar_with_no_discordant_pairs_is_not_evidence():
    """Concordant pairs carry no information, so an empty comparison is p=1."""
    assert mcnemar_exact(0, 0) == (0, 1.0)


def test_bh_returns_q_values_in_input_order():
    q = benjamini_hochberg([0.04, 0.01, 0.03])
    assert len(q) == 3
    # smallest p keeps the smallest q
    assert q[1] <= q[2] <= q[0]


def test_bh_on_the_evenly_spaced_family():
    q = benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05])
    assert all(abs(v - 0.05) < 1e-12 for v in q)


def test_bh_is_monotone_and_never_below_the_raw_p():
    ps = [0.001, 0.009, 0.04, 0.2, 0.5, 0.9]
    q = benjamini_hochberg(ps)
    assert all(a <= b + 1e-12 for a, b in zip(q, q[1:]))
    assert all(qi >= pi - 1e-12 for qi, pi in zip(q, ps))


def test_bh_removes_a_marginal_result_from_a_large_family():
    """The 14 Aug review: over 32 tests BH kept RQ3 and dropped two marginals.

    The strong result survives; a p just under 0.05 does not, once it is one of
    eight comparisons rather than one.
    """
    q = benjamini_hochberg([0.001, 0.04, 0.045, 0.05, 0.2, 0.3, 0.5, 0.9])
    assert q[0] < 0.05
    assert q[1] > 0.05 and q[2] > 0.05


def test_bh_of_nothing_is_nothing():
    assert benjamini_hochberg([]) == []


def test_balanced_accuracy_ignores_the_majority_lean():
    """Always answering the majority class scores 0.5, not 0.84."""
    rows = ([{"violation_label": "violation", "prediction": "violation"}] * 84
            + [{"violation_label": "no_violation", "prediction": "violation"}] * 16)
    assert abs(balanced_accuracy(rows) - 0.5) < 1e-12


def test_balanced_accuracy_counts_failed_calls_as_wrong():
    rows = ([{"violation_label": "violation", "prediction": None}] * 2
            + [{"violation_label": "no_violation", "prediction": "no_violation"}] * 2)
    assert abs(balanced_accuracy(rows) - 0.5) < 1e-12


def test_balanced_accuracy_needs_both_classes():
    rows = [{"violation_label": "violation", "prediction": "violation"}]
    assert balanced_accuracy(rows) is None


def test_flip_direction_distinguishes_the_two_directions():
    assert flip_direction("no_violation", "violation") == "no_violation->violation"
    assert flip_direction("violation", "no_violation") == "violation->no_violation"
    assert flip_direction("violation", "violation") is None
    assert flip_direction(None, "violation") is None
    assert flip_direction("violation", None) is None


def test_balanced_accuracy_on_rq3_shaped_rows():
    """RQ3 rows name their two answers separately, not `prediction`.

    The analysis maps them before calling the shared helpers. This is asserted
    because the omission survived one run: the slice it ran on held a single
    class, and balanced_accuracy returns early before it reaches the field.
    """
    rq3 = [{"violation_label": "violation", "original_prediction": "violation",
            "challenged_prediction": "no_violation"},
           {"violation_label": "no_violation", "original_prediction": "no_violation",
            "challenged_prediction": "no_violation"}]
    mapped = [{**r, "prediction": r["challenged_prediction"]} for r in rq3]
    assert balanced_accuracy(mapped) == 0.5      # missed one class, kept the other


def test_balanced_accuracy_needs_the_field_it_reads():
    """A row without the prediction key is a bug, not a None result."""
    import pytest
    rows = [{"violation_label": "violation"}, {"violation_label": "no_violation"}]
    with pytest.raises(KeyError):
        balanced_accuracy(rows)
