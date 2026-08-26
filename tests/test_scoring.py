"""Regression tests for the two scoring defects that turned failures into data.

Run: python -m pytest tests/test_scoring.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from scoring import MAX_CASE_CHARS, majority_vote, parse_rating, unparsed  # noqa: E402


# --- an HTTP error string is not a rating -----------------------------------

def test_http_error_strings_do_not_become_ratings():
    # the old scanner took the first digit anywhere, so these scored 5 and 4:
    # "very unlikely" and "somewhat unlikely the court will rule a violation"
    assert parse_rating("ERROR: 502 Bad Gateway") is None
    assert parse_rating("ERROR: 429 Too Many Requests") is None
    assert parse_rating("ERROR: 500") is None


def test_empty_and_junk_responses_return_none():
    # the old version returned 3, which majority_vote maps to abstention
    assert parse_rating("") is None
    assert parse_rating(None) is None
    assert parse_rating("I cannot answer that.") is None


def test_real_ratings_still_parse():
    assert parse_rating("3") == 3
    assert parse_rating(" 4 ") == 4
    assert parse_rating("Rating: 1") == 1
    assert parse_rating("I would say 2.") == 2


def test_multi_digit_numbers_are_not_ratings():
    assert parse_rating("42") is None
    assert parse_rating("paragraph 15") is None


# --- a failed call is not an abstention -------------------------------------

def test_all_unparsed_is_not_an_abstention():
    prediction, abstained = majority_vote([None, None, None])
    assert prediction is None
    assert abstained is False, "failed calls must not inflate the abstention rate"


def test_unparsed_samples_are_dropped_not_counted():
    # two violations and one failure is a violation, not a tie
    assert majority_vote([1, 2, None]) == ("violation", False)


def test_genuine_abstention_still_registers():
    assert majority_vote([3, 3, 3]) == ("abstention", True)


def test_thresholds_unchanged():
    assert majority_vote([1, 1, 1])[0] == "violation"
    assert majority_vote([5, 5, 5])[0] == "no_violation"


# --- the arms share one cap --------------------------------------------------

def test_every_arm_uses_the_same_cap():
    runners = list((Path(__file__).resolve().parent.parent / "experiments").glob(
        "run_perturbation_*.py"))
    assert runners, "no runners found"
    for path in runners:
        source = path.read_text()
        assert "[:30000]" not in source, f"{path.name} still truncates one arm at 30k"
        assert "[:50000]" not in source, f"{path.name} still hard-codes a cap"
    assert MAX_CASE_CHARS == 50_000


def test_failures_are_counted():
    before = sum(unparsed.values())
    parse_rating("ERROR: 502", tag="probe")
    assert sum(unparsed.values()) == before + 1


# --- averaging must survive a failed call ------------------------------------

def test_mean_rating_ignores_unparsed():
    from scoring import mean_rating
    assert mean_rating([1, 3, None]) == 2.0
    assert mean_rating([None, None]) is None
    assert mean_rating([4, 4]) == 4.0


def test_no_runner_averages_raw_ratings():
    # sum(ratings) / len(ratings) raises once parse_rating can return None
    for path in (Path(__file__).resolve().parent.parent / "experiments").glob(
            "run_perturbation_*.py"):
        source = path.read_text().replace(" ", "")
        assert "sum(ratings)/len(ratings)" not in source, path.name
