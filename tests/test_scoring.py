"""Regression tests for the two scoring defects that turned failures into data.

Run: python -m pytest tests/test_scoring.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from scoring import (MAX_CASE_CHARS, NO_VIOLATION_BELOW, VIOLATION_ABOVE,  # noqa: E402
                     majority_vote, parse_rating, unparsed)


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


def test_real_answers_parse():
    assert parse_rating("85") == 85
    assert parse_rating(" 0 ") == 0
    assert parse_rating("100") == 100
    assert parse_rating("I would say 70.") == 70
    assert parse_rating("75%") == 75


def test_numbers_in_a_preamble_are_not_the_answer():
    # anchoring at the end is what stops article and paragraph numbers quoted on
    # the way to the answer from being read as the answer
    assert parse_rating("Article 6 is engaged; my estimate is 30") == 30
    assert parse_rating("Under Article 8 and paragraph 15, I would say 80") == 80


def test_out_of_range_values_are_rejected():
    assert parse_rating("250") is None
    assert parse_rating("Article 8") is None   # no trailing number at all


# --- polarity: high means violation, the opposite of the old 1-5 scale ---------

def test_polarity_high_is_violation():
    # the 1-5 scale ran the other way (1 = very likely a violation); the prompt now
    # asks for the likelihood OF a violation, so the mapping must invert with it
    assert majority_vote([90, 95, 100])[0] == "violation"
    assert majority_vote([0, 5, 10])[0] == "no_violation"


def test_uncertain_band_is_abstention():
    assert majority_vote([50, 50, 50]) == ("abstention", True)
    assert NO_VIOLATION_BELOW < 50 < VIOLATION_ABOVE


def test_band_edges():
    assert majority_vote([VIOLATION_ABOVE] * 3)[0] == "abstention"      # boundary is exclusive
    assert majority_vote([VIOLATION_ABOVE + 1] * 3)[0] == "violation"
    assert majority_vote([NO_VIOLATION_BELOW] * 3)[0] == "abstention"
    assert majority_vote([NO_VIOLATION_BELOW - 1] * 3)[0] == "no_violation"


# --- a failed call is not an abstention -------------------------------------

def test_all_unparsed_is_not_an_abstention():
    prediction, abstained = majority_vote([None, None, None])
    assert prediction is None
    assert abstained is False, "failed calls must not inflate the abstention rate"


def test_unparsed_samples_are_dropped_not_counted():
    # two violations and one failure is a violation, not a tie
    assert majority_vote([90, 80, None]) == ("violation", False)


def test_genuine_abstention_still_registers():
    assert majority_vote([50, 45, 55]) == ("abstention", True)


# --- the arms share one cap --------------------------------------------------

def test_every_arm_uses_the_same_cap():
    runners = list((Path(__file__).resolve().parent.parent / "experiments").glob(
        "run_perturbation_*.py"))
    assert runners, "no runners found"
    for path in runners:
        source = path.read_text()
        assert "[:30000]" not in source, f"{path.name} still truncates one arm at 30k"
        assert "[:50000]" not in source, f"{path.name} still hard-codes a cap"
        assert "rate from 1 to 5" not in source, f"{path.name} still asks for a 1-5 rating"
    assert MAX_CASE_CHARS == 50_000


def test_failures_are_counted():
    before = sum(unparsed.values())
    parse_rating("ERROR: 502", tag="probe")
    assert sum(unparsed.values()) == before + 1


# --- averaging must survive a failed call ------------------------------------

def test_mean_rating_ignores_unparsed():
    from scoring import mean_rating
    assert mean_rating([10, 30, None]) == 20.0
    assert mean_rating([None, None]) is None
    assert mean_rating([80, 80]) == 80.0


def test_no_runner_averages_raw_ratings():
    # sum(ratings) / len(ratings) raises once parse_rating can return None
    for path in (Path(__file__).resolve().parent.parent / "experiments").glob(
            "run_perturbation_*.py"):
        source = path.read_text().replace(" ", "")
        assert "sum(ratings)/len(ratings)" not in source, path.name


# --- the runners must agree with the published set -----------------------------

def test_article_titles_cover_the_published_codes():
    # a code without a title renders as "Article 7 - Article 7" in the prompt
    import re
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    src = (Path(__file__).resolve().parent.parent
           / "experiments" / "run_perturbation_bedrock.py").read_text()
    titles = dict(re.findall(r'"([^"]+)":\s*"([^"]+)"',
                             re.search(r"ARTICLE_TITLES\s*=\s*\{(.*?)\n\}", src, re.S).group(1)))
    published = {"2", "3", "5", "6", "8", "10", "11", "13", "14", "34", "38", "41",
                 "4", "7", "9", "18", "P1-1", "P1-2", "P1-3", "P4-2", "P4-4",
                 "P7-2", "P7-4", "P12-1"}
    assert not published - set(titles), f"no title for {published - set(titles)}"


def test_convention_article_1_is_not_the_protocol_right():
    # the legacy lossy field collapses P1-1 to "1"; the title map must not repeat that
    import re
    src = (Path(__file__).resolve().parent.parent
           / "experiments" / "run_perturbation_bedrock.py").read_text()
    titles = dict(re.findall(r'"([^"]+)":\s*"([^"]+)"',
                             re.search(r"ARTICLE_TITLES\s*=\s*\{(.*?)\n\}", src, re.S).group(1)))
    assert titles["P1-1"] == "Protection of property"
    assert titles["1"] != titles["P1-1"], "Convention Article 1 is not Article 1 of Protocol 1"


def test_summaries_are_shared_across_instances_of_one_case():
    # the summary prompt takes only the case text, so keying the cache on the article
    # would give two instances of one judgment two different summaries -- and bill twice
    for path in (Path(__file__).resolve().parent.parent / "experiments").glob(
            "run_perturbation_*.py"):
        src = path.read_text()
        assert 'case["item_id"] + "_" + case["article"]' not in src, path.name
