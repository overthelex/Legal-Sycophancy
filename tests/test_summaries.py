"""Summaries are a shared artifact, and the runners must not make their own.

The invariants below are asserted against the runner sources rather than checked by
reading them. Reading missed a fourth runner once, and missed two files' worth of
unpatched result rows another time.
"""

import json
import os
import re
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))

from summaries import coverage, is_usable, load_summaries   # noqa: E402

EXPERIMENTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments")
RUNNERS = sorted(f for f in os.listdir(EXPERIMENTS)
                 if f.startswith("run_perturbation_") and f.endswith(".py"))


def source(name):
    with open(os.path.join(EXPERIMENTS, name)) as f:
        return f.read()


def test_there_are_runners_to_check():
    # Guards against the invariants below passing vacuously on an empty list.
    assert len(RUNNERS) >= 4


@pytest.mark.parametrize("runner", RUNNERS)
def test_no_runner_summarises(runner):
    """Summarising with the judge model cost 8x and had each model grade itself."""
    assert "SUMMARY_TEMPLATE" not in source(runner)


@pytest.mark.parametrize("runner", RUNNERS)
def test_summarization_arm_is_given_its_summaries(runner):
    sig = re.search(r"def run_summarization\(([^)]*)\)", source(runner))
    assert sig, f"{runner} has no run_summarization"
    assert "summaries" in sig.group(1)


@pytest.mark.parametrize("runner", RUNNERS)
def test_runner_loads_shared_summaries(runner):
    assert "load_summaries_for" in source(runner)


@pytest.mark.parametrize("runner", RUNNERS)
def test_challenge_uses_the_same_scale_as_the_question(runner):
    """RQ3 asked for a number 1-5 after asking for a percentage.

    A model that had not changed its mind still had to change its answer, so the
    reported change rate measured the rescaling rather than any reconsideration.
    """
    src = source(runner)
    if "RECONSIDERATION_PROMPT =" not in src:
        return                      # vllm imports it from the bedrock runner
    prompt = re.search(r"RECONSIDERATION_PROMPT = (.*?)\n\n", src, re.S).group(1)
    assert "1-5" not in prompt
    assert "0 to 100" in prompt


def test_failed_calls_are_not_mistaken_for_summaries():
    assert is_usable("The applicant complained of conditions of detention.")
    assert not is_usable("ERROR: connection reset")
    assert not is_usable("")
    assert not is_usable(None)


def test_load_accepts_the_wrapped_file():
    blob = {"summarizer": "x-ai/grok-4.6", "versions": 3,
            "summaries": {"001-1": ["a", "b", "c"]}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(blob, f)
    summaries, meta = load_summaries(f.name)
    assert summaries == {"001-1": ["a", "b", "c"]}
    assert meta["summarizer"] == "x-ai/grok-4.6"
    assert "summaries" not in meta
    os.unlink(f.name)


def test_load_accepts_a_bare_mapping():
    """The shape the runners wrote before summarisation was split out."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"001-1": ["a"]}, f)
    summaries, meta = load_summaries(f.name)
    assert summaries == {"001-1": ["a"]}
    assert meta == {}
    os.unlink(f.name)


def test_coverage_counts_failures_as_missing():
    summaries = {"a": ["real summary"], "b": ["ERROR: rate limited"]}
    cases = [{"item_id": "a"}, {"item_id": "b"}, {"item_id": "c"}]
    assert coverage(summaries, cases) == (1, 3)


def test_digest_identifies_the_file_not_its_name():
    from summaries import file_digest
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as a:
        json.dump({"summaries": {"x": ["one"]}}, a)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as b:
        json.dump({"summaries": {"x": ["two"]}}, b)
    assert file_digest(a.name) != file_digest(b.name)
    assert file_digest(a.name) == file_digest(a.name)
    os.unlink(a.name); os.unlink(b.name)


# --- the summariser can recognise a case and supply the verdict -----------------

def test_recalled_verdict_is_caught():
    """Real examples from the first frozen build, whose sources had none of this."""
    from summaries import asserts_outcome
    source = "The applicant complained about the length of the proceedings."
    for leaked in [
        "In its judgment of 10 March 2020 the Court found no violation.",
        "Consequently the Court found no violation of Article 8.",
        "The Court found a violation of Article 6 § 1: the restriction was disproportionate.",
        "It therefore held that there had been no violation of Article 6.",
    ]:
        assert asserts_outcome(leaked, source), leaked


def test_an_outcome_the_source_reports_is_not_a_leak():
    """Domestic and prior-judgment outcomes belong to the factual record."""
    from summaries import asserts_outcome
    source = ("The Constitutional Court found no violation of the applicant's right "
              "to a fair hearing and dismissed the appeal.")
    summary = "The domestic court found no violation and the applicant appealed."
    assert not asserts_outcome(summary, source)


def test_an_ordinary_summary_is_left_alone():
    from summaries import asserts_outcome
    source = "The applicant was detained in a cell of 2.5 square metres for 14 months."
    summary = ("The applicant was held in a cell affording 2.5 square metres of personal "
               "space for fourteen months and complained under Article 3.")
    assert not asserts_outcome(summary, source)


def test_missing_input_is_handled_the_safe_way():
    from summaries import asserts_outcome
    assert not asserts_outcome(None, "text")
    assert not asserts_outcome("", "text")
    # No source to justify it means nothing justifies it, so it is a leak.
    assert asserts_outcome("The Court found a violation.", None)
