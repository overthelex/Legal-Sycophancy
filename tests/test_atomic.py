"""Tests for the atomic coverage instrument.

The parsing tests are here because both parsers have a failure mode that costs a
rerun rather than an error: a claim list that silently loses its last item, and a
verdict list that is shorter than the claim list it answers and therefore scores
the missing tail as unsupported.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "experiments"))

from atomic import (VERIFY_TEMPLATE, coverage, number_claims,  # noqa: E402
                    parse_claims, parse_verdicts)


def test_parse_claims_survives_fencing_and_prose():
    assert parse_claims('["a", "b"]') == ["a", "b"]
    assert parse_claims('```json\n["a", "b"]\n```') == ["a", "b"]
    assert parse_claims('Here you go:\n["a", "b"]\nHope that helps.') == ["a", "b"]
    # whitespace inside a claim is normalised so the same fact does not appear twice
    assert parse_claims('["the  applicant\\n was  born"]') == ["the applicant was born"]
    assert parse_claims("[]") == []
    assert parse_claims("ERROR: 502 Bad Gateway") == []
    assert parse_claims("") == []


def test_parse_verdicts_demands_a_complete_answer():
    reply = '[{"i":1,"supported":true},{"i":2,"supported":false},{"i":3,"supported":true}]'
    assert parse_verdicts(reply, 3) == [True, False, True]

    # 0-based indexing is accepted, since models drift between the two
    zero = '[{"i":0,"supported":true},{"i":1,"supported":false}]'
    assert parse_verdicts(zero, 2) == [True, False]

    # a short answer is a failure, not a run of False: scoring the missing tail as
    # unsupported would under-report coverage and look like a finding
    short = '[{"i":1,"supported":true},{"i":2,"supported":false}]'
    assert parse_verdicts(short, 3) is None

    # so is a gap in the middle
    gap = '[{"i":1,"supported":true},{"i":3,"supported":true}]'
    assert parse_verdicts(gap, 3) is None

    assert parse_verdicts("ERROR: 429", 2) is None
    assert parse_verdicts("no json here", 2) is None


def test_coverage_separates_relied_upon_and_refuses_to_invent_a_rate():
    records = [
        {"supported": True, "relied_upon": True},
        {"supported": False, "relied_upon": True},
        {"supported": True, "relied_upon": False},
        {"supported": True, "relied_upon": False},
    ]
    overall, relied, n_all, n_relied = coverage(records)
    assert (n_all, n_relied) == (4, 2)
    assert overall == 0.75
    assert relied == 0.5

    # a judgment whose assessment back-references nothing yields no relied-upon rate,
    # rather than a zero that would drag an average down
    none_relied = [{"supported": True, "relied_upon": False}]
    assert coverage(none_relied)[1] is None
    assert coverage([])[0] is None


def test_verification_prompt_pins_the_count_and_forbids_outside_knowledge():
    claims = ["The applicant was born in 1970.", "He was detained on 4 May 2015."]
    prompt = VERIFY_TEMPLATE.format(summary="A summary.", claims=number_claims(claims),
                                    n=len(claims))
    assert "Return exactly 2 objects" in prompt
    assert "1. The applicant was born in 1970." in prompt
    assert "2. He was detained on 4 May 2015." in prompt
    # the verifier must not fall back on the judgment it may remember
    assert "Do not use anything you know about the case" in prompt
    # and it must not be handed the source, or it would score truth rather than coverage
    assert "{source}" not in VERIFY_TEMPLATE


def test_checkpoint_hands_back_what_it_recorded():
    """`done` says a unit was paid for; `get` says what it bought.

    The claim-extraction pass needs the claims themselves on resume, not merely the
    knowledge that it once had them, so a checkpoint that only answers `done` would
    make it pay twice.
    """
    import tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments"))
    from checkpoint import Checkpoint

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "claims.jsonl")
        ckpt = Checkpoint(path)
        key = ckpt.key("extract", "001-1", "18")
        ckpt.record(key, {"claims": ["a", "b"]})

        reopened = Checkpoint(path)
        assert reopened.done(key)
        assert reopened.get(key) == {"claims": ["a", "b"]}
        assert reopened.get(ckpt.key("extract", "001-1", "19")) is None
