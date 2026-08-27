"""The extractive arm rests on one guarantee: every word came from the source."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))

from extractive import (assemble, is_verbatim, omitted,   # noqa: E402
                        parse_selection, split_paragraphs)

JUDGMENT = """PROCEDURE

1.  The case originated in an application against Croatia.

2.  The applicant was born in 1970 and lives in Zagreb.

THE FACTS

3.  On 4 May 2015 the applicant was placed in cell no. 7, which measured
11 square metres and held four detainees.

10.  The applicant complained to the prison governor.

11.  The Constitutional Court dismissed the complaint on 2 March 2017."""


REAL = ("PROCEDURE\n\n1.\u00a0\u00a0The case originated in an application against Croatia.\n\n"
        "2.\u00a0\u00a0The applicant was born in 1970.\n\n"
        "3.\u00a0\u00a0The cell measured 11 square metres.\n\n"
        "4.\u00a0\u00a0The Constitutional Court dismissed the complaint.")


def test_real_hudoc_separator_is_a_non_breaking_space():
    """HUDOC writes "1.\\xa0\\xa0The case...", not "1. The case...".

    A pattern using [ \\t] matches nothing on any real judgment, which a test
    written with ordinary spaces cannot catch -- this one did not, and the pilot
    failed on 12 of 12 cases before it was found.
    """
    assert [n for n, _ in split_paragraphs(REAL)] == ["1", "2", "3", "4"]


def test_numbered_paragraphs_are_found():
    paras = split_paragraphs(JUDGMENT)
    assert [n for n, _ in paras] == ["1", "2", "3", "10", "11"]
    assert "11 square metres" in dict(paras)["3"]


def test_headings_are_not_mistaken_for_paragraphs():
    assert "PROCEDURE" not in dict(split_paragraphs(JUDGMENT)).values()


def test_selection_survives_fenced_json():
    valid = {"1", "2", "3", "10", "11"}
    for reply in ('```json\n[3, 10, 11]\n```', '[3, 10, 11]', 'I choose 3, 10 and 11.'):
        assert parse_selection(reply, valid) == ["3", "10", "11"], reply


def test_selection_keeps_the_last_element():
    """A salvage that cuts back to the last complete element loses the final item."""
    assert parse_selection("```\n[1, 2, 3, 10, 11]\n```", {"1", "2", "3", "10", "11"}) == \
        ["1", "2", "3", "10", "11"]


def test_selection_rejects_numbers_that_are_not_paragraphs():
    assert parse_selection("[3, 99, 1970]", {"1", "2", "3"}) == ["3"]


def test_selection_of_a_failed_call_is_empty():
    assert parse_selection("ERROR: 429 rate limited", {"1", "2"}) == []
    assert parse_selection("", {"1", "2"}) == []


def test_assembled_extract_is_verbatim_and_in_source_order():
    paras = split_paragraphs(JUDGMENT)
    text = assemble(paras, ["11", "3"])          # asked out of order
    assert text.index("11 square metres") < text.index("Constitutional Court")
    assert is_verbatim(text, JUDGMENT)


def test_a_paraphrase_is_not_verbatim():
    """The guarantee the arm rests on, so it is checked rather than assumed."""
    assert not is_verbatim("The applicant was held in an 11 m2 cell.", JUDGMENT)


def test_empty_extract_is_not_verbatim():
    assert not is_verbatim("", JUDGMENT)


def test_omission_is_counted_exactly_not_estimated():
    paras = split_paragraphs(JUDGMENT)
    assert omitted(paras, ["3", "10", "11"]) == ["1", "2"]
    assert omitted(paras, []) == ["1", "2", "3", "10", "11"]
