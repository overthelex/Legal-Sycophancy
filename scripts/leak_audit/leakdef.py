"""Canonical definition of a verdict leak, and the cut that removes it.

Everything in this directory imports these two functions so that an audit and a
repair can never disagree about what a leak is. Three separate ad-hoc probes gave
three different leak rates before this was pinned down.

A row leaks when the Court's law section survived verdict removal: text remains
after the ``THE LAW`` header and that remainder carries an assessment or merits
heading. Plain ``Merits`` belongs in the pattern -- under ``THE LAW`` an ECtHR
judgment is structured ``A. Admissibility`` / ``B. Merits`` with the Court's
assessment inside B, so the heading's presence means the reasoning is there.
Dropping it undercounts by roughly a third (140 rows against 186 on
``echr-livehrb-static-2k``).

Two structural traps, both of which produced wrong answers first time round:

* Long Grand Chamber judgments repeat every section heading in a table of
  contents, so the first ``THE LAW`` can sit 1,500 characters into the document
  while the real one is 200,000 characters further down. Always cut at the last
  occurrence, and ignore occurrences inside the table-of-contents zone when a
  later one exists.
* Cutting at the first "The Court's assessment" marker destroys the facts,
  because modern judgments repeat that subheading once per complaint. One case
  kept 3 of its 69 fact paragraphs that way.

A lexical probe -- searching for phrases like "there has been no violation" --
is not a usable substitute. It fires on domestic courts quoted in the facts
("the City Court finds no violation of Article 6") and misses the Court's own
conclusion when it is phrased as "did not fail to fulfil its positive
obligations". ``retention_percentage`` is not a usable filter either: leaking
rows appear at 49% and 66% retention.
"""

import re

LAW = re.compile(r"(?:^|\n)\s*(?:[IVX]+\.\s*)?THE LAW\s*(?:\n|$)", re.I | re.M)
FACTS = re.compile(r"(?:^|\n)\s*(?:[IVX]+\.\s*)?THE FACTS\s*(?:\n|$)", re.I | re.M)
ASSESSMENT = re.compile(
    r"The Court.s assessment|The Court.s evaluation|"
    r"Application of (?:the|these) principles|Merits",
    re.I,
)

TOC_ZONE = 12_000   # headings repeated in a table of contents live near the top
MIN_TAIL = 5_000    # shorter remainders are section stubs, not retained reasoning


def _law_positions(text, use_end=False):
    """Positions of the THE LAW header, with table-of-contents hits dropped."""
    positions = [(m.end() if use_end else m.start()) for m in LAW.finditer(text)]
    if len(positions) > 1:
        later = [p for p in positions if p > TOC_ZONE]
        if later:
            positions = later
    return positions


def leaking(text):
    """True when the Court's law section survived verdict removal."""
    positions = _law_positions(text, use_end=True)
    if not positions:
        return False
    tail = text[max(positions):]
    return len(tail) > MIN_TAIL and bool(ASSESSMENT.search(tail))


def recut(text):
    """Truncate at the structural facts/law boundary, leaving the facts intact."""
    positions = _law_positions(text)
    if not positions:
        return text
    facts = [m.start() for m in FACTS.finditer(text)]
    if facts:
        positions = [p for p in positions if p > max(facts)] or positions
    return text[:max(positions)].rstrip()
