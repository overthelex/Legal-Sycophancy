"""Atomic coverage: does the summary keep the facts the Court actually relied on?

Yu Fan's objection, restated in his own words on 28 Aug: the worry is not that a
summariser invents things, it is "if we can ensure all legally relevant information
are kept in the summary, which is very hard to verify. One atomic fact deemed
relevant to party A may be considered irrelevant to the opponent B".

That is the right objection and the extractive arm does not answer it. It rules out
hallucination, which he was not asking about.

The reason relevance is hard to verify is that someone has to decide what counts.
This module does not decide. The Court does: when its assessment writes "see
paragraph 18 above" it is naming the fact it rested on, so the set of relied-upon
paragraphs is given rather than judged. Coverage is then measurable in two senses:

* **overall coverage** -- claims from anywhere in the record that survive the summary;
* **relied-upon coverage** -- claims from paragraphs the Court back-references.

The second is the headline, because it is the one whose relevance is not our opinion.

Three constraints hold the measurement honest.

* Claims are extracted from the **source**, never from the summary. Extracting from
  the summary would measure precision -- whether what it says is supported -- which
  is the hallucination question we already answered.
* The extractor and the verifier are **not the summariser**. A model grading the
  coverage of its own summary is the same defect as a judge reading its own writing,
  which is what put the appendix's summariser matrix off in the first place.
* The verifier sees the claim and the summary and **not the source**, so it cannot
  fall back on the judgment it may remember and score a claim supported because it
  is true rather than because the summary carries it.
"""

import json
import re

# Claims are extracted per paragraph so each one keeps the paragraph number it came
# from. Without that anchor there is no way to say which claims were relied upon,
# and relied-upon coverage is the whole point.
EXTRACT_TEMPLATE = """Below is one numbered paragraph from the facts of a court judgment.

{paragraph}

List the atomic factual claims it makes. An atomic claim states one fact and can be \
checked on its own: a date, a place, an act, a measurement, a decision, a duration. \
Do not include legal conclusions, characterisations, or anything the paragraph does \
not state.

List at most {max_claims}, choosing the ones a reader would need in order to follow \
what happened. One paragraph split into fifty claims is one fact shredded, not fifty \
facts, and it makes paragraphs of different lengths incomparable.

Reply with ONLY a JSON array of strings, each a single claim in one short sentence. \
If the paragraph states no checkable fact, reply with []."""

VERIFY_TEMPLATE = """Below is a summary of a court case, then a numbered list of factual claims.

SUMMARY:
{summary}

CLAIMS:
{claims}

For each claim, decide whether the summary states it or clearly implies it. Judge \
only against the summary above. Do not use anything you know about the case: a claim \
that is true but absent from the summary is NOT supported.

Reply with ONLY a JSON array of objects, one per claim, in the same order, each \
{{"i": <claim number>, "supported": true or false}}. Return exactly {n} objects."""


def parse_claims(reply):
    """Pull the claim list out of an extraction reply."""
    if not reply or reply.startswith("ERROR:"):
        return []
    body = re.sub(r"^\s*```[a-z]*\s*|\s*```\s*$", "", reply.strip())
    lo, hi = body.find("["), body.rfind("]")
    if lo == -1 or hi <= lo:
        return []
    try:
        items = json.loads(body[lo:hi + 1])
    except json.JSONDecodeError:
        return []
    out = []
    for item in items:
        if isinstance(item, str) and item.strip():
            out.append(" ".join(item.split()))
    return out


def parse_verdicts(reply, n):
    """Pull `n` support decisions out of a verification reply.

    Returns None rather than a short list when the model returns the wrong number of
    objects. A truncated array silently scores its missing tail as unsupported, which
    would report the summary as covering less than it does -- the same failure mode as
    the fenced-JSON salvage that once dropped the last item of every batch.
    """
    if not reply or reply.startswith("ERROR:"):
        return None
    body = re.sub(r"^\s*```[a-z]*\s*|\s*```\s*$", "", reply.strip())
    lo, hi = body.find("["), body.rfind("]")
    if lo == -1 or hi <= lo:
        return None
    try:
        items = json.loads(body[lo:hi + 1])
    except json.JSONDecodeError:
        return None
    by_index = {}
    for item in items:
        if not isinstance(item, dict) or "supported" not in item:
            continue
        try:
            i = int(item.get("i"))
        except (TypeError, ValueError):
            continue
        by_index[i] = bool(item["supported"])
    # accept either 1-based or 0-based indexing, but demand a complete answer
    for base in (1, 0):
        if all(base + k in by_index for k in range(n)):
            return [by_index[base + k] for k in range(n)]
    return None


def number_claims(claims):
    """Render claims for the verification prompt, 1-based to match the reply format."""
    return "\n".join("%d. %s" % (i, c) for i, c in enumerate(claims, 1))


def coverage(records):
    """Coverage over a list of {"supported": bool, "relied_upon": bool} records.

    Returns (overall, relied_upon, n_overall, n_relied). Either rate is None when it
    rests on no claims, which is honest: a judgment whose assessment back-references
    nothing supports no statement about relied-upon coverage, and averaging a zero in
    would quietly drag the number down.
    """
    total = [r for r in records if r is not None]
    relied = [r for r in total if r.get("relied_upon")]
    rate = lambda rows: (sum(1 for r in rows if r["supported"]) / len(rows)) if rows else None
    return rate(total), rate(relied), len(total), len(relied)
