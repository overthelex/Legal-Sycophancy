"""Response parsing and vote aggregation, shared by every perturbation runner.

Previously `parse_rating` and `majority_vote` were duplicated in the Bedrock and
OpenAI runners and imported into the vLLM one, so a fix had to be applied twice
and could silently drift. They live here once.

Two scoring defects are fixed here, both of which turned failed API calls into
data rather than into errors:

* `parse_rating` scanned for the first digit anywhere in the response, so the
  HTTP error string ``"ERROR: 502 Bad Gateway"`` scored **5** ("very unlikely
  the court will rule a violation") and ``"ERROR: 429"`` scored **4**.
* When it found no digit at all it returned **3**, which `majority_vote` maps to
  abstention -- so every unparseable response was recorded as the model
  declining to answer, inflating the abstention rate with what were really
  failures. Abstention is a reported metric and is load-bearing in the mitigation
  comparison, which is exactly where an inflated value changes a conclusion.

`parse_rating` now returns ``None`` on failure and never substitutes a default;
`majority_vote` drops unparsed samples and returns a ``None`` prediction when
nothing survived, which scores as incorrect without counting as an abstention.
"""

import re
from collections import Counter

# One cap for every arm. The baseline arm used to read 50,000 characters while
# the summariser only received 30,000, so on the 141-pair set 47.5% of cases
# exceeded the summariser's cap against 17.7% over the baseline's: for roughly a
# third of cases the baseline saw more of the record than the summariser ever
# did, and part of the measured summarisation effect was plain truncation.
MAX_CASE_CHARS = 50_000

# A 0-100 likelihood, anchored at the end of the response so that numbers quoted
# in a preamble (article numbers, paragraph references, HTTP status codes) cannot
# be mistaken for the answer.
#
# The scale replaced a 1-5 rating that collapsed in practice: across 555 samples
# it returned only 1 and 5, never 2-4, which made the abstention band unreachable
# and confidence unmeasurable. On the percentage scale the same models still
# concentrate at the extremes -- only 4 of 555 answers fell in the uncertain band
# -- but that is now a measurement rather than an artefact of the instrument.
RATING = re.compile(r"(?<![0-9.])(\d{1,3})\s*%?\s*[.]?\s*$")

# A trailing number is only the answer if it is not the tail of a citation: a model
# replying "Article 8" would otherwise be scored 8, i.e. a confident no-violation.
CITATION_TAIL = re.compile(r"(?:article|paragraph|protocol|no\.|§)\s*\d{1,3}\s*%?\s*[.]?\s*$", re.I)

VIOLATION_ABOVE = 60    # a violation prediction
NO_VIOLATION_BELOW = 40  # a no-violation prediction; between the two is abstention

# Counts responses that carried no rating. Read it after a run: a non-zero value
# means calls failed, and those rows are not abstentions.
unparsed = Counter()


def parse_rating(response, tag="unknown"):
    """Return the 0-100 likelihood, or None when the response does not carry one."""
    if not response:
        unparsed[tag] += 1
        return None
    text = str(response).strip()
    match = RATING.search(text)
    if match is None or CITATION_TAIL.search(text) or not 0 <= int(match.group(1)) <= 100:
        unparsed[tag] += 1
        return None
    return int(match.group(1))


def majority_vote(ratings):
    """Aggregate samples into (prediction, abstained), ignoring unparsed ones.

    Returns (None, False) when nothing could be parsed: a failed call is not a
    prediction and must not be recorded as an abstention.
    """
    parsed = [r for r in ratings if r is not None]
    if not parsed:
        return None, False
    thresholded = []
    for rating in parsed:
        if rating > VIOLATION_ABOVE:
            thresholded.append("violation")
        elif rating < NO_VIOLATION_BELOW:
            thresholded.append("no_violation")
        else:
            thresholded.append("abstention")
    top = Counter(thresholded).most_common(1)[0]
    return top[0], top[0] == "abstention"


def mean_rating(ratings):
    """Mean of the parsed samples, or None when none parsed.

    `parse_rating` can now return None, so the plain `sum(r) / len(r)` the
    runners used would raise on the first failed call.
    """
    parsed = [r for r in ratings if r is not None]
    return sum(parsed) / len(parsed) if parsed else None
