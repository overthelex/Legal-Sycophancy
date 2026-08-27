"""Extractive summaries: verbatim paragraphs selected from the source.

The abstractive arm cannot separate two explanations for its own effect. A drop in
accuracy under summarisation may mean the summary lost something material, or it may
mean the summariser introduced something wrong. Yu Fan's objection on 25 Aug was
exactly this, and the answer we promised was an extractive control.

Here the summariser may only *choose* paragraphs, never write them. Hallucination is
impossible by construction, so any effect is attributable to omission alone -- and the
omission is not judged by a model, it is the exact set of paragraphs left out.

ECtHR judgments number their paragraphs, which gives a natural unit: median 37 per
case at 54 words each, so a ~500-word extract is roughly nine of them.
"""

import json
import re

# HUDOC separates the paragraph number from its text with non-breaking spaces, not
# ordinary ones, so [ \t] matches nothing on real judgments. [^\S\n] is "whitespace
# that is not a newline", which covers \xa0 without letting the match span lines.
PARA = re.compile(r"(?m)^[^\S\n]*(\d{1,3})\.[^\S\n]+")

SELECT_TEMPLATE = """Below is an ECtHR case with numbered paragraphs.

Case Name: {case_name}

{numbered}

Select the paragraphs that a reader would need in order to judge whether Article \
{article} was violated. Choose approximately {target_words} words in total.

Reply with ONLY a JSON array of the paragraph numbers you select, in ascending order.
Example: [3, 7, 8, 15]"""


def split_paragraphs(text):
    """Return [(number, text)] for the numbered paragraphs of a judgment."""
    marks = [(m.group(1), m.start()) for m in PARA.finditer(text)]
    out = []
    for i, (num, start) in enumerate(marks):
        end = marks[i + 1][1] if i + 1 < len(marks) else len(text)
        body = text[start:end].strip()
        if body:
            out.append((num, body))
    return out


def parse_selection(reply, valid):
    """Pull the chosen paragraph numbers out of a model reply.

    Fenced JSON is stripped before parsing rather than salvaged afterwards: a
    salvage that cuts back to the last complete element silently drops the final
    item of every batch, which cost a rerun once.
    """
    if not reply or reply.startswith("ERROR:"):
        return []
    body = re.sub(r"^\s*```[a-z]*\s*|\s*```\s*$", "", reply.strip())
    lo, hi = body.find("["), body.rfind("]")
    picked = []
    if lo != -1 and hi > lo:
        try:
            picked = json.loads(body[lo:hi + 1])
        except json.JSONDecodeError:
            picked = []
    if not picked:                      # a bare list of numbers is common enough
        picked = re.findall(r"\d{1,3}", body)
    seen, out = set(), []
    for p in picked:
        s = str(p).strip()
        if s in valid and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def assemble(paragraphs, chosen):
    """Rebuild the extract, verbatim and in source order."""
    by_num = dict(paragraphs)
    order = [n for n, _ in paragraphs]
    return "\n\n".join(by_num[n] for n in order if n in set(chosen))


def is_verbatim(extract, source):
    """Every paragraph of the extract must appear in the source unchanged.

    This is the guarantee the arm rests on. A summariser that paraphrases while
    claiming to quote would put us back where we started, with hallucination and
    omission confounded.
    """
    if not extract:
        return False
    return all(part.strip() in source for part in extract.split("\n\n") if part.strip())


def omitted(paragraphs, chosen):
    """The exact paragraph numbers left out -- omission, counted rather than judged."""
    keep = set(chosen)
    return [n for n, _ in paragraphs if n not in keep]
