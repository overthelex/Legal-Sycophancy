#!/usr/bin/env python3
"""Build the back-reference validation sheet.

Materiality does not have to be judged by the annotator. The Court states it: when
its assessment says "see paragraph 18 above", it is naming the fact it relied on.
That is what makes non-experts sufficient here, which is the point Terry raised on
25 Aug and Yu Fan endorsed with the jury argument -- the annotator validates a
mapping, they do not decide what is legally material.

So each item shows one paragraph of the Court's reasoning and one fact paragraph it
cites, and asks only whether the reasoning really rests on that fact.

    python scripts/build_annotation.py --full-texts full_texts.csv \
        --cases data/processed/livehrb_1k.json --out annotation

Writes annotation_sheet.csv, annotation_INSTRUCTIONS.md and annotation_key.csv.
The key holds the control answers and must not go to the annotators.
"""

import argparse, csv, json, os, random, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from extractive import split_paragraphs   # noqa: E402

csv.field_size_limit(10 ** 9)

LAW = re.compile(r"(?:^|\n)\s*(?:[IVX]+\.\s*)?THE LAW\s*(?:\n|$)", re.I | re.M)

# Everything before THE LAW is not "the facts": it also holds the statute quotations
# and international material. 26% of the Court's back-references point there. Asking
# whether reasoning "rests on" a provision the Court itself quoted is a different and
# near-trivial question, so those paragraphs are excluded rather than mixed in.
#
# The heading is not stable enough to name: RELEVANT DOMESTIC LAW, RELEVANT DOMESTIC
# LAW AND PRACTICE, RELEVANT LEGAL FRAMEWORK (2019 on), RELEVANT INTERNATIONAL
# MATERIAL, COMPARATIVE LAW and a dozen more all occur. Match the family and take the
# earliest, since the legal material always follows the facts.
FRAMEWORK = re.compile(
    r"(?:^|\n)[ \t]*(?:[IVX]+\.[ \t]*)?"
    r"(?:RELEVANT|COMPARATIVE|INTERNATIONAL|EUROPEAN)[A-Z ,\-–&']{4,70}[ \t]*(?:\n|$)")

HUDOC = "https://hudoc.echr.coe.int/eng?i={item_id}"

# "see paragraph 18 above" is a back-reference to this judgment. A bare "§ 137" is
# usually a citation to another case ("Muršić, cited above, § 137") -- requiring the
# word "above" cuts the share of fact paragraphs marked as relied upon from 34% to
# 18%, which is the difference between a mapping worth validating and noise.
BACKREF = re.compile(
    r"(?:see\s+)?paragraphs?\s+(\d{1,3})(?:\s*(?:[-–]|to|and|,)\s*(\d{1,3}))?"
    r"(?:\s*(?:[-–]|and|,)\s*(\d{1,3}))?\s+above", re.I)

INSTRUCTIONS = """# Validating what the Court relied on

Thank you for helping. **No legal training is needed.** You are not deciding what is
legally important -- the Court already did that. You are checking whether our
automatic extraction read it correctly.

## What you have

One spreadsheet file, `{fname}`, with about {n} rows. Open it in Excel, Numbers or
Google Sheets. Fill in the **`label`** column, save, and send the file back. Leave
every other column as it is.

Expect roughly an hour. Most rows take well under a minute.

## The task

Each row shows two pieces of text from the same judgment:

- **`reasoning`** -- one paragraph from the section where the Court explains its
  decision. Somewhere in it is a phrase like *"see paragraph 18 above"*.
- **`cited_fact`** -- the paragraph our tool believes the Court was pointing at. Its
  number is in `cited_number`.
- **`full_judgment`** -- a link to the whole judgment on the Court's own site, if you
  ever want the surrounding context. You are not expected to open it for every row.

Your question is only this:

> **Does that reasoning actually rest on the fact shown?**

| `label` | when |
|---|---|
| `yes` | the reasoning uses that fact, or clearly refers to it |
| `no` | the fact is not what the reasoning is resting on |
| `unclear` | you genuinely cannot tell |

Anything odd about a row goes in `notes` -- wrong-looking numbers, text that stops
mid-sentence, anything. Those notes are useful to us even when the label is easy.

One thing that is **not** odd: a paragraph containing `...`. Those are the Court's own
abridgements, mostly where it quotes a long statute and skips the parts that do not
matter. We reproduce the text exactly as published, so nothing has been cut by us.

## The distinction that matters most

Both texts always come from the same case, so almost everything will look *related*.
Related is not the question. The question is whether the reasoning **leans on** that
particular fact.

### `yes`

> **reasoning:** "The applicant was held in a cell affording him 2.5 square metres of
> personal space (see paragraph 18 above), which is below the minimum standard."
>
> **cited_fact (18):** "On 4 May 2015 the applicant was placed in cell no. 7, which
> measured 11 square metres and held four detainees."

The reasoning uses the space figure and that is where it comes from.

### `no`

> **reasoning:** "In January 2007 the applicant underwent a DNA test which excluded
> his paternity (see paragraph 31 above). It was nevertheless open to him to ask the
> prosecutor to bring an action on his behalf."
>
> **cited_fact (6):** "On 27 October 1995 R, with whom the applicant had been in a
> relationship, gave birth to a son."

Same story, same people, and the birth is why the case exists at all -- but the
reasoning is resting on the DNA test and the prosecutor route, not on the birth.
Background is not the same as relied upon.

### `unclear`

> **reasoning:** "The domestic courts examined the evidence and gave reasons (see
> paragraphs 24 to 29 above)."
>
> **cited_fact (27):** "The hearing was adjourned until 3 March."

The Court pointed at a run of six paragraphs and we are showing you one of them. If
you cannot tell whether this particular one is doing any work, `unclear` is the
honest answer. Do not guess.

## What to ignore

- Whether you agree with the Court.
- Whether the fact seems important **to you**.
- Legal terminology you do not recognise. If the reasoning mentions a date, a place, a
  measurement or an event, and the cited paragraph is where that appears, that is
  `yes`.

## Please

- Work in order. Do not skip a hard row -- mark it `unclear` and move on.
- Some rows are checks on the process rather than on you. You will not be able to tell
  which, and you are not meant to.
- Do not discuss individual rows with the other annotators. Independent answers are
  the entire point of having three of you.

Questions about the task itself are very welcome -- ask before doing eighty rows the
wrong way.
"""


# An annotator cannot read a 15,000-character block, and a block that long is a
# paragraph-splitting failure rather than a paragraph. Items outside this band are
# dropped rather than truncated: truncation can cut away the very sentence the
# reasoning points at, which would turn a good mapping into a false negative.
MIN_CHARS, MAX_CHARS = 80, 3000


def find_pairs(full_text):
    """Return [(assessment paragraph, cited number, cited paragraph)] for one case."""
    marks = list(LAW.finditer(full_text))
    if not marks:
        return []
    cut = marks[-1].start()
    pre = full_text[:cut]
    fw = FRAMEWORK.search(pre)
    facts = {}
    for num, body in split_paragraphs(pre):
        if fw and pre.find(body) >= fw.start():
            continue                 # a statute quotation, not a fact
        facts[num] = body
    out = []
    for _, body in split_paragraphs(full_text[cut:]):
        for m in BACKREF.finditer(body):
            for g in m.groups():
                if g and g in facts:
                    out.append((body, g, facts[g]))
    return out


def main():
    p = argparse.ArgumentParser(description="Build the back-reference annotation sheet")
    p.add_argument("--full-texts", required=True, help="CSV with item_id, doc_name, full_text")
    p.add_argument("--cases", required=True)
    p.add_argument("--out", default="annotation")
    p.add_argument("--items", type=int, default=100, help="genuine pairs")
    p.add_argument("--controls", type=int, default=20,
                   help="pairs whose cited paragraph is swapped for an unrelated one; "
                        "an annotator who answers yes to these is not reading")
    p.add_argument("--annotators", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260828)
    args = p.parse_args()

    article = {}
    for c in json.load(open(args.cases)):
        article.setdefault(c["item_id"], c["article"])

    rng = random.Random(args.seed)
    cases = [r for r in csv.DictReader(open(args.full_texts)) if r.get("full_text")]
    rng.shuffle(cases)

    genuine, controls = [], []
    for r in cases:
        pairs = find_pairs(r["full_text"])
        if not pairs:
            continue
        pairs = [(a, n, c) for a, n, c in pairs
                 if MIN_CHARS <= len(a) <= MAX_CHARS and MIN_CHARS <= len(c) <= MAX_CHARS]
        if not pairs:
            continue
        assessment, num, cited = rng.choice(pairs)
        row = {"case_name": r["doc_name"], "item_id": r["item_id"],
               "article": article.get(r["item_id"], "?"),
               "reasoning": assessment.strip(), "cited_number": num,
               "cited_fact": cited.strip(),
               "full_judgment": HUDOC.format(item_id=r["item_id"])}
        if len(genuine) < args.items:
            genuine.append(row)
        elif len(controls) < args.controls:
            pre = r["full_text"][:LAW.search(r["full_text"]).start()]
            fw = FRAMEWORK.search(pre)
            facts = {n: b for n, b in split_paragraphs(pre)
                     if not (fw and pre.find(b) >= fw.start())}
            # the swapped paragraph must not itself be cited by this reasoning, or the
            # honest answer is "yes" and the control scores its own annotator wrong
            cited_here = {g for m in BACKREF.finditer(assessment) for g in m.groups() if g}
            other = [n for n, body in facts.items()
                     if n not in cited_here and MIN_CHARS <= len(body.strip()) <= MAX_CHARS]
            if not other:
                continue
            swapped = rng.choice(other)
            controls.append({**row, "cited_number": swapped, "cited_fact": facts[swapped].strip()})
        else:
            break

    # every control must be a genuine negative, checked rather than assumed
    for c in controls:
        cited = {g for m in BACKREF.finditer(c["reasoning"]) for g in m.groups() if g}
        assert c["cited_number"] not in cited, "control %s shows a paragraph the reasoning cites" % c["case_name"]

    items = genuine + controls
    rng.shuffle(items)
    for i, it in enumerate(items, 1):
        it["pair_id"] = "P%03d" % i

    key_rows = [{"pair_id": it["pair_id"],
                 "kind": "control" if it in controls else "genuine",
                 "expected": "no" if it in controls else ""} for it in items]

    # Interleave rather than slice. Contiguous thirds left the control share at
    # 10%, 19% and 21%, so the check on the first annotator was half as strong as on
    # the third. Assigning each item to two adjacent annotators in rotation, and
    # doing it separately for controls and genuine items, spreads both evenly.
    def rotate(subset, offset=0):
        out = {a: [] for a in range(args.annotators)}
        for i, it in enumerate(subset):
            first = (i + offset) % args.annotators
            out[first].append(it)
            out[(first + 1) % args.annotators].append(it)
        return out

    gen_by = rotate([it for it in items if it not in controls])
    ctl_by = rotate([it for it in items if it in controls], offset=1)

    fields = ["pair_id", "case_name", "article", "cited_number",
              "reasoning", "cited_fact", "full_judgment", "label", "notes"]
    for a in range(args.annotators):
        mine = gen_by[a] + ctl_by[a]
        rng.shuffle(mine)                      # so controls are not clustered
        path = f"{args.out}_sheet_annotator{a+1}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for it in mine:
                w.writerow({**it, "label": "", "notes": ""})
        print(f"  {path}: {len(mine)} rows, {len(ctl_by[a])} controls")
    per = max(len(gen_by[a]) + len(ctl_by[a]) for a in range(args.annotators))

    with open(f"{args.out}_key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "kind", "expected"])
        w.writeheader(); w.writerows(key_rows)
    with open(f"{args.out}_INSTRUCTIONS.md", "w") as f:
        f.write(INSTRUCTIONS.format(n=per, fname=os.path.basename(f"{args.out}_sheet_annotatorN.csv")))

    print(f"\n  {args.out}_key.csv: {len(controls)} controls among {len(items)} items"
          f" -- do not send this to the annotators")
    print(f"  {args.out}_INSTRUCTIONS.md")
    print(f"\neach item carries {args.annotators * per // len(items)} independent labels")


if __name__ == "__main__":
    main()
