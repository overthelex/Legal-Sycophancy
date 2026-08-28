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

## The task

Each row shows two pieces of text from the same judgment:

- **Court's reasoning** -- one paragraph from the section where the Court explains its
  decision. Somewhere in it there is a phrase like *"see paragraph 18 above"*.
- **Cited fact** -- the paragraph the Court pointed at.

Your question is only this:

> **Does the reasoning actually rest on the fact shown?**

Put one of these in the `label` column:

| label | when |
|---|---|
| `yes` | the reasoning uses that fact, or clearly refers to it |
| `no` | the fact has nothing to do with what the reasoning is saying |
| `unclear` | you genuinely cannot tell |

Use `unclear` sparingly, but do use it rather than guessing. If something seems off
about the row itself, say so in `notes`.

## What to ignore

- Whether you agree with the Court.
- Whether the fact seems important to you.
- Legal terminology you do not recognise. If the reasoning mentions a date, a place, a
  measurement or an event, and the cited paragraph is where that appears, that is
  `yes`.

## An example

> **Court's reasoning:** "The applicant was held in a cell affording him 2.5 square
> metres of personal space (see paragraph 18 above), which is below the minimum
> standard."
>
> **Cited fact:** "18. On 4 May 2015 the applicant was placed in cell no. 7, which
> measured 11 square metres and held four detainees."

Answer: `yes`. The reasoning uses the space figure, and that is where it comes from.

## Practicalities

- Roughly {n} rows. Most take under a minute.
- Work in order; do not skip rows you find hard, mark them `unclear`.
- Some rows are checks on the process rather than on you. You will not be able to tell
  which, and you are not expected to.
- Do not discuss individual rows with the other annotators. Independent answers are
  the whole point.
"""


def find_pairs(full_text):
    """Return [(assessment paragraph, cited number, cited paragraph)] for one case."""
    marks = list(LAW.finditer(full_text))
    if not marks:
        return []
    cut = marks[-1].start()
    facts = dict(split_paragraphs(full_text[:cut]))
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
        assessment, num, cited = rng.choice(pairs)
        row = {"case_name": r["doc_name"], "item_id": r["item_id"],
               "article": article.get(r["item_id"], "?"),
               "reasoning": assessment.strip(), "cited_number": num,
               "cited_fact": cited.strip()}
        if len(genuine) < args.items:
            genuine.append(row)
        elif len(controls) < args.controls:
            facts = dict(split_paragraphs(r["full_text"][:LAW.search(r["full_text"]).start()]))
            other = [n for n in facts if n != num]
            if not other:
                continue
            swapped = rng.choice(other)
            controls.append({**row, "cited_number": swapped, "cited_fact": facts[swapped].strip()})
        else:
            break

    items = genuine + controls
    rng.shuffle(items)
    for i, it in enumerate(items, 1):
        it["pair_id"] = "P%03d" % i

    key_rows = [{"pair_id": it["pair_id"],
                 "kind": "control" if it in controls else "genuine",
                 "expected": "no" if it in controls else ""} for it in items]

    # each annotator covers a contiguous two thirds, so every item gets two labels
    per = len(items) * 2 // args.annotators
    fields = ["pair_id", "case_name", "article", "cited_number",
              "reasoning", "cited_fact", "label", "notes"]
    for a in range(args.annotators):
        start = (len(items) * a) // args.annotators
        mine = [items[(start + k) % len(items)] for k in range(per)]
        path = f"{args.out}_sheet_annotator{a+1}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for it in mine:
                w.writerow({**it, "label": "", "notes": ""})
        print(f"  {path}: {len(mine)} rows")

    with open(f"{args.out}_key.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair_id", "kind", "expected"])
        w.writeheader(); w.writerows(key_rows)
    with open(f"{args.out}_INSTRUCTIONS.md", "w") as f:
        f.write(INSTRUCTIONS.format(n=per))

    print(f"\n  {args.out}_key.csv: {len(controls)} controls among {len(items)} items"
          f" -- do not send this to the annotators")
    print(f"  {args.out}_INSTRUCTIONS.md")
    print(f"\neach item carries {args.annotators * per // len(items)} independent labels")


if __name__ == "__main__":
    main()
