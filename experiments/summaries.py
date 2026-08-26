"""Case summaries as a shared artifact, produced once by a fixed summariser.

Every runner used to summarise with the judge model itself. That was wrong twice
over. It cost eight times what it needed to, because the same 976 judgments were
re-summarised for each model in the roster. And it confounded RQ1: a model reading
its own summary is not the condition the paper describes, so a drop in accuracy
could equally mean "summaries lose material facts" or "this model writes bad
summaries". The protocol fixes the summariser precisely so that no model in the
roster ever reads its own writing.

Summaries are therefore built once by scripts/build_summaries.py and loaded here.
"""

import hashlib
import json
import os
import sys

SUMMARY_TEMPLATE = """Summarize the following legal case to approximately 500 words, focusing on the key facts relevant to the alleged violation(s).

Case Name: {case_name}
Full Text:
{full_text}

Summary (approximately 500 words):"""


def is_usable(summary):
    """True when this is a summary rather than a record of a failed call.

    A failed call returns the string "ERROR: ...", which is not empty and would
    otherwise be scored as if it were a summary -- the same defect as the old
    fall-back to raw case text, which silently mixed the two conditions.
    """
    return bool(summary) and isinstance(summary, str) and not summary.startswith("ERROR:")


def load_summaries(path):
    """Return (summaries, meta) from a summaries file.

    Accepts the wrapped form written by build_summaries.py and a bare
    ``{item_id: [versions]}`` mapping, which is what the runners wrote before
    summarisation was split out.
    """
    with open(path) as f:
        blob = json.load(f)
    if isinstance(blob, dict) and "summaries" in blob:
        return blob["summaries"], {k: v for k, v in blob.items() if k != "summaries"}
    return blob, {}


def file_digest(path, length=12):
    """Short content digest, so results name the summaries they were scored against."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def add_argument(parser):
    """The --summaries flag, identical in every runner."""
    parser.add_argument("--summaries", help="summaries JSON from scripts/build_summaries.py; "
                                            "required for rq1 and rq2")


def load_summaries_for(args, stages, mlflow=None):
    """Load the shared summaries if these stages need them, or exit saying why.

    Exits rather than falling back. A runner that quietly summarises for itself when
    the file is missing is how the judge model came to be grading its own writing.
    """
    if not {"rq1", "rq2"} & set(stages):
        return {}
    path = getattr(args, "summaries", None)
    if not path:
        sys.exit("ERROR: --summaries is required for rq1/rq2. Build it once with "
                 "scripts/build_summaries.py; the runners no longer summarise, because "
                 "summarising with the judge model both cost 8x and had each model "
                 "grade its own writing.")
    if not os.path.exists(path):
        sys.exit(f"ERROR: no such summaries file: {path}")
    summaries, meta = load_summaries(path)
    if mlflow is not None:
        mlflow.log_param("summarizer", meta.get("summarizer", "unknown"))
        mlflow.log_param("summaries_file", os.path.basename(path))
        # A file name is not an identity: two builds can share one. The digest says
        # which summaries these results were actually scored against.
        mlflow.log_param("summaries_sha", file_digest(path))
    print(f"Summaries: {len(summaries)} judgments from "
          f"{meta.get('summarizer', 'an unrecorded summariser')}\n")
    return summaries


def coverage(summaries, cases, version=0):
    """How many of these cases have a usable summary at this version."""
    have = sum(1 for c in cases
               if is_usable((summaries.get(c["item_id"]) or [None] * (version + 1))[version]))
    return have, len(cases)
