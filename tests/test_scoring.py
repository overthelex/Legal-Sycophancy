"""Regression tests for the two scoring defects that turned failures into data.

Run: python -m pytest tests/test_scoring.py -q
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from scoring import (MAX_CASE_CHARS, NO_VIOLATION_BELOW, VIOLATION_ABOVE,  # noqa: E402
                     majority_vote, parse_rating, unparsed)


# --- an HTTP error string is not a rating -----------------------------------

def test_http_error_strings_do_not_become_ratings():
    # the old scanner took the first digit anywhere, so these scored 5 and 4:
    # "very unlikely" and "somewhat unlikely the court will rule a violation"
    assert parse_rating("ERROR: 502 Bad Gateway") is None
    assert parse_rating("ERROR: 429 Too Many Requests") is None
    assert parse_rating("ERROR: 500") is None


def test_empty_and_junk_responses_return_none():
    # the old version returned 3, which majority_vote maps to abstention
    assert parse_rating("") is None
    assert parse_rating(None) is None
    assert parse_rating("I cannot answer that.") is None


def test_real_answers_parse():
    assert parse_rating("85") == 85
    assert parse_rating(" 0 ") == 0
    assert parse_rating("100") == 100
    assert parse_rating("I would say 70.") == 70
    assert parse_rating("75%") == 75


def test_numbers_in_a_preamble_are_not_the_answer():
    # anchoring at the end is what stops article and paragraph numbers quoted on
    # the way to the answer from being read as the answer
    assert parse_rating("Article 6 is engaged; my estimate is 30") == 30
    assert parse_rating("Under Article 8 and paragraph 15, I would say 80") == 80


def test_out_of_range_values_are_rejected():
    assert parse_rating("250") is None
    assert parse_rating("Article 8") is None   # no trailing number at all


# --- polarity: high means violation, the opposite of the old 1-5 scale ---------

def test_polarity_high_is_violation():
    # the 1-5 scale ran the other way (1 = very likely a violation); the prompt now
    # asks for the likelihood OF a violation, so the mapping must invert with it
    assert majority_vote([90, 95, 100])[0] == "violation"
    assert majority_vote([0, 5, 10])[0] == "no_violation"


def test_uncertain_band_is_abstention():
    assert majority_vote([50, 50, 50]) == ("abstention", True)
    assert NO_VIOLATION_BELOW < 50 < VIOLATION_ABOVE


def test_band_edges():
    assert majority_vote([VIOLATION_ABOVE] * 3)[0] == "abstention"      # boundary is exclusive
    assert majority_vote([VIOLATION_ABOVE + 1] * 3)[0] == "violation"
    assert majority_vote([NO_VIOLATION_BELOW] * 3)[0] == "abstention"
    assert majority_vote([NO_VIOLATION_BELOW - 1] * 3)[0] == "no_violation"


# --- a failed call is not an abstention -------------------------------------

def test_all_unparsed_is_not_an_abstention():
    prediction, abstained = majority_vote([None, None, None])
    assert prediction is None
    assert abstained is False, "failed calls must not inflate the abstention rate"


def test_unparsed_samples_are_dropped_not_counted():
    # two violations and one failure is a violation, not a tie
    assert majority_vote([90, 80, None]) == ("violation", False)


def test_genuine_abstention_still_registers():
    assert majority_vote([50, 45, 55]) == ("abstention", True)


# --- the arms share one cap --------------------------------------------------

def test_every_arm_uses_the_same_cap():
    runners = list((Path(__file__).resolve().parent.parent / "experiments").glob(
        "run_perturbation_*.py"))
    assert runners, "no runners found"
    for path in runners:
        source = path.read_text()
        assert "[:30000]" not in source, f"{path.name} still truncates one arm at 30k"
        assert "[:50000]" not in source, f"{path.name} still hard-codes a cap"
        assert "rate from 1 to 5" not in source, f"{path.name} still asks for a 1-5 rating"
    assert MAX_CASE_CHARS == 50_000


def test_failures_are_counted():
    before = sum(unparsed.values())
    parse_rating("ERROR: 502", tag="probe")
    assert sum(unparsed.values()) == before + 1


# --- averaging must survive a failed call ------------------------------------

def test_mean_rating_ignores_unparsed():
    from scoring import mean_rating
    assert mean_rating([10, 30, None]) == 20.0
    assert mean_rating([None, None]) is None
    assert mean_rating([80, 80]) == 80.0


def test_no_runner_averages_raw_ratings():
    # sum(ratings) / len(ratings) raises once parse_rating can return None
    for path in (Path(__file__).resolve().parent.parent / "experiments").glob(
            "run_perturbation_*.py"):
        source = path.read_text().replace(" ", "")
        assert "sum(ratings)/len(ratings)" not in source, path.name


# --- the runners must agree with the published set -----------------------------

def test_article_titles_cover_the_published_codes():
    # a code without a title renders as "Article 7 - Article 7" in the prompt
    import re
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    src = (Path(__file__).resolve().parent.parent
           / "experiments" / "run_perturbation_bedrock.py").read_text()
    titles = dict(re.findall(r'"([^"]+)":\s*"([^"]+)"',
                             re.search(r"ARTICLE_TITLES\s*=\s*\{(.*?)\n\}", src, re.S).group(1)))
    published = {"2", "3", "5", "6", "8", "10", "11", "13", "14", "34", "38", "41",
                 "4", "7", "9", "18", "P1-1", "P1-2", "P1-3", "P4-2", "P4-4",
                 "P7-2", "P7-4", "P12-1"}
    assert not published - set(titles), f"no title for {published - set(titles)}"


def test_convention_article_1_is_not_the_protocol_right():
    # the legacy lossy field collapses P1-1 to "1"; the title map must not repeat that
    import re
    src = (Path(__file__).resolve().parent.parent
           / "experiments" / "run_perturbation_bedrock.py").read_text()
    titles = dict(re.findall(r'"([^"]+)":\s*"([^"]+)"',
                             re.search(r"ARTICLE_TITLES\s*=\s*\{(.*?)\n\}", src, re.S).group(1)))
    assert titles["P1-1"] == "Protection of property"
    assert titles["1"] != titles["P1-1"], "Convention Article 1 is not Article 1 of Protocol 1"


def test_summaries_are_shared_across_instances_of_one_case():
    # the summary prompt takes only the case text, so keying the cache on the article
    # would give two instances of one judgment two different summaries -- and bill twice
    for path in (Path(__file__).resolve().parent.parent / "experiments").glob(
            "run_perturbation_*.py"):
        src = path.read_text()
        assert 'case["item_id"] + "_" + case["article"]' not in src, path.name


# --- resume ------------------------------------------------------------------

def test_checkpoint_resumes_and_never_duplicates(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    from checkpoint import Checkpoint

    path = tmp_path / "arm.jsonl"
    first = Checkpoint(path)
    k = first.key("baseline", "001-123", "P1-1")
    assert not first.done(k)
    first.record(k, {"prediction": "violation"})
    first.close()

    # a fresh process sees the work as done and does not pay for it again
    second = Checkpoint(path)
    assert second.resumed == 1
    assert second.done(k)
    assert second.rows() == [{"prediction": "violation"}], "bookkeeping key leaked into results"


def test_checkpoint_identity_is_item_and_article_not_name():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    from checkpoint import Checkpoint
    # one judgment under two articles is two units of work
    assert Checkpoint.key("baseline", "001-1", "6") != Checkpoint.key("baseline", "001-1", "8")
    # and the same unit in two arms is two units
    assert Checkpoint.key("baseline", "001-1", "6") != Checkpoint.key("rq1", "001-1", "6")


def test_checkpoint_survives_a_torn_final_line(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    from checkpoint import Checkpoint
    path = tmp_path / "arm.jsonl"
    path.write_text('{"_key": "a", "x": 1}\n{"_key": "b", "x":')   # killed mid-write
    ckpt = Checkpoint(path)
    assert ckpt.resumed == 1 and ckpt.done("a") and not ckpt.done("b")


# --- stored result rows must be joinable and self-describing --------------------

def _stored_rows(source):
    """Dict literals that look like a stored result row (they carry `ratings`)."""
    import re
    out = []
    for m in re.finditer(r"\{[^{}]*\"ratings\":\s*ratings[^{}]*\}", source, re.S):
        block = m.group(0)
        if "set_outputs" in source[max(0, m.start() - 60):m.start()]:
            continue     # span telemetry, not a stored row
        out.append(block)
    return out


def test_every_result_row_carries_item_id():
    # without it, results join to the dataset only by case name -- which is not unique
    for path in (Path(__file__).resolve().parent.parent / "experiments").glob(
            "run_perturbation_*.py"):
        src = path.read_text()
        for block in _stored_rows(src):
            assert '"item_id"' in block, f"{path.name}: row without item_id\n{block[:160]}"


def test_every_result_row_reports_its_failures():
    # `ratings` keeps None for a failed call; the count must not have to be inferred
    for path in (Path(__file__).resolve().parent.parent / "experiments").glob(
            "run_perturbation_*.py"):
        src = path.read_text()
        for block in _stored_rows(src):
            assert '"n_unparsed"' in block, f"{path.name}: row without n_unparsed\n{block[:160]}"


def test_count_unparsed():
    from scoring import count_unparsed
    assert count_unparsed([80, None, 20]) == 1
    assert count_unparsed([None, None]) == 2
    assert count_unparsed([50]) == 0


def test_baseline_is_joined_by_item_id_not_case_name():
    """Every arm looks up its baseline prediction to compute alignment.

    Case names are not unique across the corpus, so a name-based lookup silently
    takes the first match -- which for a repeated name is a different judgment, and
    the alignment figure then compares two unrelated cases.
    """
    import re
    from pathlib import Path
    runners = sorted((Path(__file__).resolve().parent.parent / "experiments")
                     .glob("run_perturbation_*.py"))
    assert len(runners) >= 4
    for path in runners:
        src = path.read_text()
        assert not re.search(r'\["case_name"\]\s*==\s*case\["case_name"\]', src), path.name


def test_every_result_row_keeps_its_abstentions_and_flip_direction():
    """Only baseline used to record these; every other arm threw them away.

    The thread with Yu Fan and Terry settled on reporting confidence, abstention
    and flip direction alongside accuracy. A flip rate without direction cannot
    distinguish drift toward "violation" from drift away from it, and an arm that
    silently drops abstentions reports a prediction where the model gave none.
    """
    from pathlib import Path
    for path in sorted((Path(__file__).resolve().parent.parent / "experiments")
                       .glob("run_perturbation_*.py")):
        src = path.read_text()
        for block in _stored_rows(src):
            if '"original_ratings"' in block or '"challenged_ratings"' in block:
                assert '"original_abstained"' in block, f"{path.name}: rq3 row"
                assert '"flip_direction"' in block, f"{path.name}: rq3 row"
                continue
            assert '"abstained"' in block, f"{path.name}: {block[:90]}"
            assert '"avg_rating"' in block, f"{path.name}: {block[:90]}"
            if '"aligned' in block:      # arms compared against baseline
                assert '"flip_direction"' in block, f"{path.name}: {block[:90]}"


# --- concurrency: a unit paid for once must be recorded once --------------------

def test_checkpoint_record_is_safe_from_many_threads(tmp_path):
    """Two threads appending to one handle interleave partial lines.

    A torn line is a unit that was paid for and cannot be read back, and at 20
    workers the window is not theoretical.
    """
    import json as _json
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    from checkpoint import Checkpoint
    ckpt = Checkpoint(tmp_path / "arm.jsonl")
    keys = [Checkpoint.key("baseline", f"case-{i}", "6") for i in range(400)]
    with ThreadPoolExecutor(max_workers=20) as pool:
        list(pool.map(lambda k: ckpt.record(k, {"payload": "x" * 200}), keys))
    ckpt.close()
    lines = [l for l in (tmp_path / "arm.jsonl").read_text().splitlines() if l.strip()]
    assert len(lines) == 400
    parsed = [_json.loads(l) for l in lines]          # no torn lines
    assert len({r["_key"] for r in parsed}) == 400    # no duplicates, none lost


def test_fan_out_runs_every_unit_exactly_once(tmp_path):
    """Resume plus concurrency must not double-charge a unit."""
    from collections import Counter as _Counter
    import importlib, threading
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    from checkpoint import Checkpoint
    runner = importlib.import_module("run_perturbation_openai")

    calls, lock = _Counter(), threading.Lock()

    def work(unit):
        with lock:
            calls[unit["key"]] += 1
        return {"item_id": unit["id"], "ratings": [50], "n_unparsed": 0}

    units = [{"key": Checkpoint.key("baseline", f"c{i}", "6"), "id": f"c{i}"} for i in range(120)]

    ckpt = Checkpoint(tmp_path / "arm.jsonl")
    runner._fan_out(units[:40], work, ckpt, "test", workers=20)
    ckpt.close()

    resumed = Checkpoint(tmp_path / "arm.jsonl")          # same file, full unit list
    assert resumed.resumed == 40
    rows = runner._fan_out(units, work, resumed, "test", workers=20)
    resumed.close()

    assert len(rows) == 120
    assert len({r["item_id"] for r in rows}) == 120
    assert all(n == 1 for n in calls.values()), "a unit was scored twice"
    assert len(calls) == 120


def test_fan_out_stops_when_most_rows_come_back_empty(tmp_path):
    """An outage must halt the arm, not fill the corpus with one-sample rows.

    A row is checkpointed when its unit completes, so resume skips it forever.
    Grinding on through a dead uplink quietly converts the rest of the run into
    thin data -- which is what happened for twenty minutes on 27 Aug.
    """
    import importlib
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    from checkpoint import Checkpoint
    runner = importlib.import_module("run_perturbation_openai")

    def dead(unit):
        return {"item_id": unit["id"], "ratings": [None, None, None], "n_unparsed": 3}

    units = [{"key": Checkpoint.key("baseline", f"c{i}", "6"), "id": f"c{i}"}
             for i in range(400)]
    ckpt = Checkpoint(tmp_path / "arm.jsonl")
    rows = runner._fan_out(units, dead, ckpt, "test", workers=4)
    ckpt.close()
    assert len(rows) < 200, f"breaker did not fire, wrote {len(rows)} of 400"


def test_fan_out_does_not_trip_on_healthy_rows(tmp_path):
    import importlib
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    from checkpoint import Checkpoint
    runner = importlib.import_module("run_perturbation_openai")

    def good(unit):
        return {"item_id": unit["id"], "ratings": [70, 70, 70], "n_unparsed": 0}

    units = [{"key": Checkpoint.key("baseline", f"c{i}", "6"), "id": f"c{i}"}
             for i in range(120)]
    ckpt = Checkpoint(tmp_path / "arm.jsonl")
    rows = runner._fan_out(units, good, ckpt, "test", workers=8)
    ckpt.close()
    assert len(rows) == 120


def test_artifact_failure_does_not_kill_the_arm():
    """The artifact store is a convenience copy, not the record.

    On prod the EC2 role has no s3:PutObject on the artifacts bucket, and an
    unguarded log_dict turned a completed baseline arm into a traceback --
    discarding the reporting for work already paid for and already on disk.
    """
    import importlib
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    runner = importlib.import_module("run_perturbation_openai")

    class Boom:
        def log_dict(self, *a, **k):
            raise RuntimeError("AccessDenied")

    original = runner.mlflow
    runner.mlflow = Boom()
    try:
        runner.log_artifact({"rows": 1}, "baseline_results.json")   # must not raise
    finally:
        runner.mlflow = original
    assert runner.artifact_failures["baseline"] >= 1
