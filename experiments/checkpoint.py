"""Append-only checkpoint so an interrupted run resumes instead of restarting.

A full-roster run is hours long and costs real money; without this, one dropped
connection or one rate-limit storm at hour three throws away everything. Rows are
written and flushed as each unit completes, so the file is always a valid record of
what has been paid for.

    ckpt = Checkpoint(Path(out_dir) / "baseline.jsonl")
    for case in cases:
        key = ckpt.key("baseline", case["item_id"], case["article"])
        if ckpt.done(key):
            continue
        ...
        ckpt.record(key, row)
    results = ckpt.rows()

Identity is (stage, item_id, article, variant) rather than the case name, because
names are not unique across the corpus and a judgment can appear under several
articles.
"""

import json
from pathlib import Path


class Checkpoint:
    def __init__(self, path, enabled=True):
        self.path = Path(path)
        self.enabled = enabled
        self._rows = []
        self._done = set()
        if not enabled:
            self._handle = None
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue      # a torn final line from a hard kill
                self._rows.append(row)
                if "_key" in row:
                    self._done.add(row["_key"])
        self._handle = self.path.open("a")

    @staticmethod
    def key(stage, item_id, article, variant=""):
        return "|".join((stage, str(item_id), str(article), str(variant)))

    def done(self, key):
        return key in self._done

    def record(self, key, row):
        row = {**row, "_key": key}
        self._rows.append(row)
        self._done.add(key)
        if self._handle:
            self._handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._handle.flush()
        return row

    def rows(self):
        """Recorded rows without the bookkeeping key."""
        return [{k: v for k, v in r.items() if k != "_key"} for r in self._rows]

    @property
    def resumed(self):
        return len(self._done)

    def close(self):
        if self._handle:
            self._handle.close()
