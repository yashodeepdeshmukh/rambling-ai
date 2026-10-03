"""Replay gold decisions and write Laya-shaped training records.

    python scripts/export_training.py   ->  data/train/{solver,nodered}.jsonl

One record per decider call: {task, id, state, questions: {qid: {type, instructions, options}},
labels: {qid: value | null}}. A null label means "don't care" on this path; skip it in the loss.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rambling import datasets  # noqa: E402
from rambling.deciders import OracleDecider, Recorder  # noqa: E402
from rambling.nodered import build as nr  # noqa: E402
from rambling.solvers import linear  # noqa: E402


def export(name, examples, gold_fn, build_fn, skip=lambda ex: None):
    out = ROOT / "data" / "train" / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rec = n_lab = skipped = 0
    with open(out, "w") as f:
        for ex in examples:
            if skip(ex):
                skipped += 1
                continue
            rec = Recorder(OracleDecider(gold_fn(ex)), {"id": ex["id"]})
            build_fn(ex["text"], rec)
            for r in rec.records:
                f.write(json.dumps(r) + "\n")
                n_rec += 1
                n_lab += sum(v is not None for v in r["labels"].values())
    print(f"{out.relative_to(ROOT)}: {n_rec} records, {n_lab} labels, {skipped} unrepresentable skipped")


if __name__ == "__main__":
    export("solver", datasets.solver_problems(), linear.gold_answers, linear.build, linear.gold_representable)
    export("nodered", datasets.nodered_requests(), nr.gold_answers, nr.build)
