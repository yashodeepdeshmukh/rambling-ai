"""Train the CPU stand-in decider on synthetic data and fit its calibration and cascade gates.

    python scripts/train.py                 # ~5 min on 4 CPUs -> models/decider.pkl, reports/training.md

Splits are by phrasing variant (see rambling/synth.py). Variants 0-2 are used for training and
cross-phrasing calibration; variant 3 is reserved for scripts/poc_eval.py.
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rambling import cascade, training  # noqa: E402
from rambling.deciders import question_kind  # noqa: E402
from rambling.learned import LearnedDecider, _softmax  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-variant", type=int, default=80, help="samples per family per phrasing variant")
    ap.add_argument("--target-precision", type=float, default=0.98)
    ap.add_argument("--out", default="models/decider.pkl")
    args = ap.parse_args()

    t0 = time.time()
    by_variant = {v: [] for v in ("v0", "v1", "v2")}
    for task in training.TASKS:
        for v in by_variant:
            by_variant[v] += training.records_for(task, training.examples(task, v, args.per_variant))
    print(f"records per phrasing: {[len(r) for r in by_variant.values()]} ({time.time() - t0:.0f}s)")

    # Cross-phrasing calibration: each fold model never sees one phrasing; its scores on that
    # phrasing set the temperatures and the (conservative) cascade gates.
    folds = []
    for held in by_variant:
        fold_model = LearnedDecider()
        fold_model.fit([r for v, recs in by_variant.items() if v != held for r in recs])
        folds.append(fold_model.scored(by_variant[held]))
        print(f"fold without {held} done ({time.time() - t0:.0f}s)")
    calib_scored = [x for f in folds for x in f]

    model = LearnedDecider()
    rows = model.fit([r for recs in by_variant.values() for r in recs])
    model.calibrate(scored=calib_scored)
    model.thresholds = cascade.fit_thresholds(folds, model.temps, target=args.target_precision)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"saved {args.out} ({time.time() - t0:.0f}s)")
    train = [r for recs in by_variant.values() for r in recs]

    # out-of-fold accuracy and gate coverage per decision kind
    acc, tot, kept, kept_ok = Counter(), Counter(), Counter(), Counter()
    for k, scores, gold in calib_scored:
        p = _softmax(scores, model.temps.get(k, 1.0))
        ok = int(p.argmax()) == gold
        tot[k] += 1
        acc[k] += ok
        if p.max() >= model.thresholds.get(k, 1.01):
            kept[k] += 1
            kept_ok[k] += ok
    lines = ["# Training report", "",
             f"Final model: {len(train)} records over phrasing variants 0-2 ({rows} option rows). "
             f"Temperatures and gates come from 3 fold models, each scored on the phrasing it never saw; "
             f"a gate must reach {args.target_precision:.0%} precision in every fold. "
             "Accuracy columns are out-of-fold (unseen phrasing).", "",
             "| task | decision kind | out-of-fold accuracy | temperature | gate | kept by gate | precision when kept |",
             "|---|---|---|---|---|---|---|"]
    solver_kinds = ("n_", "var", "c.", "obj", "domain", "goal", "nonneg", "target")
    for k in sorted(tot, key=lambda k: (not k.startswith(solver_kinds), k)):
        task = "solver" if k.startswith(solver_kinds) else "nodered"
        gate = model.thresholds.get(k, 1.01)
        prec = f"{kept_ok[k] / kept[k]:.1%}" if kept[k] else "–"
        lines.append(f"| {task} | `{k}` | {acc[k] / tot[k]:.1%} | {model.temps.get(k, 1.0)} | "
                     f"{'never' if gate > 1 else gate} | {kept[k] / tot[k]:.0%} | {prec} |")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/training.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
