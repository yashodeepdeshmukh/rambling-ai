"""Zero-shot PoC: a pretrained decider, no fine-tuning, behind the same cascade.

    python scripts/zero_shot_eval.py --backend laya                 # real Laya (downloads ~1.7 GB once)
    python scripts/zero_shot_eval.py --backend laya --laya-model-dir ./onnx
    python scripts/zero_shot_eval.py --backend wordllama            # runs anywhere, weights ship in the wheel
    python scripts/zero_shot_eval.py --backend laya --no-calibration --gate 0.9

Model weights are never updated. Unless --no-calibration is given, per-kind temperatures and
cascade gates are fitted on synthetic phrasing variants 0-2 (each variant is one fold), and the
tests use variant 3 plus the hand-written sets. Writes reports/zero_shot_<backend>.md.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rambling import datasets, training, zero_shot  # noqa: E402
from rambling.report import HEADER, nodered_item, solver_item, summarize  # noqa: E402
from rambling.solvers import linear  # noqa: E402


def make_backend(args):
    if args.backend == "wordllama":
        from rambling.backends import WordLlamaDecider
        return WordLlamaDecider()
    from rambling.backends import LayaDecider, bridge_available
    if not bridge_available():
        sys.exit("Laya bridge not installed: run `npm install` in laya_bridge/ (see LOCAL_RUN.md)")
    print("starting Laya (first run downloads the ONNX bundle, ~1.7 GB)...", flush=True)
    return LayaDecider(model_dir=args.laya_model_dir, threads=args.threads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["laya", "wordllama"], default="laya")
    ap.add_argument("--laya-model-dir", help="local ONNX bundle instead of the Hugging Face download")
    ap.add_argument("--threads", type=int, help="ONNX Runtime intra-op threads")
    ap.add_argument("--calib-per-variant", type=int, default=5, help="samples per family per calibration phrasing")
    ap.add_argument("--test-per-variant", type=int, default=10, help="samples per family in the test phrasing")
    ap.add_argument("--target-precision", type=float, default=0.98)
    ap.add_argument("--no-calibration", action="store_true", help="pure zero-shot: temperature 1, one fixed gate")
    ap.add_argument("--gate", type=float, default=0.9, help="gate used with --no-calibration")
    ap.add_argument("--live", action="store_true", help="deploy auto-accepted hand-written flows to Node-RED")
    args = ap.parse_args()

    t0 = time.time()
    backend = make_backend(args)
    decider, temps, notes = backend, {}, []
    if args.no_calibration:
        kinds = {k for k in ("template", "broker", "topic", "topic_in", "topic_out", "property", "operator", "threshold",
                             "method", "url", "path", "file", "interval", "n_vars", "n_cons", "domain", "nonneg", "goal",
                             "var", "c.const", "c.rel", "c.rhs", "c.v.sign", "c.v.mag", "target", "obj.v.sign", "obj.v.mag")}
        thresholds = {k: args.gate for k in kinds}
        notes.append(f"No calibration: temperature 1 and a fixed gate of {args.gate} for every decision kind.")
        fold_acc = None
    else:
        folds = []
        for v in ("v0", "v1", "v2"):
            recs = []
            for task in training.TASKS:
                recs += training.records_for(task, training.examples(task, v, args.calib_per_variant))
            folds.append(recs)
        print(f"calibrating on {sum(map(len, folds))} records ({time.time() - t0:.0f}s)", flush=True)
        temps, thresholds, scored = zero_shot.calibrate(backend, folds, args.target_precision)
        decider = zero_shot.Tempered(backend, temps)
        acc, tot = Counter(), Counter()
        for f in scored:
            for k, s, g in f:
                tot[k] += 1
                acc[k] += int(s.argmax()) == g
        fold_acc = (acc, tot)
        cal = Path("models") / f"zero_shot_{args.backend}.json"
        cal.parent.mkdir(exist_ok=True)
        cal.write_text(json.dumps({"temps": temps, "thresholds": thresholds}, indent=1))
        notes.append(f"Calibrated (model frozen): per-kind temperatures and gates at "
                     f"{args.target_precision:.0%} precision in every one of 3 phrasing folds.")
    print(f"evaluating ({time.time() - t0:.0f}s)", flush=True)

    solver_syn = training.examples("solver", "test", args.test_per_variant)
    nodered_syn = training.examples("nodered", "test", args.test_per_variant)
    solver_hand = [ex for ex in datasets.solver_problems() if not linear.gold_representable(ex)]
    table = [*HEADER,
             summarize("solver, synthetic unseen phrasing", [solver_item(decider, ex, thresholds) for ex in solver_syn]),
             summarize("solver, hand-written", [solver_item(decider, ex, thresholds) for ex in solver_hand]),
             summarize("Node-RED, synthetic unseen phrasing", [nodered_item(decider, ex, None, thresholds) for ex in nodered_syn])]
    if args.live:
        from rambling.nodered import live
        with live.NodeRedServer() as srv:
            table.append(summarize("Node-RED, hand-written (auto-accepted flows deployed + probed)",
                                   [nodered_item(decider, ex, srv, thresholds) for ex in datasets.nodered_requests()]))
    else:
        table.append(summarize("Node-RED, hand-written",
                               [nodered_item(decider, ex, None, thresholds) for ex in datasets.nodered_requests()]))

    lines = [f"# Zero-shot PoC: `{args.backend}` (no fine-tuning)", "", *notes, "", *table, ""]
    if fold_acc:
        acc, tot = fold_acc
        lines += ["## Per-decision accuracy on the calibration phrasings (zero-shot)", "",
                  "| decision kind | accuracy | temperature | gate |", "|---|---|---|---|"]
        lines += [f"| `{k}` | {acc[k] / tot[k]:.1%} ({tot[k]}) | {temps.get(k, 1.0)} | "
                  f"{'never' if thresholds.get(k, 1.01) > 1 else thresholds[k]} |" for k in sorted(tot)]
    lines += ["", "- Final accuracy uses a *simulated* LLM fallback (gold answers): an upper bound.",
              "- Token savings are chars/4 estimates of an LLM answering the same typed questions.",
              f"- Runtime: {time.time() - t0:.0f}s."]
    out = Path("reports") / f"zero_shot_{args.backend}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if hasattr(backend, "close"):
        backend.close()


if __name__ == "__main__":
    main()
