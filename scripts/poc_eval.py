"""PoC evaluation: trained decider alone vs. the confidence-gated cascade.

    python scripts/poc_eval.py [--live]   -> reports/poc.md

Test sets: synthetic variant 3 (a phrasing never seen in training or calibration) and the
hand-written datasets (different author, different style). The LLM fallback is SIMULATED by
the oracle (gold answers), so "final accuracy" is an upper bound on what a real LLM fallback
would give; the numbers that matter are the zero-LLM rate and its precision, and the share of
decisions answered locally. Token counts are estimates (chars / 4).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rambling import cascade, datasets, training  # noqa: E402
from rambling.decide import NotRepresentable  # noqa: E402
from rambling.deciders import OracleDecider  # noqa: E402
from rambling.learned import LearnedDecider  # noqa: E402
from rambling.nodered import build as nr  # noqa: E402
from rambling.solvers import linear  # noqa: E402


def solver_item(model, ex):
    gold = linear.gold_answers(ex)
    alone = linear.check(ex, linear.solve(linear.build(ex["text"], model)))
    out = cascade.run_solver(ex["text"], model, model.thresholds, OracleDecider(gold))
    return alone, out, linear.check(ex, out.result)


def nodered_item(model, ex, server=None):
    gold = nr.gold_answers(ex)
    t, s, _ = nr.build(ex["text"], model)
    alone = t == ex["template"] and s == ex["slots"]
    out = cascade.run_nodered(ex["text"], model, model.thresholds, OracleDecider(gold))
    final = out.result["template"] == ex["template"] and out.result["slots"] == ex["slots"]
    if server is not None and out.accepted_automatically and ex.get("probes"):
        deployed, _ = server.deploy(out.artifact)
        final = final and deployed and all(server.probe(p)[0] for p in ex["probes"])
    return alone, out, final


def summarize(name, rows):
    n = len(rows)
    auto = [r for r in rows if r[1].accepted_automatically]
    led = [r[1].ledger for r in rows]
    q, fq = sum(l.questions for l in led), sum(l.fallback_questions for l in led)
    full = sum(r[1].escalated == "full" for r in rows)
    llm_only, spent = sum(l.llm_only_tokens for l in led), sum(l.fallback_tokens for l in led)
    return (f"| {name} | {n} | {sum(r[0] for r in rows) / n:.0%} | {len(auto) / n:.0%} | "
            f"{(f'{sum(r[2] for r in auto) / len(auto):.1%}') if auto else '–'} | "
            f"{1 - fq / q:.0%} | {full / n:.0%} | {sum(r[2] for r in rows) / n:.0%} | "
            f"{1 - spent / llm_only:.0%} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/decider.pkl")
    ap.add_argument("--test-per-variant", type=int, default=40)
    ap.add_argument("--live", action="store_true", help="deploy auto-accepted hand-written flows and probe them")
    args = ap.parse_args()
    model = LearnedDecider.load(args.model)

    solver_syn = training.examples("solver", "test", args.test_per_variant)
    nodered_syn = training.examples("nodered", "test", args.test_per_variant)
    solver_hand = [ex for ex in datasets.solver_problems() if not linear.gold_representable(ex)]
    skipped = len(datasets.solver_problems()) - len(solver_hand)

    table = [
        "| test set | items | learned alone, end-to-end | zero-LLM (auto-accepted) | precision of auto-accepted | decisions answered locally | full escalations | final accuracy (simulated LLM fallback) | est. LLM tokens saved vs all-LLM |",
        "|---|---|---|---|---|---|---|---|---|",
        summarize("solver, synthetic unseen phrasing", [solver_item(model, ex) for ex in solver_syn]),
        summarize("solver, hand-written", [solver_item(model, ex) for ex in solver_hand]),
        summarize("Node-RED, synthetic unseen phrasing", [nodered_item(model, ex) for ex in nodered_syn]),
    ]
    if args.live:
        from rambling.nodered import live
        with live.NodeRedServer() as srv:
            table.append(summarize("Node-RED, hand-written (auto-accepted flows deployed + probed)",
                                   [nodered_item(model, ex, srv) for ex in datasets.nodered_requests()]))
    else:
        table.append(summarize("Node-RED, hand-written", [nodered_item(model, ex) for ex in datasets.nodered_requests()]))

    text = "\n".join([
        "# PoC results", "",
        "Trained CPU stand-in decider (hashed logistic scorer, `rambling/learned.py`) behind the "
        "confidence-gated cascade (`rambling/cascade.py`). Gates were fitted on a phrasing variant "
        "that is neither in training nor in these test sets.", "",
        *table, "",
        "- **Zero-LLM**: every decision passed its gate and the hard checks passed; no LLM call at all.",
        "- **Decisions answered locally**: share of typed questions the trained model answered; "
        "the rest went to the fallback.",
        "- **Final accuracy** uses a *simulated* LLM fallback (gold answers), so it is an upper bound.",
        "- **Tokens** are chars/4 estimates of an LLM answering the same typed questions, not API measurements.",
        f"- {skipped} hand-written solver problem is outside the v0 template and is excluded.", ""])
    Path("reports").mkdir(exist_ok=True)
    Path("reports/poc.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
