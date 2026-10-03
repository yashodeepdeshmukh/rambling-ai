"""Per-item evaluation shared by scripts/poc_eval.py and scripts/zero_shot_eval.py."""

from . import cascade
from .deciders import OracleDecider
from .nodered import build as nr
from .solvers import linear


def solver_item(model, ex, thresholds=None):
    gold = linear.gold_answers(ex)
    alone = linear.check(ex, linear.solve(linear.build(ex["text"], model)))
    out = cascade.run_solver(ex["text"], model, thresholds if thresholds is not None else model.thresholds, OracleDecider(gold))
    return alone, out, linear.check(ex, out.result)


def nodered_item(model, ex, server=None, thresholds=None):
    gold = nr.gold_answers(ex)
    t, s, _ = nr.build(ex["text"], model)
    alone = t == ex["template"] and s == ex["slots"]
    out = cascade.run_nodered(ex["text"], model, thresholds if thresholds is not None else model.thresholds, OracleDecider(gold))
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


HEADER = [
    "| test set | items | decider alone, end-to-end | zero-LLM (auto-accepted) | precision of auto-accepted | "
    "decisions answered locally | full escalations | final accuracy (simulated LLM fallback) | "
    "est. LLM tokens saved vs all-LLM |",
    "|---|---|---|---|---|---|---|---|---|",
]

