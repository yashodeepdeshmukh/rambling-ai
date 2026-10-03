"""Score deciders on both domains.

    python scripts/evaluate.py            # oracle + rule baseline, static checks
    python scripts/evaluate.py --live     # also deploy flows to headless Node-RED and probe them

Two numbers per decider:
  end-to-end   the decider alone builds the artifact (solver answer correct / flow matches gold)
  per-decision teacher-forced accuracy: gold drives the build, the decider is scored on each question
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rambling import datasets  # noqa: E402
from rambling.decide import NotRepresentable  # noqa: E402
from rambling.deciders import OracleDecider, Recorder, RuleDecider, Shadow  # noqa: E402
from rambling.nodered import build as nr, validate as nrv  # noqa: E402
from rambling.solvers import linear  # noqa: E402

DECIDERS = {"oracle": None, "rule": RuleDecider}  # None = per-example oracle


def make(name, gold):
    return OracleDecider(gold) if name == "oracle" else DECIDERS[name]()


def eval_solver(name):
    rows, shadow_c, shadow_t = [], defaultdict(int), defaultdict(int)
    for ex in datasets.solver_problems():
        gold = linear.gold_answers(ex)
        rec = Recorder(make(name, gold))
        try:
            model = linear.build(ex["text"], rec)
            result = linear.solve(model)
            ok, status = linear.check(ex, result), result["status"]
            if result["status"] == "ok" and not ok:
                status = f"solved, wrong answer {result['value']} (expected {ex['answer']})"
        except NotRepresentable:
            ok, status = False, "not representable"
        n_q = sum(len(r["questions"]) for r in rec.records)
        rows.append((ex["id"], ok, status, n_q, len(rec.records)))
        if name != "oracle" and not linear.gold_representable(ex):
            sh = Shadow(OracleDecider(gold), make(name, gold))
            linear.build(ex["text"], sh)
            for k in sh.total:
                shadow_c[k] += sh.correct[k]; shadow_t[k] += sh.total[k]
    return rows, shadow_c, shadow_t


def eval_nodered(name, server=None, live_types=None):
    rows, shadow_c, shadow_t = [], defaultdict(int), defaultdict(int)
    for ex in datasets.nodered_requests():
        gold = nr.gold_answers(ex)
        rec = Recorder(make(name, gold))
        template, slots, nodes = nr.build(ex["text"], rec)
        exact = template == ex["template"] and slots == ex["slots"]
        errors = nrv.validate(nodes, live_types)
        status = "static ok" if not errors else f"static: {errors[0]}"
        ok = exact and not errors
        if server and not errors:
            deployed, msg = server.deploy(nodes)
            probes = [server.probe(p) for p in ex.get("probes", [])] if deployed else []
            ok = ok and deployed and all(p for p, _ in probes)
            status = msg + (f", probes {sum(p for p, _ in probes)}/{len(probes)}" if probes else "")
        if not exact:
            wrong = ["template"] if template != ex["template"] else [k for k in ex["slots"] if slots.get(k) != ex["slots"][k]]
            status += "; wrong " + ", ".join(wrong)
        n_q = sum(len(r["questions"]) for r in rec.records)
        rows.append((ex["id"], ok, status, n_q, len(rec.records)))
        if name != "oracle":
            sh = Shadow(OracleDecider(gold), make(name, gold))
            nr.build(ex["text"], sh)
            for k in sh.total:
                shadow_c[k] += sh.correct[k]; shadow_t[k] += sh.total[k]
    return rows, shadow_c, shadow_t


def report(title, rows, sc, st):
    n_ok = sum(r[1] for r in rows)
    lines = [f"### {title}: {n_ok}/{len(rows)} end-to-end",
             "", "| example | ok | status | questions | decider calls |", "|---|---|---|---|---|"]
    lines += [f"| {i} | {'✅' if ok else '❌'} | {s} | {q} | {c} |" for i, ok, s, q, c in rows]
    if st:
        tc, tt = sum(sc.values()), sum(st.values())
        lines += ["", f"Teacher-forced decision accuracy: **{tc}/{tt} = {tc/tt:.1%}**", "",
                  "| decision kind | accuracy |", "|---|---|"]
        lines += [f"| `{k}` | {sc[k]}/{st[k]} = {sc[k]/st[k]:.0%} |" for k in sorted(st)]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="deploy flows to a headless Node-RED")
    ap.add_argument("--out", default="reports/baseline.md")
    args = ap.parse_args()

    out = ["# Baseline evaluation", "",
           "`oracle` replays gold decisions (pipeline ceiling, and the source of training labels). "
           "`rule` is the lexical-overlap baseline a fine-tuned Laya must beat.", ""]
    for name in DECIDERS:
        out.append(f"## Decider: `{name}`\n")
        out.append(report("Solver (linear/ILP via Z3)", *eval_solver(name)))
        if args.live:
            from rambling.nodered import live
            with live.NodeRedServer() as srv:
                out.append(report("Node-RED (live deploy + probes)", *eval_nodered(name, srv, srv.node_types())))
        else:
            out.append(report("Node-RED (static checks)", *eval_nodered(name)))
    text = "\n".join(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
