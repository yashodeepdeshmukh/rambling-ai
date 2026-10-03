"""Run one request through the trained decider and the cascade, showing every decision.

    python scripts/demo.py "A farm has chickens and cows. There are 30 heads and 74 legs. How many cows?"
    python scripts/demo.py "Expose /check: reply ALERT if the temperature is above 40, else OK." --deploy

No LLM is attached here: decisions below their gate are marked "→ LLM" (what the cascade would
send to the fallback) and the trained model's own answer is used provisionally.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rambling import cascade, extract  # noqa: E402
from rambling.deciders import Recorder, question_kind  # noqa: E402
from rambling.learned import LearnedDecider  # noqa: E402
from rambling.nodered import build as nr  # noqa: E402
from rambling.solvers import linear  # noqa: E402


class Trace:
    """Records each decision with its probability and gate."""

    def __init__(self, inner, thresholds):
        self.inner, self.thresholds, self.rows = inner, thresholds, []
        self.name = inner.name

    def decide(self, task, state, questions):
        ans = self.inner.decide(task, state, questions)
        for qid, a in ans.items():
            gate = self.thresholds.get(question_kind(qid), 1.01)
            self.rows.append((qid, a.value, max(a.probs), gate))
        return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--task", choices=["auto", "solver", "nodered"], default="auto")
    ap.add_argument("--model", default="models/decider.pkl")
    ap.add_argument("--deploy", action="store_true", help="deploy a Node-RED flow to a local headless Node-RED")
    args = ap.parse_args()

    model = LearnedDecider.load(args.model)
    task = args.task
    if task == "auto":
        t = args.text
        task = "nodered" if (extract.urls(t) or extract.topics(t) or extract.http_paths(t)
                             or "mqtt" in t.lower() or "endpoint" in t.lower()) else "solver"
    tr = Trace(model, model.thresholds)

    if task == "solver":
        m = linear.build(args.text, tr)
        result = linear.solve(m)
        problems = cascade.solver_checks(args.text, m, result)
        artifact = m.render() + f"\n→ {result.get('status')}: {result.get('value')}"
    else:
        template, slots, nodes = nr.build(args.text, tr)
        problems = cascade.nodered_checks(args.text, template, slots, nodes)
        out = Path("reports/demo_flow.json")
        out.write_text(json.dumps(nodes, indent=2))
        artifact = f"template: {template}\nslots: {json.dumps(slots)}\nflow JSON: {out} ({len(nodes) - 1} nodes)"

    print(f"task: {task}\n")
    print(f"{'decision':<16} {'answer':<44} {'p':>6} {'gate':>6}")
    for qid, val, p, gate in tr.rows:
        keep = "kept" if p >= gate else "→ LLM"
        print(f"{qid:<16} {str(val)[:44]:<44} {p:6.3f} {('never' if gate > 1 else f'{gate:.3f}'):>6}  {keep}")
    low = sum(p < g for _, _, p, g in tr.rows)
    print(f"\n{artifact}\n")
    print("checks:", "pass" if not problems else "; ".join(problems))
    verdict = ("ACCEPT with zero LLM tokens" if not low and not problems else
               "FULL ESCALATION (checks failed)" if problems else
               f"PARTIAL: {low}/{len(tr.rows)} decisions would go to the LLM")
    print("cascade:", verdict)

    if args.deploy and task == "nodered" and not problems:
        from rambling.nodered import live
        with live.NodeRedServer() as srv:
            ok, msg = srv.deploy(nodes)
            print(f"live Node-RED: {msg}")
            if ok and slots.get("path"):
                body = {slots["property"].split(".", 1)[1]: 1} if "property" in slots else {"demo": 1}
                method = slots.get("method", "POST")
                print(f"{method} {slots['path']} {json.dumps(body)} ->", srv.request(method, slots["path"], body))


if __name__ == "__main__":
    main()
