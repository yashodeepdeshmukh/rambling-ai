"""Confidence-gated cascade: the trained decider answers what it is sure about,
an LLM answers the rest, and hard checks decide whether the result is accepted.

    question batch ─► learned decider ─┬─ max prob ≥ threshold[kind] ─► keep
                                       └─ below ─► fallback (LLM) answers only those
    built artifact ─► hard checks ─┬─ pass ─► accept
                                   └─ fail ─► full fallback rebuild

Thresholds are per decision kind, chosen on held-out data so that kept decisions
reach a target precision. Token counts are estimates (chars / 4), not API measurements.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from . import extract
from .decide import to_request
from .deciders import question_kind
from .nodered import build as nr_build, validate as nr_validate
from .solvers import linear

PROMPT_OVERHEAD = 60      # system prompt + answer-format instructions per LLM call
TOKENS_PER_ANSWER = 6


def estimate_tokens(state: str, questions: dict) -> int:
    req = to_request(state, questions)
    chars = len(state) + sum(len(q["instructions"]) + sum(len(o) + 2 for o in q.get("options", []))
                             for q in req["questions"].values())
    return PROMPT_OVERHEAD + chars // 4 + TOKENS_PER_ANSWER * len(questions)


def fit_thresholds(folds: list[list[tuple]], temps: dict, target: float = 0.98,
                   grid=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)) -> dict:
    """Per decision kind, the lowest max-prob gate that reaches `target` precision in *every* fold.

    `folds` holds, per held-out phrasing, (kind, option scores, gold index) triples from a model
    that never saw that phrasing. Requiring every fold makes the gate robust to the one wording
    the model is most confidently wrong on, not just the average one.
    """
    from .learned import _softmax

    per_kind: dict[str, list[list[tuple[float, bool]]]] = defaultdict(lambda: [[] for _ in folds])
    for i, fold in enumerate(folds):
        for k, scores, gold in fold:
            p = _softmax(scores, temps.get(k, 1.0))
            per_kind[k][i].append((float(p.max()), int(p.argmax()) == gold))
    out = {}
    for k, by_fold in per_kind.items():
        out[k] = 1.01  # never trusted unless some gate works in every fold
        for t in grid:
            ok = True
            for items in by_fold:
                kept = [c for pm, c in items if pm >= t]
                if kept and sum(kept) / len(kept) < target:
                    ok = False
            if ok and sum(pm >= t for items in by_fold for pm, _ in items) >= 10:
                out[k] = t
                break
    return out


@dataclass
class Ledger:
    questions: int = 0
    fallback_questions: int = 0
    fallback_calls: int = 0
    fallback_tokens: int = 0
    llm_only_tokens: int = 0
    uncertain: list = field(default_factory=list)


class GatedDecider:
    """Keeps confident answers from `primary`; asks `fallback` (if any) for the rest."""

    def __init__(self, primary, thresholds: dict, fallback=None):
        self.primary, self.thresholds, self.fallback = primary, thresholds, fallback
        self.ledger = Ledger()
        self.name = f"gated({primary.name})"

    def decide(self, task, state, questions):
        answers = self.primary.decide(task, state, questions)
        self.ledger.questions += len(questions)
        self.ledger.llm_only_tokens += estimate_tokens(state, questions)
        low = {qid: q for qid, q in questions.items()
               if max(answers[qid].probs) < self.thresholds.get(question_kind(qid), 1.01)}
        if low:
            self.ledger.uncertain += [(qid, answers[qid].value, round(max(answers[qid].probs), 3)) for qid in low]
            self.ledger.fallback_questions += len(low)
            self.ledger.fallback_calls += 1
            self.ledger.fallback_tokens += estimate_tokens(state, low)
            if self.fallback is not None:
                answers.update(self.fallback.decide(task, state, low))
        return answers


# ---------------------------------------------------------------- hard checks

def solver_checks(text: str, model, result: dict) -> list[str]:
    problems = []
    if result.get("status") != "ok":
        problems.append(f"solver status {result.get('status')}")
    used = {abs(c) for con in model.constraints for c in con.coeffs + [con.const, con.rhs]}
    used |= {abs(c) for c in (model.objective or [])}
    unused = [extract.fmt(n) for n in extract.numbers(text) if n not in used]
    if unused:
        problems.append(f"numbers in the text not used by the model: {', '.join(unused)}")
    return problems


def nodered_checks(text: str, template: str, slots: dict, nodes: list) -> list[str]:
    problems = list(nr_validate.validate(nodes))
    values = set(slots.values())
    for kind, found in (("url", extract.urls(text)), ("topic", extract.topics(text)),
                        ("path", extract.http_paths(text)), ("file", extract.files(text)),
                        ("host", [h for h in extract.hosts(text) if h != "localhost"])):
        missing = [x for x in found if x not in values]
        if missing:
            problems.append(f"{kind} in the request not used by the flow: {', '.join(missing)}")
    if extract.durations(text) and not any(v.endswith(" seconds") for v in values):
        problems.append("the request mentions a time interval but the flow has none")
    return problems


# ---------------------------------------------------------------- one item through the cascade

@dataclass
class Outcome:
    accepted_automatically: bool   # no LLM involvement at all
    escalated: str                 # "", "partial" or "full"
    problems: list
    ledger: Ledger
    artifact: object = None
    result: object = None


def run_solver(text, decider, thresholds, fallback=None) -> Outcome:
    g = GatedDecider(decider, thresholds, fallback)
    model = linear.build(text, g)
    result = linear.solve(model)
    problems = solver_checks(text, model, result)
    escalated = "partial" if g.ledger.fallback_questions else ""
    if problems:
        escalated = "full"
        if fallback is not None:
            model = linear.build(text, fallback)
            result = linear.solve(model)
        g.ledger.fallback_tokens = g.ledger.llm_only_tokens  # a full rebuild costs about an all-LLM build
    return Outcome(not g.ledger.fallback_questions and not problems, escalated, problems, g.ledger, model, result)


def run_nodered(text, decider, thresholds, fallback=None) -> Outcome:
    g = GatedDecider(decider, thresholds, fallback)
    template, slots, nodes = nr_build.build(text, g)
    problems = nodered_checks(text, template, slots, nodes)
    escalated = "partial" if g.ledger.fallback_questions else ""
    if problems:
        escalated = "full"
        if fallback is not None:
            template, slots, nodes = nr_build.build(text, fallback)
        g.ledger.fallback_tokens = g.ledger.llm_only_tokens
    return Outcome(not g.ledger.fallback_questions and not problems, escalated, problems, g.ledger,
                   nodes, {"template": template, "slots": slots})
