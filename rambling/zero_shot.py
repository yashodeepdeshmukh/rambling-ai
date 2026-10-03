"""Use a pretrained decider as-is: no weight updates, only calibration.

Two things are fitted from labelled data, both outside the model:
  - a softmax temperature per decision kind (rescales the model's own probabilities), and
  - per-kind cascade gates, cross-phrasing: a gate must reach the target precision on every
    held-out phrasing variant (see cascade.fit_thresholds).
`--no-calibration` in scripts/zero_shot_eval.py skips both and uses one fixed gate.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np

from . import cascade
from .decide import Answer, from_request, options_of
from .deciders import question_kind

GRID = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0, 8.0)


class Tempered:
    """Applies per-kind temperatures to a frozen decider's probabilities."""

    def __init__(self, inner, temps: dict):
        self.inner, self.temps = inner, temps
        self.name = inner.name

    def decide(self, task, state, questions):
        out = self.inner.decide(task, state, questions)
        for qid, a in out.items():
            t = self.temps.get(question_kind(qid), 1.0)
            if t != 1.0:
                z = np.log(np.clip(np.array(a.probs), 1e-12, 1)) / t
                e = np.exp(z - z.max())
                out[qid] = Answer(list(e / e.sum()), a.question)
        return out


def score_records(decider, records, progress: str = "") -> list[tuple]:
    """(kind, log-probs, gold index) for each labelled question, one decide() call per record."""
    out, t0 = [], time.time()
    for i, r in enumerate(records):
        qs = {qid: from_request(b) for qid, b in r["questions"].items() if r["labels"].get(qid) is not None}
        if not qs:
            continue
        ans = decider.decide(r["task"], r["state"], qs)
        for qid, q in qs.items():
            opts = list(options_of(q))
            label = r["labels"][qid]
            if label in opts:
                out.append((question_kind(qid), np.log(np.clip(np.array(ans[qid].probs), 1e-12, 1)), opts.index(label)))
        if progress and (i + 1) % 200 == 0:
            print(f"  {progress}: {i + 1}/{len(records)} records ({time.time() - t0:.0f}s)", flush=True)
    return out


def fit_temperatures(scored) -> dict:
    by_kind = defaultdict(list)
    for k, s, g in scored:
        by_kind[k].append((s, g))
    temps = {}
    for k, items in by_kind.items():
        def nll(t):
            tot = 0.0
            for s, g in items:
                z = s / t
                z = z - z.max()
                tot -= z[g] - math.log(np.exp(z).sum())
            return tot
        temps[k] = min(GRID, key=nll)
    return temps


def calibrate(decider, folds: list[list[dict]], target: float = 0.98):
    """Returns (temps, thresholds, per-fold scores) for a frozen decider."""
    scored = [score_records(decider, recs, progress=f"calibration fold {i}") for i, recs in enumerate(folds)]
    temps = fit_temperatures([x for f in scored for x in f])
    thresholds = cascade.fit_thresholds(scored, temps, target=target)
    return temps, thresholds, scored
