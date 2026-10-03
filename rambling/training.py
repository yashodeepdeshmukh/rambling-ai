"""Shared helpers: synthetic splits -> Laya-shaped records via the oracle."""

from __future__ import annotations

from . import synth
from .deciders import OracleDecider, Recorder
from .nodered import build as nr
from .solvers import linear

TASKS = {
    "solver": (synth.solver_split, linear.gold_answers, linear.build),
    "nodered": (synth.nodered_split, nr.gold_answers, nr.build),
}


def examples(task: str, split: str, n: int, seed: int = 0):
    return TASKS[task][0](split, n, seed)[0]


def records_for(task: str, exs) -> list[dict]:
    _, gold, build = TASKS[task]
    out = []
    for ex in exs:
        rec = Recorder(OracleDecider(gold(ex)), {"id": ex["id"]})
        build(ex["text"], rec)
        out += rec.records
    return out
