"""Typed questions and answers, mirroring the Laya / Jev request shape.

A *decider* takes a state (text) plus a dict of typed questions and returns one
answer per question with a probability distribution. Builders in this package
only ever talk to a decider, so the heuristic deciders used today can later be
swapped for a fine-tuned Laya checkpoint without touching the builders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Union


@dataclass(frozen=True)
class Choice:
    instructions: str
    options: tuple[str, ...]
    type: str = field(default="choice", init=False)

    def __post_init__(self):
        if not self.options:
            raise ValueError(f"choice without options: {self.instructions!r}")
        if len(set(self.options)) != len(self.options):
            raise ValueError(f"duplicate options: {self.options}")


@dataclass(frozen=True)
class Score:
    instructions: str
    levels: tuple[str, ...]
    type: str = field(default="score", init=False)


@dataclass(frozen=True)
class Noul:
    instructions: str
    type: str = field(default="noul", init=False)


Question = Union[Choice, Score, Noul]


@dataclass
class Answer:
    """`probs` is over options (choice), levels (score) or [false, true] (noul)."""

    probs: list[float]
    question: Question

    @property
    def value(self):
        q = self.question
        if isinstance(q, Noul):
            return self.probs[1] >= 0.5
        best = max(range(len(self.probs)), key=self.probs.__getitem__)
        return q.options[best] if isinstance(q, Choice) else best

    @property
    def confidence(self) -> float:
        """1 - H(p)/ln K: 0 for a uniform guess, 1 for certainty."""
        k = len(self.probs)
        if k < 2:
            return 1.0
        h = -sum(p * math.log(p) for p in self.probs if p > 0)
        return 1.0 - h / math.log(k)


def options_of(q: Question) -> tuple:
    if isinstance(q, Choice):
        return q.options
    if isinstance(q, Score):
        return tuple(range(len(q.levels)))
    return (False, True)


def one_hot(q: Question, value) -> Answer:
    opts = options_of(q)
    if value not in opts:
        raise NotRepresentable(f"{value!r} not among {opts} for {q.instructions!r}")
    return Answer([1.0 if o == value else 0.0 for o in opts], q)


class NotRepresentable(Exception):
    """The gold answer cannot be expressed with the options the builder offered."""


class Decider(Protocol):
    def decide(self, task: str, state: str, questions: dict[str, Question]) -> dict[str, Answer]: ...


def to_request(state: str, questions: dict[str, Question]) -> dict:
    """Serialize to the Jev/Laya-style request body: {state, questions: {id: {...}}}."""
    out = {}
    for qid, q in questions.items():
        body = {"type": q.type, "instructions": q.instructions}
        if isinstance(q, Choice):
            body["options"] = list(q.options)
        elif isinstance(q, Score):
            body["levels"] = list(q.levels)
        out[qid] = body
    return {"state": state, "questions": out}
