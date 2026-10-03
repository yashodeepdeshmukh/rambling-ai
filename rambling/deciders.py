"""Deciders usable today, before a Laya checkpoint exists.

- OracleDecider replays gold labels: proves the builders end to end and emits training data.
- RuleDecider is a generic lexical-overlap baseline: the floor a trained model must beat.
- Recorder logs every request/answer pair (Laya-shaped) for training and auditing.
- Shadow drives the build with one decider while scoring another on the same questions.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from .decide import Answer, Choice, Noul, Question, Score, one_hot, options_of, to_request

DONT_CARE = None  # gold label for a question whose answer is irrelevant on this path


class OracleDecider:
    name = "oracle"

    def __init__(self, gold: dict):
        self.gold = gold

    def decide(self, task, state, questions):
        out = {}
        for qid, q in questions.items():
            if qid not in self.gold:
                raise KeyError(f"no gold label for question {qid!r}")
            g = self.gold[qid]
            if g is DONT_CARE:
                n = len(options_of(q))
                out[qid] = Answer([1.0 / n] * n, q)
            else:
                out[qid] = one_hot(q, g)
        return out


def _toks(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _softmax(xs, temp=1.0):
    m = max(xs)
    es = [math.exp((x - m) / temp) for x in xs]
    z = sum(es)
    return [e / z for e in es]


class RuleDecider:
    """Pick the option sharing the most words with the state; ties go to the earliest option."""

    name = "rule"

    def decide(self, task, state, questions):
        st = _toks(state)
        out = {}
        for qid, q in questions.items():
            if isinstance(q, Choice):
                it = _toks(q.instructions)
                scores = [len(_toks(o) & st) + 0.5 * len(_toks(o) & it) - 0.01 * i
                          for i, o in enumerate(q.options)]
                out[qid] = Answer(_softmax(scores, temp=0.5), q)
            elif isinstance(q, Score):
                n = len(q.levels)
                out[qid] = Answer([1.0 / n] * n, q)
            else:
                out[qid] = Answer([0.5, 0.5], q)
        return out


class Recorder:
    """Wraps a decider and keeps every call as a Laya-shaped training/audit record."""

    def __init__(self, inner, task_meta: dict | None = None):
        self.inner = inner
        self.meta = task_meta or {}
        self.records: list[dict] = []

    @property
    def name(self):
        return self.inner.name

    def decide(self, task, state, questions):
        answers = self.inner.decide(task, state, questions)
        labels = {}
        for qid, a in answers.items():
            uniform = len(set(a.probs)) == 1 and len(a.probs) > 1
            labels[qid] = None if uniform else a.value
        self.records.append({"task": task, **self.meta, **to_request(state, questions),
                             "labels": labels, "decider": self.inner.name})
        return answers


def question_kind(qid: str) -> str:
    """'c2.v1.sign' -> 'c.v.sign': groups accuracy by kind of decision."""
    return re.sub(r"\d+", "", qid)


class Shadow:
    """Teacher-forced evaluation: `driver` makes the decisions, `candidate` is scored on them."""

    def __init__(self, driver, candidate):
        self.driver, self.candidate = driver, candidate
        self.correct: dict[str, int] = defaultdict(int)
        self.total: dict[str, int] = defaultdict(int)

    @property
    def name(self):
        return self.driver.name

    def decide(self, task, state, questions):
        gold = self.driver.decide(task, state, questions)
        guess = self.candidate.decide(task, state, questions)
        for qid, a in gold.items():
            if len(set(a.probs)) == 1 and len(a.probs) > 1:
                continue  # don't-care
            k = question_kind(qid)
            self.total[k] += 1
            self.correct[k] += guess[qid].value == a.value
        return gold
