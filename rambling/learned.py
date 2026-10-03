"""A small trained decider: a CPU stand-in for fine-tuned Laya with the same interface.

It scores every (state, question, option) triple with a logistic model over hashed,
domain-agnostic features, then turns the option scores into a distribution with a softmax
and a per-decision-kind temperature fitted on held-out wordings.

The features are generic: where an option is mentioned in the text, the words around
that mention, its order among the options, its distance to the entity named in the
question, whether the partial build already used it, and option×text crosses for
small fixed option sets. Nothing is specific to one problem family.

When Laya weights are reachable, replace `LearnedDecider` with a wrapper around the
fine-tuned checkpoint; training records (data/train, scripts/train.py) and evaluation
stay the same.
"""

from __future__ import annotations

import math
import pickle
import re
from collections import defaultdict
from fractions import Fraction

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier

from . import extract
from .decide import Answer, Choice, Noul, Score, options_of
from .deciders import question_kind

N_FEATURES = 2 ** 21
_TOKEN = re.compile(r"\$?\d[\d,]*(?:\.\d+)?|[a-z][a-z0-9_]*|[.!?;:]", re.I)


# ---------------------------------------------------------------- text analysis

class Doc:
    """Tokenized request text (before the first blank line) and build context (after it)."""

    def __init__(self, state: str):
        text, _, context = state.partition("\n\n")
        self.text, self.context = text, context
        self.toks, self.starts, self.sent = [], [], []
        s = 0
        for m in _TOKEN.finditer(text):
            t = m.group().lower()
            self.toks.append(t)
            self.starts.append(m.start())
            self.sent.append(s)
            if t in ".!?;":
                s += 1
        self.values = [_value(t) for t in self.toks]
        self.words = [_norm(t) for t in self.toks]
        words = [w for w in self.words if w not in ".!?;:"]
        self.bag = sorted(set(words) | {f"{a}_{b}" for a, b in zip(words, words[1:])})
        self.ctx_values = [_value(m.group()) for m in _TOKEN.finditer(context)]
        anchor = next((ln[len("Anchor:"):] for ln in context.splitlines() if ln.startswith("Anchor:")), "")
        avals = {v for v in (_value(t) for t in anchor.split()) if v is not None}
        self.anchor_pos = [i for i, v in enumerate(self.values) if v is not None and v in avals]
        self.n_numbers = sum(v is not None for v in self.values)
        self.n_sent = s + 1

    def tok_at(self, char_pos: int) -> int:
        lo, hi = 0, len(self.starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.starts[mid] <= char_pos:
                lo = mid + 1
            else:
                hi = mid
        return max(lo - 1, 0)


def _value(tok: str):
    t = tok.lower().lstrip("$").replace(",", "")
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return Fraction(t)
    if t in extract.WORD_NUMBERS:
        return Fraction(extract.WORD_NUMBERS[t])
    return None


def _norm(tok: str) -> str:
    return "<num>" if _value(tok) is not None and not tok.isalpha() else tok


def _option_value(opt: str):
    try:
        return Fraction(opt.lstrip("+"))
    except (ValueError, ZeroDivisionError):
        return None


def _surface(opt: str) -> str:
    """The part of an option string that would appear in the request text."""
    if " = " in opt:
        opt = opt.split(" = ", 1)[1]
    if opt.startswith("payload."):
        opt = opt.split(".", 1)[1]
    return opt.lower()


def mentions(doc: Doc, opt: str) -> list[int]:
    val = _option_value(opt)
    if val is not None:
        return [i for i, v in enumerate(doc.values) if v is not None and v == abs(val)]
    s = _surface(opt)
    if not s:
        return []
    pat = re.compile(r"(?<![a-z0-9])" + re.escape(s) + r"(?![a-z0-9])")
    return [doc.tok_at(m.start()) for m in pat.finditer(doc.text.lower())]


def _name_positions(doc: Doc, name: str) -> list[int]:
    stem = name.lower()[:max(4, len(name) - 2)]
    return [i for i, t in enumerate(doc.toks) if t.startswith(stem) or t == name.lower()]


def _bucket(d: int) -> str:
    return "0-2" if d <= 2 else "3-5" if d <= 5 else "6-10" if d <= 10 else "11+"


# ---------------------------------------------------------------- features

def question_features(doc: Doc, qid: str, q) -> list[list[str]]:
    """One feature list per option of question `q`."""
    k = question_kind(qid)
    opts = [str(o) for o in options_of(q)] if not isinstance(q, Noul) else ["no", "yes"]
    n = len(opts)
    instr = q.instructions
    j = (m := re.search(r"#(\d+)", instr)) and m.group(1)
    vi = (m := re.search(r"\bx(\d+)\b", instr)) and m.group(1)
    name = (m := re.search(r"\(([^)]+)\)", instr)) and m.group(1)
    name_pos = _name_positions(doc, name) if name else []
    small = n <= 6
    descriptive = [len(re.findall(r"[a-z]+", o.lower())) >= 3 for o in opts]

    ment = [mentions(doc, o) if not isinstance(q, Noul) else [] for o in opts]
    firsts = sorted({m[0] for m in ment if m})
    lasts = sorted({m[-1] for m in ment if m}, reverse=True)
    dists = [min((abs(p - npos) for p in m for npos in name_pos), default=None) for m in ment]
    finite = [d for d in dists if d is not None]
    closest = min(finite) if finite else None

    out = []
    for idx, o in enumerate(opts):
        f = [f"{k}|bias", f"{k}|idx={min(idx, 10)}", f"{k}|o={o}" if small else f"{k}|n>6"]
        if idx == n - 1:
            f.append(f"{k}|lastopt")
        val = _option_value(o)
        if val is not None and o[:1] in "+-":
            f.append(f"{k}|signed={o[0]}")
        for t in re.findall(r"[a-z]+", o.lower())[:12]:
            f.append(f"{k}|ot={t}")
        if small or descriptive[idx]:  # option × text crosses for fixed vocabularies
            keys = [f"o={o}"] if small else [f"ot={t}" for t in re.findall(r"[a-z]+", o.lower())[:12]]
            for key in keys:
                for w in doc.bag:
                    f.append(f"{k}|{key}|w={w}")
                if j or vi:
                    for w in doc.bag:
                        f.append(f"{k}|{key}|j={j}|v={vi}|w={w}")
                f.append(f"{k}|{key}|nnum={min(doc.n_numbers, 9)}|nsent={min(doc.n_sent, 6)}")
                if name:
                    f.append(f"{k}|{key}|name={name.lower()}")

        m = ment[idx]
        f.append(f"{k}|in_text={bool(m)}")
        if m:
            first, last = m[0], m[-1]
            r_first = firsts.index(first)
            r_last = lasts.index(last)
            f += [f"{k}|mrank={min(r_first, 5)}", f"{k}|mrank_last={min(r_last, 5)}",
                  f"{k}|nment={min(len(m), 3)}", f"{k}|sent={min(doc.sent[first], 6)}"]
            if j:
                f += [f"{k}|j={j}|mrank={min(r_first, 5)}", f"{k}|j={j}|sent={min(doc.sent[first], 6)}"]
            if vi:
                f += [f"{k}|v={vi}|mrank={min(r_first, 5)}"]
                for p in m[:2]:
                    for d in (1, 2):
                        if p - d >= 0:
                            f.append(f"{k}|v={vi}|L{d}={doc.words[p - d]}")
                        if p + d < len(doc.words):
                            f.append(f"{k}|v={vi}|R{d}={doc.words[p + d]}")
            if doc.anchor_pos:
                ad = min(abs(p - a) for p in m for a in doc.anchor_pos)
                same = any(doc.sent[p] == doc.sent[a] for p in m for a in doc.anchor_pos)
                f += [f"{k}|adist={_bucket(ad)}", f"{k}|asame={same}",
                      f"{k}|asame={same}|dist={_bucket(d) if (d := dists[idx]) is not None else 'na'}"]
                if dists[idx] is not None and dists[idx] == closest:
                    f.append(f"{k}|asame={same}|closest")
            for p in m[:3]:
                for d in (1, 2, 3):
                    if p - d >= 0:
                        f.append(f"{k}|L{d}={doc.words[p - d]}")
                    if p + d < len(doc.words):
                        f.append(f"{k}|R{d}={doc.words[p + d]}")
                for w in doc.words[max(0, p - 6): p + 7]:
                    f.append(f"{k}|near={w}")
                    if j:
                        f.append(f"{k}|j={j}|near={w}")
        if val is not None and not m and not small:  # constants absent from the text
            f.append(f"{k}|const_o={o}|j={j}")
            if name:
                f.append(f"{k}|const_o={o}|name={name.lower()}")
            for w in doc.bag:
                f.append(f"{k}|const_o={o}|w={w}")
        d = dists[idx]
        if d is not None:
            f.append(f"{k}|dist={_bucket(d)}")
            if d == closest:
                f.append(f"{k}|closest")
        if val is not None:
            used = sum(v == abs(val) for v in doc.ctx_values if v is not None)
            f.append(f"{k}|used={min(used, 2)}")
            if j:
                f.append(f"{k}|j={j}|used={min(used, 2)}")
        out.append(f)
    return out


# ---------------------------------------------------------------- model

def _softmax(z, temp):
    z = np.asarray(z, dtype=float) / temp
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


ALPHA = {"solver.linear": 2e-6, "nodered.flow": 1e-4}  # chosen on the calibration split


class LearnedDecider:
    """One hashed logistic scorer per task, with a softmax temperature per decision kind."""

    name = "learned"

    def __init__(self, clfs=None, temps=None, thresholds=None):
        self.hasher = FeatureHasher(n_features=N_FEATURES, input_type="string", alternate_sign=False)
        self.clfs = clfs or {}
        self.temps = temps or {}
        self.thresholds = thresholds or {}  # cascade gates, fitted by cascade.fit_thresholds

    # -- training -------------------------------------------------------------
    @staticmethod
    def _rebuild_question(body):
        if body["type"] == "choice":
            return Choice(body["instructions"], tuple(body["options"]))
        if body["type"] == "score":
            return Score(body["instructions"], tuple(body["levels"]))
        return Noul(body["instructions"])

    def iter_labelled(self, records):
        """Yields (task, qid, question, per-option features, gold index) for labelled questions."""
        for r in records:
            doc = Doc(r["state"])
            for qid, body in r["questions"].items():
                label = r["labels"].get(qid)
                if label is None:
                    continue
                q = self._rebuild_question(body)
                opts = list(options_of(q))
                if label not in opts:
                    continue
                yield r["task"], qid, q, question_features(doc, qid, q), opts.index(label)

    def fit(self, records, epochs: int = 10, seed: int = 0, alpha: dict | None = None):
        alpha = {**ALPHA, **(alpha or {})}
        X, y = defaultdict(list), defaultdict(list)
        for task, _, _, feats, gold in self.iter_labelled(records):
            for i, f in enumerate(feats):
                X[task].append(f)
                y[task].append(int(i == gold))
        for task in X:
            self.clfs[task] = SGDClassifier(loss="log_loss", alpha=alpha.get(task, 1e-5), max_iter=epochs,
                                            tol=None, class_weight="balanced", random_state=seed)
            self.clfs[task].fit(self.hasher.transform(X[task]), np.array(y[task]))
        return {t: len(v) for t, v in y.items()}

    def scored(self, records):
        """(kind, option scores, gold index) for every labelled question in `records`."""
        return [(question_kind(qid), self._scores(task, feats), gold)
                for task, qid, q, feats, gold in self.iter_labelled(records)]

    def calibrate(self, records=None, scored=None,
                  grid=(0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)):
        """Fit one softmax temperature per decision kind (min NLL) on held-out scores."""
        by_kind = defaultdict(list)
        for k, s, g in (scored if scored is not None else self.scored(records)):
            by_kind[k].append((s, g))
        for k, items in by_kind.items():
            def nll(t):
                return -sum(math.log(max(_softmax(s, t)[g], 1e-12)) for s, g in items)
            self.temps[k] = min(grid, key=nll)
        return dict(self.temps)

    # -- inference ------------------------------------------------------------
    def _scores(self, task, feats):
        return self.clfs[task].decision_function(self.hasher.transform(feats))

    def decide(self, task, state, questions):
        doc = Doc(state)
        out = {}
        for qid, q in questions.items():
            feats = question_features(doc, qid, q)
            probs = _softmax(self._scores(task, feats), self.temps.get(question_kind(qid), 1.0))
            out[qid] = Answer([float(p) for p in probs], q)
        return out

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"clfs": self.clfs, "temps": self.temps, "thresholds": self.thresholds}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["clfs"], d["temps"], d.get("thresholds"))
