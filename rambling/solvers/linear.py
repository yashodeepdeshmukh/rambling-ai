"""Linear / integer word problems -> Z3, built only from typed decisions.

Template (v0):
    unknowns x1..xn (integers or reals, optionally nonnegative)
    constraints   sum(sign_i * mag_i * x_i) + const  (= | <= | >=)  rhs
    goal          value of one unknown | max/min of sum(sign_i * mag_i * x_i)

Every coefficient, constant and right-hand side is chosen from numbers found in the
text plus a few common constants, so a decider never has to write a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import z3

from .. import extract
from ..decide import Choice, NotRepresentable, Noul
from ..deciders import DONT_CARE

TASK = "solver.linear"
CONSTANTS = [1, 2, 3, 4, 5, 6, 10, 12, 24, 60, 100]
MAX_VARS, MAX_CONS = 3, 4
GOALS = {
    "value": "the value of one unknown",
    "max": "the largest possible value of an expression",
    "min": "the smallest possible value of an expression",
}
DOMAINS = ("integers", "real numbers")
SIGNS = ("+", "-", "none")
RELS = ("=", "<=", ">=")
# Short descriptions sent with the options (Laya "criteria"); they matter most without fine-tuning.
HINTS = {
    "goal": ("how many, what is, find the number or value",
             "maximum, largest, most, maximize, greatest profit",
             "minimum, smallest, least, cheapest, minimize cost"),
    "domain": ("whole countable things: animals, people, items, coins, tickets",
               "measurements: lengths, hours, money amounts, weights that can be fractional"),
    "signs": ("the quantity is added or counted in this constraint",
              "the quantity is subtracted: difference, fewer, less than, times as many as",
              "the quantity does not appear in this constraint"),
    "rels": ("is, equals, total, exactly, in all", "at most, no more than, up to, available, limit",
             "at least, no less than, needs, requires, minimum"),
}


@dataclass
class Constraint:
    coeffs: list[Fraction]
    const: Fraction
    rel: str
    rhs: Fraction


@dataclass
class LinearModel:
    vars: list[str]
    domain: str = "integers"
    nonneg: bool = True
    constraints: list[Constraint] = field(default_factory=list)
    goal: str = "value"
    target: int | None = None
    objective: list[Fraction] | None = None

    def render(self) -> str:
        kind = self.domain + (", nonnegative" if self.nonneg else "")
        lines = ["Unknowns: " + ", ".join(f"x{i+1} = {v}" for i, v in enumerate(self.vars)) + f" ({kind})"]
        if self.constraints:
            lines.append("Constraints:")
            lines += [f"  {j+1}. {_render(c.coeffs, c.const)} {c.rel} {extract.fmt(c.rhs)}"
                      for j, c in enumerate(self.constraints)]
        if self.goal == "value" and self.target is not None:
            lines.append(f"Find: x{self.target+1}")
        elif self.objective is not None:
            lines.append(f"{'Maximize' if self.goal == 'max' else 'Minimize'}: {_render(self.objective, 0)}")
        return "\n".join(lines)


def _render(coeffs, const) -> str:
    parts = []
    for i, c in enumerate(coeffs):
        if c:
            mag = "" if abs(c) == 1 else extract.fmt(abs(c)) + "*"
            parts.append(("- " if c < 0 else "+ ") + f"{mag}x{i+1}")
    if const:
        parts.append(("- " if const < 0 else "+ ") + extract.fmt(abs(const)))
    s = " ".join(parts) or "0"
    return s[2:] if s.startswith("+ ") else s


# ---------------------------------------------------------------- option lists

def magnitude_options(text: str) -> tuple[str, ...]:
    mags = []
    for v in [n for n in extract.numbers(text) if n > 0] + [Fraction(c) for c in CONSTANTS]:
        s = extract.fmt(v)
        if s not in mags:
            mags.append(s)
    return tuple(mags[:20])


def const_options(text: str) -> tuple[str, ...]:
    opts = ["0"]
    for v in extract.numbers(text):
        opts += [f"+{extract.fmt(v)}", f"-{extract.fmt(v)}"]
    return tuple(dict.fromkeys(opts))[:19]


def rhs_options(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(["0"] + [extract.fmt(v) for v in extract.numbers(text)]))[:20]


def var_label(i: int, name: str) -> str:
    return f"x{i+1} = {name}"


# ---------------------------------------------------------------- builder

def header_questions() -> dict:
    return {
        "n_vars": Choice("How many unknown quantities must be found or chosen?",
                         tuple(str(i) for i in range(1, MAX_VARS + 1))),
        "domain": Choice("The unknowns are which kind of number?", DOMAINS, HINTS["domain"]),
        "nonneg": Noul("Every unknown quantity is zero or positive."),
        "goal": Choice("What does the question ask for?", tuple(GOALS.values()), HINTS["goal"]),
        "n_cons": Choice("How many equations or inequalities does the text state, "
                         "not counting nonnegativity?", tuple(str(i) for i in range(1, MAX_CONS + 1))),
    }


def var_questions(text: str, n: int) -> dict:
    ents = tuple(extract.entities(text))
    return {f"var{i+1}": Choice(f"Unknown x{i+1} is the amount of what? Unknowns are numbered "
                                f"in order of first mention.", ents) for i in range(n)}


def term_questions(prefix: str, label: str, model: LinearModel, mags) -> dict:
    qs = {}
    for i, v in enumerate(model.vars):
        qs[f"{prefix}.v{i+1}.sign"] = Choice(f"{label}: sign of the x{i+1} ({v}) term", SIGNS, HINTS["signs"])
        qs[f"{prefix}.v{i+1}.mag"] = Choice(f"{label}: size of the x{i+1} ({v}) coefficient", mags)
    return qs


def anchor_questions(text: str, j: int) -> dict:
    """First round of a constraint: the parts that anchor it to a place in the text."""
    label = f"Constraint #{j}"
    return {
        f"c{j}.const": Choice(f"{label}: constant added to the left side", const_options(text)),
        f"c{j}.rel": Choice(f"{label}: relation between the left and right side", RELS, HINTS["rels"]),
        f"c{j}.rhs": Choice(f"{label}: right-hand side value", rhs_options(text)),
    }


def constraint_questions(text: str, model: LinearModel, j: int) -> dict:
    return {**anchor_questions(text, j),
            **term_questions(f"c{j}", f"Constraint #{j}", model, magnitude_options(text))}


def _coeffs(ans: dict, prefix: str, n: int) -> list[Fraction]:
    out = []
    for i in range(n):
        sign = ans[f"{prefix}.v{i+1}.sign"].value
        mag = Fraction(ans[f"{prefix}.v{i+1}.mag"].value)
        out.append(Fraction(0) if sign == "none" else (mag if sign == "+" else -mag))
    return out


def build(text: str, decider) -> LinearModel:
    h = decider.decide(TASK, text, header_questions())
    n_vars, n_cons = int(h["n_vars"].value), int(h["n_cons"].value)
    goal = next(k for k, v in GOALS.items() if v == h["goal"].value)

    v = decider.decide(TASK, text, var_questions(text, n_vars))
    model = LinearModel(vars=[v[f"var{i+1}"].value for i in range(n_vars)],
                        domain=h["domain"].value, nonneg=h["nonneg"].value, goal=goal)

    for j in range(1, n_cons + 1):
        # Two rounds: pick the right-hand side (and constant) first, then the coefficients,
        # so the coefficient questions can be anchored to where that number sits in the text.
        state = f"{text}\n\n{model.render()}\nNow write constraint #{j}; constraints follow the order of the text."
        a = decider.decide(TASK, state, anchor_questions(text, j))
        const, rel, rhs = (a[f"c{j}.const"].value, a[f"c{j}.rel"].value, a[f"c{j}.rhs"].value)
        anchor = " ".join(x for x in (rhs, const.lstrip("+-")) if x != "0")
        state = (f"{state}\nAnchor: {anchor}\n"
                 f"Constraint #{j} so far: ...{'' if const == '0' else ' ' + const} {rel} {rhs}")
        b = decider.decide(TASK, state, term_questions(f"c{j}", f"Constraint #{j}", model, magnitude_options(text)))
        model.constraints.append(Constraint(
            coeffs=_coeffs(b, f"c{j}", n_vars), const=Fraction(const), rel=rel, rhs=Fraction(rhs)))

    state = f"{text}\n\n{model.render()}\nNow state what to solve for."
    if goal == "value":
        labels = tuple(var_label(i, name) for i, name in enumerate(model.vars))
        a = decider.decide(TASK, state, {"target": Choice("Which unknown does the question ask for?", labels)})
        model.target = labels.index(a["target"].value)
    else:
        a = decider.decide(TASK, state, term_questions("obj", "Objective", model, magnitude_options(text)))
        model.objective = _coeffs(a, "obj", n_vars)
    return model


# ---------------------------------------------------------------- gold labels

def gold_answers(ex: dict) -> dict:
    """Translate a gold formalization into the decisions that build it."""
    text, names = ex["text"], ex["vars"]
    n = len(names)
    g = {"n_vars": str(n), "domain": ex["domain"], "nonneg": ex["nonneg"],
         "goal": GOALS[ex["goal"]], "n_cons": str(len(ex["constraints"]))}
    for i, name in enumerate(names):
        g[f"var{i+1}"] = name

    def terms(prefix, coeffs):
        for i, c in enumerate(coeffs):
            c = Fraction(str(c))
            g[f"{prefix}.v{i+1}.sign"] = "none" if c == 0 else ("+" if c > 0 else "-")
            g[f"{prefix}.v{i+1}.mag"] = DONT_CARE if c == 0 else extract.fmt(abs(c))

    for j, c in enumerate(ex["constraints"], start=1):
        terms(f"c{j}", c["coeffs"])
        const = Fraction(str(c.get("const", 0)))
        g[f"c{j}.const"] = "0" if const == 0 else ("+" if const > 0 else "-") + extract.fmt(abs(const))
        g[f"c{j}.rel"] = c["rel"]
        g[f"c{j}.rhs"] = extract.fmt(Fraction(str(c["rhs"])))
    if ex["goal"] == "value":
        g["target"] = var_label(names.index(ex["target"]), ex["target"])
    else:
        terms("obj", ex["objective"])
    return g


# ---------------------------------------------------------------- solving

def _to_fraction(v) -> Fraction:
    if z3.is_int_value(v):
        return Fraction(v.as_long())
    if z3.is_rational_value(v):
        return Fraction(v.numerator_as_long(), v.denominator_as_long())
    raise ValueError(f"not a numeral: {v}")


def solve(model: LinearModel, timeout_ms: int = 5000) -> dict:
    """Returns {'status': 'ok'|'unsat'|'not_unique'|'unbounded'|'unknown'|'invalid', ...}."""
    if len(set(model.vars)) != len(model.vars):
        return {"status": "invalid", "reason": "duplicate unknowns"}
    mk = z3.Int if model.domain == "integers" else z3.Real
    xs = [mk(f"x{i+1}") for i in range(len(model.vars))]

    def lin(coeffs, const=Fraction(0)):
        return z3.Sum([z3.RealVal(str(c)) * x if model.domain != "integers" or c.denominator != 1
                       else z3.IntVal(c.numerator) * x for c, x in zip(coeffs, xs)]) + z3.RealVal(str(const))

    facts = [x >= 0 for x in xs] if model.nonneg else []
    for c in model.constraints:
        lhs, rhs = lin(c.coeffs, c.const), z3.RealVal(str(c.rhs))
        facts.append({"=": lhs == rhs, "<=": lhs <= rhs, ">=": lhs >= rhs}[c.rel])

    if model.goal == "value":
        s = z3.Solver()
        s.set("timeout", timeout_ms)
        s.add(*facts)
        r = s.check()
        if r != z3.sat:
            return {"status": "unsat" if r == z3.unsat else "unknown"}
        m = s.model()
        x = xs[model.target]
        val = m.eval(x, model_completion=True)
        assignment = {name: _to_fraction(m.eval(xi, model_completion=True)) for name, xi in zip(model.vars, xs)}
        s.add(x != val)
        if s.check() != z3.unsat:
            return {"status": "not_unique", "value": _to_fraction(val), "assignment": assignment}
        return {"status": "ok", "value": _to_fraction(val), "assignment": assignment}

    opt = z3.Optimize()
    opt.set("timeout", timeout_ms)
    opt.add(*facts)
    obj = lin(model.objective)
    h = opt.maximize(obj) if model.goal == "max" else opt.minimize(obj)
    r = opt.check()
    if r != z3.sat:
        return {"status": "unsat" if r == z3.unsat else "unknown"}
    if "oo" in str(h.value()) or "epsilon" in str(h.value()):
        return {"status": "unbounded"}
    m = opt.model()
    return {"status": "ok", "value": _to_fraction(m.eval(obj, model_completion=True)),
            "assignment": {name: _to_fraction(m.eval(xi, model_completion=True)) for name, xi in zip(model.vars, xs)}}


def check(ex: dict, result: dict) -> bool:
    return result.get("status") == "ok" and result["value"] == Fraction(str(ex["answer"]))


def gold_representable(ex: dict) -> str | None:
    """None if the template can express the gold formalization, else the reason it can't."""
    from ..deciders import OracleDecider
    try:
        build(ex["text"], OracleDecider(gold_answers(ex)))
    except NotRepresentable as e:
        return str(e)
    return None
