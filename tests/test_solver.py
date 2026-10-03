from fractions import Fraction

import pytest

from rambling import datasets, extract
from rambling.deciders import OracleDecider, Recorder, RuleDecider
from rambling.decide import NotRepresentable
from rambling.solvers import linear

PROBLEMS = datasets.solver_problems()


@pytest.mark.parametrize("ex", PROBLEMS, ids=[p["id"] for p in PROBLEMS])
def test_gold_decisions_solve_to_the_expected_answer(ex):
    if ex.get("note", "").startswith("Known v0 gap"):
        with pytest.raises(NotRepresentable):
            linear.build(ex["text"], OracleDecider(linear.gold_answers(ex)))
        return
    model = linear.build(ex["text"], OracleDecider(linear.gold_answers(ex)))
    assert linear.check(ex, linear.solve(model)), model.render()


def test_every_option_list_fits_laya_limits():
    rec = Recorder(RuleDecider())
    for ex in PROBLEMS:
        linear.build(ex["text"], rec)
    for r in rec.records:
        for q in r["questions"].values():
            assert len(q.get("options", [])) <= 20


def test_non_unique_answer_is_flagged():
    m = linear.LinearModel(vars=["a", "b"], constraints=[linear.Constraint([Fraction(1), Fraction(1)], Fraction(0), "=", Fraction(10))], target=0)
    assert linear.solve(m)["status"] == "not_unique"


def test_unbounded_objective_is_flagged():
    m = linear.LinearModel(vars=["a"], goal="max", objective=[Fraction(1)],
                           constraints=[linear.Constraint([Fraction(1)], Fraction(0), ">=", Fraction(1))])
    assert linear.solve(m)["status"] == "unbounded"


def test_extract_numbers_handles_words_commas_and_urls():
    assert extract.numbers("twice $1,240 and half, see http://x.io/v2/9") == [2, 1240, Fraction(1, 2)]
