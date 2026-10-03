import pytest

from rambling import cascade, synth, training
from rambling.decide import Answer, Choice
from rambling.deciders import OracleDecider
from rambling.learned import LearnedDecider
from rambling.nodered.templates import TEMPLATES
from rambling.solvers import linear


@pytest.mark.parametrize("variant", [0, 1, 2, 3])
def test_every_synthetic_family_and_variant_produces_valid_examples(variant):
    import random
    r = random.Random(variant)
    for fam in synth.SOLVER_FAMILIES:
        ex = next(e for e in (synth.solver_example(r, fam, variant, i) for i in range(20)) if e)
        model = linear.build(ex["text"], OracleDecider(linear.gold_answers(ex)))
        assert linear.check(ex, linear.solve(model)), ex["text"]
    for t in TEMPLATES:
        assert any(synth.nodered_example(r, t.name, variant, i) for i in range(20)), t.name


def test_splits_never_share_a_phrasing_variant():
    assert set(synth.SPLITS["trainval"]).isdisjoint(synth.SPLITS["test"])
    assert set(synth.SPLITS["train"]).isdisjoint(synth.SPLITS["calib"])


@pytest.fixture(scope="module")
def tiny_model():
    recs = []
    for task in training.TASKS:
        recs += training.records_for(task, training.examples(task, "train", 4))
    m = LearnedDecider()
    m.fit(recs, epochs=5)
    m.calibrate(recs)
    return m


def test_learned_decider_returns_distributions_for_every_question(tiny_model):
    ex = training.examples("nodered", "test", 1)[0]
    qs = {"template": Choice("Which flow pattern does the request describe?", tuple(t.summary for t in TEMPLATES))}
    ans = tiny_model.decide("nodered.flow", ex["text"], qs)["template"]
    assert len(ans.probs) == len(TEMPLATES) and abs(sum(ans.probs) - 1) < 1e-6


def test_save_and_load_round_trip(tmp_path, tiny_model):
    tiny_model.thresholds = {"template": 0.9}
    path = tmp_path / "m.pkl"
    tiny_model.save(path)
    loaded = LearnedDecider.load(path)
    assert loaded.thresholds == {"template": 0.9} and loaded.temps == tiny_model.temps


class Fixed:
    name = "fixed"

    def __init__(self, p):
        self.p = p

    def decide(self, task, state, questions):
        return {qid: Answer([self.p, 1 - self.p], q) for qid, q in questions.items()}


def test_gate_sends_only_low_confidence_questions_to_the_fallback():
    q = {"a": Choice("?", ("x", "y")), "b": Choice("?", ("x", "y"))}
    gated = cascade.GatedDecider(Fixed(0.6), {"a": 0.5, "b": 0.9}, fallback=Fixed(0.0))
    ans = gated.decide("t", "state", q)
    assert ans["a"].value == "x" and ans["b"].value == "y"
    assert gated.ledger.fallback_questions == 1 and gated.ledger.fallback_tokens < gated.ledger.llm_only_tokens


def test_thresholds_must_hold_in_every_fold():
    import numpy as np
    good = [("k", np.array([5.0, 0.0]), 0)] * 20
    bad = [("k", np.array([5.0, 0.0]), 1)] * 20  # confidently wrong on the other phrasing
    assert cascade.fit_thresholds([good], {}) ["k"] <= 0.99
    assert cascade.fit_thresholds([good, bad], {})["k"] > 1


def test_checks_catch_unused_numbers_and_unused_topics():
    m = linear.LinearModel(vars=["a"], constraints=[linear.Constraint([1], 0, "=", 5)], target=0)
    assert cascade.solver_checks("a is 5 and b is 9", m, linear.solve(m))
    assert cascade.nodered_checks("bridge a/b to c/d", "mqtt_monitor", {"broker": "localhost", "topic": "a/b"}, [])
