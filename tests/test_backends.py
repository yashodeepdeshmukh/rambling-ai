import importlib.util
import shutil

import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rambling.backends import from_laya_answer, to_laya_question  # noqa: E402
from rambling.decide import Choice, Noul, Score  # noqa: E402


def test_questions_round_trip_through_the_laya_format():
    c = Choice("Pick", ("gt", "lt"), ("above", None))
    assert to_laya_question(c) == {"type": "choice", "instructions": "Pick", "criteria": {"gt": "above", "lt": None}}
    a = from_laya_answer(c, {"probabilities": {"gt": 0.8, "lt": 0.2}})
    assert a.value == "gt"
    assert from_laya_answer(Noul("Yes?"), {"noul": 0.7}).value is True
    s = Score("How?", ("low", "high"))
    assert from_laya_answer(s, {"probabilities": {"0": 0.1, "1": 0.9}}).value == 1


@pytest.mark.skipif(importlib.util.find_spec("onnx") is None or importlib.util.find_spec("tokenizers") is None,
                    reason="needs onnx + tokenizers to build the dummy bundle")
def test_laya_bridge_plumbing_with_a_dummy_bundle(tmp_path):
    from rambling.backends import LayaDecider, bridge_available
    if not bridge_available():
        pytest.skip("run `npm install` in laya_bridge/")
    import laya_dummy
    lay = LayaDecider(model_dir=laya_dummy.build(tmp_path / "bundle"))
    try:
        ans = lay.decide("t", "Bridge messages from a/b to c/d.", {
            "c": Choice("Which?", ("first", "second", "third")), "n": Noul("Urgent?"),
            "s": Score("Level?", ("low", "mid", "high"))})
        assert ans["c"].value == "first"  # the dummy prefers the earliest option
        assert abs(sum(ans["c"].probs) - 1) < 1e-3 and len(ans["s"].probs) == 3
        assert lay.input_tokens > 0
    finally:
        lay.close()


@pytest.mark.skipif(importlib.util.find_spec("wordllama") is None, reason="pip install wordllama")
def test_wordllama_zero_shot_answers_extractive_slots(tmp_path):
    from rambling.backends import WordLlamaDecider, load_wordllama
    from rambling.nodered import build as nr
    d = WordLlamaDecider(load_wordllama(tmp_path))
    text = "Poll https://api.example.com/health every 30 seconds and show it in debug."
    t = nr.BY_NAME["poll_http_debug"]
    ans = d.decide(nr.TASK, text, nr.slot_questions(text, t))
    assert ans["url"].value == "https://api.example.com/health"
