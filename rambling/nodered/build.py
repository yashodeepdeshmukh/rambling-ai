"""Natural-language request -> Node-RED flow JSON, via two batched decision calls."""

from __future__ import annotations

from ..decide import Choice
from .templates import BY_NAME, BY_SUMMARY, OPERATOR_HINTS, SOURCES, TEMPLATES, FlowBuilder

TASK = "nodered.flow"


def template_question() -> dict:
    return {"template": Choice("Which flow pattern does the request describe?",
                               tuple(t.summary for t in TEMPLATES))}


def slot_questions(text: str, template) -> dict:
    return {s.name: Choice(s.question, SOURCES[s.source](text),
                           OPERATOR_HINTS if s.source == "operator" else None)
            for s in template.slots}


def build(text: str, decider) -> tuple[str, dict, list[dict]]:
    """Returns (template name, slot values, flow nodes)."""
    t = BY_SUMMARY[decider.decide(TASK, text, template_question())["template"].value]
    state = f"{text}\n\nFlow pattern: {t.summary}"
    answers = decider.decide(TASK, state, slot_questions(text, t))
    slots = {name: a.value for name, a in answers.items()}
    fb = FlowBuilder(label=text[:60])
    t.build(fb, slots)
    return t.name, slots, fb.nodes


def gold_answers(ex: dict) -> dict:
    return {"template": BY_NAME[ex["template"]].summary, **ex["slots"]}
