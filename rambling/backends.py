"""Pretrained deciders used as-is (no fine-tuning).

- LayaDecider: Convai's Laya via @receptron/laya (ONNX Runtime, Node), through laya_bridge/.
- WordLlamaDecider: a zero-shot embedding-similarity decider on WordLlama, whose 16 MB weights
  ship inside its PyPI wheel. Much weaker than Laya; it exists so the zero-shot harness can be
  exercised end to end where Laya's weights are unavailable.

Both return the same Answer objects as every other decider, so the builders, cascade and
evaluation code are unchanged.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from .decide import Answer, Choice, Noul, Score

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "laya_bridge" / "bridge.mjs"


# ---------------------------------------------------------------- Laya

def to_laya_question(q) -> dict:
    """Our typed question -> Laya/Jev `system_one` question."""
    if isinstance(q, Choice):
        hints = q.hints or (None,) * len(q.options)
        return {"type": "choice", "instructions": q.instructions,
                "criteria": {o: h for o, h in zip(q.options, hints)}}
    if isinstance(q, Score):
        return {"type": "score", "instructions": q.instructions, "criteria": list(q.levels)}
    return {"type": "noul", "instructions": q.instructions}


def from_laya_answer(q, a: dict) -> Answer:
    if isinstance(q, Choice):
        probs = [float(a["probabilities"].get(o, 0.0)) for o in q.options]
    elif isinstance(q, Score):
        probs = [float(a["probabilities"].get(str(i), 0.0)) for i in range(len(q.levels))]
    else:
        p = float(a["noul"])
        probs = [1.0 - p, p]
    z = sum(probs) or 1.0
    return Answer([p / z for p in probs], q)


def bridge_available() -> bool:
    return (ROOT / "laya_bridge" / "node_modules" / "@receptron" / "laya").exists() and shutil.which("node") is not None


class LayaDecider:
    """Runs Laya through a long-lived Node process (one forward pass per decide() call)."""

    name = "laya"

    def __init__(self, model_dir: str | None = None, threads: int | None = None, timeout: float = 600):
        env = dict(os.environ)
        if model_dir:
            env["LAYA_MODEL_DIR"] = str(model_dir)
        if threads:
            env["LAYA_THREADS"] = str(threads)
        self.proc = subprocess.Popen(["node", str(BRIDGE)], cwd=BRIDGE.parent, env=env, text=True,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=1)
        ready = json.loads(self.proc.stdout.readline() or "{}")
        if not ready.get("ready"):
            raise RuntimeError("Laya bridge failed to start (see stderr above)")
        self.model_dir = ready.get("modelDir")
        self._id = 0
        self.input_tokens = 0
        self.calls = 0

    def decide(self, task, state, questions):
        self._id += 1
        req = {"id": self._id, "state": state,
               "questions": {qid: to_laya_question(q) for qid, q in questions.items()}}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        res = json.loads(self.proc.stdout.readline())
        if "error" in res:
            raise RuntimeError(f"Laya: {res['error']}")
        self.calls += 1
        self.input_tokens += res.get("usage", {}).get("input_tokens", 0)
        return {qid: from_laya_answer(q, res["answers"][qid]) for qid, q in questions.items()}

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------- WordLlama (zero-shot embeddings)

def load_wordllama(cache_dir: str | Path | None = None):
    """Load WordLlama from the weights bundled in its wheel, without network access."""
    import wordllama
    from wordllama import WordLlama

    cache = Path(cache_dir or Path.home() / ".cache" / "wordllama")
    tok_dir = WordLlama.get_file_path("tokenizer", cache)
    tok_dir.mkdir(parents=True, exist_ok=True)
    bundled = Path(wordllama.__file__).parent / "tokenizers" / "l2_supercat_tokenizer_config.json"
    if bundled.exists() and not (tok_dir / bundled.name).exists():
        shutil.copy(bundled, tok_dir)  # the wheel ships it under tokenizers/, the loader looks in tokenizer/
    return WordLlama.load(cache_dir=cache)


_WORD = re.compile(r"\S+")


class WordLlamaDecider:
    """Zero-shot scoring with frozen sentence embeddings.

    - If two or more options appear in the request (extractive questions: topics, numbers, names),
      each mentioned option is scored by how well the words around its mention match the question;
      options that do not appear are pushed down.
    - Otherwise (semantic questions: flow pattern, operator, goal), each option plus its
      description is scored by similarity to the request.
    - Yes/no questions compare the statement with the request.
    Scores become probabilities with a softmax at a fixed sharpness; per-kind temperatures can be
    fitted later without touching the model.
    """

    name = "wordllama"
    SHARPNESS = 20.0

    def __init__(self, model=None):
        self.wl = model or load_wordllama()

    @lru_cache(maxsize=200_000)
    def _emb(self, text: str):
        return self.wl.embed([text], norm=True)[0]

    def _sim(self, a: str, b: str) -> float:
        return float(self._emb(a) @ self._emb(b))

    def decide(self, task, state, questions):
        text = state.split("\n\n", 1)[0]
        words = _WORD.findall(text)
        low = [w.lower().strip(".,;:!?()$'\"") for w in words]
        out = {}
        for qid, q in questions.items():
            if isinstance(q, Noul):
                s = self._sim(text, q.instructions)
                logits = [0.0, s]
            else:
                opts = list(q.options) if isinstance(q, Choice) else list(q.levels)
                hints = (q.hints if isinstance(q, Choice) and q.hints else (None,) * len(opts))
                mentions = [self._mentions(low, str(o)) for o in opts]
                if sum(bool(m) for m in mentions) >= 2:
                    logits = []
                    for o, m in zip(opts, mentions):
                        if not m:
                            logits.append(-1.0)
                            continue
                        logits.append(max(self._sim(q.instructions, " ".join(words[max(0, i - 6): i + 7]))
                                          for i in m[:3]))
                else:
                    logits = [self._sim(text, f"{o}: {h}" if h else str(o)) for o, h in zip(opts, hints)]
            m = max(logits)
            e = [math.exp(self.SHARPNESS * (x - m)) for x in logits]
            z = sum(e)
            out[qid] = Answer([x / z for x in e], q)
        return out

    @staticmethod
    def _mentions(low_words: list[str], opt: str) -> list[int]:
        o = opt.lower()
        if " = " in o:
            o = o.split(" = ", 1)[1]
        if o.startswith("payload."):
            o = o.split(".", 1)[1]
        o = o.lstrip("+")
        return [i for i, w in enumerate(low_words) if w == o or w.replace(",", "") == o]
