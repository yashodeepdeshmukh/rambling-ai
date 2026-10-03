# rambling-ai

Building **solver models** (Z3 linear/integer problems) and **Node-RED flows** from typed
decisions only, so a System One decision model such as
[Laya](https://github.com/receptron/laya) can produce them with zero LLM tokens.

The builders ask Laya/Jev-shaped questions (`choice` / `score` / `noul`). The options come from
deterministic extraction (numbers, entities, URLs, MQTT topics, paths…) or fixed lists, so the
decider never generates text. A fine-tuned Laya plugs into the same `Decider` interface.
See [PLAN.md](PLAN.md).

**Running on your own machine (needed for the real Laya model): see [LOCAL_RUN.md](LOCAL_RUN.md).**

## Zero-shot PoC: pretrained model, no fine-tuning

`scripts/zero_shot_eval.py --backend laya` runs Convai's **Laya** unchanged through
`laya_bridge/` (Node, `@receptron/laya`, ONNX Runtime) behind the same cascade. Only per-kind
temperatures and cascade gates are calibrated; the weights are never updated. Laya's weights come
from Hugging Face, which the cloud build environment blocks, so the Laya numbers have to be produced
locally. The bridge is tested here against a dummy model with Laya's exact interface.

The same harness runs **WordLlama** (a frozen embedding model whose weights ship in its PyPI wheel)
as a floor (`reports/zero_shot_wordllama.md`). It is near-perfect on extraction slots (URLs,
paths, topics, files) and near random on reasoning decisions (coefficients, number of unknowns,
message field). The cross-phrasing gates stop it from auto-accepting anything unreliable: every item
it handled with no LLM call was correct, but that was only 2% of the synthetic flow requests.

## Proof of concept: trained stand-in

Laya weights and an LLM API aren't reachable from the build environment, so the PoC trains a
**CPU stand-in decider** (hashed logistic scorer over generic features, `rambling/learned.py`) on
synthetic data and runs it behind a **confidence-gated cascade** (`rambling/cascade.py`):
confident decisions are kept, the rest go to an LLM fallback (simulated here by gold answers),
and hard checks (solver status, unused numbers/topics/URLs, flow validation) can force a full
escalation. Results on a phrasing never used in training or calibration, plus the hand-written sets
(`reports/poc.md`):

| test set | zero-LLM items | precision of zero-LLM items | decisions answered locally | est. LLM tokens saved |
|---|---|---|---|---|
| Node-RED, unseen phrasing (400) | 38% | 100% | 86% | 69% |
| Node-RED, hand-written (20, deployed + probed live) | 80% | 100% | 95% | 92% |
| Solver, unseen phrasing (480) | 0% | – | 33% | 19% |
| Solver, hand-written (14) | 0% | – | 44% | 29% |

The solver side never runs fully automatic because coefficient choices (~50% accurate on unseen
phrasing) are never trusted; that is the main thing a pretrained encoder like Laya should fix.

```
rambling/
  decide.py          typed questions/answers, Laya-shaped request serialization
  deciders.py        OracleDecider, RuleDecider, Recorder (training logs), Shadow (per-decision eval)
  learned.py         trained stand-in decider (per-task logistic scorers, per-kind temperatures)
  backends.py        pretrained deciders: Laya (via laya_bridge/) and WordLlama, no fine-tuning
  zero_shot.py       calibration of a frozen decider (temperatures + cross-phrasing gates)
  cascade.py         confidence gates, LLM fallback hook, hard checks, token ledger
  synth.py           synthetic data: 12 solver families + 10 flow templates x 4 phrasings
  extract.py         candidate extraction: the option lists every decision chooses from
  solvers/linear.py  word problem -> decisions -> Z3 model -> unique/bounded solution
  nodered/           10 flow templates, builder, static validator, live headless Node-RED harness
data/                solver_problems.jsonl, nodered_requests.jsonl, train/ (exported records)
laya_bridge/         Node JSON-lines bridge to @receptron/laya
scripts/             zero_shot_eval.py, train.py, poc_eval.py, demo.py, evaluate.py, export_training.py
reports/             zero_shot_wordllama.md, poc.md, training.md, baseline.md
```

## Quickstart

```bash
pip install -e '.[dev]'
(cd nodered && npm install)          # optional: enables live Node-RED deploy + HTTP probes
pytest                               # live test is skipped without nodered/node_modules
python scripts/train.py              # ~3 min CPU -> models/decider.pkl, reports/training.md
python scripts/poc_eval.py --live    # -> reports/poc.md
python scripts/demo.py "A farm has chickens and cows. There are 30 heads and 74 legs. How many cows?"
python scripts/demo.py "Expose /check: reply ALERT if the temperature in the request is above 40, else OK." --deploy
python scripts/evaluate.py --live    # oracle vs lexical baseline -> reports/baseline.md
python scripts/export_training.py    # Laya-shaped records -> data/train/{solver,nodered}.jsonl
```
