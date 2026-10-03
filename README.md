# rambling-ai

Building **solver models** (Z3 linear/integer problems) and **Node-RED flows** from typed
decisions only, so a System One decision model such as
[Laya](https://github.com/receptron/laya) can produce them with zero LLM tokens.

The builders ask Laya/Jev-shaped questions (`choice` / `score` / `noul`). The options come from
deterministic extraction (numbers, entities, URLs, MQTT topics, paths…) or fixed lists, so the
decider never generates text. Today the decisions come from an oracle (gold labels) and a lexical
baseline; a fine-tuned Laya plugs into the same `Decider` interface. See [PLAN.md](PLAN.md).

```
rambling/
  decide.py          typed questions/answers, Laya-shaped request serialization
  deciders.py        OracleDecider, RuleDecider, Recorder (training logs), Shadow (per-decision eval)
  extract.py         candidate extraction: the option lists every decision chooses from
  solvers/linear.py  word problem -> decisions -> Z3 model -> unique/bounded solution
  nodered/           10 flow templates, builder, static validator, live headless Node-RED harness
data/                solver_problems.jsonl, nodered_requests.jsonl, train/ (exported records)
scripts/             evaluate.py (baseline report), export_training.py (Laya training records)
reports/baseline.md  latest evaluation
```

## Quickstart

```bash
pip install -e '.[dev]'
(cd nodered && npm install)          # optional: enables live Node-RED deploy + HTTP probes
pytest                               # live test is skipped without nodered/node_modules
python scripts/evaluate.py --live    # writes reports/baseline.md
python scripts/export_training.py    # writes data/train/{solver,nodered}.jsonl
```
