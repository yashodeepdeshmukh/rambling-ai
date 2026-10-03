# Run on your own machine

Use this to run the PoCs on your own PC, where Hugging Face is reachable and your CPU or GPU can
run the real **Laya** model. The cloud session that built this couldn't download Laya's weights.
Everything else was run and tested there.

## TODO checklist

- [ ] 1. Install prerequisites: Python ≥ 3.10, Node.js ≥ 20, git, about 3 GB free disk, 4 GB RAM.
- [ ] 2. Clone the repo and check out `claude/laya-decision-prototype`.
- [ ] 3. Install the Python and Node dependencies (below).
- [ ] 4. Run the test suite: everything should pass, with no skips once the optional parts are installed.
- [ ] 5. **Smoke-test Laya**: the first run downloads about 1.7 GB.
- [ ] 6. Run the **zero-shot Laya PoC**, starting with the quick settings.
- [ ] 7. Optional: run the full-size zero-shot PoC, on a GPU if you have one.
- [ ] 8. Share `reports/zero_shot_laya.md` back in the Claude session, and decide the next step
      (see "What to look at").

## 1–3. Setup

```bash
git clone https://github.com/yashodeepdeshmukh/rambling-ai.git
cd rambling-ai
git checkout claude/laya-decision-prototype

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev,poc,zeroshot]"

(cd laya_bridge && npm install)      # Laya runner (@receptron/laya + ONNX Runtime)
(cd nodered && npm install)          # optional: live Node-RED deploy + HTTP probes
```

On Windows PowerShell, run the two `npm install` lines as `cd laya_bridge; npm install; cd ..`,
and set environment variables with `$env:NAME="value"` instead of a `NAME=value` prefix.

If `npm install` in `laya_bridge/` fails while downloading CUDA binaries, and you don't need an
NVIDIA GPU, skip that step:

```bash
cd laya_bridge && ONNXRUNTIME_NODE_INSTALL_CUDA=skip npm install
```

## 4. Tests

```bash
pytest -q
```

The Laya test uses a generated *dummy* model with Laya's exact input/output format. It checks the
plumbing (Python → Node bridge → ONNX Runtime → typed answers), not answer quality.

## 5. Smoke-test Laya

```bash
python scripts/demo.py "A farm has chickens and cows. There are 30 heads and 74 legs in total. How many cows are there?" --backend laya
```

The first call downloads the ONNX bundle (about 1.7 GB) to `~/.cache/receptron-laya`, or to
`LAYA_CACHE` if you set it. Later runs work offline.

The output lists every decision Laya made, with its probability and whether the cascade would
keep it or send it to the LLM. Then it shows the built model, its solution and the cascade's
verdict. Before step 6 has calibrated the gates, every decision kind uses one fixed gate
(`--gate`, default 0.9).

Node-RED example, deployed to a local headless Node-RED:

```bash
python scripts/demo.py "Expose /check: reply ALERT if the temperature in the request is above 40, else OK." --backend laya --deploy
```

## 6. Zero-shot Laya PoC (no fine-tuning)

Quick run first. On a laptop CPU, expect very roughly 10–30 minutes; the timing hasn't been
measured.

```bash
python scripts/zero_shot_eval.py --backend laya --calib-per-variant 2 --test-per-variant 3
```

Then the default size, roughly 4 times as many model calls:

```bash
python scripts/zero_shot_eval.py --backend laya --live
```

What it does:
1. **Calibration (the model stays frozen).** It scores Laya on 3 synthetic phrasing variants, then
   fits one softmax temperature per decision kind and the per-kind cascade gates. A gate must reach
   98% precision on every variant, or that decision kind always goes to the LLM.
   - The result is saved to `models/zero_shot_laya.json`; `demo.py --backend laya` uses it.
2. **Test.** It runs the cascade on a 4th phrasing that calibration never saw, plus the
   hand-written problems and flows.
3. **Report.** It writes `reports/zero_shot_laya.md`, including per-decision zero-shot accuracy.

Pure zero-shot, without even calibration:

```bash
python scripts/zero_shot_eval.py --backend laya --no-calibration --gate 0.9
```

## 7. Speed: threads, GPU and other variants

| Setting | Effect |
|---|---|
| `--threads 8` | ONNX Runtime CPU threads |
| `LAYA_PROVIDERS=cuda,cpu` | NVIDIA GPU on Linux (needs the CUDA install step, i.e. **don't** skip it) |
| `LAYA_PROVIDERS=dml,cpu` | DirectML GPU on Windows |
| `LAYA_PROVIDERS=coreml,cpu` | Apple Neural Engine/GPU on macOS (may fall back to CPU for some ops) |
| `LAYA_SUBFOLDER=multilingual` | Laya's multilingual checkpoint |
| `--laya-model-dir ./onnx` | Use an ONNX bundle you exported yourself (see the `@receptron/laya` README) |

Example: `LAYA_PROVIDERS=cuda,cpu python scripts/zero_shot_eval.py --backend laya --live`

## What to look at in `reports/zero_shot_laya.md`

- **Zero-LLM rate and its precision.** The share of items Laya handles with no LLM call, and
  whether those are actually right. This is the headline number.
- **Decisions answered locally / estimated tokens saved.** Partial savings when not every decision
  passes its gate.
- **Per-decision accuracy table.** Which decision kinds Laya handles zero-shot and which it doesn't.
  Compare it with `reports/zero_shot_wordllama.md` (a frozen embedding model, the floor) and
  `reports/poc.md` (the CPU model trained on synthetic data).
- **Next step depends on the result.**
  - If Laya's zero-shot accuracy is good on extraction slots (topics, URLs, numbers) but weak on
    coefficients, fine-tuning on the synthetic data is the next step (PLAN.md, Phase 2).
  - If it's weak across the board, the option descriptions (`hints` in `rambling/solvers/linear.py`
    and `rambling/nodered/templates.py`) and question wording are the cheapest thing to tune.

## Other PoCs you can run locally

```bash
python scripts/evaluate.py --live                 # gold-answer replay vs. lexical baseline -> reports/baseline.md
python scripts/train.py                           # CPU stand-in trained on synthetic data -> models/decider.pkl
python scripts/poc_eval.py --live                 # its cascade results -> reports/poc.md
python scripts/zero_shot_eval.py --backend wordllama --live   # frozen embedding model (no download)
```

## Troubleshooting

- **`Laya bridge not installed`**: run `npm install` in `laya_bridge/`.
- **Download fails or hangs**: check that `huggingface.co` is reachable. If you're behind a proxy,
  set `HTTPS_PROXY`. If you have the files already, point `--laya-model-dir` at a local bundle.
- **`options do not fit in head_max_len`**: a question had too many or too long options for Laya's
  192-token option budget; report which question it was.
- **Out of memory**: Laya needs about 2 GB of RAM plus a few hundred MB per batch of questions.
  Close other apps or use `--threads 2`.
- **Node-RED tests are skipped**: run `npm install` in `nodered/`.
