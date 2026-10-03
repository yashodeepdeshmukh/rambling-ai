# PoC results

Trained CPU stand-in decider (hashed logistic scorer, `rambling/learned.py`) behind the confidence-gated cascade (`rambling/cascade.py`). Gates were fitted on a phrasing variant that is neither in training nor in these test sets.

| test set | items | learned alone, end-to-end | zero-LLM (auto-accepted) | precision of auto-accepted | decisions answered locally | full escalations | final accuracy (simulated LLM fallback) | est. LLM tokens saved vs all-LLM |
|---|---|---|---|---|---|---|---|---|
| solver, synthetic unseen phrasing | 480 | 5% | 0% | – | 33% | 8% | 100% | 19% |
| solver, hand-written | 14 | 36% | 0% | – | 44% | 7% | 100% | 29% |
| Node-RED, synthetic unseen phrasing | 400 | 89% | 38% | 100.0% | 86% | 0% | 100% | 69% |
| Node-RED, hand-written (auto-accepted flows deployed + probed) | 20 | 100% | 80% | 100.0% | 95% | 0% | 100% | 92% |

- **Zero-LLM**: every decision passed its gate and the hard checks passed; no LLM call at all.
- **Decisions answered locally**: share of typed questions the trained model answered; the rest went to the fallback.
- **Final accuracy** uses a *simulated* LLM fallback (gold answers), so it is an upper bound.
- **Tokens** are chars/4 estimates of an LLM answering the same typed questions, not API measurements.
- 1 hand-written solver problem is outside the v0 template and is excluded.
