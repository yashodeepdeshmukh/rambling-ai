# Zero-shot PoC: `wordllama` (no fine-tuning)

Calibrated (model frozen): per-kind temperatures and gates at 98% precision in every one of 3 phrasing folds.

| test set | items | decider alone, end-to-end | zero-LLM (auto-accepted) | precision of auto-accepted | decisions answered locally | full escalations | final accuracy (simulated LLM fallback) | est. LLM tokens saved vs all-LLM |
|---|---|---|---|---|---|---|---|---|
| solver, synthetic unseen phrasing | 240 | 0% | 0% | – | 0% | 8% | 100% | 0% |
| solver, hand-written | 14 | 0% | 0% | – | 1% | 0% | 93% | 1% |
| Node-RED, synthetic unseen phrasing | 200 | 30% | 2% | 100.0% | 56% | 0% | 99% | 25% |
| Node-RED, hand-written (auto-accepted flows deployed + probed) | 20 | 40% | 0% | – | 56% | 0% | 100% | 23% |

## Per-decision accuracy on the calibration phrasings (zero-shot)

| decision kind | accuracy | temperature | gate |
|---|---|---|---|
| `broker` | 99.7% (360) | 0.35 | 0.5 |
| `c.const` | 1.9% (1320) | 8.0 | never |
| `c.rel` | 63.1% (1320) | 1.4 | never |
| `c.rhs` | 31.6% (1320) | 8.0 | never |
| `c.v.mag` | 16.0% (2520) | 8.0 | never |
| `c.v.sign` | 67.3% (2520) | 1.4 | never |
| `domain` | 42.5% (720) | 8.0 | never |
| `file` | 100.0% (60) | 0.25 | 0.5 |
| `goal` | 61.9% (720) | 1.4 | 0.9 |
| `interval` | 40.4% (240) | 1.0 | never |
| `method` | 94.2% (120) | 0.35 | 0.95 |
| `n_cons` | 27.4% (720) | 8.0 | never |
| `n_vars` | 37.1% (720) | 8.0 | never |
| `nonneg` | 80.7% (720) | 1.0 | never |
| `obj.v.mag` | 17.7% (300) | 8.0 | never |
| `obj.v.sign` | 90.3% (300) | 0.5 | 0.9 |
| `operator` | 61.7% (120) | 1.0 | 0.9 |
| `path` | 100.0% (180) | 0.25 | 0.5 |
| `property` | 17.5% (120) | 5.0 | never |
| `target` | 87.0% (540) | 0.5 | never |
| `template` | 58.3% (600) | 1.0 | 0.95 |
| `threshold` | 95.0% (120) | 8.0 | 0.6 |
| `topic` | 100.0% (300) | 0.25 | 0.5 |
| `topic_in` | 61.7% (60) | 1.0 | 0.7 |
| `topic_out` | 38.3% (60) | 8.0 | never |
| `url` | 100.0% (240) | 0.25 | 0.5 |
| `var` | 21.1% (1320) | 8.0 | never |

- Final accuracy uses a *simulated* LLM fallback (gold answers): an upper bound.
- Token savings are chars/4 estimates of an LLM answering the same typed questions.
- Runtime: 13s.
