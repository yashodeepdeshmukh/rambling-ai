# Plan: train Laya to build solver models and Node-RED flows

## Where v0 stands

Both pipelines build their output **only from typed decisions**: `choice`, `score` and `noul`
(yes/no) questions, in the same shape as Laya and Jev requests. Every option comes from
deterministic extraction or a fixed list, so the decider never writes text. Today the decisions
come from two stand-ins:

| | Solver (linear/ILP → Z3) | Node-RED (10 templates) |
|---|---|---|
| Dataset | 15 hand-checked problems | 20 requests, 8 with live HTTP probes |
| `oracle` (gold decisions) end-to-end | 14/15; 1 known template gap | 20/20 deployed to real Node-RED, all probes pass |
| `rule` (lexical baseline) end-to-end | 0/15 | 8/20 |
| `rule` per-decision accuracy (gold-fed) | 52.2% | 79.1% |
| Decisions per item | 12–36 questions in 4–6 batched calls | 3–8 questions in 2 calls |

See `reports/baseline.md` for the per-example and per-decision tables.

**Most important finding:** a wrong model usually still *runs*. With the rule decider, 11 of
15 solver models solved cleanly to a wrong answer, and wrong flows deployed without errors.
"The solver returned ok" or "the flow deployed" is not evidence of correctness, and the cascade
below must not treat it as such.

## PoC status (stand-in decider + cascade)

Built and measured without Laya (see `README.md` and `reports/poc.md`):
- Synthetic data generator: 12 solver families and 10 flow templates × 4 phrasings, split by
  phrasing so tests measure generalization to unseen wording.
- Trained CPU stand-in decider with the Laya interface, per-kind temperatures, and
  **cross-phrasing calibration**: three fold models, each scored on the phrasing it never saw;
  a gate must reach 98% precision in every fold.
- Cascade with per-decision gates, partial and full LLM fallback, hard checks, token ledger.

Lessons that change the plan:
1. **Calibrate across phrasings, not on one held-out phrasing.** With a single calibration
   phrasing, the Node-RED gates looked safe (field choice 97.5% accurate) but auto-accepted flows
   were only 88% correct on the test phrasing, because a new wording ("the flow *reported on*
   topic …") fooled the field choice with high confidence. Cross-phrasing calibration measured that
   decision at 49% on unseen wording, so it is never trusted, and auto-accepted precision is 100%.
   Expect the same jaggedness from Laya; keep this calibration scheme.
2. **Order decisions so later ones can anchor on earlier ones.** Asking for a constraint's
   right-hand side first and feeding it back as an anchor raised coefficient accuracy; batching
   everything at once left the model nothing to localize on.
3. **Coefficient choice is the bottleneck for solvers** (~50% on unseen phrasing for the stand-in).
   This is what a pretrained encoder has to fix before solver items can run with zero LLM tokens.
4. **Paraphrase diversity matters more than volume.** The stand-in fits training data almost
   perfectly; errors come from wording it never saw. Phase 1 should spend effort on paraphrases
   (LLM-generated, verified) rather than more samples of the same phrasing.

## Phase 1: scale up labelled data (no GPU needed)

The labels are the decisions; `scripts/export_training.py` already turns any gold formalization
into Laya-shaped records (`data/train/*.jsonl`).

**Solver**
- Convert public word-problem datasets that come with equation annotations: MAWPS, ALG514,
  Dolphin18K and SVAMP for equations; NL4Opt for LP/ILP optimization. Parse each equation
  into `Σ cᵢxᵢ + k (= | ≤ | ≥) b` and map it to decisions.
- Report **coverage**: the share of problems whose gold model is representable with the
  extracted options (the 20%-discount problem is the first known gap).
- Use an LLM as a teacher for unlabelled problems. It answers the *same typed questions*;
  keep only items where the solved answer matches the known answer.

**Node-RED**
- Generate synthetic requests: template × sampled slot values × LLM paraphrases. Labels are
  exact by construction; keep only flows that pass validation and probes.
- Mine flows.nodered.org: match shared flows to templates, then caption them once with an LLM.
- Add MQTT behaviour probes with an embedded broker (e.g. aedes), so MQTT flows get tested
  for behaviour, not just for deploying.

**Hard negatives (for the faithfulness check below)**
- Take a gold build, flip one decision (sign, coefficient, relation, topic, operator…),
  render it, and label it "not faithful".
- This targets exactly the mistakes a decider makes, and costs nothing to produce.

Split train/validation/test **by example id and by paraphrase family**, never by record,
or the sequential records of one problem leak across splits.

## Phase 2: fine-tune

- **Model:** Laya (421M, ModernBERT-large, Apache 2.0), starting from its typed-decision
  checkpoint.
- **Fallback:** if its fine-tuning code doesn't support our question set, train ModernBERT-large
  directly as a cross-encoder: `[state] [SEP] [instructions] [SEP] option` → score.
  - Choice: softmax over the option scores (handles option lists that change per example).
  - Noul: one logit with binary cross-entropy.
  - Score: an ordinal head.
- **Labels:** skip null labels ("don't care" on that path) in the loss.
- **Calibration:** fit one temperature per decision kind on validation. The Laya card reports
  ECE 0.466 → 0.081 from this alone, and the cascade thresholds depend on it.
- **Metrics:** teacher-forced accuracy per decision kind (already reported by
  `scripts/evaluate.py`), then free-running end-to-end accuracy on held-out data.
- **Hardware:** a single 24 GB GPU is plenty for 421M parameters at 512 tokens.

## Phase 3: confidence-gated cascade

```
request ─► Laya decisions ─► build ─► checks ─► faithfulness noul ─┬─► accept (0 LLM tokens)
                                                                    └─► LLM finishes the partial build
```

Accept without an LLM only if **all** of these hold:
1. every decision's calibrated confidence is above its kind's threshold;
2. hard checks pass:
   - solver: status ok, unique answer, integer when the domain is integers, bounded;
   - Node-RED: static validation, deploy, and probes;
3. Laya's *faithfulness* noul on the rendered model or flow ("this matches the request")
   passes. It is trained on the hard negatives from Phase 1.

Otherwise, hand the **partly built** model or flow to the LLM. Its prompt is shorter because
most slots are already decided.

Measure: a risk–coverage curve (share handled with zero LLM tokens vs error rate) and LLM
tokens saved per item at a fixed accuracy.

## Template gaps to close (v1)

**Solver**
- Derived coefficients: percentages, `1 - p`, unit conversions. Option: add derived-number
  candidates (`100 - n`, `n/100`).
- Constants on both sides.
- More than 3 unknowns, via hierarchical questions.
- Rates (work and distance problems).

**Node-RED**
- Function nodes: use a snippet library first, and fall back to an LLM that writes only the
  function body.
- Composing several templates into one flow.
- Contrib nodes: retrieve likely candidates first, then let Laya rerank them.
- The 20-option limit: split large catalogues into category → template.

## Proposed milestones (targets to agree on, not results)

| Milestone | Exit criterion |
|---|---|
| M1 data | ≥ 1k solver problems with coverage report; ≥ 2k verified Node-RED requests |
| M2 fine-tune | Laya beats `rule` on every decision kind; per-kind teacher-forced accuracy ≥ 95% on held-out data |
| M3 cascade | Choose a threshold where accepted items have ≤ 1% error; report the share handled with zero LLM tokens |
| M4 Lean | Reuse the same decider interface for solver routing in Lean proofs (tactic choice, give-up, premise rerank) |

## Environment needs

- Hugging Face access (`convaiinnovations/laya` weights). It's blocked in the current cloud
  environment's network policy.
- A GPU machine for Phase 2.
- An LLM API key for the teacher, paraphrases and fallback.
