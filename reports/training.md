# Training report

Final model: 24000 records over phrasing variants 0-2 ({'solver.linear': 342533, 'nodered.flow': 47530} option rows). Temperatures and gates come from 3 fold models, each scored on the phrasing it never saw; a gate must reach 98% precision in every fold. Accuracy columns are out-of-fold (unseen phrasing).

| task | decision kind | out-of-fold accuracy | temperature | gate | kept by gate | precision when kept |
|---|---|---|---|---|---|---|
| solver | `c.const` | 92.4% | 5.0 | 0.95 | 74% | 100.0% |
| solver | `c.rel` | 98.5% | 3.0 | 0.95 | 90% | 100.0% |
| solver | `c.rhs` | 84.9% | 8.0 | 0.995 | 4% | 100.0% |
| solver | `c.v.mag` | 51.8% | 20.0 | never | 0% | – |
| solver | `c.v.sign` | 89.0% | 8.0 | 0.98 | 6% | 100.0% |
| solver | `domain` | 91.9% | 5.0 | 0.9 | 88% | 100.0% |
| solver | `goal` | 97.2% | 3.0 | 0.9 | 89% | 100.0% |
| solver | `n_cons` | 90.4% | 5.0 | 0.9 | 69% | 100.0% |
| solver | `n_vars` | 90.6% | 5.0 | 0.95 | 61% | 100.0% |
| solver | `nonneg` | 97.2% | 2.0 | 0.8 | 97% | 100.0% |
| solver | `obj.v.mag` | 48.1% | 12.0 | never | 0% | – |
| solver | `obj.v.sign` | 100.0% | 0.1 | 0.5 | 100% | 100.0% |
| solver | `target` | 100.0% | 0.1 | 0.5 | 100% | 100.0% |
| solver | `var` | 78.6% | 8.0 | 0.95 | 34% | 99.8% |
| nodered | `broker` | 100.0% | 0.1 | 0.5 | 100% | 100.0% |
| nodered | `file` | 100.0% | 0.1 | 0.5 | 100% | 100.0% |
| nodered | `interval` | 100.0% | 0.1 | 0.5 | 100% | 100.0% |
| nodered | `method` | 100.0% | 0.1 | 0.5 | 100% | 100.0% |
| nodered | `operator` | 100.0% | 0.1 | 0.5 | 100% | 100.0% |
| nodered | `path` | 100.0% | 0.1 | 0.5 | 100% | 100.0% |
| nodered | `property` | 49.2% | 1.0 | never | 0% | – |
| nodered | `template` | 80.8% | 1.0 | 0.95 | 43% | 99.2% |
| nodered | `threshold` | 100.0% | 0.1 | 0.5 | 100% | 100.0% |
| nodered | `topic` | 100.0% | 0.1 | 0.5 | 100% | 100.0% |
| nodered | `topic_in` | 100.0% | 0.1 | 0.5 | 100% | 100.0% |
| nodered | `topic_out` | 100.0% | 0.1 | 0.5 | 100% | 100.0% |
| nodered | `url` | 100.0% | 0.1 | 0.5 | 100% | 100.0% |
