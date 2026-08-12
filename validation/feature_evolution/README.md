# Feature-Evolution Validation Record

## Protocol

The INSECTS loader preserves temporal order, splits the stream sequentially, standardizes the original 33 input features, and creates two views:

```text
S1 = old-only features + shared features
S2 = shared features + new-only features
```

Feature membership is determined by a seeded permutation. Final runs must save the permutation seed and all old/shared/new indices in their metadata.

Three explicit scenarios are supported:

| Scenario | Purpose | Expected relation |
|---|---|---|
| `balanced` | Near-even division of non-shared features | dimensions may differ by one |
| `s2_expands` | Feature-space expansion | `dimension2 > dimension1` |
| `s2_contracts` | Feature-space contraction | `dimension2 < dimension1` |

For the current INSECTS data (`33` original features, `shared_frac=0.5`), the verified partitions are:

| Scenario | Old only | Shared | New only | S1 dimension | S2 dimension |
|---|---:|---:|---:|---:|---:|
| `balanced` | 8 | 16 | 9 | 24 | 25 |
| `s2_expands` | 5 | 16 | 12 | 21 | 28 |
| `s2_contracts` | 11 | 16 | 6 | 27 | 22 |

## Verified Invariants

- Old-only, shared, and new-only index sets are mutually disjoint.
- Their union covers every original feature exactly once.
- S1 and S2 arrays select the recorded indices in the recorded order.
- A fixed feature seed reproduces the same partition.
- Different feature seeds change feature identity without changing scenario counts.
- Both expansion and contraction configurations complete a short end-to-end S1-to-S2 run.
- Unequal input spaces are encoded into a common latent dimension.
- The transfer mapper output has the dimension expected by the historical classifier.
- MoE router weights are non-negative, at most one, and sum to one.
- Protocol metadata, including exact feature indices, is preserved in `all_metrics.npz`.

## Commands Used

```bash
python -m unittest discover -s tests -v
python -m compileall -q model tests
cd model
python train.py -DataName insects -feature_scenario s2_expands -T1 8 -t 3 -eval_window 3 -seed 7
python train.py -DataName insects -feature_scenario s2_contracts -T1 8 -t 3 -eval_window 3 -seed 7
```

## Current Evidence Boundary

These checks establish structural compatibility and reproducibility. They do **not** yet establish that the learned transfer mapper aligns semantically equivalent samples or improves predictions. That requires the paired transfer experiments and quantitative latent-alignment analysis specified in `RESEARCH_QUESTIONS.md`.

## Audit Findings to Resolve Before Final Runs

1. The current framework lacks independent switches for the transfer mapper, historical knowledge, prototype memory, fixed fusion, and individual experts.
2. No forgetting metric or fixed Stream 1 diagnostic reference set is implemented.
3. No latent-alignment metric is logged before and after transfer training.
4. Known drift annotations are not represented separately from detector alarms.
5. The current prototype experiment runner varies a single seed per invocation and covers prototype-score weights only.
6. The short-run metric warnings are expected because tiny windows contain too few classes; final pilot windows must be large enough for meaningful Kappa and class metrics.

## Contiguity and Preprocessing Correction

The original training path selected the first `B` rows of S1 and then the first `t` rows after an 80% split. For long streams this skipped a large unreported interval. That protocol is invalid for temporal recovery analysis and all results produced under it must be treated as superseded.

The corrected path selects the final `B` observations immediately before the boundary and the first `t` observations immediately after it. It records the original start, transition, and end indices. An automated test verifies boundary contiguity.

The earlier loader also standardized features using the whole stream. The corrected loader fits `StandardScaler` on S1 only and transforms S1 and S2 with those pre-transition statistics, preventing future S2 observations from influencing calibration. This remains a fixed pre-transition scaler rather than an online scaler and must be described as such.
