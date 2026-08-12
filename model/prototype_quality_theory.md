# Prototype Quality Formula

This file records the research justification for the prototype ranking score used in `model.py`.

## Formula

Each prototype receives a quality score:

```text
Q(p) = A(p) * O(p)^w_o * F(p)^w_f
```

where:

```text
A(p) =
  w_rep * rep_norm
  + w_drift * drift_norm
  + w_minority * minority_norm
  + w_uncertainty * uncertainty_norm
  ------------------------------------------------
  w_rep + w_drift + w_minority + w_uncertainty
```

The additive components are normalized to `[0, 1]` before weighting. This makes the weights interpretable and comparable.

## Components

`representativeness`

A prototype should be useful if it reliably represents its class. In the code, this is stored as `proto["rep"]` and increases when the prototype helps correct predictions. It is normalized from `[0.05, 4.0]` to `[0, 1]`.

`drift relevance`

A prototype should receive more importance if it is associated with locally unstable or changing regions. This helps the memory adapt after concept drift or feature evolution. In the code, this is stored as `proto["drift"]` and normalized from `[0, 3]` to `[0, 1]`.

`minority-class importance`

In imbalanced streams, minority examples can be underrepresented. This term gives extra value to prototypes from under-seen classes so the memory does not become dominated by majority-class regions. In the code, this is stored as `proto["minority"]` and normalized from `[0, 3]` to `[0, 1]`.

`uncertainty`

Uncertain regions are useful places to keep prototypes because the neural classifier is less confident there. The uncertainty score is based on local prototype-label entropy and is already in `[0, 1]`.

`obsolescence`

After the S1 to S2 transition, old-space prototypes may become less reliable because some old features become obsolete. This term downweights S1 prototypes during S2 according to age and the expected shared feature fraction:

```text
O(p) = shared_frac + (1 - shared_frac) * exp(-age / 250)
```

S2 prototypes keep `O(p) = 1`. Setting `w_o = 0` removes obsolescence decay for ablation.

`freshness`

Older prototypes may become stale in a stream even if they were useful before. Freshness decays with the time since the prototype was last updated:

```text
F(p) = exp(-age / 800)
```

Setting `w_f = 0` removes freshness decay for ablation.

## Current Defaults

```text
w_rep         = 1.0
w_drift       = 0.75
w_minority    = 0.5
w_uncertainty = 0.35
w_o           = 1.0
w_f           = 1.0
```

These are not claimed to be universal. They are theoretically motivated starting values that should be evaluated with ablation and sensitivity analysis.

## Ablation Interpretation

To test whether each component matters, set its weight to zero and compare downstream performance:

```bash
python3 model/run_prototype_experiments.py --suite ablation --name latest_ablation
```

The ablation study produces:

```text
model/data/prototype_quality_experiments/<name>/ablation_summary.csv
```

## Sensitivity Interpretation

To test robustness to weight choices, vary one weight at a time while holding the others fixed:

```bash
python3 model/run_prototype_experiments.py --suite sensitivity --quick --name latest_sensitivity_quick
```

The sensitivity study produces:

```text
model/data/prototype_quality_experiments/<name>/sensitivity_summary.csv
```

## Thesis Claim

The score is defensible because each component has a clear theoretical role and because the components can be empirically tested. The final weight values should be presented as dataset-dependent hyperparameters selected through ablation and sensitivity analysis, not as universally optimal constants.
