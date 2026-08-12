# Manual Drift Weight 0.5 Conclusion

This experiment tested the following prototype-quality weights:

```text
w_rep         = 1.0
w_drift       = 0.5
w_minority    = 0.5
w_uncertainty = 0.35
w_o           = 1.0
w_f           = 1.0
```

The goal was to check whether lowering the drift-relevance weight from `0.75` to `0.5` improves the model.

## Comparison Against Baseline

The baseline is the all-components setting from `latest_ablation`, where:

```text
w_drift = 0.75
```

| Metric | Baseline drift 0.75 | Manual drift 0.5 | Change |
|---|---:|---:|---:|
| Accuracy | 65.80% | 66.80% | +1.00 |
| OCA | 74.28% | 74.04% | -0.24 |
| KappaM | 62.50% | 63.60% | +1.10 |
| G-Mean | 52.55% | 51.80% | -0.75 |
| PR-AUC | 60.07% | 58.97% | -1.10 |
| Minority F1 | 40.00% | 38.10% | -1.90 |
| Minority recall | 39.53% | 37.21% | -2.33 |
| Majority F1 | 51.22% | 50.00% | -1.22 |
| ACR | 0.0609 | 0.0730 | worse |

Both runs detected drift at step `4031`.

## Interpretation

Lowering the drift weight to `0.5` slightly improves final rolling accuracy and KappaM. However, it weakens several metrics that are important for imbalanced stream learning:

- G-Mean decreases.
- PR-AUC decreases.
- Minority F1 decreases.
- Minority recall decreases.
- ACR increases, meaning regret is worse.

This suggests that `w_drift = 0.5` may make the model slightly more accurate overall, but less balanced and less useful for minority-class adaptation.

## Recommendation

Do not replace the default drift weight with `0.5` based on this experiment. The original value:

```text
w_drift = 0.75
```

is more balanced and easier to defend for the thesis because it better preserves minority-class performance, G-Mean, PR-AUC, and cumulative regret.

The manual `0.5` result can still be reported as part of sensitivity analysis, showing that the method is somewhat sensitive to the drift weight and that `0.75` is a better trade-off for this dataset.
