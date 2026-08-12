# Literature-standard online baseline pilot

Status: diagnostic pilot only (one seed, seed 17). Do not use these values as final thesis estimates.

## Protocol

- INSECTS incremental balanced stream.
- Identical contiguous 300-observation S1 and 200-observation S2 interval used by the thesis runner.
- Original feature identities are retained in River dictionaries, so shared features keep their names while obsolete and new features disappear or appear at the boundary.
- Baselines: Hoeffding Tree, Hoeffding Adaptive Tree, Adaptive Random Forest, and online Gaussian Naive Bayes.

## Preliminary observations

- Transition correctness was 0.52 for Hoeffding Tree, 0.52 for Hoeffding Adaptive Tree, 0.53 for Adaptive Random Forest, and 0.53 for Gaussian Naive Bayes.
- Early/stable post-transition correctness was 0.52, 0.56, 0.48, and 0.52 respectively in this short pilot.
- Adaptive Random Forest had the smallest measured adaptation loss (0.05) and shortest recovery time (50 samples), but its stable correctness was lowest and the interval is too short for a final comparison.
- Runtime ranged from about 7.9 to 10.3 seconds for 500 observations under the instrumented runner.

## Interpretation and next action

All four external baselines execute under the same feature-evolution and evaluation protocol. The final comparison must use the frozen longer stream windows and multiple random seeds, then report mean, standard deviation, paired significance/effect sizes, and computational cost.

