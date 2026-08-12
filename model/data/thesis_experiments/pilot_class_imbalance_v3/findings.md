# Protocol-v6 class-imbalance pilot

Status: one-seed diagnostic pilot (seed 17). Minority identity is fixed from S1 and all table metrics are exact within phase.

## Incremental-imbalanced stream

- Weighting improved transition accuracy from 0.560 to 0.607 and stable accuracy from 0.680 to 0.707.
- Stable minority recall improved from 0 to 0.273, precision from 0 to 0.375, and F1 from 0 to 0.316.
- Stable minority PR-AUC improved slightly from 0.254 to 0.258.
- Majority F1 improved from 0.836 to 0.855.
- Recovery shortened from 151 to 124 observations.
- Neither variant produced a minority-class true positive in the immediate transition window, although weighting had slightly better minority ranking (PR-AUC 0.122 versus 0.115).

## Gradual-imbalanced stream

- Both variants had zero minority recall/F1 in the reported transition and stable windows.
- Weighting improved stable minority PR-AUC from 0.059 to 0.091, but reduced stable overall accuracy from 0.760 to 0.733.
- Majority F1 was unchanged at 0.903 and recovery times were similar (100 versus 102).

## Interpretation

Prototype-informed minority weighting has a strong positive result on the incremental stream in this development seed, but only improves ranking—not thresholded detection—on the more severe gradual stream, with an overall-accuracy trade-off. The thesis must report this heterogeneity rather than average it into a universal claim. Final evidence needs multiple seeds, confidence intervals, per-class support, and both thresholded and ranking metrics.

