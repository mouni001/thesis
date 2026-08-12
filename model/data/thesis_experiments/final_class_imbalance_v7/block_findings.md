# Class-imbalance confirmatory findings

This block contains 20 audited protocol-v7 runs: minority weighting enabled and disabled on the incremental and gradual naturally imbalanced INSECTS streams, paired across five seeds. The reported intervals are 95% paired *t* intervals; thesis-wide multiplicity correction must also be considered before assigning confirmatory significance.

## Incremental stream

At feature transition, weighting did not change mean accuracy (0.6432 in both variants). It yielded favorable but highly uncertain minority point estimates: recall increased from 0 to 0.0333, F1 from 0 to 0.0444, G-Mean from 0 to 0.0565, and minority PR-AUC from 0.0549 to 0.0902. Every nonzero paired interval crossed zero.

In the stable post-change phase, both variants had zero minority precision, recall, F1, and G-Mean. Weighting changed minority PR-AUC by only +0.0029 (95% interval [-0.0021, 0.0079]) and macro PR-AUC by +0.0040 [-0.0029, 0.0109]. Stable accuracy was effectively unchanged (-0.0008 [-0.0185, 0.0169]). The apparent runtime reduction of 2.73 seconds is not a plausible algorithmic benefit of weighting and is treated as run-to-run timing noise.

## Gradual stream

Both variants again had zero minority precision, recall, F1, and G-Mean at transition and in the stable phase. Weighting produced an uncertain transition minority PR-AUC gain of 0.0373 [-0.0484, 0.1230]. In the stable phase it instead lowered minority PR-AUC from 0.0984 to 0.0705 (difference -0.0279 [-0.0589, 0.0031]); the interval narrowly crossed zero. Stable accuracy changed by +0.0056 [-0.0235, 0.0347], and macro PR-AUC by -0.0027 [-0.0369, 0.0316].

## Thesis claim supported by this block

Prototype minority weighting with weight 0.5 is **not validated as an effective class-imbalance remedy** in these streams. It can alter minority ranking around transition, but it does not prevent the rarest class from receiving zero predicted recall after feature evolution. Accuracy alone conceals this failure. The defensible conclusion is that the current weighting mechanism is insufficient and that thresholding, resampling, class-balanced losses, or a dedicated imbalance-aware online learner are future-work candidates rather than established contributions of the proposed method.

