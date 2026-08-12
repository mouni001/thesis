# Prototype-obsolescence confirmatory findings

Values are paired across five seeds and remain provisional until the complete 20-run suite is audited and aggregated.

## Feature obsolescence enabled versus disabled

Feature-based obsolescence changed prototype use as intended during transition. Disabling it increased the mean Stream 1 evidence fraction from 0.0828 to 0.1158 (difference +0.0330; 95% CI [0.0222, 0.0437]; paired dz 3.810). By the stable phase, the evidence fractions converged and their interval crossed zero.

The mechanism did not produce a clear predictive benefit. Transition accuracy was virtually identical; stable accuracy, G-Mean, macro PR-AUC, minority recall/F1, prototype help, and prototype harm all had intervals crossing zero. Recovery was 5.2 observations slower without feature obsolescence on average, but the interval also crossed zero. Runtime was unchanged.

Therefore, feature obsolescence demonstrably suppresses old-space evidence immediately after contraction, but the five-seed held-out stress test does not show that this suppression materially improves predictions. The thesis must distinguish mechanism validation from outcome benefit.

## Age decay enabled versus disabled

Age-based freshness also suppresses old-space evidence during transition. Disabling age decay increased Stream 1 evidence from 0.0828 to 0.1015 (difference +0.0186; 95% CI [0.0113, 0.0259]). Stable evidence fractions later converged.

Age decay produced a small stable G-Mean benefit: 0.6206 with decay versus 0.6145 without it (no-decay minus full -0.00607; 95% CI [-0.00946, -0.00267]). Stable accuracy was 0.0024 higher with decay, with an interval narrowly touching zero. Transition accuracy, macro PR-AUC, minority recall/F1, prototype help/harm, recovery time, and runtime did not clearly differ.

Temporal freshness therefore has somewhat stronger outcome evidence than feature obsolescence in this held-out stress scenario, but the demonstrated effect is small and concentrated in stable class-balanced performance.

## Full prototype system versus no prototype memory

Under low-overlap contraction, prototype memory did not clearly improve transition accuracy or recovery time. Its value appeared later: stable accuracy was 0.6456 with prototypes versus 0.6264 without them (difference -0.0192 for no memory; 95% CI [-0.0347, -0.0037]), and stable G-Mean was higher by 0.0222. Stable minority recall was higher by 0.0471 and minority F1 by 0.0362; their intervals excluded zero. Stable macro PR-AUC was unchanged.

Prototype memory added 38.93 seconds per 1,500 observations in this 512-entry configuration. The stress test therefore supports prototype memory for stable hard-decision and minority performance after severe feature contraction, but not for immediate adaptation, probability ranking, or computational efficiency.

## Overall obsolescence conclusion

The experiment shows that obsolete-prototype pressure exists and both feature obsolescence and age decay reduce old-space evidence during transition. Prototype memory itself helps later performance. However, feature obsolescence does not independently improve predictive outcomes, while age decay provides only a small stable G-Mean benefit. The evidence does not support a broad claim that obsolete prototypes necessarily become harmful; it supports cautious decay as a mechanism with scenario-specific benefit.
