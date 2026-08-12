# SHAP and LIME temporal explainability pilot

Status: one-seed additional validation (seed 17), protocol revision `thesis_protocol_2026-08-11_v6`. Six deterministic representative cases were analyzed at each snapshot with Kernel SHAP and three repeated LIME fits.

## Why phase snapshots were required

The online model changes after every labelled observation. Explanations were therefore computed from model checkpoints saved after the pre-feature, post-feature, pre-drift, during-drift, and recovery windows. Applying only the final model retrospectively to all phases would not explain the decisions actually available at those times.

Kernel SHAP was used because the complete fused prediction includes a non-differentiable prototype-neighbour branch. A gradient explainer would omit that contribution. S1 and S2 use separate explainers and original feature identities because their raw dimensions differ.

## Main observations

- Before feature evolution, immediately after it, and immediately before abrupt drift, model probabilities were effectively saturated on one class and invariant to the sampled perturbations. SHAP importance was zero or numerically negligible, masking important features produced no meaningful probability drop, and SHAP/LIME rank correlation was undefined. This is an explainability finding about model degeneracy—not evidence that no features matter in the data.
- All six selected cases in each of those saturated windows were correct because the local stream window was dominated by the predicted concept/class. The apparent perfect accuracy and zero attribution should be interpreted together as reliance on a near-constant decision.
- During the abrupt-drift window, nonzero feature dependence reappeared. The leading global SHAP features were shared features 16, 0, 29, 4, and 14.
- During recovery, shared features 13, 26, and 16 remained important and new S2-only feature 27 entered the top four. This is preliminary evidence that the adapted model uses a newly available feature.
- Mean SHAP/LIME absolute-rank Spearman agreement was about 0.436 during drift and 0.227 during recovery; top-five Jaccard agreement was 0.435 and 0.386. The methods therefore overlap partially but are not interchangeable.
- Repeated LIME stability was stronger after drift: mean rank correlation/top-five Jaccard were about 0.712/0.653 during drift and 0.769/0.696 during recovery.
- Attribution masking was faithful after drift. During drift, masking SHAP top-five features lowered predicted-class probability by 0.061 on average versus 0.005 for random features; during recovery the drops were 0.146 versus 0.038. LIME top-five masking drops were 0.027 versus 0.010 and 0.176 versus 0.037.

## Limitations and final protocol

- The pilot explains six cases per snapshot and is not a population estimate.
- LIME depends on the perturbation distribution; repeated seeds quantify instability but cannot remove it.
- Kernel SHAP is expensive and correlated features can share or exchange attribution.
- Raw obsolete S1 features do not exist in the S2 input, so their post-transition importance cannot be directly plotted. The valid S2 question is whether shared and new raw features influence the mapped/fused output and whether historical/prototype usage changes alongside them.
- Final explainability should repeat the analysis on selected final-run seeds, increase SHAP/LIME samples, present correct/error/minority/uncertain cases, and retain the zero-attribution saturation result if reproduced.

