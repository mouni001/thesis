# Confirmatory block findings

This file records paired blocks as they finish. Values are provisional until the complete suite is aggregated and multiplicity correction is applied across the declared family.

## Full model versus no transfer mapper (10 paired seeds)

Removing the mapper changed transition accuracy from 0.5064 to 0.5016 (candidate minus full: -0.0048; 95% CI [-0.0132, 0.0036]; paired Cohen dz -0.409). Stable post-change accuracy was effectively unchanged (0.6856 versus 0.6872; difference +0.0016; 95% CI [-0.0072, 0.0104]). Stable G-Mean and macro PR-AUC differences were also near zero.

The no-mapper condition required 16.1 additional observations to recover on average (95% CI [0.45, 31.75]; dz 0.736; unadjusted paired t p=0.0449; Wilcoxon p=0.0586). This is preliminary evidence that the mapper may improve adaptation speed even though its phase-average and final predictive effects are small. The no-mapper model was approximately 2.00 seconds faster over the 1,500-step run (95% CI [-2.95, -1.04]).

Interpretation must remain cautious: the recovery-time t-test is not yet multiplicity-adjusted, its Wilcoxon result is above 0.05, and the stable predictive estimates do not show a mapper benefit.

## Full model versus no historical knowledge (10 paired seeds)

Removing historical knowledge reduced transition accuracy from 0.5064 to 0.4576 (difference -0.0488; 95% CI [-0.0673, -0.0303]; dz -1.883; paired t p=0.000214; Wilcoxon p=0.001953). It also delayed accuracy recovery by 78.5 observations (95% CI [31.0, 126.0]; dz 1.182; paired t p=0.00463; Wilcoxon p=0.00391). Stable post-change accuracy and G-Mean were essentially unchanged, showing that the historical pathway primarily affects early adaptation rather than final performance.

Stable minority recall decreased from 0.5324 to 0.4971 without history (difference -0.0353; 95% CI [-0.0679, -0.0027]), although stable minority F1 had an interval crossing zero. The no-history condition was 14.52 seconds faster over 1,500 observations (95% CI [-15.60, -13.44]).

These unadjusted results provide stronger evidence for the value of historical knowledge than for the mapper alone, but their final significance status awaits the complete comparison-family Holm correction.

## Full model versus no prototype memory (10 paired seeds)

Removing prototype memory reduced transition accuracy from 0.5064 to 0.4852 (difference -0.0212; 95% CI [-0.0313, -0.0111]; dz -1.502; paired t p=0.00105; Wilcoxon p=0.00586). Early-recovery accuracy decreased by 0.0044 (95% CI [-0.0081, -0.0007]). Stable accuracy and G-Mean were lower without prototypes, but their 95% intervals narrowly crossed zero; stable macro PR-AUC was effectively unchanged.

The clearest stable predictive effect was on the historical minority class: recall decreased from 0.5324 to 0.4794 (difference -0.0529; 95% CI [-0.0855, -0.0203]) and F1 decreased from 0.5795 to 0.5404 (difference -0.0391; 95% CI [-0.0696, -0.0087]). Accuracy recovery time was unchanged within uncertainty.

Prototype memory imposed a large operational cost. Removing it reduced mean inference latency from 13.54 ms to 0.84 ms and mean training-update latency from 23.44 ms to 6.55 ms. Total runtime fell by 44.61 seconds per 1,500 observations. The full model retained exactly 256 prototype vectors (25,600 vector bytes) at the end of each run, versus zero in the ablation. These figures establish a clear adaptation/minority-performance versus latency tradeoff.

## Learned routing versus fixed fusion (10 paired seeds)

Replacing the learned router with equal fixed fusion had a small, uncertain transition effect (accuracy difference -0.0044; 95% CI [-0.0102, 0.0014]) but reduced early-recovery accuracy from 0.6504 to 0.6404 (difference -0.0100; 95% CI [-0.0168, -0.0032]; dz -1.055). Stable accuracy and G-Mean intervals crossed zero. Stable macro PR-AUC decreased from 0.6923 to 0.6848 (difference -0.00743; 95% CI [-0.01382, -0.00103]). Minority recall and F1 did not show a routing benefit.

The learned router changed its allocation substantially across Stream 2. Mean historical/adaptive/prototype weights were 0.344/0.355/0.301 in transition, 0.330/0.491/0.179 in early recovery, and 0.249/0.646/0.105 in the stable phase. The adaptive expert was the highest-weight expert for 96.1% of transition observations and 100% of early-recovery and stable observations. Router values are unavailable before the feature transition because the three-expert Stream 2 fusion is not active in Stream 1.

Learned routing increased total runtime by 23.25 seconds relative to fixed fusion. The evidence therefore supports meaningful expert reallocation and modest early-recovery/PR-AUC gains, accompanied by substantial optimization cost; it does not support a stable accuracy or minority-performance gain.

## Full model versus no historical expert (10 paired seeds)

Removing only the historical expert did not harm transition accuracy (difference +0.0020; 95% CI [-0.0080, 0.0120]) or clearly change early-recovery accuracy. It did, however, reduce stable accuracy from 0.6856 to 0.6776 (difference -0.0080; 95% CI [-0.0129, -0.0031]) and stable G-Mean from 0.6645 to 0.6545 (difference -0.00995; 95% CI [-0.01516, -0.00474]). Stable macro PR-AUC was unchanged.

Stable minority recall decreased by 0.0353 (95% CI [-0.0663, -0.0042]) and minority F1 decreased by 0.0263 (95% CI [-0.0468, -0.0057]). Recovery time was 34.6 observations longer on average, but its confidence interval crossed zero. Removing the expert saved 30.20 seconds per 1,500 observations.

Together with the no-historical-knowledge result, this indicates different roles: the broader historical system improves transition/recovery, whereas the historical expert itself contributes mainly to stable accuracy, balance, and minority performance. This distinction should be retained in the thesis rather than treating the two ablations as interchangeable.

## Full model versus no adaptive expert (10 paired seeds)

Removing the adaptive expert caused the largest expert-level degradation observed so far. Transition accuracy decreased from 0.5064 to 0.4268 (difference -0.0796; 95% CI [-0.1011, -0.0581]), early-recovery accuracy decreased by 0.1204 (95% CI [-0.1377, -0.1031]), and stable accuracy decreased by 0.0776 (95% CI [-0.0912, -0.0640]). Stable G-Mean fell by 0.0760 and stable macro PR-AUC by 0.0720; both intervals excluded zero.

Stable minority F1 decreased by 0.1048 (95% CI [-0.1581, -0.0515]). The minority-recall estimate decreased by 0.0676, but its interval narrowly crossed zero. Among the nine seed pairs with finite recovery times in both conditions, removing the adaptive expert delayed recovery by 444.6 observations (95% CI [332.9, 556.2]). It saved 31.55 seconds per run.

This strongly validates the router diagnostic: its increasing post-transition allocation to the adaptive expert corresponds to substantial predictive necessity, not merely changing alpha values.

## Full model versus no prototype expert (10 paired seeds)

Removing only the prototype expert reduced transition accuracy from 0.5064 to 0.4884 (difference -0.0180; 95% CI [-0.0261, -0.0099]). Early-recovery accuracy was unchanged within uncertainty. Stable accuracy and G-Mean were lower by roughly 0.0080 and 0.0091, respectively, but both intervals narrowly crossed zero.

Stable minority recall decreased by 0.0353 (95% CI [-0.0647, -0.0059]) and minority F1 by 0.0253 (95% CI [-0.0501, -0.0005]). In contrast, stable macro PR-AUC increased by 0.00585 without the prototype expert (95% CI [0.00205, 0.00965]), demonstrating a mixed classification-versus-ranking effect that must be reported rather than collapsed into a single claim. Recovery was 37.4 observations slower on average; its t interval narrowly crossed zero, although the signed-rank p-value was 0.0156 before multiplicity adjustment.

Removing the expert saved 30.26 seconds per run, compared with 44.61 seconds when the entire prototype memory was removed. The expert therefore explains most of prototype memory's transition/minority benefit and much, but not all, of its runtime cost. Retaining memory without its expert did not preserve those benefits and actually improved probability ranking slightly.

## Full model versus cold-start single adaptive classifier (10 paired seeds)

The cold-start single classifier reduced transition accuracy from 0.5064 to 0.3552 (difference -0.1512; 95% CI [-0.1758, -0.1266]) and early-recovery accuracy by 0.0408 (95% CI [-0.0575, -0.0241]). It required 123.4 additional observations to recover (95% CI [69.9, 176.9]). These large adaptation losses demonstrate the value of transferred/ensemble knowledge at the Stream 2 boundary.

Stable accuracy, G-Mean, and macro PR-AUC were indistinguishable within uncertainty, showing that the cold-start learner eventually caught up on aggregate outcomes. Stable minority recall remained 0.0441 lower (95% CI [-0.0759, -0.0124]), while minority F1 did not clearly differ.

The single classifier completed the stream 59.45 seconds faster than the full model. The full framework's main value relative to this baseline is therefore jump-start adaptation and minority recall rather than a superior final aggregate plateau, obtained at a large computational cost.

## Full model versus OLD3S-style fixed two-expert baseline (10 paired seeds)

The full model improved transition accuracy from 0.4892 to 0.5064 (baseline-minus-full difference -0.0172; 95% CI [-0.0292, -0.0052]). Early-recovery and stable accuracy differences were uncertain, as were stable G-Mean, minority recall/F1, and recovery time.

The simpler two-expert baseline achieved higher stable macro PR-AUC by 0.00666 (95% CI [0.00095, 0.01237]) and completed each 1,500-observation run 57.50 seconds faster. Thus, compared with this literature-inspired internal baseline, the proposed prototype/MoE extensions provide a modest transition-accuracy gain but no demonstrated later aggregate or minority advantage in this scenario, while worsening probability ranking and imposing a large runtime cost.

This baseline is an OLD3S-style reconstruction under the common implementation and evaluation protocol, not a claim of bit-for-bit reproduction of the authors' released system. That distinction must remain explicit in the thesis.

## Full model versus Hoeffding Tree (10 paired seeds)

Hoeffding Tree outperformed the full model during the transition: 0.5920 versus 0.5064 accuracy (tree-minus-full difference +0.0856; 95% CI [0.0666, 0.1046]). Because the baseline uses stable original feature identifiers, it can retain tree structure for shared variables across the feature transition; this is a legitimate and important result rather than a cold-start comparison.

The result reversed after transition. Relative to the full model, Hoeffding Tree was lower by 0.0264 early-recovery accuracy, 0.0696 stable accuracy, 0.0789 stable G-Mean, and 0.0166 stable macro PR-AUC. Stable minority recall was lower by 0.1206 and minority F1 by 0.1279. All corresponding 95% intervals excluded zero. Recovery-time estimates did not clearly differ under the t interval.

Hoeffding Tree completed the stream 63.88 seconds faster. It therefore provides excellent immediate transition behavior and efficiency, while the proposed model provides much stronger later adaptation, balance, probability ranking, and minority performance.

## Full model versus Hoeffding Adaptive Tree (10 paired seeds)

Hoeffding Adaptive Tree strengthened the immediate tree-baseline result: transition accuracy was 0.5992 versus 0.5064 for the full model (difference +0.0928; 95% CI [0.0757, 0.1099]). It also reached the protocol recovery criterion 62.5 observations sooner (95% CI [15.1, 109.9] in the tree's favor).

As with the standard Hoeffding Tree, later performance reversed. The adaptive tree was lower by 0.0268 early-recovery accuracy, 0.0632 stable accuracy, 0.0731 stable G-Mean, 0.0147 stable macro PR-AUC, 0.1147 stable minority recall, and 0.1170 stable minority F1. All reported intervals excluded zero. It was 62.88 seconds faster per run.

The tree baselines therefore dominate immediate boundary handling and computational efficiency, whereas the proposed model dominates later Stream 2 quality, especially balanced and minority outcomes. Claims of universally faster adaptation by the proposed framework would be contradicted by these results and must not appear in the thesis.

## Full model versus Adaptive Random Forest (10 paired seeds)

Adaptive Random Forest outperformed the full model at transition by 0.0736 accuracy (95% CI [0.0530, 0.0942]) and reached the recovery criterion 53.2 observations sooner (95% CI [16.8, 89.6]). It was also 59.36 seconds faster per run.

Unlike the individual tree baselines, its later differences from the full model were uncertain: early-recovery accuracy, stable accuracy, G-Mean, macro PR-AUC, minority recall, and minority F1 all had paired intervals crossing zero. Point estimates generally favored the full model for stable hard-decision metrics, but the evidence does not establish those advantages under this ten-seed scenario.

Adaptive Random Forest is therefore the strongest external baseline observed so far: it provides better immediate adaptation and recovery with comparable later outcomes and much lower runtime. Any overall superiority claim for the proposed framework would be contradicted by this experiment.

## Full model versus Gaussian Naive Bayes (10 paired seeds)

Gaussian Naive Bayes outperformed the full model at transition by 0.0576 accuracy (95% CI [0.0386, 0.0766]). The proposed model then exceeded it by 0.0224 early-recovery accuracy, 0.0696 stable accuracy, 0.0789 stable G-Mean, 0.0115 stable macro PR-AUC, 0.1206 stable minority recall, and 0.1279 stable minority F1. The Gaussian baseline was 63.06 seconds faster. Its recovery-time difference was uncertain under the paired t interval.

Like the tree baselines, Gaussian Naive Bayes is stronger immediately at the boundary and far more efficient, but markedly weaker later, particularly for balanced and minority outcomes.

## Multiplicity note

The final suite contains 195 variant-outcome paired tests under the broad prespecified family. Holm correction across that complete family is deliberately conservative. Many Wilcoxon p-values become 0.3809 because the smallest attainable exact two-sided value with ten same-direction pairs is 0.001953 and is multiplied by the large family size. Final interpretation must prioritize prespecified primary outcomes, confidence intervals, effect sizes, and replicated phase patterns; paired t results surviving the broad correction can be identified separately. No unadjusted p-value should be described as final statistical significance.
