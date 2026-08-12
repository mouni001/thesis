# Concept-drift confirmatory findings

Values are paired across five seeds and remain provisional until all 20 runs are audited and aggregated. Feature evolution occurs at local step 500; the separately annotated abrupt concept drift occurs at local step 1,000.

## ADWIN: drift relevance enabled versus disabled

ADWIN detected the known abrupt drift in every run with no false alarms. With drift relevance, delays were 23--55 observations (mean 48.6); without relevance, delays were also 23--55 (mean 42.2). The paired delay interval crossed zero, so detector behavior did not clearly differ.

Drift relevance had favorable but uncertain point estimates: during-drift accuracy was 0.5472 with relevance versus 0.5328 without it, and recovery-window accuracy was 0.6008 versus 0.5888. Recovery macro PR-AUC was 0.4632 versus 0.4469. All paired intervals crossed zero with five seeds. Pre-drift accuracy was unchanged.

G-Mean was zero in both conditions during drift and recovery because at least one class received zero recall. Neither condition regained the predefined recovery threshold within the observed horizon. This is a substantive failure mode: drift relevance does not solve post-drift class coverage, and its modest accuracy/ranking point gains are not confirmatory.

## ADWIN versus MDDM-G with drift relevance

Both detectors found the annotated drift in all five seeds with zero false alarms and zero misses. ADWIN's mean delay was 48.6 observations; MDDM-G's was 36.8. The paired MDDM-minus-ADWIN difference was -11.8 observations, but its 95% interval [-27.1, 3.5] crossed zero.

Predictive phase outcomes were exactly identical between detector choices in this experiment. The detectors are observational signals used in prototype relevance, not reset triggers, and their different alarm locations did not alter the prediction sequence under these settings. MDDM-G has a favorable delay point estimate, but neither detector improves the documented post-drift class-coverage failure.

## MDDM-G: drift relevance enabled versus disabled

With MDDM-G, enabling drift relevance left pre-drift accuracy effectively unchanged (difference 0.0009; 95% paired interval [-0.0037, 0.0055]) and produced an uncertain during-drift accuracy gain of 0.0051 [-0.0058, 0.0160]. The recovery-window estimates were more favorable: accuracy increased from 0.5764 to 0.5927 (difference 0.0163 [0.0020, 0.0305]) and macro PR-AUC increased from 0.4134 to 0.4267 (difference 0.0133 [0.0084, 0.0181]). These are nominal paired intervals over five seeds and should be interpreted alongside the thesis-wide multiplicity correction rather than as isolated confirmatory significance claims.

Detection itself was essentially unchanged: the mean delay was 36.8 observations with relevance and 36.2 without it (paired difference 0.6 [-1.1, 2.3]); both configurations had zero misses and zero false alarms. G-Mean remained zero during drift and recovery, and neither configuration met the predefined recovery threshold. Thus drift relevance improves recovery accuracy and ranking under MDDM-G in this setting, but it does not restore coverage of every class.

## Thesis claim supported by this block

The experiments support a narrow claim: ADWIN and MDDM-G reliably locate the injected abrupt drift, with MDDM-G showing a faster but uncertain delay point estimate. Drift relevance can improve post-drift accuracy and macro PR-AUC, most clearly under MDDM-G, but it is not sufficient for balanced class recovery. Any statement that drift relevance universally accelerates recovery or that one detector is definitively superior would exceed the evidence.
