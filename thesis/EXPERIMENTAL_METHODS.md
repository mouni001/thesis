# Experimental Methodology

> Status: protocol-frozen draft. Replace bracketed dataset details and insert only results generated under `thesis_protocol_2026-08-11_v7`.

## Research design

The experiments use test-then-train (prequential) evaluation. At time step \(t\), each method predicts the label of the incoming observation before receiving that label and updating its state. All paired comparisons receive the same ordered observations, feature partitions, transition point, evaluation windows, and random seeds. This prevents a method from benefiting from an easier stream realization.

The principal confirmatory suite uses ten paired seeds: 11, 17, 23, 31, 43, 59, 71, 83, 97, and 109. Model development, debugging, and protocol decisions used separate pilot runs. Those runs are implementation evidence only and are excluded from final inferential claims.

## Leakage control and stream construction

The held-out stream region begins at source index 44,500 and ends at 46,000. Feature evolution occurs at source index 45,000. Thus, the evaluated sequence contains 500 Stream 1 observations followed by 1,000 Stream 2 observations. The scaler is fitted only on the historical prefix ending before index 44,500. No covariate or label from the evaluated segment contributes to scaler fitting.

In the balanced feature-evolution scenario, Stream 1 contains 24 features and Stream 2 contains 25. Eight features are old-only, sixteen are shared, and nine are new-only. The exact source-column identities are saved in every run. Shared source variables retain the same semantic identity even if their local array positions differ. Expansion and contraction scenarios provide additional structural and sensitivity checks.

The feature-evolution event and concept-drift annotations are treated separately. Known stream annotations define analysis phases. Detector alarms are predictions and are never substituted for ground-truth change locations.

## Model comparisons

The primary method is compared with the following paired alternatives:

1. transfer mapper removed;
2. historical knowledge removed;
3. prototype memory removed;
4. learned routing replaced by fixed fusion;
5. historical, adaptive, or prototype expert removed individually;
6. a cold-start single adaptive classifier;
7. an OLD3S-style fixed two-expert model; and
8. Hoeffding Tree, Hoeffding Adaptive Tree, Adaptive Random Forest, and incremental Gaussian Naive Bayes.

Dedicated suites additionally compare prototype obsolescence, drift relevance, ADWIN and MDDM, minority weighting, feature-evolution scenarios, and prespecified hyperparameter values. All alternatives retain the same preprocessing and stream boundaries whenever the method permits it.

## Evaluation phases

For a known change point \(c\) and window length \(w=250\), results are summarized in four prespecified phases:

- pre-change: the window ending at \(c-1\);
- transition: the window beginning at \(c\);
- early recovery: the following window; and
- stable post-change: the final non-overlapping window after the change.

The complete per-step sequence is retained so that phase aggregation does not conceal short-lived failures.

## Outcomes

Predictive outcomes include accuracy, Cohen's kappa, macro F1, G-Mean, per-class precision/recall/F1, minority recall, minority F1, and PR-AUC. PR-AUC is computed from predicted probabilities rather than hard labels. Minority metrics use the minority identity determined from the historical Stream 1 prefix and preserve that identity throughout the run.

Adaptation loss is the difference between the pre-change value and the worst transition value. Recovery accuracy is the mean accuracy in the early-recovery phase. Recovery time is the number of observations after a known change required for a smoothed metric to regain and maintain the declared fraction of its pre-change value. Forgetting is evaluated on a fixed Stream 1 reference set without updating model parameters.

Operational outcomes include inference latency, update latency, peak resident memory, parameter count, persistent prototype-memory size, and prototype count. Timing measurements distinguish prediction from label-dependent updating.

## Mechanism-specific diagnostics

Transfer is evaluated using post-transition performance, recovery, forgetting, mapper behavior, and latent-space alignment. Prototype diagnostics retain source stream, class, age, quality, obsolescence, selection, removal, and estimated byte size. Router diagnostics retain all expert weights, entropy, selected expert, and expert-transition counts. Drift experiments retain known drift intervals, detector scores and alarms, detection delay, false alarms, and missed detections.

## Statistical analysis

For each metric and method, report the mean, sample standard deviation, and 95% confidence interval over the ten seeds. Comparisons with the full model are paired by seed. Report the mean paired difference, Cohen's \(d_z\), a paired t-test, and a Wilcoxon signed-rank test when defined. Apply Holm's correction within each declared family of comparisons. Statistical significance is interpreted alongside effect magnitude, uncertainty, phase behavior, and computational cost; it is not used as the sole criterion of practical value.

## Explainability

SHAP and LIME are post-hoc validation tools rather than components of the learning algorithm. Explanations target the complete fused prediction at frozen snapshots before feature evolution, immediately afterward, during drift, and after recovery. Kernel SHAP uses a documented background sample from the relevant feature space. LIME uses tabular perturbations and repeated random seeds. Analyses separate shared, old-only, and new-only features; include correct, incorrect, minority, and low-confidence cases; compare ranking agreement and repeatability; and test faithfulness by masking highly ranked features and measuring the change in predicted probability.

## Reproducibility rule

Every reported number must be traceable to a v7 run directory containing the configuration, seed, command, protocol metadata, raw per-step outputs, dependency versions, code-change record, and generated-artifact manifest. Pilot suites may motivate interpretation but may not supply final effect estimates.
