# Thesis Completion Checklist

This checklist tracks the remaining research, validation, writing, and reproducibility work required before the thesis is submission-ready.

## 1. Validate Feature Evolution

- [x] Verify and log that feature evolution truly occurs (`dimension1 != dimension2`).
- [x] Record which features are shared, obsolete, and newly introduced.
- [ ] Document the Stream 1 to Stream 2 transition point and feature-overlap fraction.
- [ ] Verify that samples are routed through the correct encoder before and after the transition.
- [ ] Verify that the transfer mapper aligns the two latent feature spaces correctly.
- [ ] Verify that classifier knowledge learned in Stream 1 is transferred correctly and remains usable in Stream 2.
- [x] Add automated checks for feature partitioning, tensor dimensions, and transfer output shapes.
- [ ] Test multiple feature-evolution scenarios, including different old/new dimensions, overlap fractions, transition timings, and feature-emergence/obsolescence patterns. Expansion and contraction scenarios pass structural smoke tests; full evaluations remain.
- [ ] Evaluate multiple feature-overlap fractions and transition timings.
- [ ] Explain whether concept drift is observed naturally or imposed synthetically.
- [ ] Add an established feature-evolution benchmark or rigorously justify the synthetic protocol.

## 2. Transfer Learning Experiments

Run multi-seed comparisons of:

- [ ] Full transfer mapper.
- [ ] No transfer mapper.
- [ ] No historical knowledge.

Report:

- [ ] Overall and Stream 2 performance.
- [ ] Performance immediately after feature evolution.
- [ ] Recovery time after the Stream 2 transition.
- [ ] Adaptation speed and final accuracy with and without transfer.
- [ ] Forgetting on retained Stream 1 knowledge, using an explicitly defined forgetting metric.
- [ ] Mean, standard deviation, and statistical comparisons across seeds.

Goal: quantify the contribution of cross-space transfer and retained historical knowledge.

## 3. Prototype Memory and Obsolescence Validation

Compare:

- [ ] Full prototype memory with obsolescence.
- [ ] Prototype memory without obsolescence.
- [ ] No prototype memory.

Measure:

- [ ] Performance after entering Stream 2.
- [ ] Performance immediately after feature evolution and later in Stream 2.
- [ ] Prototype age, quality, source stream, class, and obsolescence values over time.
- [ ] Whether obsolete Stream 1 prototypes hurt predictions in Stream 2.
- [ ] Prototype-bank size, replacement frequency, runtime, and memory cost.
- [ ] Adaptation speed, minority-class performance, and prediction stability with prototype memory on versus off.

## 4. Concept Drift Analysis

Produce experiments and plots showing:

- [ ] Detected and known drift locations.
- [ ] Performance before drift.
- [ ] Performance during the drift interval.
- [ ] Performance during recovery.
- [ ] Recovery time and post-drift performance.

Compare:

- [ ] ADWIN versus MDDM using matched experimental conditions.
- [ ] Prototype scoring with drift relevance versus without drift relevance.
- [ ] Detection delay, false alarms, missed drifts, and predictive performance.

Do not treat detector alarms as ground-truth drift locations; evaluate them against known or documented stream changes.

## 5. Class-Imbalance Experiments

Compare the model with and without minority weighting using:

- [ ] Minority recall.
- [ ] Minority F1.
- [ ] G-Mean.
- [ ] PR-AUC.
- [ ] Per-class precision, recall, and F1.
- [ ] Minority precision in addition to minority recall and F1.

Also:

- [ ] Test multiple imbalance levels or INSECTS variants.
- [ ] Report results before and after drift or feature evolution.
- [ ] Check whether gains for minority classes reduce majority-class performance.

## 6. Mixture-of-Experts Validation

Show:

- [ ] Router alpha values over time.
- [ ] Expert selection before and after feature evolution.
- [ ] Expert selection before, during, and after concept drift.
- [ ] Transitions between historical, adaptive, and prototype experts.

Compare against:

- [ ] Learned MoE router.
- [ ] Fixed fusion.
- [ ] No router / MoE.
- [ ] Removal of each individual expert.
- [ ] Different router hidden sizes.

## 7. Baselines

Run under identical data splits, seeds, and evaluation protocols:

- [ ] Full proposed model.
- [ ] Original OLD3S implementation.
- [ ] Fixed-fusion baseline.
- [ ] Model without prototypes.
- [ ] Model without MoE routing.
- [ ] A single-classifier baseline.
- [ ] Relevant online learning, concept-drift, and feature-evolution methods.
- [ ] Global-majority and cumulative-majority sanity-check baselines.
- [ ] All major ablation models listed below.

### OLD3S-comparable baseline extension (added 2026-08-12)

- [ ] Implement and validate FOBOS using the feature-padding treatment documented by OLD3S.
- [ ] Implement and validate OLSF, documenting any mismatch between its incremental-feature assumptions and the thesis streams.
- [ ] Implement and validate FESL under the same feature-evolution protocol.
- [ ] Obtain and run the official OLD3S implementation if reproducible.
- [ ] Add OLD-Linear and OLD-FD if their official definitions/implementation can be reproduced faithfully.
- [ ] Record implementation provenance and clearly label each method as official, faithful reimplementation, or literature-inspired approximation.
- [ ] Run all added baselines with the same stream order, preprocessing boundary, feature partitions, seeds, and prequential protocol.
- [ ] Audit all new runs and incorporate them into paired statistical and computational comparisons.

## 8. Ablation Studies

Individually remove or modify:

- [ ] Transfer mapper.
- [ ] Prototype memory.
- [ ] Obsolescence.
- [ ] Freshness.
- [ ] Representativeness.
- [ ] Uncertainty.
- [ ] Drift relevance.
- [ ] Minority weighting.
- [ ] Each individual expert.
- [ ] Router / MoE.
- [ ] Historical knowledge.

Run all thesis ablations with multiple seeds. Report mean, standard deviation, confidence intervals, and appropriate statistical tests. Discuss mixed or negative findings rather than claiming that every component helps.

## 9. Sensitivity Analysis

Vary:

- [ ] Prototype fusion weight.
- [ ] Prototype-bank size and nearest-neighbour count.
- [ ] Prototype-memory capacity.
- [ ] Obsolescence weight.
- [ ] Freshness weight.
- [ ] Drift weight.
- [ ] Minority weight.
- [ ] Representativeness and uncertainty weights.
- [ ] Router hidden size.
- [ ] Relevant encoder, classifier, mapper, and router learning rates.
- [ ] Feature-overlap fraction.

Show that conclusions are reasonably robust and identify dataset-dependent hyperparameters. Complete and save the sensitivity runs; command files alone are not experimental results.

## 10. Thesis Figures and Tables

Generate publication-ready plots for:

- [ ] Overall predictive performance over time.
- [ ] Prequential accuracy, Kappa, G-Mean, and PR-AUC time series.
- [ ] Feature-evolution and concept-drift markers.
- [ ] Performance before, during, and after drift.
- [ ] Recovery after drift.
- [ ] Prototype count, composition, quality, age, and evolution.
- [ ] Prototype obsolescence values over time.
- [ ] Router alpha values and expert transitions.
- [ ] Minority-class metrics over time.
- [ ] Ablation results with uncertainty/error bars.
- [ ] Sensitivity curves.
- [ ] Runtime and memory comparisons.

### OLD3S-comparable plot extension (added 2026-08-12)

- [ ] Generate multi-panel OCA/prequential-accuracy curves using the OLD3S paper's comparison structure.
- [ ] Plot the proposed model, FOBOS, OLSF, FESL, official OLD3S when reproducible, and selected strong general baselines together.
- [ ] Shade the old/new feature overlap period or mark the feature transition explicitly when no overlap interval exists.
- [ ] Add known-drift markers and visually distinguish detector alarms.
- [ ] Plot seed-mean curves with uncertainty bands and document smoothing/window choices.
- [ ] Use consistent colors, line styles, axes, method order, and legends across datasets/scenarios.
- [ ] Add companion Kappa, G-Mean, macro/minority PR-AUC, and minority recall/F1 plots.
- [ ] Add runtime and cumulative/additional-error comparison tables analogous to the OLD3S efficacy-versus-cost presentation.

For every thesis figure and table, record the source experiment, configuration, seeds, metric definition, and generation command.

## 11. SHAP and LIME Explainability Analysis

Use both SHAP and LIME as additional validation, as suggested by the professor. Treat explainability as analysis of the trained framework rather than as a core algorithmic contribution. Explain how predictions and important features change across the stream.

### SHAP

- [ ] Select a SHAP explainer compatible with the PyTorch model and document why it is appropriate.
- [ ] Define exactly which output is explained: the full fused prediction, individual experts, or both.
- [ ] Compute global feature importance for Stream 1 and Stream 2 separately.
- [ ] Compare SHAP values before feature evolution, immediately after it, and after recovery.
- [ ] Compare SHAP values before, during, and after concept drift.
- [ ] Separate shared, obsolete, and newly introduced features in SHAP plots.
- [ ] Compare explanations for majority and minority classes.
- [ ] Produce class-specific SHAP summary, beeswarm, bar, and dependence plots where appropriate.
- [ ] Explain representative correct predictions, errors, and drift-region samples with local SHAP plots.

### LIME

- [ ] Configure LIME for the tabular feature spaces used in Stream 1 and Stream 2.
- [ ] Explain representative predictions from before feature evolution, immediately afterward, and after recovery.
- [ ] Explain samples before, during, and after detected concept drift.
- [ ] Include correct predictions, incorrect predictions, minority-class samples, and low-confidence samples.
- [ ] Compare LIME explanations for the full model and relevant individual experts.
- [ ] Repeat LIME with multiple random seeds or perturbation samples to assess explanation stability.

### SHAP/LIME Comparison and Validation

- [ ] Compare whether SHAP and LIME identify similar influential features for the same samples.
- [ ] Measure explanation stability across nearby time steps, random seeds, and repeated runs.
- [ ] Test explanation faithfulness by masking or perturbing highly ranked features and measuring the prediction change.
- [ ] Relate feature-attribution changes to router alpha values, prototype usage, and known stream changes.
- [ ] Show whether the model reduces reliance on obsolete features and adopts new Stream 2 features.
- [ ] Document the background/reference data used by SHAP and the perturbation distribution used by LIME.
- [ ] Report computational cost and use a justified sample of stream instances if explaining every step is infeasible.
- [ ] Discuss limitations: correlated features, unstable local explanations, evolving feature dimensions, and explanations of a changing online model.

Generate thesis-ready figures and tables including:

- [ ] Stream 1 versus Stream 2 global SHAP importance.
- [ ] SHAP importance by drift phase and class.
- [ ] Local SHAP and LIME case studies for selected samples.
- [ ] Attribution trajectories for important shared, obsolete, and new features.
- [ ] SHAP/LIME agreement and stability results.

## 12. Computational Evaluation

Measure:

- [ ] End-to-end runtime.
- [ ] Online training/update time per stream instance.
- [ ] Inference latency per stream instance.
- [ ] Peak and average memory usage.
- [ ] Model parameter count and prototype-memory footprint.
- [ ] Scaling behaviour as the prototype bank and stream length increase.
- [ ] Computational overhead added separately by prototype memory, transfer mapping, and MoE routing.

Compare the full model with simpler baselines under the same hardware and software conditions. Report both predictive performance and computational tradeoffs.

## 13. Experimental Rigor and Reproducibility

- [ ] Run principal experiments with at least five random seeds.
- [ ] Report means, standard deviations, confidence intervals, and statistical tests.
- [ ] Freeze train/evaluation splits and prevent tuning on the reported test stream.
- [ ] Add automated tests for loading, metrics, drift detection, prototype updates, MoE routing, and saved outputs.
- [ ] Validate Hedge updates, prototype scores, and ACR using hand-computed examples.
- [ ] Reproduce all results from a clean environment.
- [ ] Add a dependency lock file and hardware/software details.
- [ ] Provide a single command or script that regenerates all final tables and figures.

## 14. Thesis Writing

- [ ] Create the thesis manuscript and bibliography files.
- [ ] Write the problem statement, research questions, hypotheses, and contributions.
- [ ] Complete the literature review.
- [ ] Formalize the proposed method with equations, algorithms, and pseudocode.
- [ ] Document datasets, feature-evolution protocols, baselines, metrics, and hyperparameter selection.
- [ ] Present main results, ablations, sensitivity analysis, and computational cost.
- [ ] Present and interpret the SHAP and LIME analyses.
- [ ] Discuss negative and mixed findings.
- [ ] Write limitations and threats to validity.
- [ ] Write the conclusion and future work.
- [ ] Complete citations, appendices, formatting, proofreading, and institutional submission checks.

## 15. Repository Cleanup

- [ ] Repair the README Markdown wrapper and code fences.
- [ ] Update the README for the transfer mapper, prototype memory, MoE, and final experiment commands.
- [ ] Add SHAP and LIME dependencies plus reproducible explanation-generation commands.
- [ ] Document dataset acquisition and preprocessing.
- [ ] Organize or remove temporary and duplicate experiment outputs.
- [ ] Intentionally commit the current implementation and required research artifacts.
- [ ] Ensure generated results can be traced to a code version and configuration.
