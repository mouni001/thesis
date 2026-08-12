# Research Questions and Evidence Matrix

This document freezes the questions that the experimental work must answer. Final claims must be based on repeated runs, not on a single favorable execution.

## Common Evaluation Protocol

- Use prequential evaluation: predict each instance before updating the online model with its label.
- Use identical stream order, feature partitions, transition locations, evaluation windows, and seeds for paired model comparisons.
- Run each principal comparison with at least five seeds. Report mean, standard deviation, 95% confidence intervals, paired effect sizes, and a paired significance test where its assumptions are defensible.
- Report the complete time series and predefined phase summaries rather than only the final time step.
- Primary predictive metrics: prequential accuracy, Kappa, G-Mean, macro/per-class F1, minority recall, minority F1, and PR-AUC.
- Operational metrics: adaptation time, recovery time, forgetting, update latency, inference latency, peak memory, and persistent model/memory size.
- Preserve the configuration, code revision, raw per-step outputs, summary, plots, and written finding for every final experiment.

## Predefined Stream Phases

For a known change at time `c`, use protocol-defined windows recorded before viewing model results:

1. `pre-change`: the evaluation window ending at `c - 1`.
2. `transition`: the evaluation window beginning at `c`.
3. `early recovery`: the next evaluation window.
4. `stable post-change`: the final evaluation window, provided no later known change overlaps it.

If a dataset supplies gradual-drift intervals, use those annotated intervals instead of treating drift as a single point. Detector alarms are predictions and must not be used as ground truth.

## Outcome Definitions to Implement

- `adaptation loss`: reduction in a metric from pre-change to the worst transition value.
- `recovery time`: number of instances after a known change until a smoothed metric reaches and maintains a declared fraction of its pre-change level.
- `recovery accuracy`: mean prequential accuracy over the early-recovery window.
- `forgetting`: loss of performance on a retained, fixed reference set from Stream 1 after learning Stream 2. This requires a separate diagnostic evaluation and must not update the model.
- `stability`: variability of a rolling metric within a phase, reported as standard deviation and worst-window degradation.
- `detection delay`: first valid detector alarm after a known drift minus the known drift start.
- `false alarm`: an alarm outside the declared tolerance interval around any known drift.
- `explanation stability`: agreement of feature rankings across repeated SHAP/LIME configurations or nearby samples, using a declared rank-correlation or top-k overlap measure.

## RQ1 — Does the Framework Operate Correctly Under Feature Evolution?

### Hypothesis

The framework supports unequal Stream 1 and Stream 2 feature dimensions and correctly transfers information through a common latent representation.

### Required validation

- Confirm `dimension1 != dimension2` in unequal-dimension scenarios.
- Record old-only, shared, and new feature indices; recording counts alone is insufficient.
- Test scenarios where Stream 2 has more features and where Stream 2 has fewer features.
- Validate encoder, mapper, classifier, prototype, and router tensor shapes.
- Verify that the historical classifier consumes mapped Stream 2 representations.
- Quantify latent alignment using a held-out overlap period or paired synthetic samples, not only dimensional compatibility.

### Evidence

- Automated protocol and shape tests.
- Scenario table containing dimensions, overlap, transition, seed, and feature indices.
- Latent-alignment measurement before and after mapper training.
- Architecture and feature-partition diagrams.

## RQ2 — Does Cross-Space Transfer Improve Adaptation?

### Hypothesis

The transfer mapper and historical knowledge reduce transition loss and recovery time after feature evolution without causing excessive forgetting.

### Paired comparisons

1. Full mapper and historical knowledge.
2. No mapper.
3. No historical knowledge.

### Primary outcomes

- Transition accuracy/Kappa/G-Mean.
- Adaptation loss and recovery time.
- Stable Stream 2 performance.
- Stream 1 forgetting.

### Evidence

- Phase-summary table with uncertainty and paired tests.
- Metric curves aligned at the feature transition.
- Latent-alignment plot and forgetting plot.

## RQ3 — Does Prototype Memory Improve Online Learning?

### Hypothesis

Prototype memory improves post-transition adaptation, minority-class performance, and stability relative to the same model without prototype memory.

### Paired comparisons

1. Prototype memory enabled.
2. Prototype memory disabled.

### Primary outcomes

- Recovery time and stable Stream 2 G-Mean.
- Minority recall and minority F1.
- Phase stability.
- Runtime and memory overhead.

### Evidence

- Phase-summary table and paired tests.
- Prototype count/composition trajectory.
- Performance and cost tradeoff plot.

## RQ4 — Does Prototype Obsolescence Prevent Negative Transfer?

### Hypothesis

Obsolescence decay reduces harmful reliance on old-space prototypes after feature evolution and improves Stream 2 recovery or stability.

### Paired comparisons

1. Obsolescence enabled.
2. Obsolescence disabled.
3. No prototype memory as a reference condition.

### Primary outcomes

- Stream 2 phase metrics.
- Prediction errors involving Stream 1 prototypes.
- Prototype source, age, obsolescence, selection, and removal over time.

### Evidence

- Performance curves aligned at feature evolution.
- Obsolescence and prototype-source trajectories.
- Case analysis showing when retained obsolete prototypes help or hurt.

The conclusion may be that obsolescence is unnecessary under some scenarios. Do not presuppose that outdated prototypes must become harmful.

## RQ5 — Does Drift Relevance Improve Recovery from Concept Drift?

### Hypothesis

Drift-weighted prototype selection reduces adaptation loss or recovery time after known concept drift.

### Paired comparisons

- Drift relevance enabled versus disabled.
- ADWIN versus MDDM under matched settings.

### Primary outcomes

- Detection delay, false alarms, and missed drifts.
- During-drift and recovery metrics.
- Recovery time and post-recovery stability.

### Evidence

- Metrics with known drift intervals and detector alarms shown separately.
- Detector-quality table.
- Drift-relevance ablation table and recovery plots.

## RQ6 — Does Minority Weighting Improve Imbalanced-Stream Performance?

### Hypothesis

Minority weighting improves minority recall, minority F1, G-Mean, and PR-AUC without an unacceptable loss in overall or majority-class performance.

### Paired comparisons

- Minority weighting enabled versus disabled.
- Multiple imbalance severities where available.

### Primary outcomes

- Per-class precision, recall, and F1.
- Minority recall and F1.
- G-Mean and PR-AUC.
- Overall and majority-class tradeoffs.

### Evidence

- Per-class and aggregate tables with uncertainty.
- Phase-specific minority-performance plots.
- Class-distribution and prediction-distribution diagnostics.

## RQ7 — Does Learned Mixture-of-Experts Routing Add Value?

### Hypothesis

The learned router changes expert allocation meaningfully across feature evolution and drift and outperforms fixed or reduced fusion strategies.

### Paired comparisons

1. Learned router with all experts.
2. Fixed fusion.
3. Single classifier.
4. Historical expert removed.
5. Adaptive expert removed.
6. Prototype expert removed.
7. Each expert evaluated individually.

### Primary outcomes

- Predictive phase metrics and recovery time.
- Historical, adaptive, and prototype alpha values by phase.
- Router entropy and expert-transition frequency.

### Evidence

- Alpha time series aligned at feature evolution and drift.
- Expert-weight phase table.
- Fusion ablation table and performance curves.

Meaningful routing requires more than changing alpha values: allocation changes should be interpretable and accompanied by competitive predictive performance.

## RQ8 — Which Components Are Necessary?

### Hypothesis

The complete design provides a better overall performance/adaptation tradeoff than component-removed alternatives.

### One-at-a-time ablations

- Transfer mapper and historical knowledge.
- Prototype memory.
- Representativeness, drift relevance, minority weighting, uncertainty, obsolescence, and freshness.
- Historical, adaptive, and prototype experts.
- Router/MoE.

### Evidence

- Multi-dataset, multi-seed ablation table.
- Effect sizes and uncertainty.
- Explicit discussion of components whose removal improves a metric.

## RQ8a — How Does the Method Compare with the OLD3S Evaluation Family?

### Objective

Establish direct comparability with the feature-evolution literature used by OLD3S, rather than relying only on general-purpose online classifiers or an internal OLD3S-style approximation.

### Required comparison group

- FOBOS with the OLD3S paper's documented treatment of emerging and vanished features.
- OLSF under its supported streaming-feature assumptions.
- FESL with its old/new feature-space mapping and ensemble procedure.
- The official OLD3S implementation, if it can be reproduced under the common protocol.
- OLD-Linear and OLD-FD when their official definitions or implementation can be reproduced faithfully.
- The proposed method and the already evaluated general stream baselines.

### Fair-comparison requirements

- Use identical stream order, feature identities, feature-transition locations, preprocessing boundaries, seeds, and test-then-train evaluation wherever algorithm assumptions permit.
- Document every adaptation required to run a published method on the INSECTS protocol.
- Distinguish official reproduction, faithful reimplementation, and literature-inspired approximation.
- Never label the existing `old3s_style_fixed_two_expert` condition as official OLD3S.
- Report predictive performance and computational cost under the same hardware/software conditions.

### Evidence

- Source/implementation provenance and version for every literature baseline.
- Automated behavioral tests and run-level protocol audits.
- Multi-seed phase summaries, paired uncertainty, effect sizes, and multiplicity-aware tests.
- OLD3S-comparable OCA curves and the thesis's additional balanced/minority metrics.

## RQ9 — Is the Method Robust to Hyperparameters and Scenarios?

### Parameters

- Prototype fusion weight, memory size, neighbour count, and merge rate.
- Prototype-quality weights and decay constants.
- Router hidden dimension.
- Encoder, mapper, classifier, and router learning rates.
- Feature-overlap fraction and transition timing.

### Evidence

- Sensitivity curves with uncertainty.
- Robust ranges rather than a claim of universal optimality.
- Documented selection policy that avoids tuning on final test results.

## RQ10 — Are Model Decisions Consistent with the Evolving Features?

SHAP and LIME are post-hoc validation requested by the professor, not core components of the proposed algorithm.

### Questions

- Which features influence predictions in each stream phase?
- Does reliance move away from obsolete features and toward shared/new features?
- How do explanations change around drift and for minority classes?
- Do explanations relate coherently to expert weights and prototype usage?

### Evidence

- Stream- and class-specific global SHAP summaries.

## Required OLD3S-Comparable Performance Visualization

The main baseline visualization must preserve the visual logic of the OLD3S evaluation while improving its inferential completeness:

- one panel per dataset or declared stream scenario;
- stream observation/time on the x-axis;
- online classification accuracy (OCA/prequential accuracy) on the y-axis;
- lines for the proposed model, FOBOS, OLSF, FESL, official OLD3S when reproducible, and selected strong general baselines;
- a shaded region for the old/new feature overlap period, or an explicitly labeled feature-transition marker when the protocol has no overlap interval;
- separate markers for known concept drift and detector alarms;
- mean curves and uncertainty bands across paired seeds, with the smoothing/window definition stated in the caption;
- consistent colors, method ordering, axes, and legends across panels; and
- no visual implication that detector alarms are ground-truth drift locations.

Companion figures must show Kappa, G-Mean, macro/minority PR-AUC, and minority recall/F1 so that OCA does not conceal class-coverage failures. Runtime and error-count tables should accompany the curve comparison, following the OLD3S paper's efficacy-versus-cost analysis while using the richer computational measures defined in this thesis.
- Matched local SHAP/LIME cases before and after changes.
- Agreement, stability, and feature-masking faithfulness checks.
- Full documentation of SHAP background data and LIME perturbation settings.

## RQ11 — What Is the Computational Cost?

### Paired comparisons

- Full model versus no prototypes, no transfer, fixed fusion, and single classifier.

### Outcomes

- End-to-end runtime.
- Update and inference latency per instance.
- Peak memory, parameter count, and prototype-memory footprint.
- Scaling with prototype-bank size and stream length.

### Evidence

- Cost table using identical hardware and workload.
- Accuracy/cost tradeoff figure.

## Minimum Final Artifact Set

Every research question must have:

1. Frozen configurations and seeds.
2. Raw per-step results.
3. Aggregated phase results.
4. Statistical comparison.
5. Thesis-ready table or figure.
6. A `findings.md` stating the hypothesis, result, limitations, and defensible conclusion.
