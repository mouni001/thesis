# Detailed Annotated Thesis Table of Contents

This annotated version follows the exact hierarchy in `PROPOSED_TABLE_OF_CONTENTS.md`. Each subject has one role: Chapter 2 reviews established knowledge, Chapter 3 explains the proposed method, Chapter 4 defines the evaluation, Chapter 5 reports and interprets the evidence, and Chapter 6 closes the thesis. Later chapters should refer to earlier definitions rather than repeat them.

## Front Matter

The preliminary pages summarize the thesis and provide the lists needed to navigate its content.

### Abstract

Summarize the problem, proposed framework, experimental protocol, principal supported and unsupported findings, computational trade-offs, and conclusion. Write it after completing the thesis and use only audited final results.

### Acknowledgements

Acknowledge the supervisor, committee, institution, collaborators, technical support, funding if applicable, and personal support.

### Table of Contents

Generate the final numbered hierarchy and page references automatically from the manuscript.

### List of Figures

List the architecture, stream, performance, drift, prototype, router, sensitivity, explainability, and computational figures.

### List of Tables

List the dataset, configuration, baseline, ablation, statistical, explainability, and computational tables.

### List of Algorithms

List the pseudocode for the complete online procedure and any component that requires a separate algorithm.

### List of Abbreviations and Symbols

Define recurring abbreviations such as OLD3S, MoE, ADWIN, MDDM-G, OCA, G-Mean, PR-AUC, SHAP, and LIME, followed by the mathematical notation used in the thesis.

# 1 Introduction

This chapter motivates the research, states what the thesis investigates, and previews its contributions without reviewing every related method or presenting experimental results.

## 1.1 Motivation

Explain why online models must adapt when both data distributions and available features change. Motivate the need to reuse historical knowledge while managing stale information, minority-class failure, and computational cost.

## 1.2 Problem Statement and Objective

Define sequential classification when the feature space changes from (S_1) to (S_2), dimensions may differ, and drift or imbalance may occur. State the overall objective of developing and validating a transfer-, memory-, and expert-based framework for this setting.

## 1.3 Research Questions

Present the final questions covering feature evolution, transfer, prototype memory and obsolescence, drift, imbalance, expert routing, component necessity, sensitivity, explainability, and computational feasibility. Their experimental answers belong in Section 5.11.

## 1.4 Contributions

Preview the intended methodological, experimental, and reproducibility contributions. Revise the wording after the final audit so components with mixed or negative evidence are not presented as proven benefits.

## 1.5 Thesis Organization

Give one short paragraph explaining the purpose of Chapters 2–6 and the appendices.

# 2 Background and Related Work

This chapter develops the concepts and prior research needed to understand the framework. It should describe the literature, not the thesis implementation.

## 2.1 Data Streams and Online Learning

Introduce temporally ordered, potentially unbounded streams, incremental updating, bounded resources, test-then-train evaluation, and appropriate online performance measures.

## 2.2 Evolving Feature Spaces

Explain feature emergence, disappearance, expansion, contraction, and shared features. Review cross-space mapping, historical knowledge reuse, jump-start adaptation, and negative transfer.

## 2.3 Concept Drift

Define abrupt, gradual, incremental, and recurring drift, distinguish drift from feature evolution, and introduce detection using ADWIN and MDDM-G.

## 2.4 Imbalanced Data Streams

Explain minority-class difficulty in changing streams, common weighting and sampling approaches, and why accuracy must be accompanied by recall, F1, G-Mean, and PR-AUC.

## 2.5 Prototype-Based Learning

Review prototypes as compact class representatives used for retention and prediction. Introduce bounded memory, representativeness, recency, obsolescence, and negative transfer at a conceptual level.

## 2.6 Ensemble Learning and Mixture-of-Experts

Review fixed model combination, learned gating, expert specialization, and sample-dependent routing. Establish the need for fixed-fusion and expert-removal comparisons.

## 2.7 Existing Feature-Evolution Methods

Introduce the literature methods most closely related to the thesis and explain how they form a direct feature-evolution comparison group.

### 2.7.1 FOBOS

Describe its sparse first-order online optimization and the zero-padding treatment used to handle emerging and vanished features in the OLD3S evaluation.

### 2.7.2 OLSF

Describe its passive-aggressive treatment of streaming features and its limitations when the feature space decreases rather than only expands.

### 2.7.3 FESL

Describe its linear mapping between old and new feature spaces and its ensemble of old- and new-space classifiers.

### 2.7.4 OLD3S

Describe its shared latent representation, historical/new classifier combination, and adaptive-depth mechanism. Briefly introduce OLD-Linear, OLD-FD, and relevant extensions without creating additional main headings.

## 2.8 Explainability with SHAP and LIME

Introduce SHAP and LIME as post-hoc explanation tools, their global and local uses, and their stability, faithfulness, correlation, and evolving-feature limitations.

## 2.9 Research Gap

Synthesize what prior methods do not address jointly: transfer, bounded prototype retention, explicit obsolescence, drift and minority relevance, learned three-expert routing, and comprehensive interpretability and cost evaluation.

## 2.10 Summary

Recap only the concepts and unresolved gap that lead to the proposed framework; do not repeat the introductory motivation.

# 3 Proposed Framework

This chapter presents the proposed algorithm, mathematical formulation, and prediction/update process. It should not contain experimental settings or result claims.

## 3.1 Framework Overview

Present the architecture diagram and explain the flow from a stream observation through representation learning, historical transfer, prototype evidence, expert predictions, routing, and final output.

## 3.2 Problem Formulation

Define (S_1), (S_2), feature dimensions, old/shared/new feature sets, labels, latent representations, experts, router weights, and the online prediction objective using consistent notation.

## 3.3 Feature-Evolution and Transfer Mechanism

Explain the stream-specific encoders, latent-space relationship, transfer mapper, historical classifier path, loss functions, and alignment diagnostics used when (d_1 \ne d_2).

## 3.4 Prototype Memory

Define the bounded prototype bank, stored attributes, class-based retrieval, insertion, merging, replacement, nearest-neighbour evidence, and prototype probability construction.

## 3.5 Prototype Quality, Obsolescence, and Relevance

Present the quality formulation containing representativeness, uncertainty, freshness or age decay, feature obsolescence, drift relevance, and historical minority relevance. Explain the intended role of each term without presenting measured effects.

## 3.6 Mixture-of-Experts

Explain how three complementary probability estimates are produced and combined through sample-dependent routing.

### 3.6.1 Historical Expert

Describe the frozen Stream 1 classifier evaluated on mapped Stream 2 representations and its intended transition role.

### 3.6.2 Adaptive Expert

Describe the Stream 2 classifier that learns incrementally from newly arriving observations.

### 3.6.3 Prototype Expert

Describe the memory-based probability estimate created from selected prototypes.

### 3.6.4 Router and Expert Fusion

Define router inputs, hidden layer, softmax weights, training loss, online update, final fusion, and the fixed-fusion alternative.

## 3.7 Online Learning Procedure

Provide concise pseudocode covering initialization, Stream 1 prediction and updating, transition to Stream 2, transferred/adaptive/prototype inference, routing, detection, and memory maintenance.

## 3.8 Computational Complexity

Analyze time and space complexity by component and show how prototype-bank capacity, neighbour count, network size, and routing affect cost.

## 3.9 Summary

Connect each component to the research question that evaluates it without previewing the answer.

# 4 Experimental Design

This chapter states how the evidence is produced. It should define protocols and measurements once so Chapter 5 can focus on outcomes.

## 4.1 Experimental Hypotheses

Translate each research question into a testable hypothesis and identify its primary comparison and outcome. A compact hypothesis–experiment table will avoid repeating the full research questions.

## 4.2 Hardware and Software

Report the processor, memory, GPU if used, operating system, Python environment, key package versions, and timing method.

## 4.3 Datasets

Describe INSECTS data acquisition, features, classes, temporal order, preprocessing, balanced/imbalanced variants, and published drift annotations. Add any independent benchmark used by the final baseline extension.

## 4.4 Feature-Evolution Protocol

Explain the construction of old-only, shared, and new-only feature sets; balanced, expansion, contraction, and overlap scenarios; transition coordinates; evaluation phases; and leakage control.

## 4.5 Compared Methods

Introduce the full proposed model as the reference and group alternatives by experimental role so literature baselines are not confused with ablations.

### 4.5.1 Feature-Evolution Baselines

Describe FOBOS, OLSF, FESL, official OLD3S where reproducible, and any faithful variants. Record source, version, licence, and required protocol adaptations.

### 4.5.2 General Online-Learning Baselines

Describe Hoeffding Tree, Hoeffding Adaptive Tree, Adaptive Random Forest, and incremental Gaussian Naive Bayes under the common feature-identity protocol.

### 4.5.3 Ablation Models

List fixed fusion, the cold-start single classifier, mapper/history/prototype removals, relevance changes, and expert removals, stating what each comparison isolates.

## 4.6 Evaluation Metrics

Define OCA/prequential accuracy, Kappa, precision, recall, F1, G-Mean, PR-AUC, adaptation loss, recovery, forgetting, detector quality, explanation validation, latency, memory, and model size.

## 4.7 Statistical Analysis

Define repeated-seed summaries, confidence intervals, paired differences, effect sizes, paired tests, Holm correction, missing outcomes, and practical interpretation.

## 4.8 Experimental Setup

Use one experimental matrix to record stream boundaries, phase windows, configurations, seeds, frozen hyperparameters, baseline settings, SHAP/LIME sampling, and generation commands.

## 4.9 Reproducibility

Describe protocol freezing, preprocessing leakage controls, saved commands and configurations, code state, raw arrays, checkpoints, manifests, audits, and artifact traceability.

## 4.10 Summary

Explain briefly how the design supplies valid evidence for the research questions and transition to Chapter 5.

# 5 Experimental Results and Discussion

This chapter reports the audited evidence. Each section should present its numerical results once, discuss positive and negative outcomes, and avoid re-explaining the algorithm or protocol.

## 5.1 Feature-Evolution and Transfer Validation

Report unequal dimensions, feature partitions, shape/routing checks, classifier transfer, latent alignment, adaptation, recovery, forgetting, and predictive performance with and without mapping or historical knowledge.

## 5.2 Comparison with Baselines

Present OLD3S-style OCA/prequential curves, phase summaries, uncertainty, cumulative errors, and computational trade-offs for feature-evolution and general online baselines. Explicitly report when a simpler method is stronger.

## 5.3 Prototype Memory and Obsolescence Analysis

Compare memory enabled/disabled, obsolescence enabled/disabled, and age decay. Report adaptation, minority performance, prototype composition, old-stream evidence, help/harm, memory footprint, and latency.

## 5.4 Concept Drift Analysis

Show performance before, during, and after known drift; compare ADWIN and MDDM-G; evaluate relevance on/off; and distinguish detection success from balanced predictive recovery.

## 5.5 Class-Imbalance Analysis

Compare minority weighting on/off on incremental and gradual streams using minority precision, recall, F1, G-Mean, PR-AUC, overall accuracy, and majority trade-offs.

## 5.6 Mixture-of-Experts Analysis

Present router alpha trajectories, entropy, selected-expert proportions, transitions, learned versus fixed fusion, and the predictive and computational role of each expert.

## 5.7 Ablation Study

Provide one consolidated component-removal table with effect sizes, uncertainty, and cost. Refer to detailed findings in Sections 5.1–5.6 instead of repeating them.

## 5.8 Sensitivity Analysis

Present the completed one-factor study for prototype and relevance weights, router size, learning rate, memory capacity, neighbour count, and feature overlap. Identify robust ranges and sensitive settings.

## 5.9 SHAP and LIME Analysis

Compare global and local explanations across feature-evolution and drift phases, feature roles, classes, representative cases, and model seeds. Report agreement, repeatability, faithfulness, and explanation cost.

## 5.10 Computational Evaluation

Consolidate runtime, inference/update latency, memory, parameters, prototype footprint, and scaling results. Attribute overhead to transfer, prototypes, and routing using matched comparisons.

## 5.11 Answers to the Research Questions

Give a compact table stating whether each hypothesis is supported, partially supported, unsupported, or mixed, with pointers to the relevant result section and artifact. Do not reproduce all estimates.

## 5.12 Discussion

Interpret cross-cutting patterns, trade-offs, relationships with prior work, and practical meaning. Synthesize the evidence rather than repeating the research-question table or individual numerical results.

## 5.13 Threats to Validity

Discuss internal, construct, external, statistical, operational, and reproducibility threats, including seed count, simulated feature partitions, dataset scope, baseline adaptations, temporal dependence, multiple testing, and explanation limitations.

## 5.14 Summary

State the few findings needed to lead into the conclusion without repeating every research-question answer.

# 6 Conclusion and Future Work

This chapter closes the thesis concisely and should not introduce new results or reproduce the full Chapter 5 discussion.

## 6.1 Summary of the Thesis

Summarize the complete problem–method–evaluation story and state the central conclusion using only a few high-level findings.

## 6.2 Contributions

State which methodological, empirical, and reproducibility contributions were actually demonstrated by the final evidence. This evaluates the preview given in Section 1.4 rather than copying it.

## 6.3 Practical Implications

Explain when the full framework is worthwhile and when a simpler baseline, fixed fusion, smaller memory, or different imbalance strategy may be preferable.

## 6.4 Future Work

Derive focused directions from the threats and negative findings, including stronger imbalance handling, cheaper prototype retrieval, learned obsolescence, drift-triggered adaptation, broader benchmarks, and stable evolving-model explanations.

## 6.5 Final Remarks

End with a short, measured statement about the value of transfer, memory, and adaptive expertise for evolving feature streams and the importance of transparent trade-off reporting.

# References

Include every cited publication, dataset, software package, statistical procedure, and explainability method in the required institutional style, verifying each entry against its source.

# Appendices

The appendices preserve detailed supporting evidence without interrupting the main argument.

## Appendix A — Model and Experimental Parameters

Provide frozen protocol settings, model hyperparameters, feature partitions, stream boundaries, and baseline settings and provenance.

## Appendix B — Additional Results

Provide secondary phase, per-class, per-seed, drift, baseline, ablation, and sensitivity figures or tables omitted from Chapter 5.

## Appendix C — Statistical Results

Provide complete descriptive statistics, paired differences, confidence intervals, effect sizes, raw and adjusted tests, missing outcomes, and multiplicity families.

## Appendix D — Additional SHAP and LIME Results

Provide complete phase-, class-, seed-, agreement-, stability-, faithfulness-, and local-case outputs not selected for the main results chapter.

## Appendix E — Reproducibility and Evidence Ledger

Provide the environment, dataset acquisition details, ordered reproduction commands, artifact manifests, and mapping from every final claim, figure, and table to its authoritative source.

