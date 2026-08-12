# Threats to Validity

## Internal validity

Online experiments are sensitive to stream order, preprocessing leakage, and update timing. The protocol therefore uses predict-before-update evaluation, identical streams for paired comparisons, and a scaler fitted only on the historical prefix preceding the evaluated region. Raw predictions and probabilities are retained to audit metric calculations. Component ablations can nevertheless alter optimization dynamics as well as the intended mechanism, so they estimate the effect of the complete intervention rather than a perfectly isolated causal pathway.

The feature transition is known to the experimental framework. If deployment would not provide this boundary, performance may be optimistic unless boundary detection is evaluated separately. Detector alarms are kept separate from known annotations to prevent circular drift evaluation.

## Construct validity

Accuracy alone is inadequate for imbalanced streams; the study therefore includes minority recall/F1, G-Mean, PR-AUC, and per-class outcomes. Recovery thresholds and phase windows simplify continuous adaptation and can change conclusions, so complete time series and sensitivity to reasonable definitions should accompany summaries. Prototype help/harm and router weights are mechanism diagnostics, not direct proof of causal reasoning by the model.

SHAP and LIME explain model behavior locally under their respective background and perturbation distributions. Correlated variables, probability saturation, changing feature spaces, and the non-stationary model can make attribution rankings unstable. Agreement between explainers does not establish that an explanation is correct; perturbation faithfulness and repeated runs provide limited supporting evidence only.

## External validity

The principal held-out result uses a bounded region of one INSECTS stream and one primary feature-partition design. Generalization to other datasets, drift types, imbalance severities, feature semantics, or longer deployments is not guaranteed. Expansion, contraction, overlap, timing, and INSECTS-variant experiments reduce this risk but do not replace evaluation on independent real feature-evolution benchmarks.

The synthetic feature partition preserves source-column identity but may not capture real sensor replacement, delayed feature availability, missingness, or changes in measurement error. Conclusions should be framed as evidence under the evaluated protocol rather than universal claims about evolving feature spaces.

## Statistical conclusion validity

Ten paired seeds improve precision but provide limited power for small effects and for nonparametric tests. Multiple outcomes and ablations inflate false-positive risk; comparison families and Holm correction must be declared before interpretation. Confidence intervals, paired effect sizes, and consistency across phases should receive at least as much attention as p-values. Hyperparameters selected from pilot runs must not be reinterpreted as independently confirmed by those same pilots.

## Operational validity

Wall-clock timings depend on hardware, software versions, process load, and implementation efficiency. Report the environment and separate inference from update latency. Peak RSS includes runtime overhead, while prototype byte counts capture only persistent prototype storage; neither alone is total deployment memory. Python research-code timings may not represent an optimized production implementation.

## Reproducibility limitations

Random seeds control documented stochastic components but cannot guarantee bitwise equality across hardware, dependency versions, or nondeterministic numerical kernels. Each run therefore records its environment and code state. Final claims must remain tied to immutable v7 artifacts; regenerated outputs should receive a new manifest rather than silently replacing reported evidence.
