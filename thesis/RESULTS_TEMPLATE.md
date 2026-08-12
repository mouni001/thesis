# Results Chapter Template

> Populate this document from generated v7 tables. Do not copy values from development pilots.

## 1. Protocol and implementation validation

State the evaluated source interval, scaler-fit interval, transition location, dimensions, and exact old/shared/new feature sets. Report automated-test status and latent-alignment diagnostics. Conclude narrowly whether unequal-dimensional processing and classifier transfer function as implemented.

Required artifacts: scenario table, architecture diagram, mapper-alignment table/plot, and shape-test summary.

## 2. Principal predictive comparison

Report overall and phase-specific accuracy, Kappa, G-Mean, macro F1, minority F1, and PR-AUC for the full model, internal baselines, and external online baselines. Lead with estimates and uncertainty, then discuss statistically paired differences. Explicitly state if a simpler baseline is better.

Required artifacts: main predictive table, paired-statistics table, overall time-series plot, and adaptation/recovery plot.

## 3. Transfer and historical knowledge

Compare the full method, no transfer mapper, and no historical knowledge. Discuss transition loss, recovery time, stable Stream 2 performance, latent alignment, and Stream 1 forgetting. Separate the value of the mapper from the value of retaining historical knowledge.

## 4. Prototype memory and obsolescence

Compare memory enabled, memory disabled, and obsolescence disabled. Report minority performance, stability, recovery, prototype composition, source-stream evidence, help/harm counts, removals, memory use, and latency. Acknowledge scenario dependence if obsolescence helps only under strong feature mismatch.

## 5. Concept drift and relevance weighting

Show metrics before, during, and after each known drift. Plot known drift annotations separately from ADWIN/MDDM alarms. Report detector delay, false alarms, missed changes, recovery, and the predictive effect of relevance weighting.

## 6. Class imbalance

Report per-class precision, recall, and F1 together with minority recall/F1, G-Mean, and PR-AUC. Discuss the majority-class tradeoff and whether probability ranking improves even when the default decision threshold fails to recover the minority class.

## 7. Mixture-of-Experts routing

Report alpha trajectories, phase-average weights, entropy, selected-expert proportions, and transitions. Compare learned routing with fixed fusion and expert-removal conditions. Do not equate changing weights with usefulness unless predictive performance is also competitive.

## 8. Complete ablation and sensitivity analysis

Present one-component-at-a-time ablations with paired uncertainty. Identify components whose removal improves some outcomes. Present sensitivity curves around the frozen default and distinguish robust ranges from sensitive parameters.

## 9. Explainability

Compare SHAP importance across phases and feature roles. Present repeated LIME cases and SHAP/LIME top-k agreement. Include perturbation faithfulness and explanation runtime. Treat saturated or locally constant predictions as an interpretable diagnostic rather than hiding zero-valued explanations.

## 10. Computational evaluation

Compare prediction latency, update latency, peak RSS, parameter count, and prototype-memory size. Relate any predictive gain to its operational cost and discuss online-deployment implications.

## 11. Consolidated answers to research questions

End with one evidence-limited answer per RQ: supported, partially supported, not supported, or inconclusive. Each answer should cite the exact table/figure and include both benefit and limitation.

## Writing checklist for every result

- Name the dataset/scenario, phase, metric, methods, and number of paired seeds.
- Give the estimate, uncertainty, effect direction, and practical magnitude.
- Distinguish prespecified confirmatory analysis from exploratory follow-up.
- State negative and mixed results directly.
- Avoid causal language beyond the randomized seed-paired ablation contrast.
- Reference the artifact manifest and generation command.
