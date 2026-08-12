# Thesis Research Status — 2026-08-12

This is the authoritative orientation file for the thesis experiment workspace. It records what has been completed, what the evidence currently supports, where every result is stored, and what remains. Update this file only after checking the run artifacts and audit reports; do not infer completion from a configuration file alone.

## Current overall status

The research program is **not yet complete**. Five confirmatory protocol-v7 experiment suites are complete and audited. The sensitivity suite is in progress. The final repeated SHAP/LIME suite, cross-suite synthesis, final reproducibility run, and manuscript assembly remain.

All final claims must use protocol `thesis_protocol_2026-08-11_v7`. Earlier `pilot_*` and `smoke_*` directories are development evidence and must not be quoted as final thesis results.

## Completed confirmatory suites

| Research block | Runs | Audit | Tables/figures | Written findings |
|---|---:|---|---|---|
| Main model, baselines, MoE, and major ablations | 140/140 | Passed | Generated | `final_main_v7/block_findings.md` |
| Feature-evolution scenarios and mapper alignment | 30/30 | Passed | Generated | `final_feature_evolution_v7/block_findings.md` |
| Prototype memory and obsolescence stress tests | 20/20 | Passed | Generated | `final_prototype_obsolescence_v7/block_findings.md` |
| Concept drift, ADWIN/MDDM-G, and drift relevance | 20/20 | Passed | Generated | `final_drift_v7/block_findings.md` |
| Class imbalance and minority weighting | 20/20 | Passed | Generated | `final_class_imbalance_v7/block_findings.md` |

The suite roots are under `model/data/thesis_experiments/`. Each completed suite contains raw per-run metrics and checkpoints, `summary.csv`, `aggregate_statistics.csv`, `paired_tests.csv`, `audit_report.json`, tables, figures, manifests, and a `block_findings.md` interpretation.

## Principal findings already established

1. **Feature evolution works structurally.** The framework ran with unequal dimensions in balanced, expansion, and contraction scenarios. Mapper training consistently improved latent cosine alignment. This validates dimensional compatibility and the alignment mechanism, but not a general predictive advantage from the mapper.
2. **Historical transfer helps chiefly around adaptation.** Removing historical knowledge worsened transition performance and slowed recovery, while stable aggregate performance later converged. Removing only the mapper did not yield a broad stable predictive loss.
3. **Prototype memory helps later minority and balanced performance, at substantial cost.** Removing memory reduced stable accuracy, G-Mean, minority recall, and minority F1. Prototype-enabled runs were much slower.
4. **Obsolescence behaves mechanically as intended but has limited outcome evidence.** Stronger decay suppresses old-stream prototype evidence. Feature-obsolescence weighting did not produce a broad predictive benefit; age decay showed only a narrow stable G-Mean improvement.
5. **The adaptive expert is essential.** Its removal caused large transition, recovery, and stable-performance losses. The historical expert has smaller stable benefits. The prototype expert has mixed benefits and costs.
6. **Learned MoE routing has limited advantages over fixed fusion.** It modestly improved early recovery and stable macro PR-AUC, but did not clearly improve stable accuracy, G-Mean, or minority metrics and added runtime.
7. **External baselines prevent an overall-superiority claim.** Hoeffding Tree and HAT were stronger immediately at transition. Adaptive Random Forest was the strongest external comparator overall in the main suite, including faster recovery and much lower runtime. The proposed model was stronger than several simpler baselines later in the stream and on some minority outcomes.
8. **Drift detectors locate the injected drift, but balanced recovery remains a failure mode.** ADWIN and MDDM-G had zero misses and false alarms in the drift suite. MDDM-G had a faster delay point estimate. Drift relevance improved recovery accuracy and macro PR-AUC most clearly with MDDM-G, but recovery G-Mean remained zero because at least one class received zero recall.
9. **The tested minority weighting is not validated.** On both imbalanced streams, the rarest class had zero stable recall, F1, and G-Mean with and without weighting. Accuracy would conceal this failure.

These are concise orientations, not substitutes for the exact estimates and confidence intervals in each suite's `block_findings.md` and CSV tables.

## Work currently running

The sensitivity suite `final_sensitivity_v7` contains 105 planned runs: 21 configurations times five paired seeds. At the most recent audit, 59 run artifacts existed. Completed families included the default, prototype weight, obsolescence weight, drift weight, minority weight, and freshness weight settings; router hidden size 8 was nearly complete. The runner is resumable, so interruption does not invalidate completed runs.

Do not cite sensitivity conclusions until all 105 runs pass `audit_experiment_suite.py` and the final statistics, figures, tables, and written findings have been generated.

## Remaining work, in order

1. Implement, test, and document the OLD3S-comparable baseline group: FOBOS, OLSF, FESL, and the official OLD3S implementation if reproducible; add OLD-Linear and OLD-FD where faithful reproduction is possible.
2. Run the added literature baselines under the common protocol and audit their multi-seed statistical and computational results. Preserve the existing general baselines and ablations.
3. Produce OLD3S-style multi-panel OCA/prequential-accuracy curves with shaded overlap/transition regions, then extend them with seed uncertainty, known-drift/detector markers, and companion Kappa, G-Mean, PR-AUC, and minority-performance plots.
4. Finish all 105 sensitivity runs.
5. Audit the sensitivity suite; generate aggregate statistics, paired tests, sensitivity figures/tables, and `block_findings.md`.
6. Run the three frozen protocol-v7 explainability source models in `final_explainability_v7`.
7. Run repeated SHAP and LIME analyses across feature-evolution/drift phases, classes, correct/error/low-confidence cases, and explainer seeds.
8. Quantify SHAP/LIME agreement, stability, faithfulness under feature masking, feature-role changes, and computational cost; generate thesis-ready explainability tables and figures.
9. Consolidate computational results across full model, mapper/prototype/MoE ablations, memory-size sensitivity, OLD3S-family baselines, and general online baselines.
10. Run the full automated test suite and final protocol audits.
11. Generate a cross-suite evidence ledger and master results tables/figures.
12. Reconcile `THESIS_TODO.md`; many older checkboxes are stale and do not reflect the completed audited suites.
13. Write the full process-and-results handoff: experiment chronology, commands, design decisions, results by research question, negative findings, limitations, and chapter-ready prose.
14. Complete remaining manuscript work: literature review, method equations/pseudocode, results narrative, discussion, conclusion, appendices, bibliography checks, formatting, and proofreading.
15. Add/freeze dependency information, dataset acquisition instructions, hardware/software details, and a one-command final artifact regeneration path.
16. Clean repository outputs and intentionally version the implementation and required research artifacts.

## Approved baseline and plot modification — 2026-08-12

The user explicitly expanded the thesis objectives to require direct comparison with the baseline family used in the OLD3S paper and plots following the paper's OCA-over-stream structure. Existing completed runs remain valid, but the overall baseline and figure objectives are no longer complete until this extension is implemented, run, audited, and incorporated into the thesis materials.

## Files to use when writing the thesis

- `thesis/DETAILED_ANNOTATED_TABLE_OF_CONTENTS.md`: authoritative numbered writing blueprint, with a description of the contents of every proposed section and subsection.
- `thesis/PROPOSED_TABLE_OF_CONTENTS.md`: shorter structural outline retained for quick reference.
- `RESEARCH_QUESTIONS.md`: hypotheses, outcomes, and evidence required for RQ1–RQ10.
- `thesis/EXPERIMENTAL_METHODS.md`: protocol-frozen methodology draft.
- `thesis/REPRODUCIBILITY.md`: admissibility and reproduction rules.
- `thesis/THREATS_TO_VALIDITY.md`: limitations framework.
- `thesis/RESULTS_TEMPLATE.md`: results-chapter structure.
- `thesis/references.bib`: working bibliography.
- `model/data/thesis_experiments/final_*_v7/block_findings.md`: exact result interpretation for each completed block.
- `model/data/thesis_experiments/final_*_v7/tables/`: machine-generated CSV and LaTeX tables.
- `model/data/thesis_experiments/final_*_v7/figures/`: PNG and PDF figures plus manifests.

## Interpretation rules

- Report negative and mixed results; do not imply every proposed component helps.
- Distinguish structural/mechanistic validation from predictive benefit.
- Do not claim superiority over Adaptive Random Forest from the current evidence.
- Do not call an unadjusted paired interval or p-value definitive without considering the prespecified multiplicity family.
- State that the OLD3S-style comparator is an internal reconstruction unless a bit-for-bit official implementation is later reproduced.
- Treat SHAP and LIME as post-hoc validation, not core contributions.
- Never use pilot results as final estimates.
