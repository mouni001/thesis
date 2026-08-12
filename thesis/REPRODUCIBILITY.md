# Reproducibility Guide

## Confirmatory protocol

The only protocol authorized for final inferential claims is `thesis_protocol_2026-08-11_v7`. Earlier suites are development and audit evidence. The protocol rationale is recorded in `model/data/thesis_experiments/PROTOCOL_STATUS.md`.

## Main experiment

From the repository root, run:

```bash
python model/run_thesis_experiments.py experiments/configs/final_main.json --resume
```

The runner is resumable: completed run directories with metrics are retained and skipped. The final suite contains 14 methods and 10 paired seeds. Do not edit the configuration after beginning a confirmatory suite; create a newly named configuration and output directory for any changed design.

## Verification

Run the automated checks with:

```bash
python -m pytest -q
```

A run is admissible only if its metadata reports protocol v7, a 24-to-25 dimensional transition for the principal balanced scenario, a scaler-fit boundary no later than 44,500, 1,500 evaluated predictions, and aligned arrays for labels, predictions, probabilities, timings, phases, router values, and prototype diagnostics as applicable.

## Aggregation and artifact generation

The suite runner performs aggregation after all requested runs complete. Tables and figures can be regenerated with:

```bash
python model/analyze_experiment_suite.py model/data/thesis_experiments/final_main_v7
python model/generate_thesis_tables.py model/data/thesis_experiments/final_main_v7
python model/generate_thesis_figures.py model/data/thesis_experiments/final_main_v7
```

Generated CSV/LaTeX tables and PNG/PDF figures must be accompanied by their manifests. The manifests identify source suites, methods, seeds, metrics, and output files.

## Explainability

Run SHAP/LIME only against frozen checkpoints and snapshots produced by an admissible v7 execution. Record the explained output, target class, phase, background sample, perturbation count, explainer seeds, feature-role mapping, and runtime. Explanations are derived artifacts and must not be used to tune the already frozen confirmatory model.

## Evidence ledger

For each thesis table or figure, record:

- thesis label and caption;
- generation command;
- source suite and run directories;
- included methods and seeds;
- metric definition and phase window;
- generation timestamp and code revision; and
- any exclusions with reasons.

Never manually transcribe rounded values into an intermediate spreadsheet. Generate publication tables directly from raw run artifacts, and retain more precision in machine-readable CSV than is displayed in the thesis.
