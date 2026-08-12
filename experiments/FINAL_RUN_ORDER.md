# Final Experiment Run Order

All commands are resumable and must be executed from the repository root. Do not use results for thesis claims until the entire named suite has finished and its manifest has passed the protocol-v7 audit.

1. Principal models, baselines, and major expert/component ablations (140 runs):

   ```bash
   python model/run_thesis_experiments.py experiments/configs/final_main.json --resume
   ```

2. Feature-evolution expansion/contraction and mapper validation (30 runs):

   ```bash
   python model/run_thesis_experiments.py experiments/configs/final_feature_evolution.json --resume
   ```

3. Prototype memory and obsolescence stress test (20 runs):

   ```bash
   python model/run_thesis_experiments.py experiments/configs/final_prototype_obsolescence.json --resume
   ```

4. ADWIN/MDDM and drift-relevance comparison at the held-out abrupt change (20 runs):

   ```bash
   python model/run_thesis_experiments.py experiments/configs/final_drift.json --resume
   ```

5. Minority weighting on two imbalanced streams (20 runs):

   ```bash
   python model/run_thesis_experiments.py experiments/configs/final_class_imbalance.json --resume
   ```

6. One-factor-at-a-time sensitivity analysis (105 runs):

   ```bash
   python model/run_thesis_experiments.py experiments/configs/final_sensitivity.json --resume
   ```

7. Frozen source models for repeated SHAP/LIME analysis (3 runs):

   ```bash
   python model/run_thesis_experiments.py experiments/configs/final_explainability.json --resume
   ```

8. Run the explainability analyzer for each completed source run and aggregate phase-, class-, and feature-role results. Then regenerate all suite tables and figures.

## Gate between suites

Before moving to the next suite, verify:

- expected run count and seed coverage;
- protocol version, scaler-fit boundary, stream length, feature identities, and transition indices;
- equal per-step array lengths and finite probabilities/metrics;
- absence of run-level exceptions or silent fallback behavior;
- paired-method seed alignment;
- generated aggregate summary and artifact manifest; and
- a short findings note that records positive, negative, mixed, and anomalous outcomes.

Run the automated portion of this gate with, for example:

```bash
python model/audit_experiment_suite.py \
  experiments/configs/final_main.json \
  model/data/thesis_experiments/final_main_v7
```

The command intentionally exits with a failure status while runs are absent or invalid and writes `audit_report.json` for inspection.

If a gate fails, fix the cause and create a new versioned suite. Do not silently overwrite an already interpreted result.
