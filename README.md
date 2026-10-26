# HAP-MoE: Historical, Adaptive, and Prototype Mixture of Experts

Online classification under feature-space evolution, concept drift, and class
imbalance. HAP-MoE combines a frozen historical classifier, an adaptive classifier,
prototype memory, a residual transfer mapper, and a learned mixture-of-experts
router.

The configurations compare HAP-MoE with adapted FESL, adapted shallow OLD3S,
Hoeffding Adaptive Tree, and Adaptive Random Forest. A supplementary comparison
also evaluates reliability-gated HAP-MoE and an adaptive-only condition.

## Installation

Extract the code archive and open a terminal in the project root. Create an
environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.
The versions in `requirements.txt` come from the saved reliability experiment
manifest. Process-memory measurements additionally use the optional `psutil`
package when installed.

## Data

Place the input datasets in `model/data/` using these filenames:

```text
adult.data
arrhythmia.data
car.data
INSECTS_abrupt_balanced.csv
INSECTS_abrupt_imbalanced.csv
INSECTS_gradual_balanced.csv
INSECTS_gradual_imbalanced.csv
INSECTS_incremental_abrupt_balanced.csv
INSECTS_incremental_balanced.csv
INSECTS_incremental_imbalanced.csv
INSECTS_incremental_reoccurring_balanced.csv
```

INSECTS CSV files have no header and store the class label in the final column.
Keep the original observation order. Dataset files are local inputs and are not
included in the code archive.

The loaders handle preprocessing and feature construction. Keep Adult,
Arrhythmia, and Car in their original data-file formats. Stream boundaries,
calibration sizes, and feature-partition settings are specified by the
configuration and loader code.

## Evaluation protocol

Predictions are recorded before each observation is used for online training.
The main INSECTS comparisons use 500 Stream 1 observations followed by 1,500
Stream 2 observations, with a rolling window of 250. The five paired seeds are
11, 17, 23, 31, and 43. Development and sensitivity configurations use their own
stream lengths and seeds.

The reported minority class is fixed from the evaluated Stream 1 labels. Metrics
are calculated per run before aggregation across seeds. ADWIN and MDDM outputs
are diagnostic and do not trigger model updates or enter the router.

## Experiments

Run commands from the project root. Each configuration fixes the dataset
selection, stream boundaries, feature construction, parameters, and random seeds.

| Configuration | Purpose | Runs |
| --- | --- | ---: |
| `experiments/configs/all_insects_comparison.json` | Eight-stream comparison with four baselines | 200 |
| `experiments/configs/additional_datasets.json` | Adult, Arrhythmia, and Car comparison | 75 |
| `experiments/configs/ablation.json` | Component ablations | 320 |
| `experiments/configs/sensitivity.json` | Parameter sensitivity | 125 |
| `experiments/configs/reliability.json` | Current, reliability-gated, and adaptive-only comparison | 120 |
| `experiments/configs/prototype_tuning.json` | First-stage development search | 27 |
| `experiments/configs/prototype_tuning_stage2.json` | Second-stage development search | 81 |

To inspect the first planned run without training:

```bash
python model/experiments/run_experiment_suite.py experiments/configs/all_insects_comparison.json --dry-run --max-runs 1
```

This writes suite metadata and prints the training command.

For the main INSECTS comparison:

```bash
python model/experiments/run_experiment_suite.py experiments/configs/all_insects_comparison.json --jobs 1
```

For the supplementary comparison:

```bash
python model/experiments/run_experiment_suite.py experiments/configs/reliability.json --jobs 1
```

For another experiment, replace the configuration argument with the relevant
file from the table. `--jobs` controls concurrent runs. To resume an interrupted
suite:

```bash
python model/experiments/run_experiment_suite.py experiments/configs/all_insects_comparison.json --jobs 1 --resume
```

Resume checks the configuration, source hashes, and saved run specifications.
Changes to source files, including comments, invalidate those checks for existing
runs. Final comparison settings are already fixed in their configurations;
rerunning the development searches is not required to execute them.

### Supplementary conditions

- **Current HAP-MoE:** the complete model with reliability gating disabled.
- **Reliability-gated HAP-MoE:** historical and prototype mixture weights are
  attenuated using EWMA pre-training losses, with smoothing factor 0.02. Updates
  affect subsequent predictions; the adaptive multiplier remains one.
- **Adaptive-only:** a fresh adaptive classifier without historical
  initialization, historical or prototype prediction branches, prototype memory
  and sample weighting, or transfer mapping. Several mechanisms are removed
  together, so this condition does not isolate their individual contributions.

## Saved outputs

Each suite is written to `model/data/thesis_experiments/<configuration-name>/`.
The configuration's `name` field determines the directory name.

```text
<configuration-name>/
├── manifest.json
├── code_changes.patch
├── summary.csv
├── runs/
│   └── <condition>/seed_<seed>/
│       ├── command.json
│       ├── run_spec.json
│       ├── run.log
│       └── metrics/all_metrics.npz
└── report/
```

Model checkpoints are also saved in the run directories. The manifest records
the configuration, dependency versions, and source hashes. The runner produces
summary tables after training; the reporting command below creates `report/`.

## Analysis

Generate reports from the completed main comparison:

```bash
python model/analysis/report.py model/data/thesis_experiments/final_all_insects_v8_tuned_v1 --config experiments/configs/all_insects_comparison.json --sections audit comparison friedman transition
```

For other suites, use the `name` field and configuration of that experiment.
Reports are written beneath the suite's `report/` directory. The analysis command
can generate CSV tables, LaTeX tables, and PDF/PNG figures; generated files are not
part of the source submission.

To summarize and audit the supplementary comparison:

```bash
python model/analysis/report.py model/data/thesis_experiments/insects_reliability_comparison --config experiments/configs/reliability.json --sections audit summary comparison
```

The audit verifies run completeness and recorded metrics before producing final
comparisons. Use the raw prediction records when calculating metrics for a
specific interval; averaging rolling F1 values gives a different quantity.

For individual runs and available arguments:

```bash
python model/train.py --help
python model/experiments/run_baseline.py --help
python model/analysis/report.py --help
```

## Layout

```text
experiments/configs/       Experiment definitions
model/                    Model, data loaders, metrics, and recording
model/experiments/         Suite runner, baselines, and tuning selection
model/analysis/            Statistical analysis and plotting
model/data/                Local datasets and generated outputs
external/OLD3S_official/   Vendored OLD3S source and license
```

The OLD3S baseline loads the vendored `autoencoder.py` and `mlp.py` files. The
upstream `model.py` is retained because the suite runner includes it in the source
manifest. The upstream revision is recorded in
`model/experiments/old3s_baseline.py`.

## License

See `LICENSE` and `external/OLD3S_official/LICENSE` for the applicable MIT notices.
