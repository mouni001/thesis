import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

# Find all metric files
files = glob('./data/parameter_*/metrics/all_metrics.npz')
labels = [os.path.basename(os.path.dirname(os.path.dirname(fp))).split('_', 1)[1] for fp in files]

# List of metrics to plot
metrics_to_plot = [
    'kappa',
    'gmean',
    'f1_min',
    'pr_auc',        # renamed from 'auc'
    'accuracy',
    'rec_min',       # recall for minority
    'prec_min',      # precision for minority
    'prec_maj',
    'rec_maj',
    'f1_maj',
    'drift',
    'times',
    'mems'
]

# Prettier labels for plots
pretty_names = {
    'kappa': "Cohen's Kappa",
    'gmean': 'Geometric Mean',
    'f1_min': 'F1 Score (Minority)',
    'pr_auc': 'Precision-Recall AUC',
    'accuracy': 'Batch Accuracy',
    'rec_min': 'Recall (Minority)',
    'prec_min': 'Precision (Minority)',
    'prec_maj': 'Precision (Majority)',
    'rec_maj': 'Recall (Majority)',
    'f1_maj': 'F1 Score (Majority)',
    'drift': 'Drift Events',
    'times': 'Time per Step (s)',
    'mems': 'Memory Usage (Bytes)'
}

# Plot all metrics
for metric in metrics_to_plot:
    plt.figure(figsize=(10, 5))
    for label, fp in zip(labels, files):
        data = np.load(fp)

        if metric not in data:
            print(f"[WARNING] Metric '{metric}' not found in {label}. Skipping.")
            continue

        if metric == 'drift':
            # Convert drift indices to binary time series
            drift_indices = data[metric]
            drift_flags = np.zeros_like(data['kappa'])  # assuming 'kappa' exists
            drift_flags[drift_indices.astype(int)] = 1
            series = drift_flags
        else:
            series = data[metric]

        plt.plot(np.linspace(0, 100, len(series)), series, label=label)

    plt.xlabel('Stream progress (%)')
    plt.ylabel(pretty_names.get(metric, metric))
    plt.title(f'Prequential {pretty_names.get(metric, metric)} Across Datasets')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
