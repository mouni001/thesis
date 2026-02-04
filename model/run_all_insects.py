import subprocess
import sys

insects_files = [
    "INSECTS_abrupt_balanced.csv",
    "INSECTS_abrupt_imbalanced.csv",
    "INSECTS_incremental_balanced.csv",
    "INSECTS_incremental_imbalanced.csv",
    "INSECTS_incremental_abrupt_balanced.csv",
    "INSECTS_incremental_abrupt_imbalanced.csv",
    "INSECTS_incremental_reoccurring_balanced.csv",
    "INSECTS_incremental_reoccurring_imbalanced.csv",
    "INSECTS_gradual_balanced.csv",
    "INSECTS_gradual_imbalanced.csv",
]

for csv in insects_files:
    print(f"\n=== Running {csv} ===\n")

    subprocess.run([
        sys.executable,
        "train.py",
        "-DataName=insects",
        f"-insects_csv=data/{csv}",
        "-AutoEncoder=AE",
        "-beta=0.9",
        "-eta=-0.01",
        "-learningrate=0.001",
        "-RecLossFunc=Smooth"
    ])

    subprocess.run([sys.executable, "plot_all_metrics.py"])
