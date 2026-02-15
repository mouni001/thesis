# import subprocess
# import sys

# insects_files = [
#     "INSECTS_abrupt_balanced.csv",
#     "INSECTS_abrupt_imbalanced.csv",
#     "INSECTS_incremental_balanced.csv",
#     "INSECTS_incremental_imbalanced.csv",
#     "INSECTS_incremental_abrupt_balanced.csv",
#     "INSECTS_incremental_abrupt_imbalanced.csv",
#     "INSECTS_incremental_reoccurring_balanced.csv",
#     "INSECTS_incremental_reoccurring_imbalanced.csv",
#     "INSECTS_gradual_balanced.csv",
#     "INSECTS_gradual_imbalanced.csv",
# ]

# for csv in insects_files:
#     print(f"\n=== Running {csv} ===\n")

#     subprocess.run([
#         sys.executable,
#         "train.py",
#         "-DataName=insects",
#         f"-insects_csv=data/{csv}",
#         "-AutoEncoder=AE",
#         "-beta=0.9",
#         "-eta=-0.01",
#         "-learningrate=0.001",
#         "-RecLossFunc=Smooth"
#     ])

#     subprocess.run([sys.executable, "plot_all_metrics.py"])
import os

import itertools
import subprocess
import sys

def main():
    py = sys.executable

    base_dir = os.path.dirname(__file__)
    insects_csv = os.path.join(base_dir, "data", "INSECTS_incremental_abrupt_imbalanced.csv")


    # INSECTS usually becomes 0..5 after LabelEncoder
    classes = [0, 1, 2, 3, 4, 5]

    pairs = list(itertools.combinations(classes, 2))
    print(f"[INFO] Running {len(pairs)} one-vs-one pairs on: {insects_csv}")
    print("[INFO] Pairs:", pairs)

    for a, b in pairs:
        cmd = [
            py, "train.py",
            "-DataName", "insects",
            "-insects_csv", insects_csv,
            "-class_a", str(a),
            "-class_b", str(b),
            "-RecLossFunc", "Smooth",
        ]

        print("\n[RUN]", " ".join(cmd))
        r = subprocess.run(cmd, cwd=base_dir)
        if r.returncode != 0:
            raise SystemExit(f"[ERROR] Failed on pair {a} vs {b} (return code {r.returncode}).")

    print("\n[DONE] All pairs finished successfully.")

if __name__ == "__main__":
    main()
