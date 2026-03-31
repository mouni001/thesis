"""Legacy compatibility wrapper.

The project now uses `train.py` as the single canonical experiment entry point.
This file stays in place only so older commands do not fail unexpectedly.
"""

from train import main


if __name__ == "__main__":
    print("[WARN] train_driver.py is deprecated. Redirecting to train.py.")
    main()
