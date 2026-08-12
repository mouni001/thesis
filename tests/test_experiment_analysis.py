import math
import sys
import unittest
from pathlib import Path

import numpy as np


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from run_thesis_experiments import (  # noqa: E402
    exact_classification_metrics,
    phase_slices,
    recovery_time,
    score_detector,
)
from analyze_experiment_suite import holm_adjust, mean_ci, paired_comparison  # noqa: E402


class ExperimentAnalysisTests(unittest.TestCase):
    def test_repeated_run_statistics_use_paired_differences(self):
        result = paired_comparison(
            np.asarray([0.4, 0.5, 0.6, 0.7]),
            np.asarray([0.5, 0.58, 0.73, 0.79]),
        )
        self.assertEqual(result["n_pairs"], 4)
        self.assertAlmostEqual(result["mean_paired_difference"], 0.1)
        self.assertGreater(result["cohen_dz"], 1.0)

    def test_holm_adjustment_is_monotone_in_sorted_p_values(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03, np.nan])
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[2], 0.06)
        self.assertAlmostEqual(adjusted[1], 0.06)
        self.assertTrue(np.isnan(adjusted[3]))

    def test_single_seed_interval_is_explicitly_unavailable(self):
        mean, std, low, high = mean_ci([0.5])
        self.assertEqual(mean, 0.5)
        self.assertTrue(np.isnan(std))
        self.assertTrue(np.isnan(low))
        self.assertTrue(np.isnan(high))

    def test_constant_paired_difference_has_point_interval(self):
        result = paired_comparison(
            np.asarray([256.0, 256.0, 256.0, 256.0, 256.0]),
            np.zeros(5),
        )
        self.assertEqual(result["mean_paired_difference"], -256.0)
        self.assertEqual(result["difference_ci95_low"], -256.0)
        self.assertEqual(result["difference_ci95_high"], -256.0)
        self.assertTrue(np.isneginf(result["cohen_dz"]))
        self.assertEqual(result["paired_t_p"], 0.0)

    def test_exact_phase_metrics_are_not_rolling_curve_averages(self):
        y_true = np.asarray([0, 0, 1, 1])
        y_pred = np.asarray([0, 1, 1, 1])
        y_proba = np.asarray(
            [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]],
            dtype=float,
        )
        result = exact_classification_metrics(
            y_true, y_pred, y_proba, minority_class=1, majority_class=0
        )
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["rec_min"], 1.0)
        self.assertAlmostEqual(result["prec_min"], 2.0 / 3.0)
        self.assertAlmostEqual(result["pr_auc_min"], 1.0)
        self.assertIn("f1_c0", result)

    def test_phases_do_not_cross_the_declared_boundary(self):
        phases = phase_slices(n=500, boundary=300, window=100)
        self.assertEqual((phases["pre_change"].start, phases["pre_change"].stop), (200, 300))
        self.assertEqual((phases["transition"].start, phases["transition"].stop), (300, 400))
        self.assertEqual((phases["early_recovery"].start, phases["early_recovery"].stop), (400, 500))

    def test_recovery_uses_only_post_change_observations(self):
        correct = np.concatenate([np.ones(40), np.ones(40)])
        # A recovery of zero would reveal contamination by the pre-change rolling window.
        self.assertEqual(recovery_time(correct, boundary=40, window=20), 10.0)

    def test_recovery_can_remain_unreached(self):
        correct = np.concatenate([np.ones(40), np.zeros(40)])
        self.assertTrue(math.isnan(recovery_time(correct, boundary=40, window=20)))

    def test_detector_scoring_separates_delay_misses_and_false_alarms(self):
        result = score_detector(
            known_points=[100, 300],
            detected_points=[50, 112, 250, 450],
            tolerance=25,
        )
        self.assertEqual(result["detector_matched_changes"], 1)
        self.assertEqual(result["detector_missed_changes"], 1)
        self.assertEqual(result["detector_false_alarms"], 3)
        self.assertEqual(result["detector_mean_delay"], 12.0)


if __name__ == "__main__":
    unittest.main()
