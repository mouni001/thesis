import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from loaddatasets import (  # noqa: E402
    _apply_feature_evolution,
    _resolve_feature_evolution_counts,
    load_insects_from_csv,
)
from train import (  # noqa: E402
    select_contiguous_stream,
    set_global_min_maj_from_reference,
)


class FeatureEvolutionProtocolTests(unittest.TestCase):
    def test_scaler_excludes_the_evaluated_s1_window(self):
        # With split=4 and exclusion=2, only rows [0, 1] may calibrate scaling.
        values = np.asarray(
            [
                [0.0, 10.0, 0],
                [2.0, 14.0, 1],
                [100.0, 200.0, 0],
                [200.0, 400.0, 1],
                [300.0, 600.0, 0],
            ],
            dtype=float,
        )
        with tempfile.NamedTemporaryFile(suffix=".csv") as handle:
            np.savetxt(handle.name, values, delimiter=",")
            _, _, _, _, metadata = load_insects_from_csv(
                handle.name,
                split_index=4,
                feature_protocol="same_features",
                scaler_exclusion_size=2,
                return_metadata=True,
            )
        np.testing.assert_allclose(metadata["scaler_mean"], [1.0, 12.0])
        self.assertEqual(metadata["scaler_fit_end_exclusive"], 2)
        self.assertEqual(
            metadata["scaling_protocol"],
            "standard_scaler_fit_on_historical_prefix_before_evaluated_s1",
        )

    def test_minority_identity_uses_only_s1_reference_labels(self):
        # Class 1 is the S1 minority even though a hypothetical future S2 block
        # could reverse the full-stream frequencies.
        minority, majority = set_global_min_maj_from_reference(
            torch.tensor([0, 0, 0, 1], dtype=torch.long)
        )
        self.assertEqual(minority, 1)
        self.assertEqual(majority, 0)

    def test_scenarios_control_dimension_direction(self):
        balanced = _resolve_feature_evolution_counts(33, 0.5, "balanced")
        expands = _resolve_feature_evolution_counts(33, 0.5, "s2_expands")
        contracts = _resolve_feature_evolution_counts(33, 0.5, "s2_contracts")

        old_b, shared_b, new_b = balanced
        self.assertEqual((old_b, shared_b, new_b), (8, 16, 9))

        old_e, shared_e, new_e = expands
        self.assertGreater(shared_e + new_e, old_e + shared_e)

        old_c, shared_c, new_c = contracts
        self.assertLess(shared_c + new_c, old_c + shared_c)

        for counts in (balanced, expands, contracts):
            self.assertEqual(sum(counts), 33)
            self.assertTrue(all(value >= 1 for value in counts))

    def test_partition_is_complete_disjoint_and_reproducible(self):
        x1 = np.arange(4 * 33, dtype=np.float32).reshape(4, 33)
        x2 = np.arange(4 * 33, dtype=np.float32).reshape(4, 33) + 1000

        s1, s2, metadata = _apply_feature_evolution(
            x1,
            x2,
            shared_frac=0.5,
            feature_seed=17,
            feature_scenario="s2_expands",
            return_metadata=True,
        )
        _, _, repeated = _apply_feature_evolution(
            x1,
            x2,
            shared_frac=0.5,
            feature_seed=17,
            feature_scenario="s2_expands",
            return_metadata=True,
        )

        old = set(metadata["old_only_indices"])
        shared = set(metadata["shared_indices"])
        new = set(metadata["new_only_indices"])
        self.assertFalse(old & shared)
        self.assertFalse(old & new)
        self.assertFalse(shared & new)
        self.assertEqual(old | shared | new, set(range(33)))
        self.assertEqual(metadata, repeated)

        np.testing.assert_array_equal(s1, x1[:, metadata["s1_indices"]])
        np.testing.assert_array_equal(s2, x2[:, metadata["s2_indices"]])
        self.assertEqual(s1.shape[1], metadata["dimension1"])
        self.assertEqual(s2.shape[1], metadata["dimension2"])

    def test_different_seeds_change_identity_not_counts(self):
        x = np.zeros((2, 33), dtype=np.float32)
        _, _, first = _apply_feature_evolution(
            x, x, feature_seed=1, return_metadata=True
        )
        _, _, second = _apply_feature_evolution(
            x, x, feature_seed=2, return_metadata=True
        )

        self.assertNotEqual(first["s1_indices"], second["s1_indices"])
        self.assertEqual(first["dimension1"], second["dimension1"])
        self.assertEqual(first["dimension2"], second["dimension2"])

    def test_invalid_protocol_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            _resolve_feature_evolution_counts(2, 0.5, "balanced")
        with self.assertRaises(ValueError):
            _resolve_feature_evolution_counts(10, 0.5, "unknown")
        with self.assertRaises(ValueError):
            _apply_feature_evolution(
                np.zeros((2, 3), dtype=np.float32),
                np.zeros((2, 4), dtype=np.float32),
            )

    def test_temporal_selection_is_contiguous_at_boundary(self):
        x1 = np.arange(80).reshape(80, 1)
        y1 = np.arange(80)
        x2 = np.arange(80, 100).reshape(20, 1)
        y2 = np.arange(80, 100)

        selected = select_contiguous_stream(x1, y1, x2, y2, B=5, t=3)
        _, selected_y1, _, selected_y2 = selected

        np.testing.assert_array_equal(selected_y1, np.arange(75, 80))
        np.testing.assert_array_equal(selected_y2, np.arange(80, 83))
        np.testing.assert_array_equal(
            np.concatenate([selected_y1, selected_y2]),
            np.arange(75, 83),
        )


if __name__ == "__main__":
    unittest.main()
