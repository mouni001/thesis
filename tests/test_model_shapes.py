import sys
import unittest
from pathlib import Path

import torch


MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from autoencoder import AutoEncoder_Shallow  # noqa: E402
from mlp import MLP  # noqa: E402
from moe import FeatureDriftRouter, MoEFusion  # noqa: E402
from model import OLD3S_Shallow, ResidualTransferMapper  # noqa: E402


class ModelShapeTests(unittest.TestCase):
    def test_residual_transfer_mapper_starts_as_exact_identity(self):
        mapper = ResidualTransferMapper(7)
        values = torch.randn(4, 7)
        torch.testing.assert_close(mapper(values), values)

    def test_unequal_spaces_share_a_classifier_latent_dimension(self):
        dimension1 = 22
        dimension2 = 27
        batch_size = 3
        classes = 6

        encoder1 = AutoEncoder_Shallow(dimension1, dimension2)
        encoder2 = AutoEncoder_Shallow(dimension2, dimension2)
        mapper = torch.nn.Linear(dimension2, dimension2)
        historical = MLP(dimension2, classes)
        adaptive = MLP(dimension2, classes)

        z1, reconstructed1 = encoder1(torch.randn(batch_size, dimension1))
        z2, reconstructed2 = encoder2(torch.randn(batch_size, dimension2))
        mapped_z2 = mapper(z2)

        self.assertEqual(z1.shape, (batch_size, dimension2))
        self.assertEqual(z2.shape, (batch_size, dimension2))
        self.assertEqual(mapped_z2.shape, z1.shape)
        self.assertEqual(reconstructed1.shape, (batch_size, dimension1))
        self.assertEqual(reconstructed2.shape, (batch_size, dimension2))
        self.assertEqual(historical(mapped_z2)[-1].shape, (batch_size, classes))
        self.assertEqual(adaptive(z2)[-1].shape, (batch_size, classes))

    def test_router_weights_form_a_convex_fusion(self):
        batch_size = 4
        classes = 6
        router = FeatureDriftRouter(input_dim=12, hidden_dim=8, num_experts=3)
        fusion = MoEFusion(router)
        experts = [torch.randn(batch_size, classes) for _ in range(3)]

        logits, alpha = fusion(torch.randn(batch_size, 12), *experts)

        self.assertEqual(logits.shape, (batch_size, classes))
        self.assertEqual(alpha.shape, (batch_size, 3))
        torch.testing.assert_close(alpha.sum(dim=1), torch.ones(batch_size))
        self.assertTrue(torch.all(alpha >= 0).item())
        self.assertTrue(torch.all(alpha <= 1).item())

    def test_router_starts_from_uniform_fusion(self):
        router = FeatureDriftRouter(input_dim=12, hidden_dim=8, num_experts=3)
        alpha = router(torch.randn(5, 12))
        torch.testing.assert_close(alpha, torch.full((5, 3), 1.0 / 3.0))

    def test_fixed_fusion_respects_expert_mask(self):
        router = FeatureDriftRouter(input_dim=5, hidden_dim=4, num_experts=3)
        fusion = MoEFusion(router)
        experts = [torch.randn(2, 3) for _ in range(3)]
        logits, alpha = fusion(
            torch.randn(2, 5),
            *experts,
            expert_mask=torch.tensor([1.0, 0.0, 1.0]),
            fixed_fusion=True,
        )

        torch.testing.assert_close(alpha, torch.tensor([[0.5, 0.0, 0.5]]).repeat(2, 1))
        torch.testing.assert_close(logits, 0.5 * experts[0] + 0.5 * experts[2])

    def test_component_switches_remove_information_paths(self):
        x1 = torch.randn(4, 5)
        x2 = torch.randn(4, 7)
        y = torch.tensor([0, 1, 0, 1])
        model = OLD3S_Shallow(
            x1,
            y,
            x2,
            y,
            T1=4,
            t=2,
            dimension1=5,
            dimension2=7,
            path="unit_test_switches",
            use_transfer_mapper=False,
            use_historical_knowledge=False,
            use_prototype_memory=False,
            fusion_mode="fixed",
            enable_historical_expert=True,
            enable_adaptive_expert=True,
            enable_prototype_expert=True,
        )
        z = torch.randn(1, 7)

        self.assertIs(model._map_to_historical(z), z)
        self.assertFalse(model.enable_historical_expert)
        self.assertFalse(model.enable_prototype_expert)
        self.assertTrue(model.enable_adaptive_expert)
        self.assertIsNone(model._prototype_logits(z, "s2"))
        model._prototype_update(z, torch.tensor([0]), 1, "s2")
        self.assertEqual(model.prototype_bank, [])
        self.assertEqual(model._sample_importance(z, 0), 1.0)

    def test_non_contiguous_window_labels_keep_valid_output_ids(self):
        x1 = torch.randn(3, 5)
        x2 = torch.randn(3, 7)
        y1 = torch.tensor([0, 3, 5])
        y2 = torch.tensor([1, 4, 5])
        model = OLD3S_Shallow(
            x1,
            y1,
            x2,
            y2,
            T1=4,
            t=2,
            dimension1=5,
            dimension2=7,
            path="unit_test_non_contiguous_labels",
        )
        self.assertEqual(model.num_classes, 6)

    def test_adaptive_hedge_update_does_not_change_historical_weights(self):
        x1 = torch.randn(4, 5)
        x2 = torch.randn(4, 7)
        y = torch.tensor([0, 1, 0, 1])
        wrapper = OLD3S_Shallow(
            x1, y, x2, y, T1=4, t=2, dimension1=5, dimension2=7,
            path="unit_test_separate_hedge",
        )
        classifier = MLP(7, 2)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
        historical = torch.nn.Parameter(wrapper.alpha.detach().clone(), requires_grad=False)
        adaptive = torch.nn.Parameter(wrapper.alpha.detach().clone(), requires_grad=False)
        historical_before = historical.detach().clone()
        adaptive_before = adaptive.detach().clone()

        wrapper.HB_Fit(
            classifier,
            torch.randn(1, 7),
            torch.tensor([1]),
            optimizer,
            alpha=adaptive,
        )

        torch.testing.assert_close(historical, historical_before)
        self.assertFalse(torch.allclose(adaptive, adaptive_before))
        torch.testing.assert_close(adaptive.sum(), torch.tensor(1.0))

    def test_prototype_quality_uses_documented_normalized_weighted_mean(self):
        x1 = torch.randn(4, 5)
        x2 = torch.randn(4, 7)
        y = torch.tensor([0, 1, 0, 1])
        wrapper = OLD3S_Shallow(
            x1, y, x2, y, T1=4, t=2, dimension1=5, dimension2=7,
            path="unit_test_quality_formula",
            prototype_rep_weight=1.0,
            prototype_drift_weight=1.0,
            prototype_minority_weight=1.0,
            prototype_uncertainty_weight=1.0,
            prototype_obsolescence_weight=0.0,
            prototype_freshness_weight=0.0,
        )
        prototype = {
            "rep": 4.0,
            "drift": 3.0,
            "minority": 3.0,
            "space": "s2",
            "last_step": 0,
        }
        self.assertAlmostEqual(
            wrapper._prototype_quality(prototype, "s2", uncertainty=1.0),
            1.0,
        )
        prototype.update({"rep": 0.05, "drift": 0.0, "minority": 0.0})
        self.assertAlmostEqual(
            wrapper._prototype_quality(prototype, "s2", uncertainty=0.0),
            1e-6,
        )

    def test_mse_reconstruction_preserves_standardized_value_range(self):
        x1 = torch.randn(4, 5)
        x2 = torch.randn(4, 7)
        y = torch.tensor([0, 1, 0, 1])
        wrapper = OLD3S_Shallow(
            x1, y, x2, y, T1=4, t=2, dimension1=5, dimension2=7,
            path="unit_test_mse_range", RecLossFunc="mse",
        )
        target = torch.tensor([[-2.0, 0.0, 3.0]])
        self.assertEqual(wrapper._reconstruction_loss(target.clone(), target).item(), 0.0)

    def test_prototype_logits_are_normalized_and_scaled_for_moe(self):
        x1 = torch.randn(4, 5)
        x2 = torch.randn(4, 7)
        y = torch.tensor([0, 1, 0, 1])
        wrapper = OLD3S_Shallow(
            x1, y, x2, y, T1=4, t=2, dimension1=5, dimension2=7,
            path="unit_test_prototype_logits", prototype_weight=0.35,
            prototype_obsolescence_weight=0.0,
            prototype_freshness_weight=0.0,
        )
        wrapper.prototype_bank = [
            {"vec": torch.zeros(7), "label": 0, "rep": 1.0, "drift": 0.0,
             "minority": 0.0, "space": "s2", "last_step": 0},
            {"vec": torch.ones(7), "label": 1, "rep": 1.0, "drift": 0.0,
             "minority": 0.0, "space": "s2", "last_step": 0},
        ]
        z = torch.zeros(1, 7)
        logits = wrapper._prototype_logits(z, "s2")
        torch.testing.assert_close(torch.exp(logits).sum(dim=1), torch.ones(1))
        scaled = wrapper._prototype_expert_logits(z, "s2", torch.zeros(1, 2))
        torch.testing.assert_close(scaled, 0.35 * logits)

    def test_prototype_help_and_historical_evidence_are_measured(self):
        x1 = torch.randn(4, 5)
        x2 = torch.randn(4, 7)
        y = torch.tensor([0, 1, 0, 1])
        wrapper = OLD3S_Shallow(
            x1, y, x2, y, T1=4, t=2, dimension1=5, dimension2=7,
            path="unit_test_prototype_diagnostics",
            prototype_obsolescence_weight=0.0,
            prototype_freshness_weight=0.0,
        )
        wrapper.prototype_bank = [
            {"vec": torch.zeros(7), "label": 0, "rep": 1.0, "drift": 0.0,
             "minority": 0.0, "space": "s1", "origin_space": "s1", "last_step": 0},
        ]
        diagnostics = wrapper._compute_prototype_diagnostics(
            torch.zeros(1, 7),
            "s2",
            y_true=0,
            full_logits=torch.tensor([[2.0, 0.0]]),
            historical_logits=torch.tensor([[0.0, 2.0]]),
            adaptive_logits=torch.tensor([[0.0, 2.0]]),
            moe_alpha=torch.tensor([[0.3, 0.3, 0.4]]),
        )
        self.assertEqual(diagnostics["prototype_help"], 1.0)
        self.assertEqual(diagnostics["prototype_harm"], 0.0)
        self.assertEqual(diagnostics["prototype_s1_neighbor_fraction"], 1.0)
        self.assertEqual(diagnostics["prototype_s1_evidence_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
