# Component-Isolation Validation Record

## Purpose

Thesis ablations must remove information paths, not merely rename a run or set an unrelated coefficient to zero. The model now exposes independent switches for:

- Learned transfer mapper.
- Historical knowledge initialization.
- Prototype memory.
- Learned MoE versus fixed fusion.
- Historical, adaptive, and prototype experts.
- Router hidden dimension.

Each run can also receive a safe, unique `run_tag`, preventing paired configurations from overwriting one another.

## Switch Semantics

| Switch | Disabled behaviour |
|---|---|
| `use_transfer_mapper` | S2 latent vectors pass directly to the historical classifier; mapper training is absent. |
| `use_historical_knowledge` | The adaptive S2 classifier is initialized independently; the historical expert is forcibly masked. |
| `use_prototype_memory` | Prototype prediction, updates, prototype-based sample weighting, and the prototype expert are disabled. |
| `fusion_mode=fixed` | Enabled experts receive equal, constant weights; router optimization is skipped. |
| `enable_*_expert` | The expert receives exactly zero fusion weight and its prediction is not computed for S2 fusion. |
| `router_hidden_dim` | Controls router capacity for sensitivity analysis. |

Removing historical or prototype knowledge overrides a contradictory request to enable its corresponding expert. Effective as well as requested settings are saved in run metadata.

## Verified Evidence

- Learned fusion weights remain non-negative and sum to one.
- Fixed three-expert fusion produces `[1/3, 1/3, 1/3]` at every S2 step.
- Removing historical knowledge produces historical alpha `0.0`.
- Removing prototype memory produces prototype alpha `0.0` and prototype count `0`.
- The single-adaptive configuration produces alpha `[0.0, 1.0, 0.0]`.
- With transfer disabled, the historical input is the unmodified S2 latent tensor.
- With prototype memory disabled, prototype lookup returns no evidence, updates are no-ops, and prototype-derived sample importance is `1.0`.
- Short end-to-end runs complete for no-transfer, no-history, no-prototype, fixed-fusion, and single-adaptive configurations.

## Transfer Diagnostics Added

The following prequential S2 diagnostics are now saved per step:

- Historical-expert cross-entropy and correctness.
- Adaptive-expert cross-entropy and correctness.
- Distance from the mapped S2 representation to the Stream 1 class centroid.
- Cosine similarity between the mapped S2 representation and Stream 1 class centroid.

These measurements permit full-transfer and no-transfer runs to test functional alignment rather than relying on tensor shapes alone.

## Historical-State and Forgetting Correction

The audit found that the supposedly frozen historical classifier and the adaptive classifier shared one Hedge head-weight vector. Although historical network parameters were frozen, S2 losses changed its head combination. Historical and adaptive Hedge vectors are now separated at the S1/S2 boundary. A test proves that adaptive updates leave the historical vector unchanged, and checkpoints preserve both vectors.

A fixed, non-updating S1 diagnostic subset is now evaluated during S2. The logs preserve historical-reference accuracy, adaptive-reference accuracy, and adaptive forgetting relative to the S2 boundary. This diagnostic never trains the model.

## Evidence Boundary

The switches and diagnostic paths are structurally verified. Their scientific effects still require adequately sized, paired, multi-seed experiments. A smoke run is not evidence that one architecture is more accurate than another.

## Core Pilot

All nine core architectures completed a 500-observation, one-seed pilot through the configuration-driven runner. The raw results and cautious interpretation are stored under `model/data/thesis_experiments/pilot_core_v3/`. The pilot exposed mixed evidence—particularly competitive reduced-expert and fixed-fusion variants—so no component-benefit claim has been promoted to a thesis conclusion.
