# Core Architecture Pilot Findings

Status: **superseded structural pilot**. It predates separate historical/adaptive Hedge states, active S2 autoencoder optimization, normalized prototype quality, and the corrected MSE reconstruction path. Its switch validation remains useful, but its numerical results must not be used in the thesis.

## Protocol

- Dataset: balanced incremental INSECTS.
- Contiguous interval `[34700, 35200)` with feature evolution at local step `300`.
- S1/S2 dimensions: `24/25`.
- Five hundred prequential observations; seed `17`.
- Exact commands, metadata, checkpoints, raw time series, and summaries are stored with each run.

## Structural Conclusions

- Every required architecture ran end to end without sharing an output directory.
- Expert masks were reflected exactly in saved alpha values.
- Fixed fusion remained exactly one third per expert.
- Single adaptive fusion remained `[0, 1, 0]`.
- Prototype-free runs kept prototype count at zero.
- All runs saved reconstructable final checkpoints.

## Preliminary Performance Observations

- The full model achieved transition/early-recovery correctness of `0.28/0.48`.
- Removing historical knowledge achieved `0.31/0.49`; removing the historical expert achieved `0.29/0.51`.
- Removing the adaptive expert achieved `0.35/0.49`, the smallest pilot adaptation loss (`0.18`), and the highest pilot early-recovery G-Mean (`0.404`).
- Fixed fusion (`0.28/0.49`) was competitive with learned MoE fusion in this short run.
- Removing prototype memory reduced early-recovery correctness to `0.25`; keeping memory but removing the prototype expert reduced it to `0.24`.
- The single adaptive classifier reached `0.35` early-recovery correctness.

## Defensible Interpretation

The pilot validates the experiment machinery and suggests that the prototype expert is influential under this configuration. It does not yet show that the historical or adaptive expert, learned router, or complete model improves performance. In fact, several reduced variants match or exceed the full model on individual pilot metrics. Final conclusions require longer multi-seed experiments, statistical comparisons, and additional stream scenarios.

## Follow-up Triggered by the Pilot

- Inspect whether the router is undertrained in the first 200 S2 observations.
- Test longer S2 periods before deciding whether the adaptive expert becomes useful later.
- Compare fixed fusion and learned routing across at least five seeds.
- Measure effective component-specific compute rather than relying on process-wide `tracemalloc`, which is dominated by common allocations.
