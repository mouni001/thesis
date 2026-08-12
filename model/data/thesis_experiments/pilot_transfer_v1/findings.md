# Pilot Transfer Findings

Status: **superseded implementation pilot**. It predates separate historical/adaptive Hedge states, active S2 autoencoder optimization, normalized prototype quality, and the corrected MSE reconstruction path. Retain it as an audit record, but do not use its numerical results in the thesis.

## Question

Does the learned transfer mapper align Stream 2 representations with Stream 1 knowledge and improve adaptation relative to no mapper or no historical knowledge?

## Protocol

- Dataset: `INSECTS_incremental_balanced.csv`.
- Contiguous original interval: `[34700, 35200)`.
- Feature transition: original row `35000`, local step `300`.
- Feature dimensions: S1 `24`, S2 `25`.
- All six classes occur on both sides of the transition.
- Seed: `17`.
- S1-only standardization.

## Observations

- The learned mapper improved mean class-centroid cosine similarity from approximately `-0.018` to `0.057` in the first S2 window and from `0.013` to `0.249` in the next window.
- Mean mapped-to-centroid distance decreased over S2 from `2.680` to `2.461`; without a learned mapper it decreased from `2.913` to `2.791`.
- Historical-expert cross-entropy was lower with the mapper in both S2 windows.
- These alignment improvements did not translate into a clear predictive advantage in this pilot. Full transfer achieved correctness `0.28` then `0.48`; no mapper achieved `0.30` then `0.45`; no historical knowledge achieved `0.31` then `0.49`.
- No configuration recovered to 90% of its pre-transition correctness within the first 200 S2 samples under the predefined recovery rule.

## Defensible Pilot Conclusion

The diagnostics are functioning and provide preliminary evidence that the mapper learns geometric/classifier alignment. The pilot does not establish that transfer improves end-to-end accuracy or recovery. Longer multi-seed runs are necessary, and the possibility that historical transfer is neutral or harmful must remain open.
