# Class-imbalance pilot

Status: diagnostic pilot only (one seed, seed 17). Protocol revision `thesis_protocol_2026-08-11_v4`.

## Protocol

- Matched minority-weighting ON/OFF runs on INSECTS incremental-imbalanced and gradual-imbalanced streams.
- The minority and majority class identities are fixed using only the evaluated S1 segment; S2 labels are not inspected when defining those classes.
- Reported table estimates are recomputed directly from samples within each phase. Rolling metrics are retained only for time-series visualization.
- `PR-AUC-min` is the one-vs-rest average precision for the fixed minority class. The existing `PR-AUC` series is macro average precision and must not be mislabeled as minority PR-AUC.
- The tested mechanism is prototype-informed minority weighting: the weight affects online classifier updates, prototype quality, and prototype selection. It is not a generic static class-weighted cross-entropy baseline.

## Preliminary observations

### Incremental-imbalanced stream

- Transition accuracy was similar with weighting ON (0.573) and OFF (0.580).
- Stable post-transition accuracy was 0.707 ON versus 0.700 OFF.
- Stable minority recall was 0.182 ON versus 0 OFF, and minority F1 was 0.222 ON versus 0 OFF.
- Stable minority PR-AUC was lower ON (0.259) than OFF (0.308), showing that the thresholded prediction gain did not improve ranking quality in this seed.
- Recovery was slightly slower ON (123 versus 115 observations).

### Gradual-imbalanced stream

- Neither variant predicted the fixed minority class in the transition or stable post-transition windows, so minority recall and F1 were zero.
- Stable minority PR-AUC was 0.125 ON versus 0.059 OFF. This is evidence of improved ranking in this seed, but not of usable thresholded minority detection.
- Stable overall accuracy was slightly lower ON (0.720 versus 0.733).

## Interpretation and next action

The pilot provides mixed evidence, not a universal improvement. Weighting can improve thresholded minority detection or ranking depending on the stream, while modestly trading off overall performance or recovery. Final claims require longer windows and multiple seeds. Report minority recall, precision, F1, minority PR-AUC, macro PR-AUC, per-class metrics, G-Mean, majority F1, and overall accuracy together. G-Mean was zero in these pilot phases because at least one present class had zero recall; that failure mode should be discussed rather than hidden.

