# Literature-standard online baseline pilot

Status: diagnostic pilot only (one seed, seed 17). Protocol revision `thesis_protocol_2026-08-11_v4`.

## Matched comparison

All methods used the same 300-observation S1 and 200-observation S2 interval, original feature identities, feature-transition boundary, seed, and exact phase metrics.

| Method | Pre accuracy | Transition accuracy | Stable post accuracy | Transition G-Mean | Transition macro PR-AUC | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|
| Hoeffding Adaptive Tree | 0.66 | 0.52 | 0.56 | 0.502 | 0.462 | 8.18 |
| Hoeffding Tree | 0.66 | 0.52 | 0.52 | 0.503 | 0.486 | 7.95 |
| Gaussian Naive Bayes | 0.66 | 0.53 | 0.52 | 0.520 | 0.474 | 7.92 |
| Adaptive Random Forest | 0.58 | 0.53 | 0.48 | 0.384 | 0.543 | 10.26 |
| Full proposed model | 0.56 | 0.30 | 0.43 | 0.000 | 0.325 | 19.28 |

## Interpretation

Every external online baseline outperformed the proposed model on transition and stable-post accuracy in this seed. The proposed model also had zero transition G-Mean and was roughly two times slower than the River baselines under the instrumented 500-observation run. Its measured peak Python allocation was much larger, though `tracemalloc` does not capture all native tensor memory and must be supplemented with a consistent process/GPU memory method in final computational evaluation.

This result rules out claiming superiority from the current pilot. The final analysis needs longer windows, multiple seeds, effect sizes, and the actual original OLD3S implementation if it can be reproduced faithfully. If the gap persists, the thesis contribution should be framed around the investigated architecture/mechanisms and transparent negative findings rather than state-of-the-art predictive performance.

