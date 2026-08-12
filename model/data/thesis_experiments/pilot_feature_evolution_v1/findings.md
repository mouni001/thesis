# Feature-evolution scenario pilot

Status: diagnostic pilot only (one seed, seed 17). Protocol revision `thesis_protocol_2026-08-11_v4`.

## Verified scenarios

| Scenario | S1 dimension | S2 dimension | Obsolete | Shared | New |
|---|---:|---:|---:|---:|---:|
| Balanced unequal | 24 | 25 | 8 | 16 | 9 |
| S2 expands | 21 | 28 | 5 | 16 | 12 |
| S2 contracts | 27 | 22 | 11 | 16 | 6 |

All six mapper/no-mapper runs completed, produced valid predictions, and used the intended common latent classifier dimension. Pre-transition accuracy was identical (0.56) across all variants, confirming matched stream rows and S1 evaluation.

## Alignment versus prediction

- Balanced: mapper distance/cosine 2.223/0.266 versus 2.413/0.210 without mapper.
- S2 expands: 2.149/0.391 versus 2.396/0.266.
- S2 contracts: 2.069/0.224 versus 2.261/0.117.

Thus the learned mapper consistently improves the geometric alignment diagnostics across all three unequal-space scenarios.

Predictive results are mixed:

- Balanced stable accuracy: 0.43 mapper versus 0.47 no mapper.
- S2 expands: 0.45 versus 0.52.
- S2 contracts: 0.51 versus 0.49.
- Historical-expert transition correctness was lower with the mapper in every scenario (0.20 vs 0.30, 0.29 vs 0.33, and 0.17 vs 0.27).

## Interpretation and next action

The framework demonstrably handles `dimension1 != dimension2`, including both expansion and contraction, and the transfer mapping changes the latent representation in the intended geometric direction. However, geometric alignment is not sufficient evidence of useful classifier transfer. The mapper improved stable final accuracy only in the contraction scenario and reduced historical-expert transition correctness in all three. Final multi-seed evaluation must preserve this separation between structural validity, latent alignment, and predictive benefit. Add overlap-fraction and transition-timing sensitivity after the main protocol is frozen.

