# Transfer-mapper initialization development pilot

Status: one-seed development result (seed 17), protocol revision `thesis_protocol_2026-08-11_v5`. This split is used for design diagnosis and must not be reused as the sole final test evidence.

## Motivation

The legacy mapper was a randomly initialized MLP. At the first S2 prediction it transformed the latent vector before receiving any S2 training signal, destroying the otherwise usable identity path. The protocol-v5 mapper is residual (`mapped_z = z + correction(z)`) with a zero-initialized final correction layer, so it begins as exact identity and learns changes online.

## Results

| Variant | Transition accuracy | Stable post accuracy | Historical transition correctness | Historical stable correctness | Recovery |
|---|---:|---:|---:|---:|---:|
| Residual transfer | 0.33 | 0.47 | 0.31 | 0.47 | 171 |
| Legacy random MLP | 0.30 | 0.43 | 0.20 | 0.37 | not reached |
| No mapper | 0.31 | 0.47 | 0.30 | 0.46 | 176 |
| No historical knowledge | 0.32 | 0.48 | n/a | n/a | not reached |

The residual mapper fixes the degradation caused by random initialization. It slightly improves transition accuracy and recovery over no mapper, and improves the historical expert by one percentage point, but it does not improve stable final accuracy. No historical knowledge remains one point better in the stable window.

The legacy MLP achieved a somewhat better geometric distance/cosine diagnostic than the residual mapper while producing much worse predictions. This reinforces that class-centroid geometry alone is not a sufficient transfer metric.

## Decision

Use the residual identity-safe mapper as the proposed protocol going forward. Preserve no-mapper and no-history ablations. Evaluate it on held-out stream regions, multiple feature scenarios, and multiple seeds; do not present this development comparison as confirmatory evidence.

