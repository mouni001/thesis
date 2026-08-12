# One-factor sensitivity pilot

Status: one-seed development pilot (seed 17), protocol v6. Twenty settings vary one factor at a time around the default; feature-overlap settings additionally change the observed feature views and are analyzed separately.

## Default

The default achieved transition/early/stable accuracy of 0.373/0.413/0.440, stable G-Mean 0.409, stable macro PR-AUC 0.508, and recovery at 143 observations.

## Predictive sensitivity

- Prototype fusion weight is sensitive: 0.10 produced stable accuracy 0.427, default 0.35 produced 0.440, and 0.60 produced 0.400 and did not reach recovery. Strong prototype influence is risky.
- Prototype neighbour count favoured a broader neighbourhood in this seed: `k=9` achieved 0.480 stable accuracy versus 0.440 for default `k=5` and `k=3`. `k=3` had similar stable accuracy but much slower recovery (210).
- Learning rate matters: 0.002 achieved 0.453 stable accuracy, default 0.001 achieved 0.440, and 0.0005 achieved 0.400 despite strong early recovery. Final selection must not be based on this one stream alone.
- Memory capacity 64 improved transition/stable accuracy to 0.387/0.453 and roughly halved inference latency (4.34 ms versus 9.16 ms). Capacity 512 behaved like default at 0.440 and retained 28.1 KB of prototype vectors versus 25.6 KB. A small bank may provide a better performance/cost trade-off.
- Obsolescence exponents 0.5, 1.0, and 2.0 produced identical phase accuracy in this normal-overlap pilot. The separate low-overlap stress test is more informative for this parameter.
- Router hidden dimensions 8, 32, and 64 produced identical phase accuracy and recovery, with only small PR-AUC differences. Router capacity is not a sensitive factor in this short stream.
- Minority weight 0.25 improved early recovery but matched default stable accuracy; weight 1.0 reduced stable accuracy to 0.427. The dedicated imbalanced-stream experiment remains authoritative for minority weighting.
- Drift weight is sensitive in both directions: 0.25 and 1.5 reduced stable accuracy to 0.400 and 0.413 versus 0.440 at 0.75. The default is locally preferable in this seed.
- Freshness exponent 2.0 improved stable accuracy to 0.453 and recovery to 127, whereas 0.5 matched default stable accuracy. This differs from the complete ablation where removing freshness was best, indicating interaction with stream length and reinforcing that freshness is not robustly beneficial.

## Feature overlap

- Configured overlap 0.25 produced dimensions 20→21, stable accuracy 0.493, and recovery at 89.
- Overlap 0.50 (default) produced 24→25, stable accuracy 0.440, and recovery at 143.
- Overlap 0.75 produced dimensions 29→29, stable accuracy 0.507, and recovery at 150. It still contains old-only and new-only features, but does not satisfy unequal dimensionality and must not be used as evidence for `dimension1 != dimension2`.
- Overlap changes both the information available and the feature partition, so these values measure scenario difficulty rather than ordinary tuning robustness.

## Robustness conclusion

The method is reasonably insensitive to router hidden size, obsolescence exponent in the normal scenario, and moderate changes in several quality weights. It is meaningfully sensitive to prototype fusion weight, learning rate, drift weight, neighbour count, memory capacity, and feature-overlap scenario. A blanket robustness claim is not supported. Final sensitivity curves should use repeated seeds for a reduced set of influential factors and report predictive/cost trade-offs rather than selecting the single best one-seed value.

