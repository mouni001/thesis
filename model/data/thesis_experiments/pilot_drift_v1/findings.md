# Concept-Drift Pilot Findings

Status: **superseded detector-protocol pilot**. The detector delays remain a pipeline check, but predictive results predate separate Hedge states, active S2 autoencoder optimization, normalized prototype quality, and the corrected MSE reconstruction path. Rerun before using numerical results.

## Protocol

- Dataset: balanced abrupt INSECTS.
- Evaluated original interval: `[13352, 14852)`.
- Feature evolution: original row `13852`, local step `500`.
- Known abrupt concept drift: original row `14352`, local step `1000`.
- The model therefore received 500 S2 observations before the concept drift.
- Detector tolerance for this pilot: 100 observations.
- Compared ADWIN and MDDM-G, each with prototype drift relevance enabled and disabled.

## Detector Results

- ADWIN detected the known change at step `1023`: delay `23`.
- MDDM-G detected it at step `1028`: delay `28`.
- Both detectors matched the one known drift and produced no other alarms inside this 1,500-observation interval.

## Performance Around Drift

With drift relevance enabled:

- Correctness changed from `1.00` before drift to `0.47` during the first post-drift window and `0.58` in the following recovery window.
- G-Mean changed from `1.00` to `0.275` and then `0.529`.
- Minority F1 changed from `0.00` to `0.050` and then `0.319`. The pre-drift minority value is not meaningful because the globally designated minority class is absent from that concept.

With drift relevance disabled:

- Correctness changed from `1.00` to `0.45` and then `0.63`.
- G-Mean changed from `1.00` to `0.272` and then `0.550`.
- Minority F1 changed from `0.00` to `0.043` and then `0.289`.

No configuration returned to 90% of its pre-drift correctness within the observed 500 post-drift samples.

## Important Architectural Finding

The detector currently monitors prediction errors and records alarms; it does not trigger a model reset, router change, or prototype update by itself. Therefore ADWIN and MDDM runs with otherwise identical settings produce identical predictions by design. Their scientifically valid comparison is detection delay, misses, and false alarms—not predictive superiority.

The separate `prototype_drift_weight` affects local potential-set/sample weighting and prototype scoring. It is not the ADWIN/MDDM alarm response. That terminology must remain distinct in the thesis.

## Defensible Pilot Conclusion

Both detectors identified the known abrupt drift, with ADWIN five observations faster in this seed. Drift relevance produced slightly better during-drift correctness and minority F1, while disabling it produced higher next-window correctness and G-Mean. The effect is mixed and requires multi-seed analysis. Detector-specific predictive comparisons would be misleading unless an explicit detector-triggered adaptation mechanism is added and justified as part of the method.
