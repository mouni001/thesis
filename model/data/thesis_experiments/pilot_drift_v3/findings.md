# Concept-drift and drift-relevance pilot

Status: diagnostic pilot only (one seed, seed 17). Protocol revision `thesis_protocol_2026-08-11_v4`.

## Protocol

- INSECTS abrupt-balanced stream with a source-documented abrupt change at local index 1,000.
- The feature-space transition occurs separately at local index 500.
- Exact 100-observation windows before, during, and after the abrupt change.
- ADWIN versus MDDM-G detector logging.
- Prototype drift-relevance weight ON (`0.75`) versus OFF (`0`).

## Detector observations

- ADWIN matched the known change with a 23-observation delay and no false alarms or misses in both weighting variants.
- MDDM-G matched it with delays of 31 observations (drift relevance ON) and 28 observations (OFF), also with no false alarms or misses.
- For a fixed drift-relevance setting, ADWIN and MDDM produced byte-equivalent prediction classes and numerically identical probability sequences. This is expected because detectors currently monitor errors but do not trigger model adaptation or resets.
- Therefore, predictive-performance differences must not be attributed to ADWIN versus MDDM. Their valid comparison is detection delay, false alarms, and misses.

## Drift-relevance observations

- Exact accuracy was 1.00 in the 100 observations before the documented change and fell to 0.34 during the following 100 observations for both settings.
- Accuracy in the next recovery window was 0.24 with drift relevance ON versus 0.36 OFF.
- Macro PR-AUC during drift was 0.258 ON versus 0.251 OFF; during recovery it was 0.288 ON versus 0.297 OFF.
- G-Mean fell from 1.00 before drift to zero during and after it because at least one present class had zero recall.
- Neither variant reached the defined recovery threshold within the available post-change interval.

## Interpretation and next action

ADWIN detected this change earlier than MDDM-G in the pilot, but this requires multi-seed/multiple-change validation. Drift relevance did not improve recovery and appears harmful in this seed. The final experiment should retain detector evaluation and weighting ablation as separate analyses, use all documented abrupt changes where possible, and report negative results if they persist. If detector-triggered adaptation is proposed later, it must be implemented and evaluated as a new algorithmic variant rather than inferred from the current monitoring-only detector.

