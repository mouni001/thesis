# Protocol-v6 concept-drift pilot

Status: one-seed diagnostic pilot (seed 17) around the source-annotated abrupt change at local index 1,000.

## Detector comparison

- ADWIN detected the change at 1,023 (delay 23).
- MDDM-G detected it at 1,024 (delay 24).
- Both had one match, zero misses, and zero false alarms.
- For a fixed drift-relevance setting, detector variants produced identical predictions and probabilities because the detectors are monitoring-only. The one-observation detection-delay difference is the only valid detector result here.

## Drift phases and relevance weighting

- Exact accuracy fell from 1.00 before drift to 0.42 during drift with relevance ON and 0.38 OFF.
- In the following recovery window, accuracy was 0.39 ON versus 0.37 OFF.
- Macro PR-AUC was similar during drift (0.325 ON, 0.328 OFF), but lower during recovery with relevance (0.423 versus 0.504).
- G-Mean was zero during drift because at least one present class had zero recall.
- Neither setting reached the defined recovery threshold in the available post-change interval.

## Interpretation

Under the stabilized architecture, drift relevance modestly improves thresholded accuracy but harms recovery-window probability ranking. ADWIN and MDDM-G detect this single change at nearly the same time. These are mixed one-seed findings; final claims require all usable annotated changes, multiple seeds, detection uncertainty, and class-sensitive metrics. Detector-triggered adaptation remains outside the current algorithm and must not be implied.

