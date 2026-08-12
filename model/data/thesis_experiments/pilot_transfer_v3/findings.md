# Transfer-learning pilot

Status: diagnostic pilot only (one seed, seed 17). Protocol revision `thesis_protocol_2026-08-11_v4`.

## Comparisons

- Full learned transfer mapper and retained historical knowledge.
- Identity/no transfer mapper while retaining the historical path.
- No historical knowledge at the S1-to-S2 boundary.

## Preliminary observations

- All variants had identical exact pre-transition accuracy (0.56), confirming a matched S1 comparison.
- Transition accuracy was 0.30 for full transfer, 0.31 without the mapper, and 0.32 without historical knowledge.
- Stable post-transition accuracy was 0.43, 0.47, and 0.48 respectively.
- The learned mapper reduced mean prototype distance during transition (2.223 versus 2.413) and increased cosine similarity (0.266 versus 0.210), so latent alignment is occurring according to the mapper objective.
- Better geometric alignment did not translate into better historical-expert predictions: transition historical-expert correctness was 0.20 with the mapper versus 0.30 without it; later it was 0.37 versus 0.46.
- The full model did not reach the recovery threshold within the short post-transition pilot. The no-mapper variant recovered at 176 observations.
- The MoE did not suppress the weak historical expert strongly during transition: its mean historical weight remained about 0.293.

## Interpretation and next action

This seed verifies that the mapper changes and improves the measured latent geometry, but it does not demonstrate beneficial classifier transfer. The current evidence contradicts a strong claim that transfer improves prediction. The next experiments must use longer S2 windows, multiple seeds, expansion/contraction feature scenarios, and report alignment and predictive transfer separately. Router behaviour should also be checked because retaining substantial weight on a poorly performing historical expert can obscure any adaptation benefit. If the negative result persists, the thesis must present it as a limitation or revise the mapper objective using validation data without tuning on the final test stream.

Because this pilot has only 200 S2 observations and a 100-observation phase window, the early-recovery and stable-post windows coincide. Final runs must use a longer S2 segment so all reported phases are distinct.

