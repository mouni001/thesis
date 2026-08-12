# Structural baselines and MoE ablation pilot

Status: diagnostic pilot only (one seed, seed 17). Protocol revision `thesis_protocol_2026-08-11_v4`.

## Main preliminary results

Exact stable post-transition accuracy, highest to lowest:

1. No historical knowledge: 0.48.
2. No transfer mapper: 0.47.
3. Fixed fusion: 0.46.
4. No historical expert: 0.45.
5. Full model: 0.43.
6. Cold-start single adaptive classifier: 0.40.
7. No adaptive expert: 0.40.
8. No prototype expert: 0.40.
9. No prototype memory: 0.32.
10. OLD3S-style fixed two-expert configuration: 0.32.
11. Transferred single adaptive classifier: 0.30.

## Component interpretation

- Prototype mechanisms are useful in this seed: removing prototype memory reduced stable accuracy from 0.43 to 0.32, while removing only the prototype expert reduced it to 0.40. Prototype memory also changes sample weighting and therefore has an effect beyond the explicit prototype expert.
- The adaptive expert is useful: removing it reduced stable accuracy from 0.43 to 0.40.
- Historical transfer is harmful in this short pilot: removing historical knowledge, the transfer mapper, or the historical expert improved stable accuracy.
- The learned MoE router did not beat fixed equal fusion (0.43 versus 0.46). During transition, the full router assigned mean weights of approximately 0.293 historical, 0.373 adaptive, and 0.334 prototype, despite the weak historical branch.
- The full model had lower transition accuracy (0.30) than no history (0.32), no historical expert (0.32), and fixed fusion/no mapper (0.31).
- The no-transfer and fixed-fusion variants were the only leading multi-expert variants to reach the pilot recovery threshold (176 and 173 observations). The full model did not recover within the short S2 interval.

## Important scope notes

- S1 performance differs for variants that disable prototype memory because prototypes also affect S1 prediction and sample importance. Expert enable/disable switches apply to the S2 MoE; the S1 path remains the common Hedge classifier.
- The 200-observation S2 interval is too short for distinct early-recovery and stable-post phases when the evaluation window is 100. Final runs need longer S2 windows.
- These are one-seed mechanism checks, not inferential results.

## Next action

Run longer, multi-seed comparisons before making claims. In particular, test whether the router learns to suppress a weak historical expert after more S2 observations. If fixed fusion or removal of history remains superior, report that negative result and consider a validated router regularization or gating change only in a separately labelled development experiment.

