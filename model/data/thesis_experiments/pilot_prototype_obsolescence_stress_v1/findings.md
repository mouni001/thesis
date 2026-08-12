# Prototype-obsolescence stress pilot

Status: one-seed mechanism stress test (seed 17), protocol revision `thesis_protocol_2026-08-11_v6`. The low-overlap stress condition is intentionally stronger than the standard benchmark and must be labelled as such.

## Stress condition

- S2 contraction from 24 to 17 dimensions.
- Only 25% configured feature overlap.
- 512-prototype capacity to retain more S1 history.
- 500 S1 and 500 S2 observations with distinct 125-observation transition, early-recovery, and stable windows.

## Results

| Variant | Transition accuracy | Early recovery | Stable post | Recovery time |
|---|---:|---:|---:|---:|
| Full decay | 0.336 | 0.512 | 0.528 | 209 |
| No feature obsolescence | 0.352 | 0.512 | 0.496 | 208 |
| No age decay | 0.352 | 0.488 | 0.528 | 429 |
| No prototype memory | 0.360 | 0.456 | 0.512 | not reached |

## Mechanism validation

- Feature obsolescence reduced S1-origin prototype evidence during transition from 0.075 to 0.029 and in the stable window from 0.070 to 0.028.
- That suppression traded away 1.6 percentage points of immediate transition accuracy but improved stable post-transition accuracy by 3.2 points (0.528 versus 0.496).
- In the stable window, full decay produced more prototype-help than harm (0.048 versus 0.040), while disabling feature obsolescence produced less help than harm (0.024 versus 0.048).
- Age decay did not change stable accuracy in this seed but substantially shortened recovery (209 versus 429 observations).
- Prototype memory reduced immediate transition accuracy relative to no memory (0.336 versus 0.360) but improved early recovery (0.512 versus 0.456), stable accuracy (0.528 versus 0.512), and achieved recovery where the no-memory variant did not.

## Interpretation

This is the first pilot that directly validates the intended obsolescence mechanism: it measurably suppresses historical prototype evidence, converts a harmful stable prototype contribution into a net helpful one, and improves later accuracy under strong feature obsolescence. The benefit is delayed rather than immediate. Final evaluation should include both the standard stream and this explicitly labelled stress scenario across multiple seeds, reporting evidence origin and help/harm alongside predictive metrics.

