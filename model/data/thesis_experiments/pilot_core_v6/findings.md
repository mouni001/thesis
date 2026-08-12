# Protocol-v6 structural and MoE pilot

Status: one-seed development pilot (seed 17). Identity-safe residual transfer and uniform router initialization are active.

## Results

| Variant | Transition accuracy | Stable post accuracy | Recovery observations |
|---|---:|---:|---:|
| No historical expert | 0.33 | 0.51 | 167 |
| Full learned MoE | 0.36 | 0.48 | 171 |
| Fixed equal fusion | 0.35 | 0.48 | 169 |
| No historical knowledge | 0.32 | 0.47 | not reached |
| No transfer mapper | 0.34 | 0.46 | 176 |
| No adaptive expert | 0.26 | 0.43 | not reached |
| No prototype expert | 0.30 | 0.43 | 184 |
| Cold-start single adaptive | 0.22 | 0.40 | 166 |
| OLD3S-style fixed two expert | 0.28 | 0.35 | not reached |
| No prototype memory | 0.27 | 0.33 | not reached |
| Transferred single adaptive | 0.27 | 0.30 | not reached |

## Interpretation

- Safe initialization materially improves the full model over protocol v4: transition/stable accuracy rose from 0.30/0.43 to 0.36/0.48 on the same development stream.
- Learned MoE is one point better than fixed fusion during transition and tied later. Its recovery is two observations slower. This is weak evidence, not proof of a router contribution.
- Router weights remain close to equal during transition (historical 0.338, adaptive 0.339, prototype 0.323), so meaningful specialization has not yet been demonstrated.
- Removing the historical expert gives the highest stable accuracy (0.51), even though removing all historical knowledge is slightly worse than the full model (0.47 versus 0.48). This suggests transferred initialization and S1-derived prototypes may help while direct historical-expert predictions hurt.
- Both adaptive and prototype experts contribute: removing either lowers stable accuracy to 0.43.
- Prototype memory is important in this development seed: disabling it lowers stable accuracy to 0.33. This switch also removes prototype-informed sample weighting, so the explicit expert and training effects must be separated in the complete ablation.

## Next action

Use longer S2 streams to test whether router weights meaningfully diverge and whether the historical expert is eventually suppressed. Keep fixed fusion, no historical expert, no historical knowledge, no prototypes, and individual-expert removals in final multi-seed comparisons. Do not claim that learned routing is superior unless the longer held-out results show consistent gains and interpretable expert transitions.

