# Complete component-ablation pilot

Status: one-seed diagnostic pilot (seed 17), frozen predictive protocol v6. The suite contains 17 matched variants over 500 S1 and 500 S2 observations. The generated inferential files correctly show that confidence intervals and paired significance are unavailable from one seed.

## Stable post-transition accuracy

| Variant | Accuracy | G-Mean | Macro PR-AUC |
|---|---:|---:|---:|
| No freshness decay | 0.584 | 0.580 | 0.590 |
| Full model | 0.560 | 0.558 | 0.582 |
| Fixed fusion | 0.552 | 0.551 | 0.569 |
| No minority weighting | 0.544 | 0.544 | 0.563 |
| No obsolescence | 0.544 | 0.540 | 0.580 |
| No uncertainty | 0.536 | 0.534 | 0.589 |
| No drift relevance | 0.536 | 0.536 | 0.575 |
| No representativeness | 0.536 | 0.537 | 0.594 |
| OLD3S-style fixed two expert | 0.536 | 0.505 | 0.573 |
| No transfer mapper | 0.528 | 0.520 | 0.562 |
| No prototype memory | 0.528 | 0.498 | 0.580 |
| No prototype expert | 0.528 | 0.501 | 0.583 |
| No historical expert | 0.520 | 0.510 | 0.563 |
| Transferred single adaptive | 0.512 | 0.483 | 0.577 |
| No historical knowledge | 0.480 | 0.461 | 0.558 |
| No adaptive expert | 0.464 | 0.456 | 0.476 |
| Cold-start single adaptive | 0.464 | 0.408 | 0.533 |

## Component findings

- Transfer has a time-dependent effect: removing the mapper improves immediate transition accuracy (0.344 versus 0.304) but lowers stable accuracy (0.528 versus 0.560). Retaining historical knowledge is beneficial later; removing all history lowers stable accuracy to 0.480.
- Prototype memory improves stable accuracy by 3.2 points (0.560 versus 0.528) and G-Mean by 6.0 points, but immediate transition accuracy is lower (0.304 versus 0.328).
- Feature-obsolescence decay improves stable accuracy by 1.6 points relative to no obsolescence, consistent with the separate low-overlap stress test.
- Freshness decay is harmful in this normal-stream seed: removing it gives the best stable accuracy (0.584). This contradicts treating age alone as a reliable indicator of prototype usefulness.
- Representativeness, uncertainty, drift relevance, and minority weighting each improve stable accuracy by 1.6–2.4 points relative to their removals, although some removals improve immediate transition accuracy or PR-AUC. Their effects are not uniformly positive across all outcomes.
- The adaptive expert is the most important explicit expert: removing it lowers stable accuracy to 0.464. Removing the prototype or historical expert lowers it to 0.528 and 0.520.
- Transferring the adaptive classifier initialization improves stable accuracy over cold start (0.512 versus 0.464), even though the complete multi-expert model is better than either single classifier.
- The OLD3S-style fixed two-expert approximation reaches 0.536 stable accuracy, below the full model but above both single-classifier baselines. It is an architectural approximation, not yet a reproduction of the original authors' implementation.

## MoE behaviour

- The learned MoE slightly outperforms fixed fusion in stable accuracy (0.560 versus 0.552) but is worse during transition (0.304 versus 0.320); recovery time is identical at 255 observations.
- Mean full-model router weights move from approximately 0.340 historical / 0.340 adaptive / 0.320 prototype during transition to 0.326 / 0.445 / 0.228 in the stable window.
- By argmax selection, the adaptive expert is selected for 81.6% of transition samples, 99.2% of early-recovery samples, and 100% of stable samples. Switch frequency falls from 3.2% to 1.6% to zero.
- Router entropy falls from 1.098 near the maximum `ln(3)=1.099` during transition to 1.056 later. The router learns a directionally meaningful preference but remains a soft mixture; the predictive advantage over fixed fusion is small.

## Computational trade-offs

- Full-model mean inference/update times were approximately 11.0/22.5 ms per observation.
- Without prototype memory they were 0.72/6.10 ms, showing that the current Python-loop prototype implementation dominates latency.
- Removing all historical knowledge reduced inference/update times to 7.28/13.47 ms.
- Full checkpoint size was about 362 KB versus 250 KB without prototypes; stored prototype vectors themselves occupied 25.6 KB. Checkpoint overhead also includes prototype metadata and model state.
- Process RSS was roughly 497–523 MB across these runs and is too coarse to isolate small components; final reporting should emphasize paired latency, persistent size, prototype bytes, and process/GPU peaks together.

## Conclusion for final evaluation

The one-seed pilot supports retaining transfer, history, prototype memory, obsolescence, representativeness, uncertainty, drift relevance, minority weighting, and all experts for confirmatory testing. It does not support freshness decay as currently defined, and the learned router's advantage is small. These conclusions require repeated held-out runs; no one-seed difference is statistically established.

