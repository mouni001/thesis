# Prototype memory and obsolescence pilot

Status: diagnostic pilot only (one seed, seed 17). Do not use these values as final thesis estimates.

## Protocol

- INSECTS incremental balanced stream.
- Contiguous interval around split index 35,000.
- 1,500 pre-transition and 1,000 post-transition observations.
- Balanced feature-evolution scenario with unequal S1 and S2 feature sets.
- Five variants: full, no prototype memory, no feature-obsolescence decay, no age decay, and neither decay.

## Preliminary observations

- Prototype memory improved immediate transition correctness from 0.380 to 0.395 in this seed.
- The no-memory variant recovered faster (349 versus 512 observations) and had slightly higher stable post-transition correctness (0.545 versus 0.535).
- Removing only feature-obsolescence decay produced nearly the same transition result and slightly lower stable post-transition correctness (0.525).
- Removing both decay terms yielded the highest stable post-transition correctness (0.570), but increased prototype help and harm simultaneously (0.155 help and 0.125 harm during transition, versus 0.100 and 0.085 for the full model).
- Historical S1 evidence among prototype neighbours was very low after transition (below 2% on average in the reported transition summaries). This pilot therefore provides limited pressure for the feature-obsolescence mechanism to demonstrate a large benefit.

## Interpretation and next action

This pilot does not support a definitive claim that obsolescence improves predictive performance. It shows that prototypes can help individual predictions while also delaying aggregate recovery. The final evaluation must use multiple seeds and add a stress scenario with a larger obsolete-feature fraction and/or a smaller bank that forces competition between old and new prototypes. Report help, harm, S1 evidence fraction, recovery time, and stable post-transition performance together; accuracy alone would hide the mechanism.

