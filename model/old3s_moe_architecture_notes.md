# OLD3S + MoE Architecture Diagram Notes

Diagram file:

```text
model/old3s_moe_architecture.drawio
```

## How to open it in diagrams.net

1. Go to `https://app.diagrams.net/`.
2. Choose `Device`.
3. Click `File -> Open From -> Device`.
4. Select:

```text
/Users/nihadzitouni/Desktop/thesis/model/old3s_moe_architecture.drawio
```

## What the Diagram Shows

The `.drawio` file contains three pages:

1. `OLD3S MoE Layer Flow Chart`: full architecture overview
2. `Prototype Memory Detail`: how prototypes are created, scored, used, and updated
3. `MoE Experts and Router Detail`: how the experts and router interact

The architecture should be read as a response to feature drift. Each major block has a specific role:

| Component | Feature-drift challenge addressed |
|---|---|
| `AE1 / AE2 encoders` | Map different feature spaces into latent representations so S1 and S2 can be compared/used after feature evolution. |
| `Historical Expert` | Preserves knowledge learned from S1 when old/shared features are still useful after the transition. |
| `Adaptive Expert` | Learns from S2 data when new features appear and the old classifier is no longer sufficient. |
| `Prototype Expert` | Stores local memory for drift-sensitive, minority, and uncertain regions where neural predictions may be unstable. |
| `Feature-Drift Router` | Decides how much to rely on Historical, Adaptive, and Prototype experts as the feature space evolves. |
| `Drift detector / uncertainty signals` | Provide evidence that the current region may no longer match the old feature distribution. |
| `Prototype quality score` | Prevents prototype memory from becoming stale by accounting for representativeness, drift relevance, minority importance, uncertainty, obsolescence, and freshness. |

## Current Architecture

In the current implementation, the model has:

- `autoencoder_1` for S1 representation learning
- `classifier_1` trained in S1 using Hedge Backpropagation
- `autoencoder_2` for S2 representation learning
- `transfer_mapper` for mapping S2 latent representations toward the old latent space
- `classifier_2`, copied from `classifier_1`, then trained online in S2
- prototype memory, weighted by the normalized prototype-quality score

At the S1 to S2 boundary, `classifier_1` is copied into `classifier_2`, then `classifier_1` is frozen.

## Proposed MoE Extension

The MoE layer treats the existing prediction sources as role-specific experts:

```text
E1 = Historical Expert
E2 = Adaptive Expert
E3 = Prototype Expert
```

The router has one focused job:

```text
Decide how much to rely on each expert as the feature space evolves.
```

The proposed prediction is:

```text
logits = alpha_hist * h_hist + alpha_adapt * h_adapt + alpha_proto * h_proto
```

where:

```text
h_hist  = Historical Expert output
h_adapt = Adaptive Expert output
h_proto = Prototype Expert output
```

The router should stay simple and interpretable. It can use a compact feature-drift state, such as:

- phase indicator: S1 or S2
- latent representation `z_t`
- drift signal
- prototype uncertainty / local prototype disagreement

## Prototype Memory Detail

The prototype detail page shows the prototype mechanism as its own subsystem.

Each incoming sample is encoded into a latent vector:

```text
z_t = AE(x_t)
```

The system predicts first, then uses the true label to update memory. A prototype stores:

```text
{vec, label, space, last_step, rep, drift, minority}
```

The quality score is:

```text
Q(p) = A(p) * O(p)^w_o * F(p)^w_f
```

where `A(p)` is the normalized weighted average of representativeness, drift relevance, minority importance, and uncertainty.

Prototype memory interacts with the rest of the architecture in two ways:

1. Directly: it produces prototype expert logits:

```text
score_class += Q(p) / (distance + epsilon)
```

2. Indirectly: it provides router features:

```text
uncertainty, nearest-prototype distances, local drift support
```

This is why prototypes are shown as both an expert and a feature-drift signal source.

## MoE Detail

The MoE detail page uses the expert names suggested by the professor:

```text
E1 = Historical Expert
E2 = Adaptive Expert
E3 = Prototype Expert
```

The router outputs soft expert weights:

```text
alpha_hist, alpha_adapt, alpha_proto
```

The final prediction is:

```text
logits = alpha_hist * h_hist + alpha_adapt * h_adapt + alpha_proto * h_proto
```

Hedge Backpropagation is shown separately because it is already used inside `classifier_1` and `classifier_2` as the online multi-head training method.

## Paper Ideas Included

`Adaptive Mixtures of Local Experts`

Supports the idea that different experts can specialize in different regions or regimes of the input space.

`Mixture-of-Experts routing papers`

Support the idea of a router/gating mechanism that learns sample-specific expert weights.

`Online Deep Learning / Hedge Backpropagation`

Supports treating different MLP heads as weighted online predictors during classifier training.

`DriftMoE-style stream routing`

Supports making the router aware of nonstationarity and drift in stream settings.

## Thesis-Friendly Explanation

The current model already contains several expert-like components: historical knowledge from S1, adaptive learning in S2, prototype memory, and Hedge Backpropagation heads. The proposed MoE extension formalizes this structure by adding a focused router whose role is to decide how much to rely on the Historical, Adaptive, and Prototype experts as feature drift occurs. This makes the architecture a response to feature drift rather than a collection of unrelated techniques.
