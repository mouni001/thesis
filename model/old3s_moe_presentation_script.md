# OLD3S + MoE Architecture Presentation Script

Use this while presenting `old3s_moe_architecture.drawio`. The file has three pages, so the script is split into three matching parts.

## Page 1: OLD3S MoE Layer Flow Chart

### Opening

Start by saying:

> This diagram shows the full proposed architecture. The main problem I am addressing is feature drift, where the feature space changes between the first phase and the second phase. The goal is not only to classify the stream, but to decide which source of knowledge should be trusted as the feature space evolves.

### Input and Encoding

Point to `Online Stream Sample`.

Say:

> The data arrives one sample at a time as an online stream. Each sample has features `x_t` and a label `y_t`. In phase 1, the sample comes from the old feature space, which contains old and shared features. In phase 2, the sample comes from the evolved feature space, which contains shared and new features.

Point to `Phase Encoder`.

Say:

> The sample is passed through a phase-specific autoencoder. In S1, I use `AE1`, and in S2, I use `AE2`. The purpose of the autoencoder is to produce a latent representation `z_t`. This helps because the raw feature dimensions can change, but the model still needs a representation where learning and comparison are possible.

### Feature-Evolution Phases

Point to `Phase 1: S1 Training`.

Say:

> During phase 1, the system trains `classifier_1` on the old feature space. This classifier is trained using Hedge Backpropagation, which means the classifier has multiple heads and the training process learns how much to trust each head.

Point to `Boundary: B = T1 - t`.

Say:

> At the boundary between S1 and S2, `classifier_1` is copied into `classifier_2`. Then `classifier_1` is frozen. This is important because `classifier_1` becomes the historical knowledge source: it preserves what was learned before feature drift.

Point to `Phase 2: S2 Adaptation`.

Say:

> During phase 2, `classifier_2` adapts online to the new feature space. The transfer mapper also maps the S2 latent representation toward the old latent space so the frozen historical classifier can still be used when old knowledge remains useful.

### Routing and Experts

Point to `Gating / Router Network`.

Say:

> The router has one focused job: it decides how much to rely on each expert as the feature space evolves. It is not meant to solve everything. It simply produces soft weights for the Historical Expert, Adaptive Expert, and Prototype Expert.

Point to `Historical Expert`.

Say:

> The Historical Expert is the frozen old classifier. It addresses the part of feature drift where some old or shared features may still be useful. Instead of discarding old knowledge, this expert preserves it.

Point to `Adaptive Expert`.

Say:

> The Adaptive Expert is the new classifier trained in phase 2. It addresses the new part of the feature space. When new features become important, this expert should receive more weight.

Point to `Prototype Expert`.

Say:

> The Prototype Expert uses memory of representative latent samples. It is especially useful for local regions affected by drift, minority classes, and uncertain areas where the neural classifiers may be unstable.

Point to `Hedge Backpropagation`.

Say:

> Hedge Backpropagation is already part of the implementation. It trains the classifier experts using multiple MLP heads. Each head receives a loss, and the model adjusts head weights based on performance. So Hedge Backpropagation works inside the classifier experts, while the MoE router works above the experts.

### Auxiliary Mechanisms

Point to `Prototype Quality`.

Say:

> Prototype quality decides which prototypes should have more influence. The formula combines a normalized usefulness score with obsolescence and freshness. This is directly related to feature drift, because old prototypes may become less reliable after the feature space changes.

Point to `Drift Signal`.

Say:

> The drift detector provides evidence that the stream behavior is changing. This can help the router and prototype memory know when the current region may not match the previous distribution.

Point to `Uncertainty`.

Say:

> Uncertainty is estimated from local prototype disagreement. If nearby prototypes disagree, that suggests the region is unstable or difficult, and the prototype expert may become more important.

### Aggregation and Output

Point to `Weighted Expert Aggregation`.

Say:

> The final prediction is a weighted combination of expert logits. The router produces weights, and the architecture combines the Historical, Adaptive, and Prototype experts.

Use the formula:

```text
logits = alpha_hist * h_hist + alpha_adapt * h_adapt + alpha_proto * h_proto
```

Point to `Softmax Prediction`.

Say:

> The combined logits are converted into class probabilities. Since this is an online stream, the model predicts first.

Point to `Online Update`.

Say:

> After the true label is available, the model updates the trainable parts: the adaptive classifier, the prototype memory, metrics, and the drift detector.

### Page 1 Closing

Say:

> The main story of this page is that feature drift creates uncertainty about which knowledge source to trust. The MoE layer makes that decision explicit by combining historical knowledge, adaptive learning, and prototype memory.

## Page 2: Prototype Memory Detail

### Opening

Say:

> This page zooms into the prototype memory. The reason for separating it is that prototypes are not just extra predictions. They also provide local evidence about drift, uncertainty, and minority regions.

### Stream and Latent Space

Point to `Incoming sample`.

Say:

> A sample arrives from the stream. The model follows a prequential setup, meaning it predicts first and updates after receiving the true label.

Point to `Autoencoder latent vector`.

Say:

> The autoencoder maps the input into a latent vector `z_t`. This latent vector is what the prototype memory stores and compares.

Point to `Prediction result`.

Say:

> After prediction, the system compares the predicted label with the true label. Whether the model was correct or wrong affects how prototype memory is updated.

### Prototype Memory Bank

Point to `Prototype Bank Entry`.

Say:

> Each prototype stores a latent vector, a class label, which space it came from, the last update step, and quality-related values. These are representativeness, drift relevance, and minority importance.

Read the prototype structure:

```text
p = {vec, label, space, last_step, rep, drift, minority}
```

Point to `Prototype Update Rules`.

Say:

> If the model is correct and the sample is close to an existing same-class prototype, the prototype is refreshed. If the model makes a mistake, or if there is local drift support, a new prototype can be added. This helps the memory adapt to local changes in the feature space.
How many???? and why??

Point to `Quality Score`.

Say:

> The quality score decides how influential each prototype should be. It is not just based on age or distance. It combines representativeness, drift relevance, minority importance, uncertainty, obsolescence, and freshness.

Use the formula:

```text
Q(p) = A(p) * O(p)^w_o * F(p)^w_f
```

Then say:

> `A(p)` is the normalized weighted average of the main usefulness terms. `O(p)` handles obsolescence, especially for old S1 prototypes in S2. `F(p)` handles freshness, so stale prototypes become less dominant.

### Interaction With Experts and MoE

Point to `Prototype Expert`.

Say:

> The Prototype Expert turns memory into logits. Nearby prototypes vote for their class, but the vote is weighted by prototype quality.

Use the formula:

```text
score_class += Q(p) / (distance + epsilon)
```

Point to `Router Inputs from Prototypes`.

Say:

> Prototype memory also provides signals to the router. For example, if nearby prototypes disagree, uncertainty is high. If potential prototypes accumulate in a region, this may indicate local drift. So prototypes help both directly through prediction and indirectly through routing.

Point to `MoE Router`.

Say:

> The router uses these signals to decide whether to rely more on the Historical Expert, Adaptive Expert, or Prototype Expert.

Point to `Final Prediction`.

Say:

> The prototype expert contributes to the final prediction through its own weight, `alpha_proto`. This is important because in drift-sensitive regions, local memory may be more reliable than either global classifier.

### Page 2 Closing

Say:

> The key idea is that prototype memory addresses local feature drift. It remembers useful local regions, downweights stale prototypes, supports minority classes, and provides uncertainty information to the router.

## Page 3: MoE Experts and Router Detail

### Opening

Say:

> This page explains the MoE layer more directly. The goal is to make the existing components of the model act as role-specific experts.

### Input to Router

Point to `Encoded sample`.

Say:

> The router receives the encoded sample and a compact feature-drift state. This can include the latent representation, the current phase, drift signal, and prototype uncertainty.

Point to `Feature-Drift Router`.

Say:

> The router outputs soft weights. These weights represent how much the model should trust each expert for the current sample.

Use:

```text
alpha = softmax(g(feature-drift state))
```

Then say:

> The output is `alpha_hist`, `alpha_adapt`, and `alpha_proto`.

### Experts

Point to `E1 Historical Expert`.

Say:

> The Historical Expert is based on the frozen S1 classifier. It is useful when old or shared features still contain predictive information after feature drift.

Point to `E2 Adaptive Expert`.

Say:

> The Adaptive Expert is the S2 classifier. It learns the new feature space online, so it is useful when new features become more predictive than historical knowledge.

Point to `E3 Prototype Expert`.

Say:

> The Prototype Expert uses local memory. It is useful when the stream enters drift-sensitive, uncertain, or minority-class regions.

### Aggregation

Point to `MoE Aggregation`.

Say:

> The three expert outputs are combined by the router weights. This makes the prediction sample-specific instead of using one fixed global combination.

Use:

```text
logits = alpha_hist * h_hist + alpha_adapt * h_adapt + alpha_proto * h_proto
```

### Training and Feedback

Point to `Hedge Backpropagation`.

Say:

> Hedge Backpropagation is the existing online training method inside the classifier experts. It trains the multi-head MLP by weighting the heads according to their losses. This is separate from the MoE router: Hedge works within the classifier, while MoE works between experts.

Point to `Online Feedback`.

Say:

> After the prediction and true label, the system updates the adaptive classifier, prototype memory, drift detector, and metrics. This feedback loop is what allows the system to keep adapting over the stream.

### Page 3 Closing

Say:

> The MoE layer is not added just because it is a modern technique. It is added because feature drift creates multiple possible sources of useful knowledge. The Historical Expert handles old knowledge, the Adaptive Expert handles new features, and the Prototype Expert handles local uncertain regions. The router decides how much to rely on each one.

## Final Summary to Say

End with:

> Overall, the architecture tells a feature-drift story. First, the model learns from the old feature space. Then, when the feature space evolves, it keeps historical knowledge, adapts to new features, and uses prototype memory for local drift-sensitive regions. The MoE router connects these parts by deciding which expert should be trusted for each sample.
