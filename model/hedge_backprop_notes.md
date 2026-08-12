# Hedge Backpropagation Notes

## What Alpha Means

`alpha` is the weight assigned to each MLP head.

The MLP has 5 heads, so the model keeps 5 head weights:

```text
alpha = [alpha_1, alpha_2, alpha_3, alpha_4, alpha_5]
```

At the beginning, all heads are trusted equally:

```python
self.alpha = Parameter(torch.Tensor(5).fill_(1 / 5), requires_grad=False).to(self.device)
```

So initially:

```text
alpha = [0.2, 0.2, 0.2, 0.2, 0.2]
```

## What Each Head Returns

Each MLP head returns one logit vector:

```text
[batch_size, num_classes]
```

For one sample and 6 classes, one head might return:

```text
[[0.2, -1.0, 2.3, 0.5, -0.4, 1.1]]
```

These are class scores, not probabilities. The predicted class is the index with the largest score.

## How Cross Entropy Works

For each head, the loss is:

```python
self.CELoss(out, y_idx)
```

where:

```text
out   = logits for all classes
y_idx = correct class index
```

PyTorch applies softmax internally, selects the probability of the true class, and computes:

```text
loss = -log(probability of the true class)
```

If the correct class has high probability, the loss is small. If the correct class has low probability, the loss is large.

## How The Head Losses Are Combined

Each head gets its own cross-entropy loss:

```text
losses = [loss_1, loss_2, loss_3, loss_4, loss_5]
```

Then the code combines them using `alpha`:

```python
loss_sum += (self.alpha[i] / alpha_sum) * loss
```

So the final loss is:

```text
loss_sum =
alpha_1 * loss_1
+ alpha_2 * loss_2
+ alpha_3 * loss_3
+ alpha_4 * loss_4
+ alpha_5 * loss_5
```

Heads with higher `alpha` contribute more to training.

## How The Prediction Is Combined

The weighted prediction is:

```python
out_ens += self.alpha[i] * out
```

So:

```text
ensemble_logits =
alpha_1 * head_1_logits
+ alpha_2 * head_2_logits
+ alpha_3 * head_3_logits
+ alpha_4 * head_4_logits
+ alpha_5 * head_5_logits
```

This makes prediction consistent with Hedge Backpropagation because all heads are used, not only the last head.

## How Alpha Is Updated

After computing the loss of each head, alpha is updated using:

```python
self.alpha[i] *= torch.pow(self.b, losses[i])
```

Mathematically:

```text
alpha_i <- alpha_i * b^(loss_i)
```

This is a Hedge-style multiplicative weight update.

Because:

```text
0 < b < 1
```

a larger loss makes the multiplier smaller.

Example with `b = 0.9`:

```text
loss = 0.1  -> 0.9^0.1 ~= 0.989
loss = 1.0  -> 0.9^1.0 = 0.900
loss = 3.0  -> 0.9^3.0 = 0.729
```

So high-loss heads lose influence faster.

## Why This Is A Hedge Equation

The common Hedge update is:

```text
w_i,t+1 = w_i,t * exp(-eta * loss_i,t)
```

Your code uses:

```text
alpha_i <- alpha_i * b^(loss_i)
```

These are equivalent because:

```text
b^loss = exp(log(b) * loss)
```

Since `b < 1`, `log(b)` is negative. Therefore:

```text
b^loss = exp(-eta * loss)
```

where:

```text
eta = -log(b)
```

For `b = 0.9`:

```text
eta = -log(0.9) ~= 0.105
```

So `b` controls how aggressively bad heads are penalized.

## Why Use b = 0.9

`b = 0.9` is a conservative choice because it is close to 1.

That means head weights adapt gradually instead of changing too aggressively.

If `b` were smaller, for example `0.5`, bad heads would lose influence much faster.

## Thesis Explanation

Hedge Backpropagation keeps a separate prediction head at multiple depths of the MLP. Each head produces class logits and receives its own cross-entropy loss. The final loss and prediction are weighted combinations of all heads using adaptive weights `alpha`. These weights are updated with a multiplicative Hedge rule:

```text
alpha_i <- alpha_i * b^(loss_i)
```

where `b` is in `(0, 1)`. Heads with larger losses are downweighted, while better-performing heads retain more influence. In this implementation, `b = 0.9` gives gradual adaptation across the heads.
