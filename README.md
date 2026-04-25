# coresets

A small library for building coresets: compact, weighted approximations of
large datasets -- by minimizing Maximum Mean Discrepancy (MMD) via gradient
descent.

---

## What is a coreset?

A coreset is a weighted point set $C = \{(x_i, w_i)\}$ of size $n$ that
approximates a much larger dataset or data stream. The goal is that any
computation run on $C$ (with weights applied) produces nearly the same result
as running it on the full data. This library finds coresets by minimizing MMD,
a kernel-based distributional distance, between the coreset and the target data.

---

## Files

| File | Role |
|------|------|
| `kernels.py` | Kernel protocol and three built-in kernel implementations |
| `mmd.py` | Stochastic MMD objective used during optimization |
| `coreset.py` | `GDBuilder`: the gradient-descent coreset construction algorithm |

---

## Quick start

```python
import torch
from kernels import RBFKernel
from coreset import GDBuilder

# Wrap your data as an iterator of mini-batches (b, d)
def make_stream(data, batch_size=256):
    n = len(data)
    while True:
        idx = torch.randint(0, n, (batch_size,))
        yield data[idx]

data = torch.randn(10000, 2)
stream = make_stream(data)

kernel = RBFKernel(sigma=0.5)
builder = GDBuilder(kernel, steps=2000, learning_rate=0.05)

coreset, mmd_history = builder.build(stream, n=50, dim=2, verbose=True)

print(coreset.points_nd.shape)   # (50, 2)
print(coreset.weights_n.sum())   # ~1.0
```

---

## Design choices

### 1. MMD as the optimization objective

MMD measures how far a weighted point set is from a target distribution in a
reproducing kernel Hilbert space (RKHS). The squared MMD between a coreset
$C = \{(x_i, w_i)\}$ and a distribution $P$ is:

$$
MMD^2(C, P) = \sum_{i,j} w_i \cdot w_j \cdot k(x_i, x_j)
              - 2 \cdot \sum_i w_i \cdot \mathbb{E}_{z\sim P}[k(x_i, z)]
              + \mathbb{E}_{z,z' \sim P}[k(z, z')]
$$

MMD was chosen because it is:
- **Differentiable** with respect to both the coreset point locations and
  their weights, enabling gradient-based optimization.
- **Kernel-parameterized**, so the notion of "similarity" is modular and
  swappable without changing the optimization loop.
- **Well-studied** for coreset and distribution compression tasks.

### 2. Dropping the constant term

The third term, $\mathbb{E}_{z,z' \sim P}[k(z, z')]$, depends only on the target distribution `P`
and is constant with respect to the coreset parameters. It is omitted from
`stochastic_mmd` because it contributes no gradient. This keeps the
objective cheap to evaluate without changing which coreset is optimal.

### 3. Stochastic estimation with mini-batches

Rather than accumulating the full dataset upfront to evaluate the exact cross
term $\mathbb{E}_{z \sim P}[k(x_i, z)]$, the library uses a streaming interface. Callers
pass an iterator that yields fresh mini-batches each step. The cross term is
estimated by averaging over the batch.

This design choice has two benefits:
- It works on datasets too large to fit in memory.
- It introduces noise that acts as implicit regularization, similar to
  stochastic gradient descent vs. full-batch gradient descent.

### 4. Joint optimization of positions and weights

Most classical coreset methods fix candidate point locations and only solve for
weights. `GDBuilder` instead **jointly optimizes both**:

- `x_nd` -- the coreset point locations (initialized uniformly at random in
  `[0, 1)^d`)
- `w_logits_n` -- raw weight logits (initialized to zero, softmaxed at the end)

Freeing the point locations allows the coreset to place points wherever the
distribution actually has mass, rather than being constrained to the original
data points. This typically produces better approximations for the same `n`.

### 5. Softmax weight parameterization

Weights must satisfy `w_i >= 0` and `sum(w_i) = 1`. Rather than enforcing
these constraints with projections or penalty terms, the optimizer works in an
unconstrained space of raw logits and applies **softmax** to convert them to
valid probabilities.

This is numerically stable (no projection steps), keeps the optimization
landscape smooth, and guarantees that constraints are satisfied exactly at
every step -- not just approximately at convergence.

### 6. Adam optimizer

Both `x_nd` and `w_logits_n` are optimized jointly with a single **Adam**
optimizer. Adam was chosen over plain SGD because:
- Adaptive per-parameter learning rates handle the different scales and
  curvatures of the position and weight parameters naturally.
- It converges reliably on the non-convex MMD landscape without careful
  learning rate tuning.

### 7. EMA-based early stopping

Raw per-step MMD values are noisy due to mini-batch variance. Stopping on the
raw value would trigger too early or require a very small tolerance. Instead,
`GDBuilder` tracks an **exponential moving average (EMA)** of the loss with a
smoothing factor of 0.05 (heavily weighted toward recent history). Training
halts when the EMA has not improved by more than `tol` for `patience`
consecutive steps.

Setting `patience=0` disables early stopping entirely and runs for the full
`steps` budget.

### 8. Kernel as a Protocol (structural typing)

The `Kernel` type in `kernels.py` is a `typing.Protocol`, not an abstract base
class. Any object that provides a `gram(X_nd, Y_md) -> Tensor` method
automatically satisfies the protocol -- no inheritance required. This makes it
trivial to plug in a custom kernel without modifying library code.

---

## Kernel reference

All kernels accept two tensors of shapes `(n, d)` and `(m, d)` and return a
Gram matrix of shape `(n, m)`.

### RBFKernel (Gaussian)

$$
k(x, y) = \exp\left(\frac{-||x - y||^2} {2 \sigma^2}\right)
$$

Infinitely differentiable (smooth). Good default choice. `sigma` controls the
length-scale: smaller values make the kernel more local.

### Matern32Kernel

$$
k(x, y) = \left(1 + \frac{\sqrt{3} \cdot ||x - y||}{\sigma}\right) \cdot \exp\left(\frac{-\sqrt{3} \cdot ||x - y||}{ \sigma}\right)
$$

Once-differentiable. A better fit than RBF when the target distribution is not
infinitely smooth (e.g., has sharp features or discontinuities). `sigma`
controls the length-scale.

### IMQKernel (Inverse Multi-Quadric)

$$
k(x, y) = \frac{c}{\sqrt{c^2 + ||x - y||^2}}
$$

Heavier tails than RBF. More robust to outliers and suitable when the
distribution has wide support. `c` controls the width of the flat region near
the origin.

---

## Dependencies

- Python 3.8+
- PyTorch
- tqdm
