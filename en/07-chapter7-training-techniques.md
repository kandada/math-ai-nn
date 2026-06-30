# Chapter 7: Training Techniques — Optimizers, Regularization & Loss Functions

> **Goal**: Master the advanced techniques that make neural network training stable, fast, and generalizable.

> © xiefujin · Contact: 490021684@qq.com · Licensed under CC BY-NC-SA 4.0
>
> **Code**: `../code/ch07/` (7 files)

---

## 📋 Chapter Learning Objectives

- [ ] Understand optimizer variants (SGD → Adam → AdamW)
- [ ] Master learning rate scheduling
- [ ] Understand regularization methods (L1/L2, dropout, early stopping)
- [ ] Understand normalization (BatchNorm, LayerNorm)
- [ ] Master weight initialization strategies
- [ ] Understand loss functions for different tasks
- [ ] Deep understanding of Softmax

---

## 7-1 Optimizer Landscape: From SGD to AdamW

### Optimizer Evolution

#### Stochastic Gradient Descent (SGD)

$$
v_{t+1} = \beta v_t + \nabla L(\theta_t)
$$
$$
\theta_{t+1} = \theta_t - \eta v_{t+1}
$$

### Which Optimizer to Choose?

| Scenario | Recommended Optimizer |
|:---------|:---------------------|
| Quick prototyping | Adam |
| CV (ImageNet-scale) | SGD + Momentum |
| NLP / Transformers | AdamW |
| Sparse data | AdaGrad |
| RL / non-stationary | RMSprop |

![Figure 7-1: Loss landscape comparison of three optimizers. Darker color = lower loss; path lines show each optimizer's trajectory to the minimum.](../images/ch07/NN07_optimizer_comparison.png)
*Figure 7-1: Optimizer comparison on the loss landscape.*

![Figure 7-2: Loss landscape visualization — the effect of different initialization points on convergence paths.](../images/ch07/NN07_loss_landscape.png)
*Figure 7-2: Loss landscape with different initialization points.*

---

## 7-2 Learning Rate Scheduling

### Common Schedulers

```python
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, CosineAnnealingLR,
    ReduceLROnPlateau
)

# Step decay: reduce by 0.1 every 30 epochs
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Cosine annealing: smooth decay
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Adaptive: reduce when loss plateaus
scheduler = ReduceLROnPlateau(optimizer, mode='min',
                              factor=0.5, patience=5)
```

### Learning Rate Best Practices

| Strategy | Use Case |
|:---------|:---------|
| Constant | Simple problems, short training |
| Step decay | Standard CV training |
| Cosine | Modern deep learning |
| One-cycle | Fast convergence |
| Warmup + decay | Transformers |

---

## 7-3 Regularization and Generalization

#### L1 vs L2 Regularization

**L2 Regularization (Weight Decay)** adds a penalty proportional to the squared magnitude of weights:

$$L_{\text{new}} = L_{\text{original}} + \frac{\lambda}{2} \sum_{w} w^2$$

The gradient update becomes:

$$w \leftarrow w - \eta \nabla L - \eta \lambda w = w(1 - \eta\lambda) - \eta \nabla L$$

The $w(1 - \eta\lambda)$ term causes **weight decay** — weights are slightly shrunk at each update. This prevents any single weight from becoming too large, encouraging the model to use all weights moderately.

**L1 Regularization** (Lasso) uses the absolute value:

$$L_{\text{new}} = L_{\text{original}} + \lambda \sum_{w} |w|$$

L1 regularization encourages **sparsity** — many weights become exactly zero. This is useful for feature selection.

| Property | L2 (Ridge) | L1 (Lasso) |
|:---------|:----------|:-----------|
| Penalty | $\frac{\lambda}{2}w^2$ | $\lambda|w|$ |
| Effect | Shrinks weights proportionally | Sets some weights to zero |
| Gradient | $\lambda w$ | $\lambda \cdot \text{sign}(w)$ |
| Best for | Preventing overfitting | Feature selection |

![Figure 7-3: L2 regularization effect — the larger the regularization coefficient, the more weights approach zero, reducing model complexity.](../images/ch07/NN07_l2_regularization.png)
*Figure 7-3: L2 regularization comparison.*

> **Little Genius says**: L2 regularization is like telling weights "don't be too confident!" L1 is like "if you're not useful, just become zero!" They're both ways to keep the model humble and prevent it from memorizing noise.

#### Dropout

Dropout randomly **drops** (sets to zero) a fraction of neurons during training:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),  # 50% of neurons randomly dropped
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(128, 10)
)
```

**Intuition**: Dropout forces the network to learn **redundant representations** — no single neuron can be essential because it might be dropped at any time. This is like having multiple experts who each learn independently, then averaging their opinions.

![Figure 7-4: Overfitting demonstration — high-degree polynomial fitting noisy data. Training error is very low but generalization is poor.](../images/ch07/NN07_overfitting_demo.png)
*Figure 7-4: Overfitting — polynomial fitting comparison.*

#### Early Stopping

Monitor validation loss and stop training when it stops improving:

```python
best_val_loss = float('inf')
patience = 5
no_improve = 0

for epoch in range(100):
    train_loss = train_one_epoch()
    val_loss = evaluate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

#### Data Augmentation

Create more training data through transformations:

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomRotation(10),       # Slight rotation
    transforms.RandomHorizontalFlip(),    # Horizontal flip
    transforms.RandomAffine(0, shear=10), # Shear
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

## 7-4 Normalization Methods

#### Batch Normalization

Normalizes each batch to have mean 0, variance 1:

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
```

### Comparison

| Method | Normalizes Across | Best For |
|:-------|:------------------|:---------|
| BatchNorm | Batch dimension | CV (fixed-size batches) |
| LayerNorm | Feature dimension | NLP / Transformers |
| InstanceNorm | Single sample | Style transfer |
| GroupNorm | Groups of channels | Small batch sizes |

---

## 7-5 Weight Initialization

#### Why Initialization Matters

Poor initialization can cause:
- **Gradient vanishing**: If weights are too small, activations shrink
- **Gradient explosion**: If weights are too large, activations explode
- **Symmetry**: All neurons learn the same features

#### Xavier (Glorot) Initialization

For **Sigmoid/Tanh** activations, Xavier initialization maintains variance across layers:

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$

```python
def xavier_init(n_in, n_out):
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))
```

#### He (Kaiming) Initialization

For **ReLU** activations, He initialization accounts for the fact that ReLU zeros out half the neurons:

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

```python
def he_init(n_in, n_out):
    std = np.sqrt(2 / n_in)
    return np.random.randn(n_in, n_out) * std
```

| Activation | Recommended Initialization | Distribution |
|:-----------|:-------------------------|:-------------|
| Sigmoid / Tanh | Xavier (Glorot) | Uniform |
| ReLU / Leaky ReLU | He (Kaiming) | Normal |
| No activation | LeCun | Normal |

> **Core insight**: The key principle is **variance preservation** — the variance of activations should remain roughly constant across layers. Too much variance growth → explosion. Too much variance shrinkage → vanishing.

Bad initialization → vanishing/exploding gradients. Good initialization → fast convergence.

#### Common Initialization Strategies in Practice

```python
# Default PyTorch (Kaiming Uniform for ReLU)
layer = nn.Linear(784, 256)  # automatically initialized

# Manual initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # for tanh/sigmoid
        # nn.init.kaiming_uniform_(m.weight)  # for ReLU
        nn.init.zeros_(m.bias)

model.apply(init_weights)
```

![Figure 7-5: Gradient flow comparison for different weight initialization methods. Good initialization keeps gradient magnitudes stable across layers; poor initialization causes vanishing or exploding gradients.](../images/ch07/NN07_weight_init_compare.png)
*Figure 7-5: Weight initialization gradient flow comparison.*

---

## 7-6 Loss Functions Overview

### Classification

| Loss | Formula | Use Case |
|:-----|:--------|:---------|
| Cross-Entropy | $-\sum t_k \log p_k$ | Multi-class |
| Binary CE | $-(t\log p + (1-t)\log(1-p))$ | Binary |
| Focal Loss | Weighted CE | Imbalanced classes |

### Regression

| Loss | Formula | Robustness |
|:-----|:--------|:-----------|
| MSE | $\frac{1}{2}(y-t)^2$ | Sensitive to outliers |
| MAE | $\|y-t\|$ | Robust to outliers |
| Huber | MSE for small error, MAE for large | Best of both |

![Figure 7-6: Different loss function surfaces — MSE (smooth convex), MAE (linear cone-shaped), Huber (smooth transition).](../images/ch07/NN07_loss_surfaces.png)
*Figure 7-6: Different loss function surfaces.*

---

## 7-7 Deep Understanding of Softmax ⭐

### Beyond the Basic Definition

Softmax converts logits to probabilities, but its behavior depends on **temperature**:

$$
p_k = \frac{e^{z_k / T}}{\sum_{j} e^{z_j / T}}
$$

| Temperature | Effect |
|:------------|:-------|
| $T \to 0$ | Becomes argmax (hard assignment) |
| $T = 1$ | Standard Softmax |
| $T \to \infty$ | Becomes uniform distribution |

### Temperature in Practice

```python
def softmax_with_temperature(logits, temperature=1.0):
    """Softmax with temperature scaling"""
    scaled_logits = logits / temperature
    exp_logits = torch.exp(scaled_logits - torch.max(scaled_logits))
    return exp_logits / exp_logits.sum(dim=-1, keepdim=True)

logits = torch.tensor([2.0, 1.0, 0.1])
for T in [0.5, 1.0, 2.0, 5.0]:
    probs = softmax_with_temperature(logits, T)
    print(f"T={T:.1f}: {probs.numpy().round(3)}")
```

---

## 7-8 Regularization Deep Dive

### Which Regularization When?

| Method | Effect | When to Add |
|:-------|:-------|:------------|
| L2 | Small weights | Always (mild) |
| Dropout | Prevents co-adaptation | Large networks, overfitting |
| Early stopping | Limits training time | Always |
| Data augmentation | More data implicitly | CV tasks |
| Label smoothing | Softens targets | Classification |

![Figure 7-7: L1 vs L2 regularization effect comparison. L1 produces sparse weights (some become zero), L2 causes all weights to decay uniformly.](../images/ch07/NN07_regularization_effect.png)
*Figure 7-7: L1 vs L2 regularization comparison.*

---

## 7-9 Hyperparameter Tuning

### Key Hyperparameters

### Tuning Strategies

1. **Grid search**: Systematic but expensive
2. **Random search**: Better for high-dimensional spaces
3. **Bayesian optimization**: Efficient but complex
4. **Learning rate finder**: Find optimal lr quickly

---

## 7-10 Gradient Clipping

Prevents exploding gradients by scaling down large gradients:

```python
# Before stepping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

## 📝 Supplementary Details

These subsections provide deeper mathematical background for the topics above.

#### Momentum

Momentum accelerates SGD by adding a velocity term:

$$v_{t+1} = \beta v_t + \eta \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

Where $\beta \approx 0.9$. This helps escape local minima and dampens oscillations.

#### NAG (Nesterov Accelerated Gradient)

NAG looks ahead: compute gradient at the approximate next position:

$$v_{t+1} = \beta v_t + \eta \nabla L(\theta_t - \beta v_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

This "look-ahead" correction often leads to faster convergence than standard Momentum.

#### AdaGrad: Adaptive Learning Rate

AdaGrad adapts per-parameter learning rates based on historical gradients:

$$\theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,ii} + \epsilon}} \cdot \nabla L(\theta_{t,i})$$

Where $G$ accumulates squared gradients. Good for sparse features, but LR monotonically decreases.

#### RMSprop: Fixing AdaGrad's LR Decay

RMSprop uses a moving average of squared gradients instead of a cumulative sum:

$$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta) g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t$$

This prevents the learning rate from vanishing.

#### Adam (Adaptive Moment Estimation) ⭐

Adam combines Momentum and RMSprop:
- **Momentum**: first moment (mean) of gradients
- **RMSprop**: second moment (uncentered variance) of gradients
- **Bias correction**: compensates for initialization at zero

Adam is the default optimizer for most deep learning tasks.

#### Why Learning Rate Scheduling?

As training progresses, the optimal learning rate decreases. Early training benefits from larger steps to descend quickly, while later stages need smaller steps to fine-tune around the minimum.

#### L1/L2 Regularization

**L2 (Weight Decay)**: $$L_{\text{reg}} = L + \frac{\lambda}{2} \sum \theta^2$$

**L1 (Lasso)**: $$L_{\text{reg}} = L + \lambda \sum |\theta|$$

L2 shrinks weights proportionally to their size; L1 drives small weights to zero (feature selection).

#### Data Augmentation

Artificially expand training data with transformations. Common techniques: rotation, flipping, cropping, color jitter, Cutout. This reduces overfitting and improves generalization.

#### Regression Tasks

For regression, **Mean Squared Error (MSE)** is the most common loss:

$$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Also used: MAE ($L_1$ loss), Huber loss (robust to outliers).

#### Classification Tasks

For classification, **Cross-Entropy Loss** is standard:

$$L = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})$$

Where $y_{i,k}$ is the one-hot label and $\hat{y}_{i,k}$ is the predicted probability for class $k$.

#### Why Softmax?

Softmax converts raw logits to probabilities that sum to 1:

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

This enables probabilistic interpretation of the model's output.

#### Mathematical Derivation

Softmax has a convenient derivative: $\frac{\partial \hat{y}_i}{\partial z_j} = \hat{y}_i(\delta_{ij} - \hat{y}_j)$.

Combined with cross-entropy, the gradient simplifies to $\hat{y}_i - y_i$ — a beautiful cancellation!

#### Derivative of Softmax

The Jacobian of softmax is:

$$\frac{\partial \hat{y}_i}{\partial z_j} = \begin{cases} \hat{y}_i (1 - \hat{y}_i) & i = j \\ -\hat{y}_i \hat{y}_j & i \neq j \end{cases}$$

#### Softmax + Cross-Entropy: A Beautiful Combination

The gradient of cross-entropy loss with softmax output simplifies elegantly:

$$\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$$

This is the same as the gradient for MSE + linear output in regression!

#### L1 vs L2 Regularization

| Aspect | L2 (Weight Decay) | L1 (Lasso) |
|:-------|:------------------|:-----------|
| Effect | Shrinks weights | Sparse weights |
| Derivative | $2\lambda\theta$ | $\lambda \cdot \text{sign}(\theta)$ |
| Solution | Non-sparse | Sparse (feature selection) |
| Stability | More stable | Can be unstable |

#### Elastic Net: L1 + L2 Combination

$$L_{\text{reg}} = L + \lambda_1 \sum |\theta| + \lambda_2 \sum \theta^2$$

Combines the benefits of both L1 (sparsity) and L2 (stability).

#### Regularization Strength Selection

Use **cross-validation** to select the regularization strength $\lambda$:
1. Try $\lambda$ values on a log scale ($10^{-4}$ to $10^0$)
2. Train with each $\lambda$ on training folds
3. Select $\lambda$ with best validation performance

#### Key Hyperparameters Overview

| Hyperparameter | Typical Range | Effect |
|:--------------|:--------------|:-------|
| Learning rate | $10^{-5}$ to $10^{-1}$ | Most important |
| Batch size | 16-512 | Memory vs stability |
| Hidden units | 64-4096 | Model capacity |
| Dropout rate | 0.1-0.5 | Regularization |

#### Grid Search vs Random Search

**Grid Search** evaluates all combinations — exponential in #parameters.
**Random Search** samples randomly — more efficient for high dimensions (Bergstra & Bengio, 2012).

Modern approach: **Bayesian Optimization** (e.g., Optuna, Hyperopt).

#### LR Range Test

Find the optimal learning rate by:
1. Start with a very small LR
2. Increase exponentially per mini-batch
3. Plot loss vs LR
4. Pick LR where loss decreases most steeply

```python
# lr_finder pattern
for lr in np.logspace(-5, -1, 100):
    optimizer.param_groups[0]['lr'] = lr
    loss = train_step()
    lrs.append(lr)
    losses.append(loss.item())
```

#### Gradient Explosion Solutions

Gradient explosion occurs when gradients grow exponentially during backprop. Solutions:
- **Gradient clipping**: Cap gradient norms
- **Weight initialization**: Xavier/Kaiming init
- **Batch normalization**: Stabilizes activations
- **Residual connections**: Improve gradient flow

#### Two Gradient Clipping Methods

**Clip by value**: $$g_i = \text{clip}(g_i, -c, c)$$

**Clip by norm**: $$g = \frac{c}{\|g\|} \cdot g \text{ if } \|g\| > c$$

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

Norm clipping is preferred as it preserves gradient direction.

## 📦 Chapter Code List

| File | Content | Key Concept |
|:----|:-----|:----------|
| `ch07/NN07_regularization.py` | L1/L2 regularization implementation & comparison | Regularization principles |
| `ch07/NN07_dropout.py` | Dropout implementation & effect analysis | Dropout mechanism |
| `ch07/NN07_batchnorm.py` | Batch Normalization manual implementation | Normalization techniques |
| `ch07/NN07_loss_functions.py` | Multiple loss function implementations & comparison | Loss function selection |
| `ch07/NN07_early_stopping.py` | Early stopping implementation & overfitting prevention | Early stopping strategy |
| `ch07/NN07_weight_init_viz.py` | Weight initialization comparison visualization | Initialization methods |

---

## 📖 Chapter Summary

- **Optimizers**: how to follow the gradient (SGD → Adam → AdamW)
- **Scheduling**: when to change step size (Step → Cosine → Plateau)
- **Regularization**: L1/L2, Dropout, BatchNorm
- **Initialization**: Xavier (tanh) vs He (ReLU)
- **Loss**: MSE (regression) → CrossEntropy (classification)

### 🧪 Exercises

#### Exercise 1: Optimizer Comparison
Train MNIST with SGD, SGD+Momentum, Adam. Compare epochs to 95% accuracy.

#### Exercise 2: Regularization
Train a network with and without dropout. Compare train vs. test accuracy.

#### Exercise 3: BatchNorm
Add BatchNorm to a deep network (5+ layers). Does it help convergence?

#### Exercise 4: Learning Rate Finder
Implement a learning rate finder: start lr=1e-5, increase exponentially each batch, plot loss vs. lr.

← [Chapter 6](06-chapter6-convolutional-neural-networks.md) | [Table of Contents](README.md) | [Chapter 8](08-chapter8-modern-architectures.md) →
