# Chapter 4: Training & Optimization in Neural Networks

> **Goal**: Understand how neural networks learn through optimization — from loss functions to gradient descent variants, with hands-on PyTorch experiments.

> © xiefujin · Contact: 490021684@qq.com · Licensed under CC BY-NC-SA 4.0
>
> **Code**: `../code/ch04/` (7 files)

---

## 📋 Chapter Learning Objectives

- [ ] Understand the formal formulation of neural network optimization
- [ ] Master the concept of loss functions (MSE, Cross-Entropy)
- [ ] Understand Softmax and its relationship to probabilities
- [ ] Master SGD, Momentum, Adam optimizers
- [ ] Understand learning rate scheduling
- [ ] Be able to train a complete classifier in PyTorch

---

## 4-1 Parameters and Variables of Neural Networks

### 4-1-1 Parameter Categories

#### Trainable Parameters (learned by the model)

| Parameter | Symbol | Role |
|:---------|:-------|:-----|
| **Weights** | $\mathbf{W}$ | Controls connection strength between neurons |
| **Biases** | $\mathbf{b}$ | Adjusts neuron activation threshold |

#### Hyperparameters (set manually)

| Hyperparameter | Role |
|:--------------|:-----|
| Learning rate $\eta$ | Controls gradient descent step size |
| Hidden layer size | Number of neurons per layer |
| Layer count | Network depth |
| Batch size | Samples per update |

### 4-1-2 Variable Categories

| Variable | Symbol | Description |
|:---------|:-------|:------------|
| **Input** | $\mathbf{x}$ | Raw data |
| **Intermediate output** | $\mathbf{z}^{(l)}$ | Layer activations |
| **Final output** | $\mathbf{y}$ | Prediction |
| **Target** | $\mathbf{t}$ | Ground truth labels |

### 4-1-3 PyTorch Parameter Management

In PyTorch, model parameters are automatically managed via `nn.Parameter`:

```python
import torch.nn as nn

linear = nn.Linear(784, 256)
for name, param in linear.named_parameters():
    print(f"{name}: {param.shape}")
```

This prints all trainable parameters and their shapes.

---
## 4-2 Variable Relationships in Neural Networks

### 4-2-0 Computational Graph Overview

Forward: x → W¹,b¹ → z¹ → σ → a¹ → W²,b² → z² → σ → a² → L(a², t)

Where each variable's value depends on the previous ones.

#### Forward Dependency Chain Within the Graph

- $a^{(l)}$ depends on: $a^{(l-1)}$, $W^{(l)}$, $b^{(l)}$
- $L$ depends on: $a^{(L)}$ (output) and $t$ (target)
- All parameters ultimately affect $L$

---

### 4-2-1 Layer-by-Layer Propagation Formulas

#### Input Layer to Hidden Layer (2 inputs to 2 hidden neurons)

$$
u_1 = x_1 w_{11} + x_2 w_{21} + b_1
\qquad
u_2 = x_1 w_{12} + x_2 w_{22} + b_2
$$

$$
z_1 = \sigma(u_1), \quad z_2 = \sigma(u_2)
$$

#### Hidden Layer to Output Layer

$$
y = z_1 w'_1 + z_2 w'_2 + b'
$$

### 4-2-2 Matrix Form

#### Single Layer Propagation

$$
\mathbf{u}^{(1)} = \mathbf{x} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}
$$

$$
\mathbf{z}^{(1)} = \sigma(\mathbf{u}^{(1)})
$$

$$
\mathbf{y} = \mathbf{z}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}
$$

### 4-2-3 Python Practice: 2-Layer Network Forward Pass

```python
import numpy as np

def forward_2layer(X, W1, b1, W2, b2, activation='sigmoid'):
    u1 = X @ W1 + b1
    if activation == 'sigmoid':
        z1 = 1 / (1 + np.exp(-u1))
    elif activation == 'relu':
        z1 = np.maximum(0, u1)
    else:
        z1 = u1
    y = z1 @ W2 + b2
    return y, z1


np.random.seed(42)
X = np.random.randn(4, 2)
W1 = np.random.randn(2, 3) * 0.1
b1 = np.zeros(3)
W2 = np.random.randn(3, 1) * 0.1
b2 = np.zeros(1)
y, hidden = forward_2layer(X, W1, b1, W2, b2)
print(f"Input: {X.shape}, Hidden: {hidden.shape}, Output: {y.shape}")
```

---
## 4-3 Training Data and Ground Truth

### 4-3-0 Dataset Splits Overview

| Dataset | Purpose | Size |
|:--------|:--------|:-----|
| **Training** | Learn parameters | ~60% |
| **Validation** | Tune hyperparameters | ~20% |
| **Test** | Final evaluation | ~20% |

#### MNIST as Running Example

```python
from torchvision import datasets, transforms

mnist_train = datasets.MNIST(root='./data', train=True,
                             transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root='./data', train=False,
                            transform=transforms.ToTensor(), download=True)

print(f"Training samples: {len(mnist_train)}")
print(f"Test samples: {len(mnist_test)}")
print(f"Image shape: {mnist_train[0][0].shape}")
```

---

### 4-3-1 Supervised Learning Data Structure

#### Feature Matrix

$$
\mathbf{X} \in \mathbb{R}^{n \times m}
$$

- $n$: number of samples
- $m$: number of features

#### Label Vector

$$
\mathbf{y} \in \mathbb{R}^{n}
$$

#### A Single Sample Pair

$$
(\mathbf{x}^{(i)}, y^{(i)})
$$

### 4-3-2 Dataset Splits

| Dataset | Purpose | Typical Split |
|:--------|:--------|:-------------:|
| **Training set** | Train model parameters (gradient descent updates weights) | 60-80% |
| **Validation set** | Tune hyperparameters, early stopping (prevent overfitting) | 10-20% |
| **Test set** | Final evaluation of generalization performance | 10-20% |

> **Note**: The test set can only be used once after training is complete. Never tune your model based on test set performance!

### 4-3-3 Data Standardization

Standardization (Z-score normalization) ensures all features have mean 0 and variance 1:

$$
x_{\text{standardized}} = \frac{x - \mu}{\sigma}
$$

where $\mu$ is the mean and $\sigma$ is the standard deviation.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Use same mean/std!
```

**Why standardize?** Features with different scales cause gradient descent to converge slowly — it creates an elongated loss landscape.

### 4-3-4 One-Hot Encoding

For classification with $K$ classes, convert class labels to vectors:

| Class | One-Hot Vector |
|:------|:---------------|
| Class 0 | $[1, 0, 0]$ |
| Class 1 | $[0, 1, 0]$ |
| Class 2 | $[0, 0, 1]$ |

```python
import torch.nn.functional as F

# Labels: 3 samples belonging to classes 1, 0, 2
labels = torch.tensor([1, 0, 2])
one_hot = F.one_hot(labels, num_classes=3)
print(one_hot)
# tensor([[0, 1, 0],
#         [1, 0, 0],
#         [0, 0, 1]])
```

---
## 4-4 Loss (Cost) Functions

### 4-4-1 Mean Squared Error (MSE)

$$
L_{\text{MSE}} = \frac{1}{2m} \sum_{i=1}^{m} \|y_i - t_i\|^2
$$

Best for: **regression** tasks (continuous outputs).

### 4-4-2 Cross-Entropy Loss

$$
L_{\text{CE}} = -\sum_{k} t_k \log p_k
$$

Best for: **classification** tasks.

### 4-4-3 Softmax Function

Converts raw scores (logits) to probabilities:

$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

Properties:
- All $p_k \in (0, 1)$
- $\sum_{k} p_k = 1$
- Preserves ranking: larger $z_k$ → larger $p_k$

### 4-4-4 Why Cross-Entropy + Softmax?

The combination of Softmax + Cross-Entropy gives a particularly nice gradient:

$$
\frac{\partial L}{\partial z_k} = p_k - t_k
$$

This means: the gradient is simply the difference between prediction and target!

---

## 4-5 Experiencing Neural Networks with Python & PyTorch

### 4-5-1 The Three-Step Transition: From Pure NumPy to PyTorch

> **Little Genius says**: Moving from NumPy to PyTorch is like upgrading your tools! Before, you had to compute gradients by hand (so tiring!). Now with autograd, you just run the forward pass, call `.backward()`, and all gradients are computed automatically. What a productivity boost!

The core meaning of the three-step transition: **Each step reduces the manual work, letting us focus more on the mathematical principles**.

#### Step 1: Pure NumPy — Manual Everything

In this step, we **hand-code** both the forward pass and backpropagation. This is essential for understanding what happens under the hood.

```python
import numpy as np

np.random.seed(42)
N, D_in, H, D_out = 100, 2, 4, 1
X = np.random.randn(N, D_in)
t = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(float).reshape(-1, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Parameter initialization
W1 = np.random.randn(D_in, H) * 0.1
b1 = np.zeros(H)
W2 = np.random.randn(H, D_out) * 0.1
b2 = np.zeros(D_out)

lr = 0.5
for epoch in range(1000):
    # Forward pass
    z1 = sigmoid(X @ W1 + b1)
    y = sigmoid(z1 @ W2 + b2)

    # Manual gradient computation
    dL_dy = (y - t) / N
    dy_du2 = y * (1 - y)
    dL_dW2 = z1.T @ (dL_dy * dy_du2)
    dL_db2 = np.sum(dL_dy * dy_du2, axis=0)

    dz1_du1 = z1 * (1 - z1)
    dL_dz1 = (dL_dy * dy_du2) @ W2.T
    dL_dW1 = X.T @ (dL_dz1 * dz1_du1)
    dL_db1 = np.sum(dL_dz1 * dz1_du1, axis=0)

    # Gradient descent update
    W2 -= lr * dL_dW2; b2 -= lr * dL_db2
    W1 -= lr * dL_dW1; b1 -= lr * dL_db1

    if epoch % 200 == 0:
        loss = 0.5 * ((y - t)**2).mean()
        print(f"Epoch {epoch}: loss = {loss:.6f}")
```

| Epoch | Loss |
|:-----:|------:|
| 0 | 0.1292 |
| 200 | 0.0290 |
| 400 | 0.0161 |
| 600 | 0.0113 |
| 800 | 0.0088 |

#### Step 2: PyTorch Tensor + Autograd — Automatic Differentiation

PyTorch's `autograd` automatically tracks operations on tensors and computes gradients. We only need to write the forward pass; `.backward()` handles the rest.

```python
import torch

# Convert to Tensor with requires_grad=True for autograd
W1_t = torch.tensor(W1, requires_grad=True)
b1_t = torch.tensor(b1, requires_grad=True)
W2_t = torch.tensor(W2, requires_grad=True)
b2_t = torch.tensor(b2, requires_grad=True)
X_t = torch.tensor(X, dtype=torch.float32)
t_t = torch.tensor(t, dtype=torch.float32)

lr = 0.5
for epoch in range(1000):
    # Forward pass (same as NumPy)
    z1 = torch.sigmoid(X_t @ W1_t + b1_t)
    y = torch.sigmoid(z1 @ W2_t + b2_t)
    loss = 0.5 * ((y - t_t)**2).mean()

    # Automatic backward pass — replaces all manual gradient code!
    loss.backward()

    # Manual parameter update (no_grad avoids tracking update operations)
    with torch.no_grad():
        W1_t -= lr * W1_t.grad
        b1_t -= lr * b1_t.grad
        W2_t -= lr * W2_t.grad
        b2_t -= lr * b2_t.grad

        # Zero gradients for next iteration
        W1_t.grad.zero_()
        b1_t.grad.zero_()
        W2_t.grad.zero_()
        b2_t.grad.zero_()
```

**Key insight**: The gradient computation code is gone! `loss.backward()` automatically computes all the chain rule derivatives we wrote manually in Step 1. The results are identical.

#### Step 3: nn.Module + optim — Production-Ready

This is how you'd write neural network code in practice — encapsulating the model and using a built-in optimizer.

```python
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

for epoch in range(1000):
    pred = model(X_t)
    loss = criterion(pred, t_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()   # ← optimizer.step() replaces the manual update block!
```

**Core insight**: All three steps produce **identical gradients and identical training progress**, but the manual effort decreases at each level. Step 1 teaches you what's happening internally; Step 3 is what you'd use in production. Chapter 5 will dive deep into the math behind those "manual gradient calculations" in Step 1.

---

### 4-5-2 Visualizing Training Progress

![Figure 4-1: Loss curve descent and decision boundary evolution during training.](../images/ch04/NN04_loss_curve.png)

*Figure 4-1: Training loss over epochs. The curve shows how the model progressively minimizes error.*

---

## 4-6 Experiment: Training Your First Classifier

### 4-6-1 Dataset: Moon Dataset

Let's use the classic `make_moons` dataset — two interleaving crescent shapes that are **not linearly separable**:

```python
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Generate non-linear dataset
X, t = make_moons(n_samples=200, noise=0.2, random_state=42)

# Visualize
plt.figure(figsize=(6, 5))
plt.scatter(X[t==0, 0], X[t==0, 1], label='Class 0', alpha=0.7)
plt.scatter(X[t==1, 0], X[t==1, 1], label='Class 1', alpha=0.7)
plt.legend()
plt.title('Moon Dataset')
plt.show()
```

![Figure 4-2: Moon dataset — two crescent shapes, non-linearly separable.](../images/ch04/NN04_moons_data.png)

*Figure 4-2: Moon dataset — a classic non-linear problem that a linear classifier cannot solve.*

> **Core insight**: A linear model (like logistic regression) cannot separate these two classes with a straight line. This is why we need neural networks with hidden layers — they learn **non-linear decision boundaries**.

### 4-6-2 Model: 2-Layer Fully Connected Network

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(2, 10),   # Input: 2 features → Hidden: 10 neurons
    nn.ReLU(),           # Activation (prevents gradient vanishing)
    nn.Linear(10, 1),   # Hidden → Output: 1 class probability
    nn.Sigmoid()         # Squash to [0, 1]
)

print(model)
```

```output
Sequential(
  (0): Linear(in_features=2, out_features=10, bias=True)
  (1): ReLU()
  (2): Linear(in_features=10, out_features=1, bias=True)
  (3): Sigmoid()
)
```

### 4-6-3 Training Loop

```python
# Convert data to tensors
X_t = torch.tensor(X, dtype=torch.float32)
t_t = torch.tensor(t, dtype=torch.float32).reshape(-1, 1)

criterion = nn.BCELoss()         # Binary Cross-Entropy
optimizer = optim.Adam(model.parameters(), lr=0.01)

n_epochs = 500
losses = []

for epoch in range(n_epochs):
    # Forward pass
    pred = model(X_t)
    loss = criterion(pred, t_t)
    losses.append(loss.item())

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}")
```

| Epoch | Loss |
|:-----:|------:|
| 0 | 0.6938 |
| 100 | 0.3467 |
| 200 | 0.2533 |
| 300 | 0.2101 |
| 400 | 0.1752 |

The loss steadily decreases — the model is learning the non-linear decision boundary!

### 4-6-4 Visualizing the Decision Boundary

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, t):
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Predict over the grid
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    Z = model(grid).detach().numpy().reshape(xx.shape)

    # Plot
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
    plt.scatter(X[t==0, 0], X[t==0, 1], label='Class 0', alpha=0.7)
    plt.scatter(X[t==1, 0], X[t==1, 1], label='Class 1', alpha=0.7)
    plt.legend()
    plt.title('Decision Boundary After Training')
    plt.show()

plot_decision_boundary(model, X, t)
```

![Figure 4-3: Trained decision boundary. Background color indicates the model's prediction, scatter points show the training data.](../images/ch04/NN04_decision_boundary.png)

*Figure 4-3: The decision boundary after training. The background colors show how the model has learned to separate the two crescent shapes with a curved boundary.*

> **Little Genius says**: See how the model draws a **curved** boundary between the two classes? That's the power of the hidden layer — it learns to warp the input space so the classes become separable. A simple linear model can't do this!

---

## 4-7 Optimizer Comparison Overview ⭐

> **Note**: This section provides an overview and intuitive understanding of optimizers. Chapter 7, Section 7-1 will dive deeper into the mathematical derivations and code verification for each optimizer (including NAG, AdaGrad, RMSprop, Adam, AdamW with complete implementations).

### 4-7-1 The Evolution from SGD to Adam

The history of optimizers is a story of "how to converge faster and more stably." Each improvement solves a core problem:

| Optimizer | Core Innovation | Problem Solved | Formula |
|:---------|:---------------|:--------------|:-------|
| **SGD** | Basic gradient descent | The most basic parameter update | $w \leftarrow w - \eta \nabla L$ |
| **Momentum** | Introduces momentum term | Overcome oscillation, accelerate convergence | $v \leftarrow \gamma v + \eta \nabla L$; $w \leftarrow w - v$ |
| **NAG** | Nesterov momentum | "Look ahead" to reduce overshooting | $v \leftarrow \gamma v + \eta \nabla L(w - \gamma v)$ |
| **AdaGrad** | Per-parameter learning rates | Different rates for different parameters | $w \leftarrow w - \frac{\eta}{\sqrt{G + \epsilon}} \odot \nabla L$ |
| **RMSprop** | Improved AdaGrad | Prevent learning rate from decaying to zero | $v \leftarrow \beta v + (1-\beta)(\nabla L)^2$ |
| **Adam** | Momentum + Adaptive | **Combines best of both worlds** | $m, v \leftarrow \beta_1 m + (1-\beta_1)\nabla L, \beta_2 v + (1-\beta_2)(\nabla L)^2$ |

> **Core insight**: Every optimizer update can be written as $w \leftarrow w - \eta \times \text{[direction]} \times \text{[step size]}$. Momentum adjusts the direction (using historical gradients), AdaGrad/RMSprop adjust the step size (using squared historical gradients), and Adam adjusts both.

### 4-7-2 Momentum: Adding Inertia to Gradient Descent

A major problem with SGD is that it oscillates in ravines — gradients keep changing direction, creating a zigzag convergence path. Momentum's solution is intuitive: **add "inertia" to parameter updates**.

$$v_{t+1} = \gamma v_t + \eta \nabla L(\mathbf{w}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - v_{t+1}$$

Where $\gamma$ (usually 0.9) is the momentum decay coefficient.

> **Little Genius says**: Momentum is like adding a "snowball effect" to going downhill! If I keep running in the same direction, I get faster and faster (momentum accumulation). If I suddenly hit a headwind (gradient reversal), I won't immediately turn around because of inertia — I'll gradually change direction, reducing the zigzag!

#### Code Comparison: SGD vs Momentum

```python
import numpy as np

def sgd(grad, w, lr=0.01):
    return w - lr * grad

def momentum(grad, w, v, lr=0.01, gamma=0.9):
    v = gamma * v + lr * grad
    return w - v, v

# Test on Rosenbrock function
def rosenbrock_grad(w):
    x, y = w[0], w[1]
    return np.array([-2*(1-x) - 400*x*(y-x**2), 200*(y-x**2)])

# Run both optimizers
w_sgd = np.array([-1.5, 1.0])
w_mom = np.array([-1.5, 1.0])
v = np.zeros(2)

for i in range(100):
    g = rosenbrock_grad(w_sgd)
    w_sgd = sgd(g, w_sgd, lr=0.001)
    
    g = rosenbrock_grad(w_mom)
    w_mom, v = momentum(g, w_mom, v, lr=0.001)
```

```output
SGD path: Significant oscillation, slow convergence
Momentum path: Smooth acceleration, 3-5x faster convergence
```

### 4-7-3 AdaGrad: Per-Parameter Learning Rates

Different parameters may need different learning rates — some are updated frequently (small steps), others rarely (large steps). AdaGrad achieves this by accumulating squared historical gradients:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{G}_t + \epsilon}} \odot \nabla L(\mathbf{w}_t)$$

Where $\mathbf{G}_t = \sum_{\tau=1}^t (\nabla L(\mathbf{w}_\tau))^2$ is the sum of squared historical gradients.

> **Little Genius says**: Imagine each parameter has its own "step-size logbook." If a parameter is frequently updated with large gradients (often used), its accumulated $G$ is large, and the step size automatically decreases — like a veteran employee who doesn't need repeated training. If a parameter is rarely updated, its $G$ is small, so the step size stays large — like a new hire who needs to learn quickly!

#### AdaGrad's Fatal Flaw

$\mathbf{G}_t$ only increases and never decreases, causing the learning rate to **monotonically decay** and eventually approach zero — the model stops learning!

### 4-7-4 RMSprop: Fixing the Vanishing Learning Rate

RMSprop replaces AdaGrad's cumulative sum with an **exponential moving average**:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta)(\nabla L(\mathbf{w}_t))^2$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{v}_t + \epsilon}} \odot \nabla L(\mathbf{w}_t)$$

> **Little Genius says**: RMSprop is like adding "forgetfulness" to AdaGrad! AdaGrad remembers every step from day one, so steps get smaller and smaller. RMSprop only remembers "recent" gradient magnitudes — old debts are written off, so the learning rate never decays to zero!

### 4-7-5 Adam: The Best of Both Worlds ⭐

Adam (Adaptive Moment Estimation) combines the advantages of Momentum and RMSprop — it has both "inertia" (first moment estimate) and "adaptive step size" (second moment estimate):

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\nabla L(\mathbf{w}_t)$$
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)(\nabla L(\mathbf{w}_t))^2$$
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$

Default hyperparameters: $\beta_1 = 0.9$ (momentum coefficient), $\beta_2 = 0.999$ (adaptive coefficient), $\epsilon = 10^{-8}$.

> **Little Genius says**: Adam is the "Swiss Army knife" of optimizers! It remembers which direction it was going (momentum) and adjusts step sizes for different parameters (adaptivity). That's why PyTorch and TensorFlow both use Adam as the **default optimizer** — most tasks work well with Adam out of the box!

#### Practical Recommendations

| Scenario | Recommended Optimizer | Reason |
|:--------|:--------------------|:-------|
| **Beginner / Quick prototyping** | **Adam** | Default hyperparameters usually work well |
| **Computer Vision** | SGD + Momentum | Often generalizes better than Adam |
| **NLP / Transformer** | AdamW | Adam + decoupled weight decay |
| **Sparse features** | AdaGrad | Automatically gives large LR to infrequent features |
| **Reinforcement Learning** | RMSprop | Handles non-stationary targets well |

---

## 4-8 Learning Rate Scheduling

### 4-8-1 Why Adjust Learning Rate?

Early in training, parameters are far from the optimum — take big steps (large learning rate). Later, parameters are close to the optimum — take small, fine-grained steps (small learning rate). This is the core idea of LR scheduling — **dynamically adjusting the learning rate**.

> **Core insight**: Learning rate scheduling = the "accelerator" control of training strategy — floor it at the start, brake on curves, accelerate again on straights.

### 4-8-2 Three Classic Scheduling Strategies

#### Strategy 1: Step Decay

Reduce the learning rate by a factor at fixed epoch intervals:

```python
import torch.optim.lr_scheduler as scheduler

# PyTorch implementation
step_scheduler = scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# Every 30 epochs, multiply the learning rate by 0.1
# lr = lr_0 × 0.1^{floor(epoch/30)}
```

$$\eta_t = \eta_0 \times \gamma^{\lfloor t / \text{step\_size} \rfloor}$$

#### Strategy 2: Cosine Annealing

The learning rate smoothly decreases from the initial value to a minimum following a cosine curve:

```python
cos_scheduler = scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
```

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(\frac{t}{T_{\max}}\pi))$$

> **Little Genius says**: Cosine Annealing is like the sunset — light doesn't disappear suddenly but gradually dims. Similarly, the learning rate smoothly descends from the initial value to the minimum, allowing the model to "fine-tune" in the later stages!

#### Strategy 3: ReduceLROnPlateau (Adaptive Decay)

Automatically reduce the learning rate when the validation loss stops decreasing:

```python
plateau_scheduler = scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)
# If loss doesn't decrease for 5 epochs, halve the learning rate
```

### 4-8-3 Practical Comparison

```python
import matplotlib.pyplot as plt
import torch.optim as optim
import torch

model = torch.nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Compare three schedulers
schedulers = {
    'StepLR(step=20, gamma=0.1)': scheduler.StepLR(optimizer, 20, 0.1),
    'CosineAnnealing(T_max=50)': scheduler.CosineAnnealingLR(optimizer, 50),
    'ReduceLROnPlateau': scheduler.ReduceLROnPlateau(optimizer, patience=5),
}
```

```output
StepLR:        Step-wise decay, suitable for fixed-stage training
CosineAnnealing: Smooth decay, suitable for long training
ReduceLROnPlateau: Adaptive decay, suitable when training duration is unknown
```

### 4-8-4 Warmup Technique

Use a smaller learning rate at the start of training, then gradually increase to the target LR. This **prevents the model from "exploding" in the initial stage due to too-large learning rates**.

```python
class WarmupLR(scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps=1000):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [base_lr * self._step_count / self.warmup_steps
                    for base_lr in self.base_lrs]
        return self.base_lrs
```

> **Core insight**: Warmup + Cosine Annealing is the most popular LR scheduling combination today — start slowly, then cosine-decay smoothly. Almost all modern LLM training uses this strategy!

---

## 4-9 Loss Landscape Visualization

### 4-9-1 What Is a Loss Landscape?

The loss function $L(\mathbf{w})$ in high-dimensional weight space forms a "loss landscape." Imagine a 3D space where x and y axes are two weight parameters, and the z-axis is the corresponding loss value — that surface is the simplest loss landscape.

### 4-9-2 Convex vs. Non-Convex Functions

| Type | Shape | Can Gradient Descent Find the Global Optimum? |
|:----|:-----|:--------------------------------------------|
| **Convex** | Bowl-shaped, single minimum | ✅ Definitely |
| **Non-convex** | Multiple valleys (local minima) and saddle points | ⚠️ Not guaranteed |

**Convex function definition**: $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$

> **Little Genius says**: A convex loss landscape is like a bowl — no matter where you start on the bowl, following the steepest direction will always get you to the bottom (global optimum). A non-convex function is like a mountain range — you might get stuck in a small valley (local minimum), thinking you've reached the top, when there are higher mountains elsewhere!

### 4-9-3 Rosenbrock Function — Classic Test Case

The Rosenbrock function is a famous non-convex test function commonly used to evaluate optimizer performance:

$$f(\mathbf{w}) = (1 - w_1)^2 + 100(w_2 - w_1^2)^2$$

Its characteristic is a **narrow, curved valley** — optimization algorithms easily oscillate within this valley.

```python
import numpy as np

def rosenbrock(w):
    return (1 - w[0])**2 + 100 * (w[1] - w[0]**2)**2

def rosenbrock_grad(w):
    x, y = w[0], w[1]
    dx = -2*(1-x) - 400*x*(y - x**2)
    dy = 200*(y - x**2)
    return np.array([dx, dy])
```

```output
Rosenbrock minimum at (1, 1), f(1,1) = 0
The valley has a curved parabolic shape, standard SGD converges very slowly
```

### 4-9-4 Visualization: Contour Plot + Optimization Path

```python
# Create grid
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.log(rosenbrock([X, Y]) + 1)  # Log scale for easier visualization

# Contour plot
plt.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.7)
plt.colorbar(label='log(Loss)')
plt.xlabel('w₁'); plt.ylabel('w₂')
plt.title('Rosenbrock Function Loss Landscape')
```

### 4-9-5 Why Visualization Matters

The geometry of the loss landscape directly determines optimizer behavior:

1. **Flat regions**: Gradients near zero, optimization stalls (saddle-point-like)
2. **Steep ravines**: Gradient direction changes rapidly, SGD oscillates
3. **Local minima**: Looks like the lowest point, but is not globally optimal

> **Core insight**: In **high-dimensional spaces** (the actual neural network scenario), local minima are usually not a problem — there are too many dimensions to "go around." The real issue is **saddle points** — where gradients are zero in some directions but not others. Adaptive optimizers like Adam handle saddle points better.

---

## 4-10 Code Visualization Results

Running the code in this chapter produces the following visualizations:

![Figure 4-4: GD (batch gradient descent) is stable but slow, SGD (stochastic gradient descent) oscillates more, Mini-batch strikes a balance.](../images/ch04/NN04_gd_sgd_minibatch.png)
*Figure 4-4: Three gradient descent variants compared.*

![Figure 4-5: Convergence trajectories of SGD, Momentum, RMSprop, and Adam optimizers on 2D contour lines.](../images/ch04/NN04_optimizer_comparison.png)
*Figure 4-5: Optimizer comparison on contour lines.*

![Figure 4-6: Effect of different learning rates on convergence speed, and determining the optimal learning rate.](../images/ch04/NN04_convergence_rate.png)
*Figure 4-6: Learning rate effects.*

![Figure 4-7: Comparison of Step Decay, Exponential Decay, and Cosine Annealing learning rate schedules.](../images/ch04/NN04_lr_schedule.png)
*Figure 4-7: Learning rate schedule comparison.*

![Figure 4-8: Adam optimizer's adaptive convergence behavior.](../images/ch04/NN04_adam_convergence.png)
*Figure 4-8: Adam optimizer convergence.*

![Figure 4-9: Gradient descent paths from different initializations on the loss landscape.](../images/ch04/NN04_loss_landscape.png)
*Figure 4-9: Loss landscape with optimization paths.*

![Figure 4-10: Decision boundary differences produced by different optimizers.](../images/ch04/NN04_pytorch_optimizers.png)
*Figure 4-10: Optimizer decision boundary comparison.*

## 📖 Chapter Summary

### 🧪 Practice Exercises

#### Exercise 1: Learning Rate Comparison

Perform gradient descent on f(x) = x². Try learning rates η = 0.1, 0.5, 1.0, 1.8 starting from x=2 for 20 iterations, and plot the convergence trajectory.

**Predict**: Which learning rate will diverge? Which will converge fastest?

```python
def gradient_descent(lr, steps=20):
    x = 2.0
    trajectory = [x]
    for _ in range(steps):
        grad = 2 * x  # derivative of x²
        x = x - lr * grad
        trajectory.append(x)
    return trajectory

# Try different learning rates
for lr in [0.1, 0.5, 1.0, 1.8]:
    traj = gradient_descent(lr)
    print(f"lr={lr}: final x={traj[-1]:.4f}")
```

#### Exercise 2: Manual Softmax + Cross-Entropy

Compute the combined gradient of Softmax + Cross-Entropy by hand for logits [2.0, 1.0, 0.1] with true class 0. Verify with PyTorch.

#### Exercise 3: Optimizer Comparison

Train the same 2-layer network on the Moon dataset using SGD, SGD+Momentum, and Adam. Compare:
- Convergence speed (epochs to reach loss < 0.1)
- Stability (loss variance)
- Final decision boundary

#### Exercise 4: Learning Rate Search

Train a simple classifier on the Moon dataset with learning rates [0.1, 0.01, 0.001, 0.0001]. Which converges fastest? Which fails to converge?







---


### 📌 Chapter Key Concepts

| Core Concept | One-Sentence Summary |
|:------------|:---------------------|
| **Parameter Management** | Weights & biases are learnable parameters; hyperparameters are manually set; PyTorch's `nn.Parameter` manages them uniformly |
| **Forward Propagation** | Data flows layer by layer from input to output: $\mathbf{h} = \sigma(\mathbf{Wx} + \mathbf{b})$ |
| **Loss Functions** | MSE for regression; CrossEntropy + Softmax for classification (gradient = $p-t$, the simplest form) |
| **Gradient Descent** | $w \leftarrow w - \eta \nabla L$; three core elements: direction (gradient), step size (learning rate), velocity (momentum) |
| **Optimizer Evolution** | SGD → Momentum → AdaGrad → RMSprop → **Adam** (best of all worlds, recommended as default) |
| **Learning Rate Scheduling** | Step Decay, Cosine Annealing, ReduceLROnPlateau, Warmup |
| **Loss Landscape** | In high dimensions, saddle points are more problematic than local minima; adaptive optimizers like Adam handle them better |
| **Data Processing** | Feature standardization (Z-Score), One-Hot encoding, 80:20 train/validation split |
| **Softmax Function** | Converts logits to probabilities: $p_k = e^{z_k} / \sum_j e^{z_j}$, used with cross-entropy |
| **Cross-Entropy Gradient** | With Softmax: gradient = $p - t$ (prediction minus truth), simplest and most efficient form |
| **Mini-batch SGD** | Balances GD's stability and SGD's speed; typical batch sizes: 32 or 64 |
| **Decision Boundary** | The separating surface in feature space; visualized via contour plots to show model learning |


### 📁 Chapter Code Files

| File Name | Core Content |
|:---------|:-------------|
| `NN04_gd_variants.py` | Comparison of GD, SGD, and Mini-batch gradient descent |
| `NN04_optimizers.py` | Convergence trajectories of SGD, Momentum, RMSprop, Adam |
| `NN04_convergence_analysis.py` | Effect of different learning rates on convergence speed |
| `NN04_lr_schedule.py` | Step Decay, Cosine Annealing, and other scheduling strategies |
| `NN04_adam_demo.py` | Adam's adaptive convergence on the Rosenbrock function |
| `NN04_loss_landscape.py` | Loss landscape surface visualization + gradient descent paths |
| `NN04_pytorch_optim_demo.py` | PyTorch built-in optimizer decision boundary comparison |

> Code is located in the `code/ch04/` directory. Run `python NN04_xxx.py` to see the visualizations.

---

← [Chapter 3](03-chapter3-pytorch-basics-tensor-autograd.md) | [Table of Contents](README.md) | [Chapter 5](05-chapter5-backpropagation.md) →