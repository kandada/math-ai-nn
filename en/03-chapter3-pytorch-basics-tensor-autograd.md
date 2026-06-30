# Chapter 3: PyTorch Basics — Tensor & Autograd

> **Goal**: Understand the two core tools PyTorch provides — **Tensor** (efficient array computation) and **Autograd** (automatic differentiation) — and how they work together to enable neural network training.

> © xiefujin · Contact: 490021684@qq.com · Licensed under CC BY-NC-SA 4.0
>
> **Code**: `../code/ch03/` (8 files)

---

## 📋 Chapter Learning Objectives

- [ ] Understand the relationship between Tensor and NumPy array
- [ ] Master Tensor creation and basic operations
- [ ] Understand broadcasting mechanism
- [ ] Understand Autograd: how PyTorch automatically computes gradients
- [ ] Master computational graph concepts
- [ ] Be able to build network modules with nn.Module
- [ ] Understand DataLoader for batch processing

---

## 3-1 Why Do We Need PyTorch?

### 3-1-1 From Excel to NumPy to PyTorch

| Tool | Pros | Cons |
|:-----|:-----|:------|
| **Excel** | Visual, accessible | Can't scale, no GPU |
| **NumPy** | Powerful, great for prototyping | No GPU, no autograd |
| **PyTorch** | GPU + autograd + ecosystem | Learning curve |

NumPy is great for fixed computations, but neural networks need **differentiation** — and PyTorch's autograd handles this automatically.

### 3-1-2 Deep Learning Framework Comparison

| Framework | Pros | Cons |
|:----------|:-----|:------|
| **PyTorch** | Dynamic graphs, Pythonic, research-friendly | Production deployment maturing |
| **TensorFlow/Keras** | Deployment, production | Steeper learning curve for custom work |
| **JAX** | Functional, fast | Newer, smaller ecosystem |

### 3-1-3 PyTorch's Design Philosophy

1. **Pythonic**: Feels like writing NumPy
2. **Dynamic graphs**: Graphs built on-the-fly (ease debugging)
3. **Tensors first**: Same interface for CPU/GPU

---

## 3-2 PyTorch Tensor Basics

### 3-2-1 What Is a Tensor?

A **tensor** is a multi-dimensional array — a generalization of:

```text
Scalar (0D) → Vector (1D) → Matrix (2D) → Tensor (3D+)
```

In neural networks: images are 4D tensors (batch × channels × height × width).

### 3-2-2 Creating Tensors

```python
import torch

# From data
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([[1, 2], [3, 4]])

# From shapes (useful for weights)
zeros = torch.zeros(2, 3)
ones = torch.ones(2, 3)
randn = torch.randn(2, 3)  # standard normal

# From NumPy
import numpy as np
np_array = np.array([1, 2, 3])
t3 = torch.from_numpy(np_array)

print(f"tensor: {t1}")
print(f"zeros: {zeros}")
print(f"random: {randn}")
```

### 3-2-3 Tensor Properties

```python
x = torch.randn(3, 4)
print(f"Shape: {x.shape}")
print(f"Dtype: {x.dtype}")
print(f"Device: {x.device}")
print(f"Numel: {x.numel()}")  # total elements
```

---

## 3-3 Tensor Operations and Broadcasting

### 3-3-1 Basic Operations

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(f"Add: {a + b}")
print(f"Mul: {a * b}")     # element-wise
print(f"Dot: {a @ b}")     # dot product (torch.dot(a, b))
print(f"Matmul: {a @ b}")  # same for 1D; use @ for 2D+
```

### 3-3-2 Broadcasting ⭐

Broadcasting automatically expands dimensions to make shapes compatible — making your code match the math formula perfectly:
$$
\mathbf{U} = \mathbf{XW} + \mathbf{b}
$$

```python
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])  # shape (2, 3)
b = torch.tensor([10, 20, 30])  # shape (3,) → broadcast to (2, 3)

c = a + b  # b is "stretched" along dimension 0
print(c)
# tensor([[11, 22, 33],
#         [14, 25, 36]])
```

### 3-3-3 Dimension Operations

```python
x = torch.randn(2, 3, 4)
print(f"Original: {x.shape}")

# Reshape
print(f"Reshaped: {x.view(6, 4).shape}")  # or x.reshape(6, 4)

# Transpose
print(f"Transposed: {x.transpose(0, 1).shape}")  # (3, 2, 4)

# Unsqueeze / Squeeze
print(f"Unsqueezed: {x.unsqueeze(0).shape}")  # (1, 2, 3, 4)
print(f"Squeezed:   {x.squeeze().shape}")     # removes dims of size 1
```

---

## 3-4 Autograd: Automatic Differentiation ⭐

### 3-4-1 What Is Autograd?

**Autograd** is PyTorch's automatic differentiation engine. It automatically computes **gradients of any computation** you define.

### 3-4-2 requires_grad: Tell PyTorch Which Tensors Need Gradients

```python
# Only tensors with requires_grad=True will accumulate gradients
w = torch.tensor([2.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
x = torch.tensor([3.0])  # input (no gradient needed)
```

### 3-4-3 Forward Pass: Building the Computation Graph

```python
# Forward: PyTorch records every operation in a graph
z = w * x + b
loss = z ** 2

print(f"z = {z.item():.2f}")
print(f"loss = {loss.item():.2f}")
```

### 3-4-4 Backward Pass: One Line for All Gradients

```python
# Backward: computes ALL gradients via chain rule
loss.backward()

print(f"d(loss)/dw = {w.grad.item():.4f}")
print(f"d(loss)/db = {b.grad.item():.4f}")

# This is equivalent to applying the chain rule automatically:
# $$
# \frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w}
# $$

# Verify manually:
# loss = (wx + b)²
# d(loss)/dw = 2(wx + b) * x = 2(2*3+1) * 3 = 42
# d(loss)/db = 2(wx + b) * 1 = 2(2*3+1) = 14
print(f"Expected: dL/dw = {2*(2*3+1)*3:.4f}, dL/db = {2*(2*3+1):.4f}")
```

### 3-4-5 How backward() Works

Autograd automatically applies the chain rule across the computation graph:
$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial w}
$$

For our example: $L = z^2$, $z = w \cdot x + b$, so:
$$
\frac{\partial L}{\partial w} = 2z \cdot x
$$

`loss.backward()` traces the computational graph backward:

1. **loss**: seed gradient = 1
2. **z²**: d(loss)/dz² = 2z
3. **z = wx + b**: dz/dw = x, dz/db = 1
4. **Accumulate**: w.grad += dz/dw, b.grad += dz/db

### 3-4-6 Gradient Accumulation ⚠️

Gradients **accumulate** by default — you must zero them before each backward pass:

```python
# WRONG: gradients accumulate!
loss.backward()
loss.backward()  # gradients DOUBLE!
print(w.grad)  # 2x expected value!

# CORRECT: zero the gradients first
w.grad.zero_()
loss.backward()  # now correct
```

---

## 3-5 Computational Graph Deep Dive ⭐

### 3-5-1 grad_fn: Tracing Operation Sources

Every tensor tracks how it was created via `grad_fn`:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y.mean()

print(f"x.grad_fn = {x.grad_fn}")  # None (leaf tensor)
print(f"y.grad_fn = {y.grad_fn}")  # <PowBackward0>
print(f"z.grad_fn = {z.grad_fn}")  # <MeanBackward0>
```

### 3-5-2 Static vs. Dynamic Graphs

| Aspect | Static (TF1) | Dynamic (PyTorch) |
|:-------|:-------------|:------------------|
| Graph built | Before execution | During execution |
| Debugging | Hard | Easy |
| Flexibility | Limited | Full Python control |

### 3-5-3 Detaching from the Graph

Sometimes you want to **stop gradient flow** through part of the graph:

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
z = y.detach()  # z is a normal tensor, no gradient tracking
w = z ** 2      # w won't track gradients through y
```

![Figure 3-1: PyTorch computational graph visualization. Green nodes are Tensors (leaf nodes), blue nodes are operations (grad_fn), arrows indicate data flow direction.](../images/ch03/NN03_computational_graph.png)
*Figure 3-1: Computational graph visualization. From x, w, b through multiplication, addition, Sigmoid, and squared to the final loss L.*

---

## 3-6 nn.Module: Building Blocks for Networks

### 3-6-1 The nn.Module Base Class

All neural network components inherit from `nn.Module`:

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = SimpleNet()
print(net)
```

### 3-6-2 nn.Sequential: Quick Stacking

```python
net = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
print(net)
```

### 3-6-3 nn.Parameter: Trainable Parameters

```python
# nn.Linear already registers its weights as parameters
for name, param in net.named_parameters():
    print(f"{name}: {param.shape}, requires_grad={param.requires_grad}")
```

### 3-6-4 Model Parameter Management

```python
# Total parameter count
total = sum(p.numel() for p in net.parameters())
print(f"Total parameters: {total:,}")

# Training mode vs eval mode
net.train()  # enables dropout, batchnorm updates
net.eval()   # disables dropout, fixes batchnorm
```

---

## 3-7 Datasets and DataLoaders

### 3-7-1 Dataset: Where the Data Lives

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

### 3-7-2 DataLoader: Batching and Shuffling

```python
dataset = MyDataset(torch.randn(1000, 784),
                    torch.randint(0, 10, (1000,)))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_x, batch_y in loader:
    print(f"Batch: x={batch_x.shape}, y={batch_y.shape}")
    break
```

### 3-7-3 MNIST Dataset Example

```python
from torchvision import datasets, transforms

mnist = datasets.MNIST(
    root='./data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

loader = DataLoader(mnist, batch_size=64, shuffle=True)
images, labels = next(iter(loader))
print(f"Images: {images.shape}")  # (64, 1, 28, 28)
print(f"Labels: {labels.shape}")  # (64,)
```

![Figure 3-2: MNIST handwritten digit dataset samples. 28x28 pixel grayscale images, containing digits 0-9 across ten classes.](../images/ch03/NN03_mnist_samples.png)
*Figure 3-2: MNIST dataset samples — the "Hello World" of deep learning.*

---

## 3-8 Complete Training Loop

### 3-8-1 The Training Loop Template

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    for batch_x, batch_y in loader:
        # 1. Flatten images
        batch_x = batch_x.view(batch_x.size(0), -1)

        # 2. Forward
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # 3. Backward
        optimizer.zero_grad()  # clear old gradients
        loss.backward()        # compute new gradients

        # 4. Update
        optimizer.step()       # apply gradients

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
```

### 3-8-2 GPU Acceleration

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

model.to(device)  # move model to GPU

for batch_x, batch_y in loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    # ... forward, backward, update (same as before)
```


## ⚠️ Common Pitfalls & Debugging Guide

### Pitfall 1: Forgetting optimizer.zero_grad()

This is the #1 mistake beginners make:

```python
for epoch in range(10):
    for batch_x, batch_y in loader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # ⚠️ Forgot optimizer.zero_grad()!
        loss.backward()
        optimizer.step()
```

**What happens**: Gradients accumulate across batches instead of being replaced. Your parameter updates will be wrong and training diverges.

**Fix**: Always call `optimizer.zero_grad()` before `loss.backward()`.

### Pitfall 2: Tensor Device Mismatch

```python
# ❌ Common mistake
model = nn.Linear(784, 10).to('cuda')      # model on GPU
batch_x = torch.randn(32, 784)              # Tensor on CPU!
outputs = model(batch_x)                    # RuntimeError!
```

**Fix**: Move all tensors to the same device.

### Pitfall 3: In-place Operations Breaking the Computation Graph

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x ** 2

# ❌ In-place modification
x.add_(1)  # changes x, but computation graph still points to old x!
y.backward()  # RuntimeError
```

**Rule**: Never modify a tensor in-place if it has `requires_grad=True`.

---

## 3-9 Autograd Deep Dive ⭐

### 3-9-1 When Is the Computation Graph Built?

PyTorch uses a **dynamic computational graph** — a new graph is built on every forward pass.

```python
import torch

x = torch.tensor(2.0, requires_grad=True)

# Each forward pass rebuilds the graph
for i in range(3):
    y = x ** 2          # new graph created each iteration
    y.backward()
    print(f"Iteration {i}: grad = {x.grad}")
    x.grad.zero_()      # manually clear gradient
```

**Key Insight**: Unlike static frameworks (TensorFlow 1.x), PyTorch's dynamic graph means:
- You can use Python control flow (if/for) naturally
- The graph structure can change each iteration
- Debugging is easier — you can inspect tensors at any point

### 3-9-2 Gradient Accumulation and Clearing

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # tensor(4.0)

# ⚠️ Gradients ACCUMULATE!
y = x ** 3
y.backward()
print(x.grad)  # tensor(16.0) = 4.0 + 12.0
```

**Use gradient accumulation** for simulating larger batch sizes:

```python
# Simulate batch_size=64 using 2 micro-batches of 32
for i, (batch_x, batch_y) in enumerate(loader):
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss = loss / 2  # normalize by number of accumulations
    loss.backward()

    if (i + 1) % 2 == 0:  # update every 2 micro-batches
        optimizer.step()
        optimizer.zero_grad()
```

### 3-9-3 detach(): Separating from the Computation Graph

Use `.detach()` when you want to use a tensor's value without tracking its gradient.

**Use cases**:
- Freezing pretrained layers during fine-tuning
- Extracting features for visualization
- GAN training (detach discriminator output when training generator)

### 3-9-4 no_grad() vs inference_mode()

Both disable gradient tracking, but in different ways:

| Context | Grad Tracking | Performance | Use Case |
|:--------|:-------------|:------------|:---------|
| `no_grad()` | Disabled | Fast | Evaluation, feature extraction |
| `inference_mode()` | Disabled | Faster | Pure inference |
| Default | Enabled | Normal | Training |

> **Core Insight**: Use `torch.no_grad()` during evaluation. It reduces memory usage and speeds up computation.

---

## 3-10 PyTorch Debugging Practice

### 3-10-1 Common Errors and Solutions

**Error 1: Shape Mismatch**
```python
x = torch.randn(32, 784)
linear = nn.Linear(784, 256)
# out = linear(x.t())  # ❌ Wrong shape
out = linear(x)  # ✅ Correct
```

**Error 2: Gradients are None**
- Cause: `backward()` was not called before accessing `.grad`
- Fix: ensure `loss.backward()` is called

**Error 3: Loss is NaN**
- Possible causes: learning rate too high, division by zero, log(0)
- Fix: clip gradients, add epsilon to denominators, check data

### 3-10-2 Debugging with Hooks

```python
def debug_hook(module, input, output):
    print(f"Layer: {module.__class__.__name__}")
    print(f"  Input shape: {input[0].shape}")
    print(f"  Output mean: {output.mean().item():.4f}")

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
hook = model[0].register_forward_hook(debug_hook)
x = torch.randn(32, 784)
out = model(x)
hook.remove()
```

### 3-10-3 Using TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_1')
for epoch in range(100):
    writer.add_scalar('Loss/train', loss, epoch)
writer.close()
```

---

## 📦 Code Checklist for This Chapter

| # | File | Description |
|:--|:-----|:-----------|
| 1 | `code/ch03/NN03_tensor_basics.py` | Basic tensor operations |
| 2 | `code/ch03/NN03_broadcasting.py` | Broadcasting examples |
| 3 | `code/ch03/NN03_autograd_demo.py` | Autograd demonstration |
| 4 | `code/ch03/NN03_computational_graph.py` | Computational graph visualization |
| 5 | `code/ch03/NN03_nn_module.py` | Building networks with nn.Module |
| 6 | `code/ch03/NN03_dataset_dataloader.py` | Data loading pipeline |
| 7 | `code/ch03/NN03_training_loop.py` | Complete training loop |
| 8 | `code/ch03/NN03_mnist_viz.py` | MNIST data visualization |

---

## 📖 Chapter Summary

### Core Concepts

- **Tensor**: Multi-dimensional array (like NumPy, but GPU-ready)
- **Autograd**: Automatic gradient computation
- **Computational Graph**: Records operations for backward pass
- **nn.Module**: Building block for network components
- **DataLoader**: Efficient batch processing

### Key Takeaways

1. PyTorch uses dynamic computational graphs — built fresh each forward pass
2. Always call `optimizer.zero_grad()` before `loss.backward()`
3. Use `.to(device)` to move both model and data to GPU
4. Use `torch.no_grad()` during evaluation to save memory
5. Register hooks for debugging intermediate layers

### Common Pitfalls Recap

| Pitfall | Symptom | Solution |
|:--------|:--------|:---------|
| Forgot zero_grad | Training diverges | Call zero_grad() every batch |
| Device mismatch | RuntimeError | Check .device of all tensors |
| In-place ops | backward() fails | Never modify requires_grad tensors in-place |

### Quick Reference: Key PyTorch APIs

| API | Purpose |
|:---|:--------|
| `torch.tensor()` | Create tensor |
| `torch.randn()` | Random normal tensor |
| `x.requires_grad_()` | Enable gradient tracking |
| `x.backward()` | Compute gradients |
| `x.detach()` | Remove from graph |
| `nn.Linear(in, out)` | Fully connected layer |
| `nn.ReLU()` | Activation function |
| `nn.CrossEntropyLoss()` | Loss for classification |
| `optim.SGD()` / `optim.Adam()` | Optimizers |
| `DataLoader(dataset, batch_size)` | Batch data loading |

← [Chapter 2](02-chapter2-mathematical-foundations.md) | [Table of Contents](README.md) | [Chapter 4](04-chapter4-optimization.md) →


