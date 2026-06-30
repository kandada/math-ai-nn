# Appendix B: PyTorch API Quick Reference

> This appendix summarizes all PyTorch APIs used in this book, organized by functionality.

> © xiefujin · Contact: 490021684@qq.com · Licensed under CC BY-NC-SA 4.0
---

## B-1 Tensor Creation & Operations

### B-1-1 Creating Tensors

| API | Description | Example |
|:----|:------------|:--------|
| `torch.tensor(data)` | From list/array | `torch.tensor([1, 2, 3])` |
| `torch.zeros(shape)` | All zeros | `torch.zeros(3, 4)` |
| `torch.ones(shape)` | All ones | `torch.ones(2, 3)` |
| `torch.randn(shape)` | Normal random | `torch.randn(128, 784)` |
| `torch.rand(shape)` | Uniform [0,1) | `torch.rand(3, 3)` |
| `torch.arange(start, end)` | Arithmetic sequence | `torch.arange(0, 10)` |
| `torch.linspace(s, e, n)` | Evenly spaced | `torch.linspace(0, 1, 100)` |
| `torch.eye(n)` | Identity matrix | `torch.eye(5)` |

### B-1-2 Tensor Properties

```python
x = torch.randn(3, 4)
x.shape       # torch.Size([3, 4])
x.dtype       # torch.float32
x.device      # cpu or cuda:0
x.numel()     # 12 (total elements)
```

---

## B-2 Autograd

```python
# Enable gradient tracking
x = torch.tensor([1.0], requires_grad=True)

# Forward (graph built automatically)
y = x ** 2
z = y.mean()

# Backward (compute all gradients)
z.backward()
print(x.grad)  # gradient dz/dx
```

---

## B-3 Neural Network Modules

| Module | Description |
|:-------|:------------|
| `nn.Linear(in, out)` | Fully connected layer |
| `nn.Conv2d(in, out, k)` | 2D convolution |
| `nn.MaxPool2d(k)` | Max pooling |
| `nn.BatchNorm1d/2d` | Batch normalization |
| `nn.Dropout(p)` | Dropout regularization |
| `nn.ReLU()` | ReLU activation |
| `nn.Sigmoid()` | Sigmoid activation |
| `nn.Softmax(dim)` | Softmax function |
| `nn.Sequential()` | Layer container |
| `nn.Module` | Base class for all networks |

---

## B-4 Loss Functions

| Loss | Description | Use Case |
|:-----|:------------|:---------|
| `nn.MSELoss()` | Mean squared error | Regression |
| `nn.CrossEntropyLoss()` | Cross-entropy | Classification |
| `nn.BCELoss()` | Binary cross-entropy | Binary classification |

---

## B-5 Optimizers

| Optimizer | Usage |
|:----------|:------|
| `optim.SGD(params, lr)` | SGD with optional momentum |
| `optim.Adam(params, lr)` | Adam optimizer |
| `optim.AdamW(params, lr, wd)` | Adam with decoupled weight decay |

---

## B-6 Training Loop Template

```python
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### torchvision Utilities

| API | Description |
|:----|:------------|
| `datasets.MNIST(root, train)` | MNIST handwritten digits |
| `datasets.CIFAR10(root, train)` | CIFAR-10 color images |
| `transforms.ToTensor()` | PIL Image -> Tensor conversion |
| `transforms.Normalize(mean, std)` | Standardization |
| `transforms.Compose(transforms)` | Chain multiple transforms |

← [Appendix A](A-mathematical-review.md) | [Table of Contents](../README.md) | [Appendix C](C-notation-conventions.md) →

