# Appendix E: Exercise Reference Answers

> This appendix provides reference answers and explanations for chapter exercises.

> © xiefujin · Contact: 490021684@qq.com · Licensed under CC BY-NC-SA 4.0
---

## Chapter 1: The Idea of Neural Networks

### Exercise 1: M-P Neuron for NAND Gate

Set threshold = 2, weights = [-1, -1]:

```python
def mp_neuron(x1, x2, w1=-1, w2=-1, threshold=-1.5):
    """NAND gate: negated AND"""
    total = w1 * x1 + w2 * x2
    return 1 if total >= threshold else 0

# Test
for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
    print(f"NAND({x1},{x2}) = {mp_neuron(x1, x2)}")
# Output: 1, 1, 1, 0
```

### Exercise 4: Parameter Count

- Layer 1: 784 × 256 + 256 = 200,960
- Layer 2: 256 × 10 + 10 = 2,570
- Total: 203,530

If hidden = 512: 784 × 512 + 512 = 401,920; 512 × 10 + 10 = 5,130; Total = 407,050 (2×)

---

## Chapter 2: Mathematical Foundations

### Exercise 1: Dot Product

$x \cdot y = 1 \times 4 + 2 \times 5 + 3 \times 6 = 32$

### Exercise 3: Numerical Derivative

$f(x) = x^3$, $f'(x) = 3x^2$, $f'(2) = 12$

```python
def numerical_derivative(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)
print(numerical_derivative(lambda x: x**3, 2))  # ~12.0
```

---

## Chapter 3: PyTorch Basics

### Exercise 2: Autograd

```python
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3*x + 1
y.backward()
print(x.grad)  # 2*2 + 3 = 7
```

---

## Chapter 5: Backpropagation

### Exercise 1: Manual Backpropagation Calculation

For $f(x) = \sigma(wx + b)$ at $x=1, w=0.5, b=0$, compute $\frac{\partial f}{\partial w}$:

$$u = 0.5 \times 1 + 0 = 0.5$$
$$y = \sigma(0.5) = 0.6225$$
$$\frac{\partial y}{\partial u} = y(1-y) = 0.6225 \times 0.3775 = 0.2350$$
$$\frac{\partial u}{\partial w} = x = 1$$
$$\frac{\partial f}{\partial w} = 0.2350 \times 1 = 0.2350$$

> **Explanation**: Using the chain rule, $\frac{\partial f}{\partial w} = \frac{\partial f}{\partial u} \cdot \frac{\partial u}{\partial w} = \sigma'(u) \cdot x$.

---

## Chapter 9: Large Language Models

### Exercise 1: Temperature Parameter Experiment

- `T → 0`: Approaches greedy decoding (highest probability token's probability → 1), output is deterministic but repetitive.
- `T → ∞`: Approaches uniform sampling, output is random but diverse.

**Best practice**: T = 0.7 ~ 1.0 balances creativity and coherence.

> **Analysis**: Temperature controls the sharpness of the softmax distribution. Lower temperatures amplify probability differences; higher temperatures flatten them.

← [Appendix D](D-common-functions.md) | [Table of Contents](../README.md) | [Appendix F](F-recommended-reading.md) →

