# Chapter 2: Mathematical Foundations of Neural Networks

> **Goal**: **Intuitively understand** the three mathematical pillars of deep learning — functions, linear algebra, and calculus. No need to memorize formulas; instead understand *why* each mathematical tool appears in neural networks.

> © xiefujin · Contact: 490021684@qq.com · Licensed under CC BY-NC-SA 4.0
>
> **Code**: `../code/ch02/` (8 files)

> **Figures**: `../images/ch02/` (9 images)

---

## 📋 Chapter Learning Objectives

- [ ] Understand the functions that appear in neural networks (linear, quadratic, exponential, logarithmic)
- [ ] Understand forward propagation as a recurrence relation
- [ ] Master the relationship between summation notation and dot products
- [ ] Understand how matrix multiplication enables batch computation
- [ ] Understand the intuitive meaning of derivatives and partial derivatives
- [ ] Master the chain rule — the mathematical engine of backpropagation
- [ ] Understand gradient descent: descending along the steepest direction

---

## 2-1 Functions in Neural Networks

### 2-1-1 Linear Functions (Linear Transformations)

#### Definition

$$
y = ax + b
$$

Where $a$ is the slope and $b$ is the intercept.

#### Geometric Meaning

- $a$ controls the **steepness** of the line (larger $a$ = steeper)
- $b$ controls the **position** on the $y$-axis

![Figure 2-1: Linear, quadratic, exponential, and logarithmic function families — the basic building blocks of neural network math.](../images/ch02/NN02_function_family.png)
*Figure 2-1: Four fundamental function families — linear (blue), quadratic (orange), exponential (green), logarithmic (red).*

![Figure 2-2: Matrix multiplication visualization — each cell in the heatmap represents a weight; color-coded blocks highlight the multiplicative relationship between corresponding elements.](../images/ch02/NN02_matrix_multiplication.png)
*Figure 2-2: Visualizing matrix multiplication — one matrix multiplication = computing all samples' weighted sums simultaneously.*

---

## 2-2 Sequences and Recurrence Relations

### 2-2-1 Sequence Basics

#### Arithmetic Sequences

Each term differs from the previous term by a constant:

$$
a_n = a_1 + (n-1)d
$$

**Example**: $2, 5, 8, 11, 14, \ldots$ (common difference $d=3$)

#### Geometric Sequences

Each term is a constant multiple of the previous term:

$$
a_n = a_1 r^{n-1}
$$

**Example**: $2, 4, 8, 16, 32, \ldots$ (common ratio $r=2$)

#### Generating Sequences with Python

```python
# Arithmetic sequence
a1, d, n = 2, 3, 10
arith_seq = a1 + np.arange(n) * d
print(f"Arithmetic sequence: {arith_seq}")

# Geometric sequence
a1, r, n = 2, 2, 10
geom_seq = a1 * r ** np.arange(n)
print(f"Geometric sequence: {geom_seq}")
```

---

### 2-2-2 Recurrence Relations

#### Definition

A recurrence relation determines the current value from previous values:

$$
a_{n+1} = f(a_n)
$$

#### Example: The Fibonacci Sequence

$$
F_{n+2} = F_{n+1} + F_n, \quad F_1 = 1, F_2 = 1
$$

```python
def fibonacci(n):
    a, b = 1, 1
    for _ in range(n):
        print(a, end=' ')
        a, b = b, a + b

fibonacci(10)  # 1 1 2 3 5 8 13 21 34 55
```

> **Little Genius says**: Recurrence is like a relay race — each little genius passes the signal to the next one, and every little genius's work builds on all the previous ones! The forward propagation of a neural network is essentially a recurrence process.

---

### 2-2-3 Forward Propagation Is Recurrence

#### The Recurrence Relation of Neural Networks

$$
\mathbf{z}^{(l+1)} = f(\mathbf{W}^{(l)} \mathbf{z}^{(l)} + \mathbf{b}^{(l)})
$$

#### Analogy: Falling Dominoes

Forward propagation is like falling dominoes: the output of layer $l$ triggers the computation of layer $l+1$, one layer toppling the next.

The output of the previous layer "topples" the next layer's computation. This recurrence relationship is key to understanding backpropagation — because backpropagation is also a recurrence, just going **backward**.

> **Core Insight**: Forward propagation is recurrence "from input to output"; backpropagation is recurrence "from output to input." Both directions use the **same recurrence logic**.

The **forward propagation** of a neural network is essentially a recurrence process!

$$
\mathbf{a}^{(0)} = \mathbf{x} \quad\text{(input layer, start of recurrence)}
$$
$$
\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)} \quad\text{(recurrence rule: weighted sum)}
$$
$$
\mathbf{a}^{(l)} = f(\mathbf{z}^{(l)}) \quad\text{(recurrence rule: activation)}
$$

Starting from the input layer, computing layer by layer until the output layer — this is the recurrence essence of forward propagation.

> **Little Genius says**: Forward propagation is like a relay race! The input is the starting line; the little geniuses at each layer complete their calculations and pass the result (activation values) to the next layer. The baton traveling from layer 0 to layer L is one complete forward pass!

---

## 2-3 Summation Notation

### 2-3-1 Introduction to Sigma Notation

#### Definition

$$
\sum_{i=1}^{n} x_i = x_1 + x_2 + \cdots + x_n
$$

#### Basic Properties

| Property | Formula |
|:---------|:--------|
| Addition distributes | $\sum (x_i + y_i) = \sum x_i + \sum y_i$ |
| Constant factoring | $\sum c x_i = c \sum x_i$ |
| Sum of a constant | $\sum_{i=1}^{n} c = nc$ |

The summation symbol $\Sigma$ (Sigma) is the mathematical shorthand for "accumulation":

**Key Rules**:

- $\sum_{i=1}^{n} (x_i + y_i) = \sum x_i + \sum y_i$ (addition can be split)
- $\sum_{i=1}^{n} c \cdot x_i = c \cdot \sum x_i$ (constants can be factored out)
- $\sum_{i=1}^{n} \sum_{j=1}^{m} x_{ij}$ (double summation = nested loop)

```python
# Double summation in Python
total = 0
for i in range(1, n+1):
    for j in range(1, m+1):
        total += x[i][j]
```

---

### 2-3-2 Summation and Neural Networks

#### Weighted Sum of a Neuron

$$
u = \sum_{i=1}^{n} w_i x_i
$$

#### Multi-Layer Network Summation

$$
u_j^{(l)} = \sum_i w_{ji}^{(l)} z_i^{(l-1)} + b_j^{(l)}
$$

> **Summation is the "universal language" of neural networks** — every neuron's input is expressed using $\sum$.

---

### 2-3-3 Python: From for Loops to Vectorization

```python
import numpy as np

n = 5
w = np.array([0.5, -0.3, 0.8, 0.1, -0.2])
x = np.array([1.0, 0.5, 2.0, 1.5, 0.3])

# Level 1: for loop (most intuitive, but slowest)
u1 = 0
for i in range(n):
    u1 += w[i] * x[i]
print(f"for loop: u = {u1:.4f}")

# Level 2: np.sum + element-wise multiplication (cleaner)
u2 = np.sum(w * x)
print(f"np.sum:   u = {u2:.4f}")

# Level 3: np.dot (most semantic — tells the reader this is 'inner product')
u3 = np.dot(w, x)
print(f"np.dot:   u = {u3:.4f}")

# Level 4: @ operator (Python 3.5+, most concise)
u4 = w @ x
print(f"@ operator: u = {u4:.4f}")
```

```output
for loop: u = 1.5100
np.sum:   u = 1.5100
np.dot:   u = 1.5100
@ operator: u = 1.5100
```

> **Core Insight**: The evolution from for loops to the @ operator embodies the mental shift from "computing element by element" to "holistic vectorization." Vectorization is the core optimization technique behind modern deep learning frameworks (PyTorch, TensorFlow).

---

## 2-4 Vector Basics

### 2-4-1 Definition of Vectors

#### Mathematical Definition

A vector $\mathbf{x} = (x_1, x_2, \cdots, x_n)$ is an ordered array of $n$ numbers.

#### Geometric Meaning

A **point** or **directed line segment** in $n$-dimensional space.

#### Python Representation

```python
import numpy as np

# Create a vector
x = np.array([1, 2, 3, 4, 5])
print(f"Vector x = {x}")
print(f"Dimension: {x.shape}")
print(f"Second element: {x[1]}")
```

A vector is an ordered list of numbers with both **magnitude** and **direction**.

$$
\mathbf{v} = (v_1, v_2, \dots, v_n)^{\top}
$$

In neural networks, vectors are everywhere:

- Input vector $\mathbf{x}$: all pixel values of an image
- Weight vector $\mathbf{w}$: all connection weights of a single neuron
- Gradient vector $\nabla L$: partial derivatives of the loss w.r.t. all parameters

**Two perspectives on vectors**:

1. **Geometric perspective**: An arrow pointing in some direction in space
2. **Data perspective**: A column of ordered numbers, stored in a Tensor or array

> **Little Genius says**: A vector is my "shopping list" — it lists everything I need to buy in order! In a neural network, a neuron's input signal $\mathbf{x} = [x_1, x_2, \dots, x_n]^T$ is a vector — all the incoming signals form a "signal checklist."

---

### 2-4-2 Vector Inner Product ⭐

> **Little Genius says**: The vector inner product is my daily work! Input signals $x_1, x_2, \dots, x_n$ are like a pile of packages, and weights $w_1, w_2, \dots, w_n$ are the "importance coefficient" for each package. My job is to compute $\mathbf{w} \cdot \mathbf{x} = \sum w_i x_i$ — the weighted sum of all packages!

#### Definition

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i
$$

#### Geometric Meaning

$$
\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos \theta
$$

where $\theta$ is the angle between the two vectors.

#### Connection to Neural Networks

**Weighted sum = inner product!**

$$
u = \mathbf{w} \cdot \mathbf{x} + b
$$

**Intuition**: The inner product measures the similarity of two vectors —

- If $\mathbf{w}$ and $\mathbf{x}$ are aligned (similar), the inner product is large → large output
- If $\mathbf{w}$ and $\mathbf{x}$ point in opposite directions (dissimilar), the inner product is small → small output

---

### 2-4-3 Vector Norms and Similarity

#### L2 Norm (Length)

$$
\|\mathbf{x}\| = \sqrt{\sum_{i=1}^{n} x_i^2}
$$

#### Cosine Similarity

$$
\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}
$$

Range $[-1, 1]$: 1 means completely aligned directions, -1 means completely opposite.

#### Python Practice

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Inner product
dot_product = np.dot(a, b)
print(f"Inner product a·b = {dot_product}")

# Norm
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
print(f"|a| = {norm_a:.4f}, |b| = {norm_b:.4f}")

# Cosine similarity
cos_sim = dot_product / (norm_a * norm_b)
print(f"Cosine similarity = {cos_sim:.4f}")
```

```output
Inner product a·b = 32
|a| = 3.7417, |b| = 8.7750
Cosine similarity = 0.9746
```

---

## 2-5 Matrix Basics

### 2-5-1 Definition of Matrices

A matrix $\mathbf{W} \in \mathbb{R}^{m \times n}$ is a 2D array with $m$ rows and $n$ columns.

```python
W = np.array([[1, 2, 3],
              [4, 5, 6]])  # Matrix with 2 rows, 3 columns
print(f"Shape: {W.shape}")  # (2, 3)
```

A matrix is a **rectangular table of numbers** arranged in rows and columns.

$$
\mathbf{W} = \begin{bmatrix} w_{11} & w_{12} & \dots & w_{1n} \\ w_{21} & w_{22} & \dots & w_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ w_{m1} & w_{m2} & \dots & w_{mn} \end{bmatrix}
$$

**Matrix dimensions**: $\mathbb{R}^{m \times n}$ means $m$ rows and $n$ columns.

In neural networks, the weight matrix $\mathbf{W}^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$ — each row corresponds to one neuron's weight vector.

> **Core Insight**: A matrix is **a group of organized, structured vectors** — each row is the weight vector of one neuron, and the whole matrix is the weight collection of all neurons in a layer!

---

### 2-5-2 Matrix Multiplication ⭐

#### Dimension Requirement

$$
(m \times n) \cdot (n \times p) \rightarrow (m \times p)
$$

**Key rule**: Number of columns of the left matrix = number of rows of the right matrix.

#### Computation Formula

$$
C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}
$$

#### Notes

- **Not commutative**: $\mathbf{AB} \neq \mathbf{BA}$
- **Associative**: $\mathbf{(AB)C} = \mathbf{A(BC)}$

---

### 2-5-3 Matrix Representation of Neural Network Propagation ⭐

#### One Layer Propagation

$$
\mathbf{u} = \mathbf{xW} + \mathbf{b}
$$

Dimension changes:

- $\mathbf{x}$: $(1 \times n)$ input vector
- $\mathbf{W}$: $(n \times m)$ weight matrix
- $\mathbf{b}$: $(1 \times m)$ bias vector
- $\mathbf{u}$: $(1 \times m)$ weighted sum output

#### Batch Processing Multiple Samples

```python
# 32 samples, each 784-dim → 128 hidden neurons
X = np.random.randn(32, 784)     # Input matrix
W = np.random.randn(784, 128)    # Weight matrix
b = np.zeros(128)                # Bias vector

# One matrix multiplication = compute all samples' weighted sums simultaneously
U = X @ W + b                    # Shape: (32, 128)
Z = np.maximum(0, U)             # ReLU activation
```

> **Core Insight**: Matrix multiplication enables "batch processing" — one matrix multiplication = simultaneously computing the weighted sums of all neurons for all samples. This is why neural networks can run efficiently on GPUs.

---

### 2-5-4 Visualizing Matrix Multiplication

```python
import numpy as np

# Matrix multiplication in a neural network
X = np.random.randn(32, 784)     # batch=32, input features=784
W1 = np.random.randn(784, 256)   # hidden layer weights
b1 = np.random.randn(256)        # hidden layer bias

# Forward propagation = matrix multiplication + broadcast addition
Z1 = X @ W1 + b1  # Result shape: (32, 256)
A1 = np.maximum(0, Z1)  # ReLU activation
```

> **Core Insight**: The essence of matrix multiplication is "batch vector inner products" — it compresses $n$ inner products in a for loop into a single matrix multiplication, allowing the GPU to compute them in parallel.

---

## 2-6 Derivative Basics

### 2-6-1 Definition of the Derivative

The **derivative** measures the instantaneous rate of change:

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

#### Intuitive Meaning

> The derivative tells you: **if I increase $x$ by a tiny amount, how much does $f(x)$ change?**

#### Connection to Neural Networks

- Derivatives tell us the "downhill direction" for gradient descent
- Each weight's derivative tells us how changing it affects the loss

### 2-6-2 Numerical Differentiation (Approximation)

We can approximate the derivative without calculus:

$$
f'(x) \approx \frac{f(x+h) - f(x)}{h}
$$

```python
def numerical_derivative(f, x, h=1e-5):
    """Numerical derivative using central difference"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Test
def f(x):
    return x**2

for x in [1.0, 2.0, 3.0]:
    approx = numerical_derivative(f, x)
    exact = 2 * x
    print(f"x={x}: approx={approx:.6f}, exact={exact:.6f}, error={abs(approx-exact):.2e}")
```

```output
x=1.0: approx=2.000000, exact=2.000000, error=1.01e-10
x=2.0: approx=4.000000, exact=4.000000, error=8.14e-11
x=3.0: approx=6.000001, exact=6.000000, error=1.09e-09
```

### 2-6-3 Basic Differentiation Rules

| Rule | Formula | Example |
|:-----|:--------|:--------|
| Constant | $\frac{d}{dx}c = 0$ | |
| Power | $\frac{d}{dx}x^n = nx^{n-1}$ | $\frac{d}{dx}x^2 = 2x$ |
| Exponential | $\frac{d}{dx}e^x = e^x$ | |
| Logarithm | $\frac{d}{dx}\ln x = \frac{1}{x}$ | |
| Sigmoid | $\sigma'(x) = \sigma(x)(1-\sigma(x))$ | |

### 2-6-4 Visualization

```python
x = np.linspace(-3, 3, 100)
y = x**2
dy = 2 * x  # analytical derivative

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', label='f(x) = x²')
plt.plot(x, dy, 'r-', label="f'(x) = 2x")
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('Function and Its Derivative')

plt.subplot(1, 2, 2)
# Tangent line at x=1
x0 = 1
y0 = x0**2
slope = 2 * x0
tangent = slope * (x - x0) + y0
plt.plot(x, y, 'b-', label='f(x) = x²')
plt.plot(x, tangent, 'r--', label=f'Tangent at x=1 (slope={slope})')
plt.plot(x0, y0, 'ro', markersize=8)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title('Geometric Meaning: Tangent Slope = Derivative')
plt.tight_layout()
plt.show()
```

![Figure 2-3: Function curve and its derivative (tangent slope). The blue line is the original function, the red dot is the tangent point, and green arrows indicate the sign and magnitude of the derivative.](../images/ch02/NN02_derivative_visual.png)
*Figure 2-3: Derivative = the instantaneous rate of change of a function = the slope of the tangent line.*

---

## 2-7 Partial Derivatives

### 2-7-1 Multivariate Functions

Neural networks have **many** inputs (weight parameters), so we need partial derivatives.

$$
f(x_1, x_2, \dots, x_n)
$$

A **partial derivative** measures the rate of change with respect to **one variable**, holding all others constant:

$$
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \dots, x_i + h, \dots, x_n) - f(x_1, \dots, x_n)}{h}
$$

### 2-7-2 The Gradient Vector ⭐

The **gradient** collects all partial derivatives into a vector:

$$
\nabla f = \begin{bmatrix}
\frac{\partial f}{\partial x_1} \\
\frac{\partial f}{\partial x_2} \\
\vdots \\
\frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

#### Gradient Points in the Direction of Steepest Ascent

This is the **single most important fact** for neural networks:

> The gradient $\nabla f$ points in the direction of **steepest increase**. Therefore, $-\nabla f$ points in the direction of **steepest decrease** (fastest way downhill).

This is why gradient descent works!

```python
# Gradient descent for a 2D quadratic bowl
def f(x, y):
    return x**2 + y**2

def gradient(x, y):
    df_dx = 2 * x
    df_dy = 2 * y
    return np.array([df_dx, df_dy])

# Initial point
pos = np.array([4.0, 3.0])
lr = 0.1
trajectory = [pos.copy()]

for _ in range(20):
    grad = gradient(pos[0], pos[1])
    pos = pos - lr * grad
    trajectory.append(pos.copy())

print(f"Start: [{trajectory[0][0]:.4f}, {trajectory[0][1]:.4f}]")
print(f"End:   [{trajectory[-1][0]:.4f}, {trajectory[-1][1]:.4f}]")
```

| | x | y |
|:---|---:|---:|
| Start | 4.0000 | 3.0000 |
| End | 0.0003 | 0.0002 |

### 2-7-3 Visualization

```python
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

# Plot trajectory
traj = np.array(trajectory)
ax.plot(traj[:, 0], traj[:, 1], traj[:, 0]**2 + traj[:, 1]**2,
        'r.-', markersize=10, linewidth=2, label='Gradient Descent Path')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.legend()
plt.show()
```

![Figure 2-4: 3D surface contour plot with gradient vector field. Arrows point in the direction of steepest ascent; the opposite direction points toward steepest descent.](../images/ch02/NN02_gradient_3d.png)
*Figure 2-4: Gradient vector field — arrow direction = direction of fastest increase; arrow length = magnitude of growth rate.*

---

## 2-8 The Chain Rule ⭐

### 2-8-1 Single-Variable Chain Rule

The chain rule tells us how to compute the derivative of a **composed function**:

If $y = f(g(x))$, then:

$$
\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
$$

#### Intuition

> If changing $x$ by 1 changes $g$ by 2 (i.e., $g'(x) = 2$), and changing $g$ by 1 changes $f$ by 3 (i.e., $f'(g) = 3$), then changing $x$ by 1 changes $f$ by $2 \times 3 = 6$.

### 2-8-2 Multi-Variable Chain Rule (The Heart of Neural Networks) ⭐

In neural networks, the loss $L$ depends on the weights through many layers:

$$
L = L(y), \quad y = \sigma(z^{(L)}), \quad z^{(L)} = W^{(L)}a^{(L-1)} + b^{(L)}, \quad \dots
$$

The chain rule for multiple variables:

$$
\frac{\partial L}{\partial w_{ji}^{(l)}} = \frac{\partial L}{\partial z_j^{(l)}} \cdot \frac{\partial z_j^{(l)}}{\partial w_{ji}^{(l)}}
$$

This single equation is the foundation of **backpropagation**.

> **Core Insight**: The chain rule allows us to "propagate" the error signal backward through the network. The error at the output layer is "chained" back through each layer, telling each weight how much it contributed to the error.

### 2-8-3 Computational Graph Concept

A **computational graph** visualizes the chain of operations:

1. **Forward pass** (left to right): x → Linear(z=wx+b) → Activation(a=σ(z)) → Loss(L=½(a-t)²) → L
2. **Backward pass** (right to left): ∂L/∂a → ∂L/∂z → ∂L/∂w, ∂L/∂b, ∂L/∂x

```python
# Visualizing the chain rule through a computational graph
import numpy as np

# Forward pass (computational graph)
x, t = 1.0, 0.0  # input, target
w, b = 0.5, 0.2

# Forward
z = w * x + b        # linear
a = 1 / (1 + np.exp(-z))  # sigmoid
L = 0.5 * (a - t)**2      # loss

print(f"Forward: x={x}, t={t}")
print(f"  z={z:.4f}, a={a:.4f}, L={L:.4f}")

# Backward (backpropagation = chain rule)
dL_da = a - t                     # ∂L/∂a
da_dz = a * (1 - a)               # ∂a/∂z (sigmoid derivative)
dz_dw = x                         # ∂z/∂w
dz_db = 1.0                       # ∂z/∂b

# Chain rule
dL_dw = dL_da * da_dz * dz_dw     # ∂L/∂w
dL_db = dL_da * da_dz * dz_db     # ∂L/∂b

print(f"\nBackward (chain rule):")
print(f"  dL/da={dL_da:.4f}, da/dz={da_dz:.4f}")
print(f"  dL/dw={dL_dw:.4f}, dL/db={dL_db:.4f}")
```

Forward: x=1.0, t=0.0, z=0.7000, a=0.6682, L=0.2232

| Gradient | Value |
|:---------|------:|
| ∂L/∂w | 0.1481 |
| ∂L/∂b | 0.1481 |

![Figure 2-5: Computational graph of the chain rule. Forward propagation (green) goes left to right; backward propagation (red) goes right to left.](../images/ch02/NN02_chain_rule_graph.png)
*Figure 2-5: The computational graph of the chain rule. Forward propagation (green) from left to right, backward propagation (red) from right to left.*

---

## 2-9 Approximation Formulas for Multivariate Functions

### 2-9-1 Taylor Expansion

The Taylor expansion approximates a function near a point:

$$
f(x + \Delta x) \approx f(x) + f'(x)\Delta x + \frac{1}{2}f''(x)(\Delta x)^2 + \dots
$$

**First-order approximation** (used in gradient descent):

$$
f(x + \Delta x) \approx f(x) + f'(x)\Delta x
$$

![Figure 2-6: Taylor expansion approximating sin(x). 0th order = constant, 1st order = linear, 3rd/5th/7th orders gradually approach the original function.](../images/ch02/NN02_taylor_approx.png)
*Figure 2-6: Taylor expansion approximating sin(x) at different orders.*

### 2-9-2 Total Differential (Multivariate)

For multivariate functions:

$$
df = \frac{\partial f}{\partial x_1}dx_1 + \frac{\partial f}{\partial x_2}dx_2 + \cdots + \frac{\partial f}{\partial x_n}dx_n
$$

#### Matrix Form

$$
df = \nabla f^{\mathsf{T}} \cdot dx
$$

This linear approximation tells us: if we change the weights slightly, how much will the loss change?

---

## 2-10 Gradient Descent ⭐

### 2-10-1 Algorithm

1. **Compute gradient**: $\nabla L(W) = \frac{\partial L}{\partial W}$
2. **Update**: $W^{(t+1)} = W^{(t)} - \eta \nabla L(W^{(t)})$
3. **Repeat** until convergence

### 2-10-2 The Role of the Learning Rate

The learning rate $\eta$ controls step size:

- **Too large**: may overshoot the minimum
- **Too small**: slow convergence
- **Just right**: efficient convergence

### 2-10-3 Gradient Descent from Scratch

```python
import numpy as np

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# Gradient descent for linear regression
w = np.random.randn(1, 1)
b = np.zeros((1, 1))
lr = 0.1

for epoch in range(100):
    # Forward: prediction
    y_pred = X @ w.T + b

    # Loss: MSE
    loss = np.mean((y_pred - y)**2)

    # Gradient (chain rule!)
    grad_w = np.mean(2 * (y_pred - y) * X, axis=0)
    grad_b = np.mean(2 * (y_pred - y), axis=0)

    # Update
    w -= lr * grad_w
    b -= lr * grad_b

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}: loss={loss:.6f}, w={w[0,0]:.4f}, b={b[0,0]:.4f}")

print(f"\nTrue: w=2.0, b=1.0")
print(f"Learned: w={w[0,0]:.4f}, b={b[0,0]:.4f}")
```

---

### 2-10-4 Python Practice: Gradient Descent Implementation

```python
def gradient_descent(f, df, x0, lr=0.1, epochs=100):
    x = x0
    history = [x]
    for i in range(epochs):
        x = x - lr * df(x)
        history.append(x)
    return x, history

# Test: f(x) = x**2, minimum at x=0
f = lambda x: x**2
df = lambda x: 2*x
x_opt, history = gradient_descent(f, df, x0=5.0, lr=0.1, epochs=20)
print(f"Optimal: x = {x_opt:.6f}")
```

```output
Optimal: x = 0.000000
Trajectory: ['5.00', '4.00', '3.20', '2.56', '2.05', ...]
```

---

### 2-10-5 Visualization: Learning Rate Effects

![Figure 2-7: Comparing gradient descent with different learning rates.](../images/ch02/NN02_learning_rate_compare.png)

*Figure 2-7: Learning rate comparison — small is slow, large diverges.*

---

## 2-11 Understanding Automatic Differentiation (Autograd)

### 2-11-1 Manual vs. Automatic Differentiation

| Method | Pros | Cons |
|:-------|:-----|:------|
| Manual derivation | Exact, educational | Error-prone, tedious |
| Numerical approx | Simple | Slow, precision issues |
| **Autograd** | Exact, fast, convenient | Less transparent |

### 2-11-2 PyTorch Autograd Demo

```python
import torch

# Create tensors with gradient tracking
x = torch.tensor([1.0], requires_grad=True)
w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.2], requires_grad=True)
t = torch.tensor([0.0])  # target

# Forward pass (PyTorch tracks the graph automatically)
z = w * x + b
a = torch.sigmoid(z)
loss = 0.5 * (a - t)**2

# Backward pass (automatic!)
loss.backward()

print(f"Manual:   dL/dw={0.1481:.4f}")
print(f"Autograd: dL/dw={w.grad.item():.4f}")
```

### 2-11-3 Visualization

The computational graph PyTorch builds internally:

Forward graph: (x,w) → multiply → add(+b) → sigmoid → (subtract t, square, mean) → L
Backward: gradients flow in reverse through the same nodes using the chain rule.

![Figure 2-8: 2D gradient descent path. Left: contour map with trajectory path; Right: coordinate values changing with iterations.](../images/ch02/NN02_gd_2d_path.png)
*Figure 2-8: 2D gradient descent path. Red arrows show the direction and magnitude of each update step.*

---

## 2-12 Optimization Problems and Regression

### 2-12-1 Least Squares Method

Least squares finds the optimal line $y = wx + b$ that minimizes:

$$
L = \sum_{i=1}^{m} (y_i - (wx_i + b))^2
$$

### 2-12-2 Python: Analytical Solution vs. Gradient Descent

```python
import numpy as np

# Data
X = np.array([[1], [2], [3], [4]])
y = np.array([[2], [4], [6], [8]])

# Analytical solution: Normal equation
X_design = np.hstack([X, np.ones_like(X)])
theta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
print(f"Analytical: w={theta[0,0]:.4f}, b={theta[1,0]:.4f}")

# Gradient descent
w = np.random.randn()
b = np.random.randn()
lr = 0.01

for _ in range(1000):
    y_pred = w * X + b
    grad_w = np.mean(2 * (y_pred - y) * X)
    grad_b = np.mean(2 * (y_pred - y))
    w -= lr * grad_w
    b -= lr * grad_b

print(f"GD:        w={w[0]:.4f}, b={b[0]:.4f}")
```

### 2-12-3 Comparison

| Method | Advantages | Disadvantages |
|:-------|:-----------|:--------------|
| **Normal Equation** | One-shot, exact | $O(n^3)$, can't handle large $n$ |
| **Gradient Descent** | Scales to millions of params | Requires tuning learning rate |
| **SGD** | Handles huge datasets | Noisy convergence |

---

### 2-12-4 Visualization: Linear Regression Fitting

![Figure 2-9: Linear regression fitting result. Scatter points are data points, the line is the fitted result.](../images/ch02/NN02_linear_regression.png)

*Figure 2-9: Linear regression — fitting a line to data points.*

---

## 2-14 Practice: Understanding Neural Networks with Math

### 2-14-1 Forward Propagation from a Mathematical Perspective

Each layer of a neural network can be viewed as a **function composition**:

$$
f_{\text{NN}}(\mathbf{x}) = f^{(L)} \circ f^{(L-1)} \circ \dots \circ f^{(1)}(\mathbf{x})
$$

Where each layer function $f^{(l)}(\mathbf{a}) = \sigma(\mathbf{W}^{(l)}\mathbf{a} + \mathbf{b}^{(l)})$ is an **affine transformation + nonlinear activation**.

### 2-14-2 Common Function Composition Patterns

| Pattern | Mathematical Form | Where It Appears |
|:--------|:-----------------|:----------------|
| **Linear->Nonlinear** | $\sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$ | All hidden layers |
| **Linear->Probability** | $\text{softmax}(\mathbf{W}\mathbf{x} + \mathbf{b})$ | Multi-class output |
| **Linear->Scalar** | $\mathbf{w}^T\mathbf{x} + b$ | Regression output |
| **Residual Connection** | $\mathbf{x} + F(\mathbf{x})$ | ResNet-style architectures |

### 2-14-3 NumPy Forward Pass Implementation

```python
import numpy as np

def forward_pass(X, weights, biases, activation='relu'):
    a = X
    for i, (W, b) in enumerate(zip(weights, biases)):
        z = a @ W + b
        if i < len(weights) - 1:  # hidden layers
            if activation == 'relu':
                a = np.maximum(0, z)
            elif activation == 'sigmoid':
                a = 1 / (1 + np.exp(-z))
        else:
            a = z
    return a

# Test: 2-layer network
X = np.random.randn(10, 784)
W1 = np.random.randn(784, 256) * 0.01
b1 = np.zeros(256)
W2 = np.random.randn(256, 10) * 0.01
b2 = np.zeros(10)
output = forward_pass(X, [W1, W2], [b1, b2])
print(f"Input: {X.shape} -> Output: {output.shape}")
```

```output
Input: (10, 784) -> Output: (10, 10)
```

### 2-14-4 Matrix Calculus for Backpropagation

Backpropagation is fundamentally an application of **matrix calculus**:

$$
\frac{\partial}{\partial \mathbf{W}} (\mathbf{W}\mathbf{x} + \mathbf{b}) = \mathbf{x}^T
$$

$$
\frac{\partial}{\partial \mathbf{b}} (\mathbf{W}\mathbf{x} + \mathbf{b}) = \mathbf{I}
$$

| Operation | Forward | Weight Gradient | Input Gradient |
|:---------|:--------|:---------------|:--------------|
| **Affine** | $\mathbf{z} = \mathbf{W}\mathbf{a} + \mathbf{b}$ | $\frac{\partial L}{\partial \mathbf{W}} = \frac{\partial L}{\partial \mathbf{z}} \cdot \mathbf{a}^T$ | $\frac{\partial L}{\partial \mathbf{a}} = \mathbf{W}^T \cdot \frac{\partial L}{\partial \mathbf{z}}$ |
| **ReLU** | $\mathbf{a} = \max(0, \mathbf{z})$ | - | $\frac{\partial L}{\partial \mathbf{z}} = \mathbb{1}[\mathbf{z} > 0] \odot \frac{\partial L}{\partial \mathbf{a}}$ |
| **Sigmoid** | $\mathbf{a} = \sigma(\mathbf{z})$ | - | $\frac{\partial L}{\partial \mathbf{z}} = \sigma(\mathbf{z}) \odot (1-\sigma(\mathbf{z})) \odot \frac{\partial L}{\partial \mathbf{a}}$ |

---

## 2-15 From Math to Code: Python Implementation of Core Formulas

### 2-15-1 Vectorization for Speed

```python
import numpy as np
import time

# Loop implementation (slow)
def dot_product_loop(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

# Vectorized implementation (much faster)
def dot_product_vec(a, b):
    return np.dot(a, b)

# Performance comparison
a = np.random.randn(10000)
b = np.random.randn(10000)

t0 = time.time()
for _ in range(1000):
    dot_product_loop(a, b)
t1 = time.time()
print(f"Loop: {(t1-t0)*1000:.1f}ms")

t0 = time.time()
for _ in range(1000):
    dot_product_vec(a, b)
t1 = time.time()
print(f"Vectorized: {(t1-t0)*1000:.1f}ms")
```

> **Key Insight**: Vectorization with NumPy is typically **50-100x faster** than Python loops. This is why neural networks use matrix operations.

### 2-15-2 Complete Code Overview

```python
import numpy as np

# 1. Vectors and Matrices
v = np.array([1, 2, 3])
M = np.array([[1, 2], [3, 4]])
dot = np.dot(v, v)

# 2. Numerical Derivative
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

print(f"Derivative of x**2 at x=3: {numerical_derivative(lambda x: x**2, 3):.4f}")

# 3. Gradient Descent
def gradient_descent(df, x0, lr=0.1, steps=100):
    x = x0
    for _ in range(steps):
        x -= lr * df(x)
    return x

x_min = gradient_descent(lambda x: 2*x, 5.0)
print(f"GD finds x ~ {x_min:.2f} (true min at 0)")

# 4. Normal Equation (Least Squares)
X = np.random.randn(100, 3)
w = np.random.randn(3, 1)
y = X @ w + 0.1 * np.random.randn(100, 1)
w_hat = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"Least squares error = {np.mean((X@w_hat - y)**2):.6f}")
```

> **Core Message**: Everything in this chapter - vectors, matrices, derivatives, gradients - comes together in $y = \mathbf{W}\mathbf{x} + \mathbf{b}$. This single formula, repeated across layers, is what makes neural networks work.

---

## 📦 Chapter Code List

| File | Content | Key Concept |
|:-----|:--------|:------------|
| `ch02/NN02_functions.py` | Linear, quadratic, exponential, logarithmic functions | Function visualization |
| `ch02/NN02_vectors.py` | Vector dot product, norms, cosine similarity | Vector operations |
| `ch02/NN02_matrices.py` | Matrix multiplication, batch forward pass | Matrix operations |
| `ch02/NN02_derivatives.py` | Numerical & analytical derivatives | Derivative computation |
| `ch02/NN02_chain_rule.py` | Chain rule computational graph | Chain rule |
| `ch02/NN02_gradient_descent.py` | Gradient descent on quadratic functions | GD implementation |
| `ch02/NN02_least_squares.py` | Normal equation vs gradient descent | Least squares |
| `ch02/NN02_autograd_demo.py` | PyTorch autograd introduction | Autograd basics |

---

## 📖 Chapter Summary

### Core Concepts Review

| Category | Key Concepts |
|:---------|:-------------|
| Functions | Linear, quadratic, exponential, logarithmic |
| Linear Algebra | Dot product, matrix multiplication, batch computation, forward propagation |
| Calculus | Derivative, partial derivative, gradient, **chain rule** ⭐, gradient descent |

### Math Tools ↔ Neural Networks Mapping

| Math Concept | Role in Neural Networks |
|:-------------|:------------------------|
| Linear function $y = wx + b$ | Neuron's weighted sum |
| Quadratic function $y = x^2$ | MSE loss function |
| Exponential $e^x$ | Sigmoid / Softmax |
| Logarithm $\ln x$ | Cross-entropy loss |
| Vector dot product | Single neuron computation |
| Matrix multiplication | Batch / layer computation |
| Derivative $f'(x)$ | Sensitivity measure |
| Gradient $\nabla f$ | Downhill direction |
| Chain rule | Backpropagation |
| Taylor expansion | Gradient descent theory |

### 🧪 Exercises

#### Exercise 1: Dot Product Practice

Given $x = [1, 2, 3]^{\mathsf{T}}$ and $y = [4, 5, 6]^{\mathsf{T}}$, compute $x \cdot y$ manually, then verify with `np.dot()`.

#### Exercise 2: Matrix Multiplication

For $A \in \mathbb{R}^{2 \times 3}$ and $B \in \mathbb{R}^{3 \times 2}$, perform $C = AB$. Verify the dimensions.

```python
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])
C = A @ B
print(f"C shape: {C.shape}")
print(C)
```

#### Exercise 3: Numerical Derivative

Use the numerical derivative formula to compute the derivative of $f(x) = x^3$ at $x = 2$. Compare with the analytical result.

#### Exercise 4: Chain Rule Practice

If $f(x) = (2x + 1)^3$, use the chain rule to find $f'(x)$. Verify with numerical differentiation.

#### Exercise 5 (Challenge): Manual 2-Layer Network Gradient

For a 2-layer network, manually compute the gradient of the loss with respect to all weights using the chain rule. Compare with PyTorch's autograd.

### Core Formula Quick Reference

| Concept | Formula | Meaning |
|:--------|:--------|:--------|
| Dot product | $x \cdot y = \sum x_i y_i$ | Neuron weighted sum |
| Matrix mult | $C = AB$ | Layer transformation |
| Layer forward | $z = Wx + b$ | Linear transform |
| Derivative | $f'(x) = \lim_{h\to 0} \frac{f(x+h)-f(x)}{h}$ | Rate of change |
| Gradient | $\nabla f = [\partial f/\partial x_1, \dots]^{\mathsf{T}}$ | Steepest direction |
| Chain rule | $\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$ | Backpropagation |
| GD update | $w^{(t+1)} = w^{(t)} - \eta \frac{\partial L}{\partial w}$ | Parameter update |

← [Chapter 1](01-chapter1-neural-network-ideas.md) | [Table of Contents](README.md) | [Chapter 3](03-chapter3-pytorch-basics-tensor-autograd.md) →
