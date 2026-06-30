# Chapter 1: The Idea of Neural Networks

> **Goal**: Starting from biological neurons, **intuitively understand** how artificial neurons make mathematical decisions — from weighted input to activated output, verified step by step with code.

> © xiefujin · Contact: 490021684@qq.com · Licensed under CC BY-NC-SA 4.0
>
> **Code**: `../code/ch01/` (5 files)

> **Figures**: `../images/ch01/` (3 visualization images)

---

## 📋 Chapter Learning Objectives

- [ ] Understand the correspondence between biological and artificial neurons
- [ ] Master the M-P neuron model and its mathematical representation
- [ ] Understand why continuous, differentiable activation functions are needed
- [ ] Understand the differences between 4 common activation functions
- [ ] Understand the three essential elements of neural networks: input layer, hidden layer, output layer
- [ ] Be able to build a simple 2-layer network in code
- [ ] Understand that "learning = parameter optimization"

---

## 1-1 Neural Networks and Deep Learning

### 1-1-1 The Background of Deep Learning

#### Three Waves of AI

Artificial intelligence has undergone three major waves:

| Period | Wave | Core Idea | Milestone |
|:-------|:-----|:----------|:----------|
| 1950s-1960s | **Symbolism** | Logical reasoning, symbolic computation | Logic Theorist, Expert Systems |
| 1980s-1990s | **Statistical Learning** | Data-driven, probabilistic modeling | SVM, Random Forests |
| 2010s-present | **Deep Learning** | End-to-end learning, representation learning | AlexNet, Transformer, GPT |

#### Why Now?

Deep learning exploded in the 2010s, driven by three factors:

1. **Data**: The internet generated massive labeled datasets (ImageNet: 14 million images)
2. **Compute**: GPU massive parallel computing made training deep networks feasible
3. **Algorithms**: Three breakthroughs — Backpropagation + Gradient Descent + ReLU

#### Deep Learning Application Landscape

| Domain | Applications |
|:-------|:-------------|
| Computer Vision (CV) | Image classification, object detection, face recognition |
| Natural Language Processing (NLP) | Machine translation, sentiment analysis, dialogue systems |
| Speech Recognition | Speech-to-text, text-to-speech |
| Recommendation Systems | Short video recommendations, product recommendations |
| Reinforcement Learning | Game AI, robot control, autonomous driving |

---

### 1-1-2 Starting with a Simple Example

#### Problem: Predicting House Prices by Square Footage

Suppose you are a real estate agent and you notice a relationship between house area and sale price:

| Area (m²) | Price (10,000 yuan) |
|:---------:|:-------------------:|
| 50 | 150 |
| 80 | 230 |
| 100 | 300 |
| 120 | 360 |

#### Three Approaches

| Method | Approach | Characteristics |
|:-------|:---------|:----------------|
| **Human Intuition** | ~30,000 yuan per m², rough estimate | Crude, unstable |
| **Mathematical Modeling** | Fit a straight line with linear regression $y = wx + b$ | Precise, but requires manual design |
| **Neural Network** | Let the network learn $w$ and $b$ automatically | General-purpose, scales to complex problems |

#### Warm-up: Experience a Neural Network with One Line of PyTorch

```python
import torch
import torch.nn as nn

# One neuron = a linear layer
neuron = nn.Linear(in_features=1, out_features=1)

# Input: area 100 m²
area = torch.tensor([[100.0]], dtype=torch.float32)

# Forward propagation
price = neuron(area)
print(f"Predicted price: {price.item():.2f} ten thousand yuan")

```

```output
Predicted price: 162.34 ten thousand yuan

```

> **Tip**: This result is not necessarily accurate — because the weights are randomly initialized.
>
> By the end of this chapter, you will understand exactly what mathematical computation this "neuron" is doing internally.

---

## 1-2 The Mathematical Representation of a Neuron

### 1-2-1 Inspiration from Biological Neurons

#### Structure of a Biological Neuron

1. **Dendrites** (receive signals) — receive electrical signals from the previous neuron
2. **Cell Body** (integrate signals) — sum + check whether threshold is exceeded
3. **Axon** (output signal) — if the threshold is exceeded, fire a pulse
4. **Synapse** (transmit to the next neuron)

#### Key Characteristics

- **"All-or-None" Law**: Fire if above threshold (emit a pulse), otherwise remain silent
- **Synaptic Plasticity**: Connection strengths (weights) can change with learning
- **Parallel Processing**: The brain has ~86 billion neurons, highly parallel

#### Mathematical Abstraction

> Input x₁, x₂, ... → weighted sum Σwᵢxᵢ → activation function f(·) → output y

---

### 1-2-2 The McCulloch-Pitts Model (M-P Model)

In 1943, Warren McCulloch and Walter Pitts proposed the first mathematical model of a neuron.

#### Mathematical Formula

**Decision function**: when $\sum w_i x_i \geq \theta$, output $1$; otherwise, output $0$.

#### Symbol Legend

| Symbol | Meaning | Biological Analogy |
|:------|:--------|:-------------------|
| $x_i$ | Input signal (0 or 1) | Electrical signal received by dendrites |
| $w_i$ | Synaptic weight (positive = excitatory, negative = inhibitory) | Synaptic connection strength |
| $\theta$ | Threshold | Firing threshold of the cell body |
| $y$ | Output (0 or 1) | Whether the axon fires a pulse |

---

### 1-2-3 Verifying the M-P Neuron with Code

#### Python Implementation

```python
import numpy as np

class MPNeuron:
    """McCulloch-Pitts neuron model"""

    def __init__(self, weights, threshold):
        self.w = np.array(weights)
        self.threshold = threshold

    def forward(self, x):
        """Forward propagation: weighted sum → threshold decision"""
        u = np.dot(self.w, x)          # Weighted sum: Σ wi × xi
        return 1 if u >= self.threshold else 0  # Threshold decision

```

#### Core Idea

$$
u = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n = \sum_{i=1}^{n} w_i x_i
$$

**Decision function**: output $1$ when $u \geq \theta$, output $0$ when $u < \theta$.

---

### 1-2-4 Worked Examples: AND / OR Logic Gates

#### Implementing the AND Gate

Truth table for AND: output is 1 only when both inputs are 1.

| $x_1$ | $x_2$ | $y_{\text{AND}}$ |
|:-----:|:-----:|:----------------:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

```python
# Implement AND gate: output 1 when x1 + x2 >= 2
and_neuron = MPNeuron(weights=[1, 1], threshold=2)

print("AND gate test:")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        y = and_neuron.forward([x1, x2])
        print(f"  {x1} AND {x2} = {y}")

```

```output
AND gate test:
  0 AND 0 = 0
  0 AND 1 = 0
  1 AND 0 = 0
  1 AND 1 = 1

```

#### Implementing the OR Gate

Truth table for OR: output is 1 as long as at least one input is 1.

```python
# Implement OR gate: output 1 when x1 + x2 >= 1
or_neuron = MPNeuron(weights=[1, 1], threshold=1)

print("OR gate test:")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        y = or_neuron.forward([x1, x2])
        print(f"  {x1} OR {x2} = {y}")

```

```output
OR gate test:
  0 OR 0 = 0
  0 OR 1 = 1
  1 OR 0 = 1
  1 OR 1 = 1

```

#### Why Can't It Solve XOR?

Truth table for XOR: output is 1 when the two inputs differ, 0 when they are the same.

| $x_1$ | $x_2$ | $y_{\text{XOR}}$ |
|:-----:|:-----:|:----------------:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Visualization**: Plot the four points on a 2D plane, coloring by output.
| Point | Coordinates | XOR Output |
|:-----:|:-----------:|:----------:|
| ● | (0,0) | 0 |
| ○ | (0,1) | 1 |
| ○ | (1,0) | 1 |
| ● | (1,1) | 0 |

**Why can't a single straight line separate them?**

Try drawing a straight line to separate ○ (should output 1) from ● (should output 0):

- If you connect top-left to bottom-right, top-right and bottom-left get mixed together
- If you connect bottom-left to top-right, top-left and bottom-right get mixed together

**Mathematical explanation**: The XOR problem is not linearly separable in 2D — no single straight line $w_1 x_1 + w_2 x_2 = \theta$ can correctly classify all four points.

**Solution**: You need **two layers** of M-P neurons! Combine AND, OR, and NAND to compute an intermediate result:

Layer 1: h₁ = NAND(x₁, x₂),  h₂ = OR(x₁, x₂)
Layer 2: y = AND(h₁, h₂) = x₁ XOR x₂

This is the fundamental reason neural networks need **multi-layer structures**.

> **Core Insight**: The XOR problem is the "Achilles' heel" of the M-P neuron — it reveals the fundamental limitation of single-layer models: they can only handle linearly separable problems. The only way to overcome this limitation is by **stacking multiple layers**, which is the very origin of neural networks.

> **Core Insight**: A single M-P neuron can only solve **linearly separable** problems.
>
> XOR requires multiple layers of neurons — this is exactly the starting point of **neural networks**.

---

## 1-3 Activation Functions: Generalizing Neuron Behavior

### 1-3-1 From Step Function to Continuous Functions

#### The Step Function

The decision function used by the M-P model is exactly the step function:

$$
f(x) = \mathbb{I}(x \ge 0)
$$

#### Problems with the Step Function

The step function is **non-differentiable** at $x=0$ (the derivative does not exist), and its derivative is 0 everywhere else.

```
Step function: when x ≥ 0, f(x)=1; when x < 0, f(x)=0. It jumps at x=0, which is non-differentiable.

```

> **Why is "non-differentiable" a big problem?**
>
> Later on, we will use **gradient descent** to train neural networks — and gradient descent requires computing derivatives.
>
> If the activation function is non-differentiable, gradient descent cannot work.

#### The Solution: Finding a "Smooth Version of the Step Function"

We need a function that satisfies:

1. **Continuously differentiable** (has a derivative everywhere)
2. **S-shaped** (similar shape to the step function)
3. **Range in (0, 1)** (interpretable as a probability)

This is the **Sigmoid function**.

---

### 1-3-2 Understanding Sigmoid: Why Do We Need Smooth Activation Functions?

#### Mathematical Definition

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

#### Important Derivative Formula

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

This derivative formula is extremely important — you will use it repeatedly later in backpropagation.

#### Characteristics

| Property | Description |
|:---------|:------------|
| **Range** | (0, 1), interpretable as a probability |
| **Monotonicity** | Monotonically increasing |
| **Continuous differentiability** | Differentiable everywhere |
| **Saturation region** | When $\vert x\vert$ is large, gradient approaches 0 (vanishing gradient problem) |

#### Intuitive Understanding

> Sigmoid is like a "soft switch":
>
> - When input is large → output approaches 1 (switch on)
> - When input is small → output approaches 0 (switch off)
> - In the middle region → output transitions smoothly

---

### 1-3-3 Understanding ReLU: Why Is It More Commonly Used Than Sigmoid?

ReLU (Rectified Linear Unit) is currently the most commonly used activation function.

#### Mathematical Definition

$$
\text{ReLU}(x) = \max(0, x)
$$

#### Derivative

**ReLU derivative**: $1$ when $x > 0$, $0$ when $x \leq 0$.

#### Why Is ReLU More Popular Than Sigmoid?

| Comparison | Sigmoid | ReLU |
|:-----------|:--------|:-----|
| Computational complexity | Requires exponential operation | Only needs a max operation |
| Vanishing gradient | Saturated at both ends (gradient → 0) | Positive half-plane gradient always 1 |
| Convergence speed | Slow | Fast (~6×) |
| Output range | (0, 1) | [0, +∞) |

#### Leaky ReLU Variant

To solve the "dead neuron" problem in ReLU's negative half-plane, Leaky ReLU gives the negative half-plane a very small slope:

$$
\text{LeakyReLU}(x) = \max(0.01x, x)
$$

---

### 1-3-4 Understanding Tanh: What Are the Benefits of a Zero-Centered Activation?

#### Mathematical Definition

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

#### Relationship with Sigmoid

$$
\tanh(x) = 2\sigma(2x) - 1
$$

#### Characteristics

| Property | Description |
|:---------|:------------|
| **Range** | (-1, 1), zero-centered |
| **Advantage** | Output has mean 0, which helps the next layer learn |
| **Disadvantage** | Also has saturation regions (vanishing gradient) |

> **Little Genius says**: Tanh is like Sigmoid's "upgraded version" — it not only tells you how strong the signal is (positive/negative), but also **centers** the signal near 0, making the signals received by the next layer's Little Geniuses more balanced.

---

### 1-3-5 Code Verification: Visualizing the Four Activation Functions with Python

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def step(x):
    return (x >= 0).astype(float)

# Generate data
x = np.linspace(-5, 5, 1000)

# Four activation functions and their derivatives
activations = {
    'Step Function': (step, None),
    'Sigmoid': (sigmoid, sigmoid_derivative),
    'Tanh': (tanh, tanh_derivative),
    'ReLU': (relu, relu_derivative),
}

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (name, (func, deriv)) in zip(axes.flat, activations.items()):
    ax.plot(x, func(x), 'b-', linewidth=2, label=name)
    if deriv is not None:
        ax.plot(x, deriv(x), 'r--', linewidth=1.5, label='Derivative')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title(name)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../images/ch01/NN01_activation_functions.png', dpi=150)
plt.show()

```

![Figure 1-1: Comparison of four activation functions and their derivatives. Blue = function, red = derivative (step function is non-differentiable).](../images/ch01/NN01_activation_functions.png)

*Figure 1-1: Comparison of four activation functions and their derivatives. Blue = function, red = derivative (step function is non-differentiable).*

> **Core Insight**: The choice of activation function determines the network's expressive power.
>
> **Rule of thumb**: Use ReLU in hidden layers, Sigmoid (binary classification) / Softmax (multi-class classification) in the output layer.

---

## 1-4 What Are Neural Networks

### 1-4-1 Three Essential Network Elements

A standard neural network consists of three parts:

#### Input Layer

- Receives raw data (feature vectors)
- Does no computation, just "passes the signal through"
- Number of nodes = feature dimensionality

#### Hidden Layer

- Feature extraction and transformation
- Can have one or multiple layers (this is the source of "depth")
- Each layer performs: weighted sum → activation function

#### Output Layer

- Final prediction result
- Number of nodes depends on the task type
  - Binary classification: 1 node (Sigmoid)
  - Multi-class classification: K nodes (Softmax)
  - Regression: 1 node (no activation function)

| Layer | Nodes | Role |
|:-----:|:------|:-----|
| **Input Layer** | ○ ○ ○ | Receives raw data, no computation |
| **Hidden Layer** | ○ ○ ○ ... | Feature extraction and transformation (multiple layers stacked = depth) |
| **Output Layer** | ○ | Final prediction result |

---

### 1-4-2 The Meaning of Dense (Fully Connected) Layers

#### What Is "Fully Connected"?

Every neuron in one layer is connected to **all** neurons in the previous layer. For example, inputs $x_1, x_2$ fully connected to hidden layer $h_1, h_2, h_3$:

$x_1$ connects to all hidden neurons (via $w_{11}, w_{12}, w_{13}$)
$x_2$ connects to all hidden neurons (via $w_{21}, w_{22}, w_{23}$)

Each hidden neuron $h_j$ receives: $u_j = w_{1j}x_1 + w_{2j}x_2 + b_j$

#### Mathematical Expression

$$
\mathbf{z}^{(l+1)} = f(\mathbf{W}^{(l)} \mathbf{z}^{(l)} + \mathbf{b}^{(l)})
$$

| Symbol | Meaning | Shape |
|:------|:--------|:------|
| $\mathbf{z}^{(l)}$ | Output of layer $l$ | $(n_{in},)$ |
| $\mathbf{W}^{(l)}$ | Weight matrix | $(n_{in}, n_{out})$ |
| $\mathbf{b}^{(l)}$ | Bias vector | $(n_{out},)$ |
| $f$ | Activation function | Element-wise operation |

#### Parameter Count Calculation

For a fully connected layer:

$$
\text{Parameter count} = (n_{in} \times n_{out}) + n_{out}
$$

- $n_{in} \times n_{out}$ weight parameters
- $n_{out}$ bias parameters

---

### 1-4-3 From a Single Neuron to a Neural Network

#### A Single Neuron

$$
y = f\left(\sum_{i=1}^{n} w_i x_i + b\right)
$$

#### One Layer of Neurons (Vector Form)

$$
\mathbf{y} = f(\mathbf{xW} + \mathbf{b})
$$

Where:

- $\mathbf{x} = [x_1, x_2, \ldots, x_n]$ is a $1 \times n$ input row vector
- $\mathbf{W}$ is an $n \times m$ weight matrix
- $\mathbf{b} = [b_1, b_2, \ldots, b_m]$ is a bias row vector
- $\mathbf{y} = [y_1, y_2, \ldots, y_m]$ is the output of $m$ neurons

> **Core Insight**: One layer of neurons = one matrix multiplication + one element-wise activation function.

#### Multi-Layer Stacking (Depth)

The transformation at each layer: hₗ = fₗ(hₗ₋₁ Wₗ + bₗ)

Stacking three layers: x → W₁,b₁ → h₁ → W₂,b₂ → h₂ → W₃,b₃ → y

> **The source of depth**: Each layer performs feature transformation; multiple stacked layers can learn progressively more abstract features.

---

### 1-4-4 Understanding a 2-Layer Network: From Math Formulas to Python Code

#### Mathematical Expression

$$
\mathbf{h} = \sigma(\mathbf{xW}_1 + \mathbf{b}_1)
$$

$$
\mathbf{y} = \sigma(\mathbf{hW}_2 + \mathbf{b}_2)
$$

#### Python Implementation

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class TwoLayerNetwork:
    """Manual implementation of a 2-layer fully connected network"""

    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights (small random numbers)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros(output_size)

    def forward(self, x):
        """Forward propagation"""
        # Layer 1: input → hidden
        self.z1 = sigmoid(np.dot(x, self.W1) + self.b1)
        # Layer 2: hidden → output
        self.y = sigmoid(np.dot(self.z1, self.W2) + self.b2)
        return self.y

    def predict(self, x):
        """Prediction (binary classification)"""
        prob = self.forward(x)
        return 1 if prob >= 0.5 else 0

# Test the network
np.random.seed(42)
net = TwoLayerNetwork(input_size=3, hidden_size=4, output_size=1)

# Random input
x = np.array([0.5, 0.3, 0.8])
output = net.forward(x)
print(f"Network output: {output[0]:.4f}")

```

> **Little Genius says**: I'm the little messenger standing between two layers of neurons!
>
> With my left hand I receive signals from the previous layer, and with my right hand I weight-sum them, activate them, and pass them on to the next layer.

---

## 1-5 Understanding Network Structure with Little Geniuses ✨

### 1-5-1 Character Roles of the Little Geniuses

#### Every Connection Line Is a Little Genius

In a fully connected network, every connection line is managed by a Little Genius. Take hidden neuron $j$ as an example — it receives signals from ALL input neurons simultaneously:

> 🧚 **Genius A** holds number $w_{1j}=0.5$, amplifies $x_1$ by 0.5× and passes to neuron $j$
> 🧚 **Genius B** holds number $w_{2j}=-0.3$, flips $x_2$ and passes to neuron $j$
> 🧚 **Genius C** holds number $w_{3j}=0.8$, amplifies $x_3$ by 0.8× and passes to neuron $j$

Each Little Genius's job: **receive the signal from the previous step, multiply by their assigned weight, and pass to the next step**.

Neuron $j$ sums all incoming signals: $u_j = 0.5x_1 + (-0.3)x_2 + 0.8x_3 + b_j$

Scaling up to a full layer, fully connected means **every input connects to every hidden neuron**:

| | $h_1$ | $h_2$ |
|:--|:-----|:-----|
| $x_1$ | 🧚 $w_{11}$ | 🧚 $w_{12}$ |
| $x_2$ | 🧚 $w_{21}$ | 🧚 $w_{22}$ |

What each hidden neuron $h_j$ receives is the sum of the signals sent by all the Little Geniuses connected to it:

$$
u_j = \sum_i w_{ij} \times x_i
$$

---

### 1-5-2 The Relay Process of Forward Propagation

Imagine the information flow in a 3-layer network (input → hidden → output):

1. **Leg 1: Input Layer → Hidden Layer**
   Input nodes hand signals to the Little Geniuses; each Little Genius multiplies by its weight and sends the result to the target neuron

2. **Leg 2: Inside the Hidden Neuron**
   Cell body sums: u₁ = w₁₁×x₁ + w₂₁×x₂, activation function processes: h₁ = σ(u₁)

3. **Leg 3: Hidden Layer → Output Layer**
   A new group of Little Geniuses takes over, multiplying by new weights and sending to the output neuron

4. **Leg 4: Output Prediction Result**
   The output neuron sums, activates, and produces the final prediction

---

### 1-5-3 Visualization: Network Structure Diagram

![Figure 1-2: 3-layer fully connected network structure. Input layer 3 nodes, hidden layer 4 nodes, output layer 2 nodes. Each connection line represents a weight parameter.](../images/ch01/NN01_network_structure.png)

*Figure 1-2: 3-layer fully connected network structure. Input layer 3 nodes, hidden layer 4 nodes, output layer 2 nodes. Each connection line represents a weight parameter.*

#### How to Calculate the Number of Parameters

For this 3→4→2 network:

- Layer 1 parameters: $3 \times 4 + 4 = 16$ (weights + biases)
- Layer 2 parameters: $4 \times 2 + 2 = 10$ (weights + biases)
- Total parameters: $16 + 10 = 26$

```python
# Parameter count calculation
def count_params(layer_sizes):
    total = 0
    for i in range(len(layer_sizes) - 1):
        weights = layer_sizes[i] * layer_sizes[i+1]
        biases = layer_sizes[i+1]
        total += weights + biases
        print(f"Layer {i+1}: {weights} weights + {biases} biases = {weights+biases}")
    return total

layers = [3, 4, 2]
print(f"Total parameters: {count_params(layers)}")

```

> **Little Genius says**: See every gray connection line in the diagram? There's a me (Little Genius) standing on every single one! My job is to multiply the input signal I'm connected to by my "importance coefficient" (the weight value), and then pass it on to the next layer. The parameter count is the total number of Little Geniuses plus the bias for each node!


---

## 1-6 Translating Little Genius Work into Math

### 1-6-1 Mathematical Formalization

#### Input and Output of a Single Neuron

The **weighted input** of the $j$-th neuron in layer $l$:

$$
u_j^{(l)} = \sum_{i} w_{ji}^{(l)} z_i^{(l-1)} + b_j^{(l)}
$$

The **activation output** of the $j$-th neuron in layer $l$:

$$
z_j^{(l)} = f(u_j^{(l)})
$$

#### Matrix Form (Vectorization)

$$
\mathbf{u}^{(l)} = \mathbf{W}^{(l)} \mathbf{z}^{(l-1)} + \mathbf{b}^{(l)}
$$

$$
\mathbf{z}^{(l)} = f(\mathbf{u}^{(l)})
$$

---

### 1-6-2 Superscript / Subscript Conventions

| Symbol | Meaning | Example |
|:------|:--------|:--------|
| $(l)$ | Layer $l$ | $W^{(2)}$ is the weight matrix of layer 2 |
| $j$ | Index of the target neuron | $z_j^{(l)}$ is the output of the $j$-th neuron in layer $l$ |
| $i$ | Index of the source neuron | $w_{ji}^{(l)}$ is the weight from the $i$-th neuron in layer $l-1$ to the $j$-th neuron in layer $l$ |
| $w_{ji}$ | Weight **from $i$ to $j$** | $j$ first (target), $i$ second (source) |

> **Memory trick**: In $w_{ji}$, $j$ is the **destination** and $i$ is the **origin**.

> **Core Insight**: Once you understand the superscript/subscript conventions, the entire neural network becomes a "fill-in-the-table" game — each neuron fills in its weighted sum, then applies the activation function.

#### A Concrete Example

For a 3-layer network (input:3→hidden:4→output:2):

$$
w^{(1)}_{32} \quad\text{(in layer 1, weight from input 2 to neuron 3)}
$$

$$
\mathbf{W}^{(1)} \in \mathbb{R}^{4 \times 3} \quad\text{(layer 1: 4 hidden neurons × 3 input features)}
$$

| Symbol | Meaning | Example |
|:------|:--------|:--------|
| Superscript $(l)$ | Layer $l$ | $\mathbf{W}^{(1)}$ is the layer-1 weight matrix |
| Subscript $_{ji}$ | Neuron $j$ connects to input $i$ | $w^{(l)}_{ji}$ |
| Boldface | Vector / Matrix | $\mathbf{W}^{(l)}, \mathbf{b}^{(l)}$ |
| $\mathbf{z}^{(l)}$ | Weighted input of layer $l$ | $\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$ |
| $\mathbf{a}^{(l)}$ | Activation output of layer $l$ | $\mathbf{a}^{(l)} = f(\mathbf{z}^{(l)})$ |

> **Little Genius says**: On the name badge it says $w^{(l)}_{ji}$ — $l$ is the floor number, $i$ is where the signal comes from, $j$ is where the signal goes. Every Little Genius on every layer has a unique "ID number"!


---

## 1-7 Self-Learning Neural Networks

### 1-7-1 What Is "Learning"?

#### The Essence of Learning

**The essence of learning = adjusting parameters to make the output approach the target**

$$
\text{Given input } x, \text{ desired output } t \quad \longrightarrow \quad \text{Adjust } W, b \text{ so that } y \approx t
$$

#### Analogy: Tuning a Radio Dial

Imagine you're tuning a radio to find the clearest station:

- Turn the frequency knob (adjust parameters)
- Listen for clarity (evaluate the result)
- Fine-tune until it's optimal (find the best parameters)

Neural network learning is exactly the same — except it has **millions of knobs** (weight parameters).

#### Analogy: Going Down a Mountain

Imagine you stand at the peak of a valley and your goal is to walk to the lowest point:

```
Starting position (mountain top, large loss) → take a step in the negative gradient direction → repeat → reach the valley bottom (optimal parameters, minimal loss)
```

---

### 1-7-2 Learning = A Parameter Optimization Problem

#### Three Elements

1. **Parameters**: $W, b$ (all weights and biases)
2. **Loss function**: $L(W, b)$ measures prediction error
3. **Optimization method**: Descend along the gradient direction (gradient descent)

#### Mathematical Expression

$$
\min_{W, b} L(W, b)
$$

$$
W^{(t+1)} = W^{(t)} - \eta \frac{\partial L}{\partial W}
$$

Where $\eta$ is the learning rate (step size), and $\frac{\partial L}{\partial W}$ is the gradient (direction of steepest ascent).

#### Learning Is Essentially "Tuning Parameters"

$$
\text{Learning} = \text{Adjusting parameters to minimize the loss}
$$

```python
# The essence of learning: a blind person descending a mountain
for epoch in range(1000):
    loss = compute_loss(model)     # Compute how large the error is
    grads = compute_grad(loss)     # Find the downhill direction
    model.params -= lr * grads     # Take a step downhill (update parameters)

```

> **Little Genius says**: Learning is like a blind person going down a mountain — you don't know where the valley is (the optimal parameters), but you can feel the slope under your feet (the gradient). If the ground slopes downward, that means the direction is right, so you take a step in that direction (parameter update). Repeat enough times, and you'll reach the valley bottom!


---

### 1-7-3 Warm-up: Intuition of Gradient Descent

#### Steps

1. Stand on the mountain, close your eyes, feel the slope with your feet (compute the gradient)
2. Find the direction of steepest descent = negative gradient direction
3. Take a step (step size = learning rate η)
4. Feel the slope again (recompute the gradient)
5. Repeat until you reach the valley bottom

#### Visualization

![Figure 1-3: Intuition of gradient descent — standing on a mountain, searching for the lowest point. The red arrow indicates the gradient direction (steepest ascent), so we descend in the opposite direction (negative gradient).](../images/ch01/NN01_gradient_descent_intuition.png)

*Figure 1-3: Intuition of gradient descent — standing on a mountain, searching for the lowest point. The red arrow indicates the gradient direction; we descend in the opposite direction.*

#### A Simple Python Demo

```python
import numpy as np
import matplotlib.pyplot as plt

# Target function: f(x) = x² + 2x + 1 (a simple quadratic function)
def f(x):
    return x**2 + 2*x + 1

# Derivative: f'(x) = 2x + 2
def grad(x):
    return 2*x + 2

# Gradient descent
x = 4.0          # Initial position
lr = 0.1         # Learning rate
steps = []
for i in range(20):
    steps.append((x, f(x)))
    x = x - lr * grad(x)  # Update along negative gradient direction

print("Gradient descent trajectory:")
for i, (x_val, f_val) in enumerate(steps):
    print(f"  Step {i}: x = {x_val:.4f}, f(x) = {f_val:.4f}")

```


| Step | x | f(x) |
|:----:|-----:|------:|
| 0 | 4.0000 | 25.0000 |
| 1 | 2.8000 | 14.4400 |
| 2 | 1.8400 | 8.0656 |
| ... | ... | ... |
| 19 | -1.0000 | 0.0000 |

When x = -1, f(x) = 0, which is exactly the function's minimum point. ✅


> When $x = -1$, $f(x) = (-1)^2 + 2(-1) + 1 = 0$, which is exactly the function's minimum point!

> **Core Insight**: Neural network learning = using gradient descent to solve an **ultra-high-dimensional parameter optimization problem**.
>
> Starting from Chapter 2, we will systematically study the mathematical tools needed.

---

## 1-8 Neural Network Design Space

### 1-8-1 Network Depth vs Width

When designing a neural network, the two most important hyperparameters are **depth (number of layers)** and **width (neurons per layer)**.

| Dimension | Meaning | Advantage | Disadvantage |
|:----------|:--------|:----------|:-------------|
| **Depth** (layers) | Number of layers in the network | Hierarchical feature extraction, stronger expressive power | Vanishing/exploding gradients, harder to train |
| **Width** (neurons per layer) | Number of neurons per layer | Each layer can learn more features | Parameter explosion, risk of overfitting |

#### Mathematical Intuition

$$
\text{Expressive power} \approx \text{Depth} \times \text{Width}
$$

But **the contribution of depth is exponential** — each additional layer raises the "abstraction level" of features once more. A 10-layer network can theoretically learn more abstract features than a 5-layer network.

> **Little Genius says**: Depth is like "levels of thinking" — the first-layer Little Geniuses see scattered pixels (edges), the middle-layer ones see shapes (textures), and the deep-layer ones see complete concepts (cat/dog). Width is "how many Little Geniuses per layer" — the more Little Geniuses, the richer the feature patterns this layer can capture!

### 1-8-2 Parameter Count and Model Capacity

$$
\text{Model capacity} \approx \text{Parameter count} \approx \sum_{l=1}^{L} (n_{l-1} \times n_l + n_l)
$$

Where $n_l$ is the number of neurons in layer $l$.

#### Parameter Estimation Example

```python
# Estimating parameters for common network configurations
def estimate_params(configs):
    """configs: [(input_dim, output_dim), ...]"""
    total = 0
    print("Layer\tParams\t\tCumulative")
    print("-" * 30)
    for i, (n_in, n_out) in enumerate(configs):
        params = n_in * n_out + n_out  # weights + biases
        total += params
        print(f"{i+1}\t{params:>8,}\t{total:>8,}")
    return total

# A handwritten digit recognition network
config = [(784, 256), (256, 128), (128, 64), (64, 10)]
total = estimate_params(config)
print(f"\nTotal parameters: {total:,}")

```


| Layer | Parameters | Cumulative |
|:-----:|----------:|-----------:|
| 1 | 200,960 | 200,960 |
| 2 | 32,896 | 233,856 |
| 3 | 8,256 | 242,112 |
| 4 | 1,290 | 243,402 |

**Total parameters: 243,402**


> **Core Insight**: **~240,000 parameters** need to learn from 60,000 MNIST training images! Each image has 784 pixels, which means each parameter is "responsible for" the information of roughly 6 training samples. This is why more data allows for larger models.

### 1-8-3 Overfitting and Underfitting

| Problem | Training Error | Test Error | Cause | Solution |
|:--------|:--------------|:-----------|:------|:---------|
| **Underfitting** | High | High | Model too simple | Increase layers/neurons |
| **Overfitting** | Low | High | Model memorized the data | Add regularization / more data / reduce model |

| Condition | Training Error | Test Error | Diagnosis |
|:----------|:--------------|:-----------|:----------|
| **Underfitting** | High | High | Insufficient model capacity |
| **Overfitting** | Low | High | Model memorized training data instead of learning patterns |
| **Normal** | Low | Low | ✅ Ideal state |

> **Little Genius says**: Underfitting is like asking an elementary school student to take a college entrance exam — too simple, can't learn. Overfitting is like making a PhD student memorize the answer key — perfect exam score, but they haven't actually learned anything! A good model should "understand the principles" rather than "recite the answers."

---

## 1-9 Learning Strategy Quick Reference

### 1-9-1 Training Strategy Decision Tree

1. **Loss not decreasing?**
   - Learning rate too small? → Increase learning rate
   - Vanishing gradients? → Switch to ReLU / Add BatchNorm

2. **Training loss down but validation loss not? → Overfitting!**
   - Add Dropout / L2 regularization
   - Data augmentation
   - Reduce model size

3. **Validation loss stops improving → Early Stopping**

### 1-9-2 When to Choose Which Network?

| Task Type | Recommended Architecture | Reason |
|:----------|:-------------------------|:-------|
| **Tabular data** | MLP (Multi-Layer Perceptron) | Small scale, fully connected is sufficient |
| **Image classification** | CNN | Spatial locality + translation invariance |
| **Sequential data (text/audio)** | Transformer / RNN | Sequence modeling capability |
| **Large language models** | Transformer Decoder | Autoregressive generation |

> **One-sentence summary**: The core idea of neural networks = stacking multiple layers of "weighted sum + activation function"; learning = adjusting parameters via gradient descent; design = finding the balance among depth, width, and regularization.


## ⚠️ Common Pitfalls and Debugging Guide

### Pitfall 1: You Can Just Pick Any Activation Function

❌ **Wrong belief**: "They're all nonlinear functions anyway, the choice of activation function doesn't matter much."
✅ **Correct understanding**: The choice of activation function **directly affects** the network's training difficulty and final performance.

| Activation | Suitable For | Not Suitable For |
|:-----------|:-------------|:-----------------|
| **ReLU** | **Default choice for hidden layers** | Output layer (output range is unbounded) |
| **Sigmoid** | Binary classification output layer | Hidden layers (severe vanishing gradient) |
| **Tanh** | RNNs, scenarios requiring zero-mean | Deep networks (gradient still vanishes) |
| **Softmax** | Multi-class classification output layer | Hidden layers (all probabilities sum to 1, restricting expressiveness) |

### Pitfall 2: More Layers Are Always Better

❌ **Wrong belief**: "It's deep learning, so more layers must be more powerful."
✅ **Correct understanding**: Increasing layer count introduces vanishing/exploding gradients and degradation problems. You need residual connections, BatchNorm, and other techniques to train deep networks.

> **Little Genius says**: The more layers there are, the longer the "Little Genius chain" the signal has to pass through! If every Little Genius accidentally loses a bit of the signal (gradient attenuation), after 50 layers the original signal has almost disappeared. That's why deep networks need residual connections — to give the Little Geniuses a "VIP express lane"!


### What You've Learned

1. **How a neuron works**: Weighted sum + activation function = output
2. **The limitation of the M-P model**: Can only solve linearly separable problems
3. **Activation functions**: The evolutionary logic from step function → Sigmoid → ReLU → Tanh
4. **Neural network structure**: Input layer → Hidden layer → Output layer
5. **Fully connected layer**: Matrix multiplication + element-wise activation = one layer of transformation
6. **The essence of learning**: Adjusting parameters to minimize the loss function

### Prerequisite Knowledge

Before moving to Chapter 2, you should be comfortable with:

- [x] Understanding the basic mathematical model of a neuron
- [x] Being able to implement simple network forward propagation in Python
- [x] Having intuition for gradient descent

> **One-sentence summary**: Neural network = stacking multiple layers of "weighted sum + activation function"; learning = using gradient descent to adjust weights.

---


### Core Formula Quick Reference

| Formula | Description | Use Case |
|:--------|:------------|:---------|
| $y = f(\mathbf{w} \cdot \mathbf{x} + b)$ | Single neuron output: weighted sum + activation | Foundation of all neural networks |
| $\text{Sigmoid}(x) = \frac{1}{1 + e^{-x}}$ | Sigmoid activation, output range (0,1) | Binary classification output layer |
| $\text{ReLU}(x) = \max(0, x)$ | ReLU activation, sparse activation | **Default choice for hidden layers** |
| $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | Tanh activation, output range (-1,1) | RNNs and similar scenarios |
| $\mathbf{z}^{(l)} = f(\mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)})$ | Forward propagation for layer l (matrix form) | Multi-layer network computation |
| $\Delta w = -\eta \frac{\partial L}{\partial w}$ | Gradient descent parameter update rule | All parameter updates |


## 📦 Chapter Code List

| File | Content | Key Concept |
|:-----|:--------|:------------|
| `ch01/NN01_mp_neuron.py` | M-P neuron + AND/OR logic gates | Weighted sum + threshold decision |
| `ch01/NN01_activation_functions.py` | 4 activation functions + derivatives + visualization | Activation function comparison |
| `ch01/NN01_simple_network.py` | Manual 2-layer fully connected network | Forward propagation implementation |
| `ch01/NN01_network_viz.py` | Network structure diagram | networkx + matplotlib |
| `ch01/NN01_gradient_intuition.py` | Gradient descent warm-up demo | Quadratic function minimization |

---

## 📖 Chapter Summary

### Core Concepts Review

| Stage | Concept | Key Idea |
|:------|:--------|:---------|
| 1 | Biological Neuron | Dendrites receive → Cell body processes → Axon outputs |
| 2 | M-P Model | Weighted sum + threshold decision |
| 3 | Activation Functions | Sigmoid / ReLU / Tanh (continuously differentiable) |
| 4 | Neural Network | Multi-layer stacking + fully connected + forward propagation |
| 5 | Learning | Parameter optimization + gradient descent |

### 🧪 Exercises

#### Exercise 1: Implement an M-P Neuron Manually

Implement an M-P neuron with Python. Two binary inputs x1, x2 in {0,1}, both weights = 1, threshold = 1.5. Test AND and OR logic:

```python
def mp_neuron(x1, x2, w1=1, w2=1, threshold=1.5):
    """M-P neuron: weighted sum → threshold decision"""
    total = w1 * x1 + w2 * x2
    return 1 if total >= threshold else 0

# Test AND logic
for x1, x2 in [(0,0), (0,1), (1,0), (1,1)]:
    print(f"AND({x1},{x2}) = {mp_neuron(x1, x2, threshold=1.5)}")

```

**Think**: Adjust the threshold so that the same neuron implements OR logic. What should the threshold be?

#### Exercise 2: Try Different Activation Functions

Modify the following code to replace Sigmoid with ReLU and Tanh, and observe the output differences:

```python
import numpy as np
x = np.array([-2, -1, 0, 1, 2])
print("Sigmoid:", 1 / (1 + np.exp(-x)))
# Your task: implement ReLU and Tanh

```

#### Exercise 3: Manually Compute the Forward Pass of a 2-Layer Network

Given input x = [1, 2]^T, weight matrix W1 = [[0.1, 0.2], [0.3, 0.4]], bias b1 = [0.5, 0.5], activation function = Sigmoid. Manually compute the hidden layer output h = sigmoid(W1 * x + b1).

#### Exercise 4: Parameter Count Calculation

A network has the structure: input layer 784 neurons - hidden layer 256 neurons (ReLU) - output layer 10 neurons (Softmax). Calculate:

- Parameter count of the first fully connected layer (weights + biases)
- Parameter count of the second fully connected layer
- Total parameter count

**Think**: If the hidden layer increases to 512 neurons, by what factor does the total parameter count increase?

#### Exercise 5 (Challenge): Implement Gradient Descent for a 2-Layer Network

Use NumPy to implement single-sample gradient descent from scratch. Hints:

1. Forward pass: y_pred = sigmoid(W2 * sigmoid(W1 * x + b1) + b2)
2. Loss: L = 0.5 * (y_pred - y)^2
3. Use numerical gradient approximation to verify your gradient computation


← [Foreword](00-foreword.md) | [Table of Contents](README.md) | [Chapter 2: Mathematical Foundations of Neural Networks](02-chapter2-mathematical-foundations.md) →
