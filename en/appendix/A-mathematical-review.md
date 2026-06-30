# Appendix A: Mathematical Formula Review

> This appendix summarizes all mathematical formulas needed for this book. Organized into calculus, linear algebra, probability & statistics, and information theory.

> © xiefujin · Contact: 490021684@qq.com · Licensed under CC BY-NC-SA 4.0
---

## A-1 Calculus

### A-1-1 Basic Derivatives

| Function | Derivative | Notes |
|:---------|:-----------|:------|
| $f(x) = c$ (constant) | $f'(x) = 0$ | Constant rule |
| $f(x) = x^n$ | $f'(x) = nx^{n-1}$ | Power rule |
| $f(x) = e^x$ | $f'(x) = e^x$ | Exponential |
| $f(x) = \ln x$ | $f'(x) = 1/x$ | Logarithm |
| $f(x) = \sigma(x) = \frac{1}{1+e^{-x}}$ | $f'(x) = \sigma(x)(1-\sigma(x))$ | Sigmoid |
| $f(x) = \tanh(x)$ | $f'(x) = 1 - \tanh^2(x)$ | Tanh |
| $f(x) = \text{ReLU}(x) = \max(0, x)$ | $f'(x) = \mathbf{1}_{\{x > 0\}}$ | ReLU |

### A-1-2 Differentiation Rules

| Rule | Formula |
|:-----|:--------|
| Sum rule | $(f + g)' = f' + g'$ |
| Product rule | $(fg)' = f'g + fg'$ |
| Quotient rule | $(f/g)' = (f'g - fg') / g^2$ |
| **Chain rule** | $(f \circ g)' = f'(g(x)) \cdot g'(x)$ |

### A-1-3 Partial Derivatives

For $f(x_1, x_2, \dots, x_n)$, the partial derivative $\frac{\partial f}{\partial x_i}$ measures the rate of change with respect to $x_i$ alone.

---

## A-2 Linear Algebra

### A-2-1 Vectors

| Operation | Definition |
|:----------|:-----------|
| Dot product | $x \cdot y = \sum_{i} x_i y_i$ |
| L2 norm | $\|x\|_2 = \sqrt{\sum_i x_i^2}$ |
| Cosine similarity | $\cos\theta = \frac{x\cdot y}{\|x\| \|y\|}$ |

### A-2-2 Matrices

| Operation | Definition |
|:----------|:-----------|
| Matrix multiplication | $(AB)_{ij} = \sum_k A_{ik} B_{kj}$ |
| Transpose | $(A^{\mathsf{T}})_{ij} = A_{ji}$ |
| Identity matrix | $I_{ij} = \delta_{ij}$ |

### A-2-3 Key Neural Network Operations

| Expression | Meaning |
|:-----------|:--------|
| $z = Wx + b$ | Single sample forward |
| $Z = XW^{\mathsf{T}} + b$ | Batch forward |
| $\delta^{(l)} = (W^{(l+1)})^{\mathsf{T}} \delta^{(l+1)}$ | Error backpropagation |

---

## A-3 Probability & Statistics

| Concept | Formula |
|:--------|:--------|
| Probability | $P(A) \in [0, 1]$ |
| Conditional prob. | $P(A\|B) = \frac{P(A \cap B)}{P(B)}$ |
| Expectation | $\mathbb{E}[X] = \sum x P(X=x)$ |
| Variance | $\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$ |

---

## A-4 Information Theory

| Concept | Formula | Meaning |
|:--------|:--------|:--------|
| Self-information | $I(p) = -\log p$ | Surprise of an event |
| Entropy | $H(p) = -\sum p_i \log p_i$ | Average uncertainty |
| Cross-entropy | $H(p, q) = -\sum p_i \log q_i$ | Prediction error |
| KL divergence | $D_{\text{KL}}(p\|q) = \sum p_i \log \frac{p_i}{q_i}$ | Distribution distance |

← [Chapter 9](../09-chapter9-large-language-models.md) | [Table of Contents](../README.md) | [Appendix B](B-pytorch-api.md) →
