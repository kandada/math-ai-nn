# Appendix D: Common Function Reference

> This appendix summarizes common activation functions and loss functions used in neural networks.

> © xiefujin · Contact: 490021684@qq.com · Licensed under CC BY-NC-SA 4.0
---

## D-1 Activation Functions

### D-1-1 Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- Range: (0, 1)
- Derivative: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
- Use case: Binary classification output, gating mechanisms
- Limitation: Gradient vanishing in saturation regions

### D-1-2 Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

- Range: (-1, 1)
- Derivative: $\tanh'(x) = 1 - \tanh^2(x)$
- Use case: RNN hidden state, zero-centered activation needed
- Limitation: Still saturates

### D-1-3 ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x)$$

- Range: [0, +∞)
- Derivative: 1 if $x > 0$, 0 if $x \leq 0$
- Use case: **Default for hidden layers**
- Limitation: Dying ReLU (neurons can become permanently inactive)

### D-1-4 Leaky ReLU

$$\text{LeakyReLU}(x) = \max(\alpha x, x)$$

- Typically $\alpha = 0.01$
- Solves the dying ReLU problem

### D-1-5 Softmax

$$p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

- Output: Probability distribution over $K$ classes
- Use case: Multi-class classification output

---

## D-2 Loss Functions

### D-2-1 Mean Squared Error (MSE)

$$L_{\text{MSE}} = \frac{1}{m} \sum_{i=1}^{m} (y_i - t_i)^2$$

- Use case: Regression
- Gradient: $\frac{\partial L}{\partial y} = \frac{2}{m}(y - t)$

### D-2-2 Cross-Entropy

$$L_{\text{CE}} = -\sum_{k} t_k \log p_k$$

- Use case: Classification
- With Softmax: $\frac{\partial L}{\partial z_k} = p_k - t_k$ (nice gradient!)

### D-2-3 Binary Cross-Entropy (BCE)

$$L_{\text{BCE}} = -\frac{1}{m} \sum_{i} (t_i \log p_i + (1-t_i) \log(1-p_i))$$

- Use case: Binary classification

---

## D-3 Summary: When to Use What

| Task | Output Activation | Loss Function |
|:-----|:------------------|:--------------|
| Binary classification | Sigmoid | BCE |
| Multi-class classification | Softmax | Cross-Entropy |
| Regression | None (linear) | MSE |
| Multi-label classification | Sigmoid | BCE (per label) |

---

## D-4 Regularization Methods

| Method | Formula | Effect |
|:------|:--------|:-------|
| L1 Regularization (Lasso) | $\lambda \sum \lvert w_i \rvert$ | Produces sparse weights (some become zero) |
| L2 Regularization (Ridge) | $\frac{\lambda}{2} \sum w_i^2$ | Weight decay (shrink but not zero) |
| Dropout | Randomly drop $p\%$ of neurons during training | Ensemble effect, prevents overfitting |
| Batch Normalization | $\hat{x} = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$ | Stabilizes training, accelerates convergence |

← [Appendix C](C-notation-conventions.md) | [Table of Contents](../README.md) | [Appendix E](E-exercise-answers.md) →

