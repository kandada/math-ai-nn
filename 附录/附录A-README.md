# 附录A：数学公式回顾

> 本附录汇总了阅读本书所需的全部数学公式。分为微积分、线性代数、概率统计、信息论四大部分。

> © xiefujin · Contact: 490021684@qq.com · Licensed under CC BY-NC-SA 4.0
---

## A-1 微积分（Calculus）

### A-1-1 导数基本公式

| 函数 | 导数 | 说明 |
|:----|:-----|:-----|
| $f(x) = c$（常数） | $f'(x) = 0$ | 常数导数为零 |
| $f(x) = x^n$ | $f'(x) = nx^{n-1}$ | 幂函数 |
| $f(x) = e^x$ | $f'(x) = e^x$ | 指数函数 |
| $f(x) = \ln x$ | $f'(x) = 1/x$ | 对数函数 |
| $f(x) = \sigma(x) = \frac{1}{1+e^{-x}}$ | $f'(x) = \sigma(x)(1-\sigma(x))$ | Sigmoid |
| $f(x) = \tanh(x)$ | $f'(x) = 1 - \tanh^2(x)$ | Tanh |
| $f(x) = \text{ReLU}(x) = \max(0, x)$ | $f'(x) = \mathbf{1}_{\{x > 0\}}$ | ReLU |

### A-1-2 求导法则

**加法法则**：$(f + g)' = f' + g'$

**乘法法则**：$(f \cdot g)' = f' \cdot g + f \cdot g'$

**链式法则（核心）**：$(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$

### A-1-3 偏导数与梯度

对于多变量函数 $f(x_1, x_2, \dots, x_n)$，偏导数 $\frac{\partial f}{\partial x_i}$ 表示固定其他变量时 $f$ 对 $x_i$ 的变化率。

**梯度向量**：$\nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n} \right)$

### A-1-4 泰勒展开

**一阶近似**：$f(x) \approx f(a) + f'(a)(x - a)$

**二阶近似**：$f(x) \approx f(a) + f'(a)(x - a) + \frac{1}{2}f''(a)(x - a)^2$

---

## A-2 线性代数（Linear Algebra）

### A-2-1 向量基础

**内积（点积）**：$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = \mathbf{a}^\top \mathbf{b}$

**L2 范数**：$\|\mathbf{a}\|_2 = \sqrt{\sum_{i=1}^{n} a_i^2}$

### A-2-2 矩阵运算

**矩阵乘法**：$(AB)_{ij} = \sum_{k=1}^{m} A_{ik} B_{kj}$

**转置**：$(A^\top)_{ij} = A_{ji}$

**神经网络中的常见形状**：

- 输入 $\mathbf{x} \in \mathbb{R}^{d}$，权重 $\mathbf{W} \in \mathbb{R}^{d \times h}$，偏置 $\mathbf{b} \in \mathbb{R}^{h}$
- 前向传播：$\mathbf{h} = \mathbf{W}^\top \mathbf{x} + \mathbf{b}$，结果 $\mathbf{h} \in \mathbb{R}^{h}$

### A-2-3 矩阵微积分基础

**常见梯度**：

- $\frac{\partial}{\partial \mathbf{x}} (\mathbf{a}^\top \mathbf{x}) = \mathbf{a}$
- $\frac{\partial}{\partial \mathbf{W}} (\mathbf{W} \mathbf{x}) = \mathbf{x}^\top$（形状对齐后）

---

## A-3 概率统计（Probability & Statistics）

### A-3-1 基本概念

**期望**：$\mathbb{E}[X] = \sum_i x_i p_i$（离散），$\mathbb{E}[X] = \int x p(x) dx$（连续）

**方差**：$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$

### A-3-2 常见分布

| 分布 | 概率密度/质量函数 | 用途 |
|:----|:-----------------|:-----|
| 伯努利分布 | $P(X=1) = p$ | 二分类输出 |
| 正态分布 | $\mathcal{N}(\mu, \sigma^2)$ | 初始化、噪声 |
| 均匀分布 | $\mathcal{U}(a, b)$ | 权重初始化 |

---

## A-4 信息论（Information Theory）

**熵（Entropy）**：$H(P) = -\sum_i P_i \log P_i$

**交叉熵（Cross Entropy）**：$H(P, Q) = -\sum_i P_i \log Q_i$

**KL 散度**：$D_{KL}(P \parallel Q) = \sum_i P_i \log \frac{P_i}{Q_i}$

**重要关系**：$H(P, Q) = H(P) + D_{KL}(P \parallel Q)$
