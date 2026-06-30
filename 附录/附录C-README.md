# 附录C：符号约定

> 本附录定义了本书使用的全部数学符号。同一符号在不同上下文中可能有不同含义，具体见各章说明。

> © xiefujin · Contact: 490021684@qq.com · Licensed under CC BY-NC-SA 4.0
---

## C-1 基本符号

| 符号 | 含义 | 示例 |
|:----|:-----|:-----|
| $x, y, z$ | 标量（小写斜体） | $x = 3.14$ |
| $\mathbf{x}, \mathbf{w}$ | 向量（小写粗体） | $\mathbf{x} \in \mathbb{R}^d$ |
| $\mathbf{W}, \mathbf{X}$ | 矩阵（大写粗体） | $\mathbf{W} \in \mathbb{R}^{d \times h}$ |
| $\mathbb{R}$ | 实数集 | $\mathbb{R}^n$ 表示 $n$ 维实数空间 |

---

## C-2 神经网络专用符号

### C-2-1 网络结构

| 符号 | 含义 | 说明 |
|:----|:-----|:-----|
| $L$ | 网络总层数 | 通常不计输入层 |
| $n^{(l)}$ | 第 $l$ 层的神经元数 | $l = 1, 2, ..., L$ |
| $\mathbf{W}^{(l)}$ | 第 $l$ 层的权重矩阵 | $\mathbf{W}^{(l)} \in \mathbb{R}^{n^{(l-1)} \times n^{(l)}}$ |
| $\mathbf{b}^{(l)}$ | 第 $l$ 层的偏置向量 | $\mathbf{b}^{(l)} \in \mathbb{R}^{n^{(l)}}$ |
| $\mathbf{u}^{(l)}$ | 第 $l$ 层的加权输入 | $\mathbf{u}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$ |
| $\mathbf{a}^{(l)}$ | 第 $l$ 层的激活输出 | $\mathbf{a}^{(l)} = f(\mathbf{u}^{(l)})$ |
| $f(\cdot)$ | 激活函数 | 如 Sigmoid, ReLU, Tanh |

### C-2-2 反向传播符号

| 符号 | 含义 | 说明 |
|:----|:-----|:-----|
| $\delta^{(l)}_j$ | 第 $l$ 层第 $j$ 个神经元的误差 | $\delta^{(l)}_j = \frac{\partial L}{\partial u^{(l)}_j}$ |
| $\boldsymbol{\delta}^{(l)}$ | 第 $l$ 层的误差向量 | |
| $C$ | 损失函数（Loss） | 全书统一使用「损失函数」，符号为 $L$（第 5 章用 $C$ 表示代价） |
| $L$ | 损失函数（Loss） | 与 $C$ 含义相同，全书统一使用 $L$ |
| $\eta$ | 学习率 | 梯度下降的步长 |
| $\nabla$ | 梯度算子 | $\nabla_\mathbf{W} L$ 表示 Loss 对 W 的梯度 |

### C-2-3 数据集符号

| 符号 | 含义 | 说明 |
|:----|:-----|:-----|
| $N$ | 训练样本总数 | |
| $\mathbf{x}_i$ | 第 $i$ 个输入样本 | |
| $\mathbf{t}_i$ | 第 $i$ 个样本的目标标签（Target） | |
| $\mathbf{y}_i$ | 第 $i$ 个样本的网络输出 | |
| $\mathcal{D}$ | 数据集 | $\mathcal{D} = \{(\mathbf{x}_i, \mathbf{t}_i)\}_{i=1}^{N}$ |

---

## C-3 上下标约定

| 标记 | 含义 | 示例 |
|:----|:-----|:-----|
| $x^{(l)}$ | 上标括号 — 第 $l$ 层 | $\mathbf{W}^{(1)}$ 第1层权重 |
| $x_i$ | 下标 — 第 $i$ 个元素 | $x_1$ 第1个输入特征 |
| $x^{(l)}_i$ | 组合 — 第 $l$ 层第 $i$ 个 | $\delta^{(2)}_3$ 第2层第3个误差 |
| $x_{ij}$ | 双下标 — 第 $i$ 行第 $j$ 列 | $W_{ij}$ 输入 $j$ 到神经元 $i$ 的权重 |

---

## C-4 常用缩写

| 缩写 | 全称 | 说明 |
|:----|:-----|:-----|
| M-P | McCulloch-Pitts | 最早的神经元数学模型 |
| SGD | Stochastic Gradient Descent | 随机梯度下降 |
| BP | Backpropagation | 反向传播 |
| CNN | Convolutional Neural Network | 卷积神经网络 |
| RNN | Recurrent Neural Network | 循环神经网络 |
| LSTM | Long Short-Term Memory | 长短期记忆网络 |
| ResNet | Residual Network | 残差网络 |
| Transformer | — | 基于注意力的架构 |
| LLM | Large Language Model | 大语言模型 |
| RLHF | Reinforcement Learning from Human Feedback | 人类反馈强化学习 |
