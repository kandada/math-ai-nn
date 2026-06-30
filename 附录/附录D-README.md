# 附录D：常见函数汇总

> 本附录汇总了神经网络中常用的激活函数和损失函数。

> © xiefujin · Contact: 490021684@qq.com · Licensed under CC BY-NC-SA 4.0
---

## D-1 激活函数（Activation Functions）

### D-1-1 Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- **输出范围**：$(0, 1)$
- **导数**：$\sigma'(x) = \sigma(x)(1 - \sigma(x))$
- **优点**：输出可解释为概率
- **缺点**：梯度饱和（两端导数接近0）、非零中心
- **适用**：二分类输出层

### D-1-2 Tanh

$$\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

- **输出范围**：$(-1, 1)$
- **导数**：$\tanh'(x) = 1 - \tanh^2(x)$
- **优点**：零中心（比 Sigmoid 好）
- **缺点**：仍存在梯度饱和
- **适用**：循环神经网络（RNN/LSTM）

### D-1-3 ReLU（Rectified Linear Unit）

$$\text{ReLU}(x) = \max(0, x)$$

- **输出范围**：$[0, \infty)$
- **导数**：$\text{ReLU}'(x) = \mathbf{1}_{\{x > 0\}}$
- **优点**：计算极快、缓解梯度消失
- **缺点**：Dying ReLU（负区域梯度为0）
- **适用**：**现代网络默认激活函数**（CNN、全连接）

### D-1-4 Leaky ReLU

$$\text{LeakyReLU}(x) = \max(\alpha x, x)$$

- **优点**：解决了 Dying ReLU 问题，$\alpha$ 通常取 0.01
- **适用**：需要避免神经元死亡时

### D-1-5 GELU（Gaussian Error Linear Unit）

$$\text{GELU}(x) = x \cdot \Phi(x)$$（$\Phi$ 是标准正态分布的 CDF）

- **优点**：Transformer 系列的标准选择
- **适用**：BERT、GPT 等现代模型

### D-1-6 Softmax（多分类激活函数）

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

- **输出**：$K$ 维概率向量（和为 1）
- **梯度**：$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$
- **适用**：多分类输出层

---

## D-2 损失函数（Loss Functions）

### D-2-1 均方误差（MSE）

$$L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - t_i)^2$$

- **特点**：对大误差的惩罚更大（平方项）
- **梯度**：$\frac{\partial L}{\partial y_i} = \frac{2}{N}(y_i - t_i)$
- **适用**：回归任务

### D-2-2 平均绝对误差（MAE）

$$L_{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - t_i|$$

- **特点**：对异常值鲁棒
- **梯度**：$\frac{\partial L}{\partial y_i} = \frac{1}{N} \cdot \text{sign}(y_i - t_i)$
- **适用**：回归任务（有离群点时）

### D-2-3 交叉熵损失（Cross Entropy）

**二分类**：$L_{BCE} = -\frac{1}{N} \sum_{i} [t_i \log y_i + (1 - t_i) \log(1 - y_i)]$

**多分类**：$L_{CE} = -\frac{1}{N} \sum_{i} \sum_{k} t_{ik} \log y_{ik}$

- **特点**：分类任务的标准损失函数
- **结合 Softmax 的梯度**：$\frac{\partial L}{\partial z_i} = y_i - t_i$（极简！）
- **适用**：分类任务

### D-2-4 Hinge Loss（SVM 损失）

$$L_{Hinge} = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)$$

- **适用**：SVM、最大间隔分类

### D-2-5 KL 散度

$$D_{KL}(P \parallel Q) = \sum_i P_i \log \frac{P_i}{Q_i}$$

- **适用**：变分自编码器（VAE）、知识蒸馏

---

## D-3 正则化方法

| 方法 | 公式 | 效果 |
|:----|:-----|:-----|
| L1 正则化（Lasso） | $\lambda \sum \lvert w_i \rvert$ | 产生稀疏权重（部分归零） |
| L2 正则化（Ridge） | $\frac{\lambda}{2} \sum w_i^2$ | 权重衰减（趋向于0但不为0） |
| Dropout | 训练时随机丢弃 $p$\\% 神经元 | 集成学习效果，防过拟合 |
| Batch Normalization | $\hat{x} = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$ | 稳定训练，加速收敛 |
