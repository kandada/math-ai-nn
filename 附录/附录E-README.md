# 附录E：习题参考答案

> 本附录提供各章课后练习的参考答案与解析。

> © xiefujin · Contact: 490021684@qq.com · Licensed under CC BY-NC-SA 4.0
---

## 第1章 神经网络的思想

### 练习 1：M-P 神经元实现 NAND 门

```python
def nand_gate(x1, x2):
    # NAND = NOT AND = 当 x1 + x2 < 2 时输出 1
    w1, w2, threshold = -1, -1, -1.5
    return 1 if w1 * x1 + w2 * x2 >= threshold else 0

# 验证
print(nand_gate(0, 0))  # 1
print(nand_gate(0, 1))  # 1
print(nand_gate(1, 0))  # 1
print(nand_gate(1, 1))  # 0
```

> **解析**：NAND 门是「与非」——只有两个输入都是1时才输出0。通过设置负权重和负阈值（-1, -1, -1.5）实现。

### 练习 2：调整阈值实现 OR 门

对于 M-P 神经元 $y = \mathbb{1}(w_1 x_1 + w_2 x_2 \geq \theta)$，OR 门需要：

- $0 + 0 = 0 < \theta$ → 输出 0，所以 $\theta > 0$
- $1 + 0 = 1 \geq \theta$ → 输出 1，所以 $\theta \leq 1$
- 取 $w_1 = w_2 = 1$时，$\theta = 0.5$ 即可

```python
def or_gate(x1, x2):
    return 1 if x1 + x2 >= 0.5 else 0
```

### 练习 3：手动计算两层网络前向传播

已知：$\mathbf{x} = [1, 2]$，$\mathbf{W}^{(1)} = [[0.1, 0.3], [0.2, 0.4]]$，$\mathbf{b}^{(1)} = [0.1, 0.2]$

$$\mathbf{u}^{(1)} = \mathbf{x} \mathbf{W}^{(1)} + \mathbf{b}^{(1)} = [1 \times 0.1 + 2 \times 0.2 + 0.1, 1 \times 0.3 + 2 \times 0.4 + 0.2] = [0.6, 1.3]$$

若激活函数为 ReLU，则 $\mathbf{a}^{(1)} = [0.6, 1.3]$。

---

## 第2章 神经网络的数学基础

### 练习 1：手动计算梯度

$f(x, y) = x^2 + 2y^2$ 在 $(3, 2)$ 处：

- $\frac{\partial f}{\partial x} = 2x = 6$
- $\frac{\partial f}{\partial y} = 4y = 8$
- 梯度 $\nabla f = (6, 8)$

### 练习 2：数值微分验证

```python
def numerical_gradient(f, x, h=1e-7):
    grad = []
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad
```

---

## 第3章 PyTorch 基础

### 练习 1：广播机制

```python
a = torch.tensor([[1, 2, 3]])  # shape (1, 3)
b = torch.tensor([[4], [5]])   # shape (2, 1)
c = a + b  # → (2, 3): [[5, 6, 7], [6, 7, 8]]
```

> **解析**：a 广播为 (2, 3)，b 广播为 (2, 3)。

### 练习 2：手动实现线性层

```python
class ManualLinear:
    def __init__(self, in_features, out_features):
        self.W = torch.randn(in_features, out_features) * 0.01
        self.b = torch.zeros(out_features)
    
    def forward(self, x):
        return x @ self.W + self.b
```

---

## 第5章 反向传播

### 练习 1：手动计算反向传播

对 $f(x) = \sigma(wx + b)$ 在 $x=1, w=0.5, b=0$ 处计算 $\frac{\partial f}{\partial w}$：

$$u = 0.5 \times 1 + 0 = 0.5$$
$$y = \sigma(0.5) = 0.6225$$
$$\frac{\partial y}{\partial u} = y(1-y) = 0.6225 \times 0.3775 = 0.2350$$
$$\frac{\partial u}{\partial w} = x = 1$$
$$\frac{\partial f}{\partial w} = 0.2350 \times 1 = 0.2350$$

---

## 第9章 大语言模型

### 练习 1：温度参数实验

`T → 0`：趋近于贪心解码（概率最大的 token 概率 → 1），输出确定但重复。
`T → ∞`：趋近于均匀采样，输出随机但多样。

**最佳实践**：T=0.7~1.0 在创造性和一致性之间取得平衡。
