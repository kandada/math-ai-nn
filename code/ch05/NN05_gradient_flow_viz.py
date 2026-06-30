"""
梯度传播可视化：展示反向传播中 δ（误差信号）逐层传播的过程 ⭐

目标：观察 5 层全连接网络中，δ 值从输出层向输入层传播时的变化规律。
与 NN05_backprop_viz.py (结构图) 互补——本文件展示数值传播过程。

关键观察：
  - 浅层网络：δ 值变化平稳，梯度可正常传播
  - 深层网络（Sigmoid）：δ 指数衰减 → 梯度消失
  - 深层网络（ReLU）：δ 保持一定强度 → 缓解梯度消失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


class SimpleMLP(nn.Module):
    """简单全连接网络，用于观察梯度传播"""
    def __init__(self, layers, activation='sigmoid'):
        super().__init__()
        self.linears = nn.ModuleList()
        self.activation = activation
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))

    def forward(self, x):
        self.activations = [x]
        self.pre_acts = []
        for i, linear in enumerate(self.linears):
            z = linear(self.activations[-1])
            self.pre_acts.append(z)
            if i < len(self.linears) - 1:  # 隐藏层
                if self.activation == 'sigmoid':
                    a = torch.sigmoid(z)
                else:
                    a = torch.relu(z)
            else:  # 输出层 - 无激活
                a = z
            self.activations.append(a)
        return self.activations[-1]


def compute_gradient_flow(model, activation='sigmoid'):
    """
    计算网络中各层的梯度 δ 值
    返回：每层的 δ L2 范数
    """
    model.zero_grad()
    x = torch.randn(32, model.linears[0].in_features)
    y = model(x)

    # 模拟回归任务，计算 MSE 损失
    target = torch.randn(32, model.linears[-1].out_features)
    loss = F.mse_loss(y, target)
    loss.backward()

    # 收集每层的梯度（δ = ∂L/∂z）
    deltas = []
    for i, linear in enumerate(model.linears):
        # 对于线性层，weight.grad ≈ δ·a_prev^T
        # 我们用 weight.grad 的范数来近似 δ 的强度
        grad_norm = linear.weight.grad.norm().item()
        deltas.append(grad_norm)

    return deltas


def experiment():
    """对比不同深度和激活函数的梯度传播"""
    print("=" * 60)
    print("梯度传播对比实验")
    print("=" * 60)

    # 不同深度配置
    configs = [
        ('3 层 (Sigmoid)', [10, 20, 1], 'sigmoid'),
        ('5 层 (Sigmoid)', [10, 20, 15, 10, 1], 'sigmoid'),
        ('8 层 (Sigmoid)', [10, 20, 18, 16, 14, 12, 10, 1], 'sigmoid'),
        ('8 层 (ReLU)', [10, 20, 18, 16, 14, 12, 10, 1], 'relu'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, (name, layers, act) in enumerate(configs):
        ax = axes[idx // 2][idx % 2]
        model = SimpleMLP(layers, act)

        # 多次采样取平均
        all_deltas = []
        for _ in range(10):
            deltas = compute_gradient_flow(model, act)
            # 归一化到 [0,1] 以便比较
            if max(deltas) > 0:
                deltas = [d / max(deltas) for d in deltas]
            all_deltas.append(deltas)

        avg_deltas = np.mean(all_deltas, axis=0)

        # 绘制
        layers_range = range(1, len(avg_deltas) + 1)
        ax.plot(layers_range, avg_deltas, 'o-', linewidth=2, markersize=6)

        # 衰减比
        decay_ratio = avg_deltas[0] / (avg_deltas[-1] + 1e-10)
        ax.set_title(f'{name}\n衰减比 = {decay_ratio:.1f}x',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Layer (from input)')
        ax.set_ylabel('Normalized δ Magnitude')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.15)

        # 标注数值
        for i, v in enumerate(avg_deltas):
            ax.annotate(f'{v:.2f}', (i+1, v),
                       textcoords='offset points', xytext=(0, 10),
                       ha='center', fontsize=8)

        print(f"  {name:<15} | δ衰减: {avg_deltas[0]:.3f} → {avg_deltas[-1]:.3f} ({decay_ratio:.1f}x)")

    plt.suptitle('Gradient (δ) Propagation: Sigmoid vs ReLU',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/ch05/NN05_gradient_flow.png', dpi=150)
    print(f"\n✅ 梯度传播图已保存")

    print(f"\n{'='*60}")
    print(f"结论:")
    print(f"  深度 Sigmoid 网络：δ 指数衰减 → 浅层学不动（梯度消失）")
    print(f"  ReLU 网络：δ 保持较好 → 缓解梯度消失")
    print(f"  这就是现代深度网络默认使用 ReLU 的原因！")
    print(f"{'='*60}")


if __name__ == '__main__':
    experiment()
