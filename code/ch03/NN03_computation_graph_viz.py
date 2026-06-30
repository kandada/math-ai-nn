"""
计算图可视化：展示前向传播 + 反向传播的完整流程 ⭐

目标：用图形方式展示 autograd 的计算图结构——前向传播构建图，
反向传播计算梯度。每个节点显示其正向输出值和反向梯度值。

与 NN03_computational_graph.py (文本追踪) 互补——本文件提供视觉化呈现。
"""

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def viz_simple_graph():
    """可视化简单的计算图：y = (w * x + b) * v"""
    # 定义输入
    x = torch.tensor([2.0], requires_grad=True)
    w = torch.tensor([0.5], requires_grad=True)
    b = torch.tensor([1.0], requires_grad=True)
    v = torch.tensor([3.0], requires_grad=True)

    # 前向传播：构建计算图
    u = w * x + b   # 中间节点
    y = u * v       # 输出
    L = (y - 1.0) ** 2  # 损失

    # 记录前向值
    fwd_vals = {
        'x': x.item(), 'w': w.item(), 'b': b.item(), 'v': v.item(),
        'u': u.item(), 'y': y.item(), 'L': L.item()
    }

    # 反向传播：计算梯度
    L.backward()

    # 记录梯度值
    grad_vals = {
        'x': x.grad.item(), 'w': w.grad.item(),
        'b': b.grad.item(), 'v': v.grad.item()
    }

    print("=" * 50)
    print("计算图可视化 - 前向值与梯度")
    print("=" * 50)
    print(f"  前向传播: L = (y-1)², y = u·v, u = w·x + b")
    print(f"  x={fwd_vals['x']:.1f}  w={fwd_vals['w']:.1f}  b={fwd_vals['b']:.1f}  v={fwd_vals['v']:.1f}")
    print(f"  u={fwd_vals['u']:.2f}  y={fwd_vals['y']:.2f}  L={fwd_vals['L']:.4f}")
    print(f"\n  反向传播梯度:")
    print(f"  ∂L/∂w = {grad_vals['w']:.4f}    ∂L/∂x = {grad_vals['x']:.4f}")
    print(f"  ∂L/∂b = {grad_vals['b']:.4f}    ∂L/∂v = {grad_vals['v']:.4f}")

    # 绘制计算图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 4)
    ax.axis('off')
    ax.set_title('Computation Graph: Forward (blue) + Backward (red)',
                 fontsize=14, fontweight='bold')

    # 节点布局 (x, y)
    nodes = {
        'x': (0, 2), 'w': (0, 1), 'b': (0, 0), 'v': (3, 3),
        'u': (2, 1.5), 'y': (4, 2), 'L': (5, 1)
    }

    # 边：前向传播（蓝色箭头）
    edges_fwd = [
        ('x', 'u'), ('w', 'u'), ('b', 'u'),
        ('u', 'y'), ('v', 'y'),
        ('y', 'L')
    ]

    # 边：反向传播（红色虚线箭头）
    edges_bwd = [
        ('L', 'y'), ('y', 'u'), ('y', 'v'),
        ('u', 'x'), ('u', 'w'), ('u', 'b')
    ]

    # 绘制前向边
    for src, dst in edges_fwd:
        sx, sy = nodes[src]
        dx, dy = nodes[dst]
        ax.annotate('', xy=(dx, dy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color='#2196F3',
                                    lw=2, connectionstyle='arc3,rad=0.15'))

    # 绘制反向边
    for src, dst in edges_bwd:
        sx, sy = nodes[src]
        dx, dy = nodes[dst]
        ax.annotate('', xy=(dx, dy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color='#F44336',
                                    lw=1.5, linestyle='dashed',
                                    connectionstyle='arc3,rad=-0.15'))

    # 绘制节点
    leaf_color = '#E3F2FD'    # 叶节点（输入）
    inner_color = '#FFF9C4'   # 中间节点
    loss_color = '#FFCDD2'    # 损失节点

    for name, (nx, ny) in nodes.items():
        if name in ('x', 'w', 'b', 'v'):
            color = leaf_color
        elif name == 'L':
            color = loss_color
        else:
            color = inner_color

        circle = plt.Circle((nx, ny), 0.4, color=color, ec='#333', lw=1.5)
        ax.add_patch(circle)

        # 节点名 + 前向值
        if name in fwd_vals:
            label = f"{name}={fwd_vals[name]:.2f}"
        else:
            label = name
        ax.text(nx, ny, label, ha='center', va='center', fontsize=9, fontweight='bold')

        # 梯度值（如���有）
        if name in grad_vals:
            ax.text(nx, ny - 0.55,
                    f"g={grad_vals[name]:.3f}",
                    ha='center', va='top', fontsize=8, color='#D32F2F')

    # 添加图例
    ax.text(-0.5, 3.8, '→ Forward  ', fontsize=10, color='#2196F3',
            bbox=dict(facecolor='white', edgecolor='none'))
    ax.text(-0.5, 3.5, '- - → Backward (gradient)', fontsize=10, color='#F44336',
            bbox=dict(facecolor='white', edgecolor='none'))

    plt.tight_layout()
    plt.savefig('code/ch03/images/NN03_computation_graph.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 计算图已保存至 code/ch03/images/")


if __name__ == '__main__':
    viz_simple_graph()
