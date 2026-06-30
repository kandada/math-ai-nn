"""
注意力模式可视化：展示 Self-Attention 的注意力权重矩阵 ⭐

目标：直观观察 Transformer 中不同位置的 token 如何「关注」其他 token。
展示三种典型的注意力模式：局部关注、全局关注、对角关注。

关键观察：
  - 注意力权重矩阵的热力图直接展示了 token 之间的依赖关系
  - 不同 head 可能学习到不同的注意力模式（局部 vs 全局）
  - 这是 Transformer 能捕捉长距离依赖的核心机制
"""

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def compute_attention_patterns(seq_len=12, d_model=32, num_heads=4):
    """
    生成随机 query/key 并计算注意力权重矩阵
    模拟不同 head 学习到不同的注意力模式
    """
    # 随机生成 Q, K (batch=1, heads=4)
    Q = torch.randn(1, num_heads, seq_len, d_model // num_heads)
    K = torch.randn(1, num_heads, seq_len, d_model // num_heads)

    # 为每个 head 添加不同的偏置，模拟不同的注意力模式
    for h in range(num_heads):
        # Head 0: 偏向关注近邻（局部）
        if h == 0:
            for i in range(seq_len):
                for j in range(seq_len):
                    dist = abs(i - j)
                    Q[0, h, i, :] += torch.exp(-torch.tensor(dist / 2.0))
        # Head 1: 偏向关注开头和结尾（全局）
        elif h == 1:
            Q[0, h, :, :] += 0.5
            K[0, h, 0, :] += 2.0  # 开头 token
            K[0, h, -1, :] += 2.0  # 结尾 token
        # Head 2: 偏向对角线（自关注）
        elif h == 2:
            for i in range(seq_len):
                Q[0, h, i, :] += 1.0
                K[0, h, i, :] += 1.0
        # Head 3: 随机模式
        else:
            pass  # 保持随机

    # 计算注意力分数
    scale = (d_model // num_heads) ** 0.5
    scores = Q @ K.transpose(-2, -1) / scale
    attn_weights = F.softmax(scores, dim=-1)

    return attn_weights[0]  # (num_heads, seq_len, seq_len)


def visualize_patterns():
    """生成并可视化注意力模式"""
    print("=" * 60)
    print("注意力模式可视化")
    print("=" * 60)

    seq_len = 12
    num_heads = 4
    attn = compute_attention_patterns(seq_len, 32, num_heads)

    head_names = [
        'Local Attention\n(focus on neighbors)',
        'Global Attention\n(focus on first/last)',
        'Self Attention\n(strong diagonal)',
        'Mixed Pattern\n(random)',
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    for h in range(num_heads):
        ax = axes[h // 2][h % 2]
        im = ax.imshow(attn[h].detach().numpy(), cmap='YlOrRd', vmin=0, vmax=attn[h].max().item())
        ax.set_title(head_names[h], fontsize=11, fontweight='bold')
        ax.set_xlabel('Key Position (attended to)')
        ax.set_ylabel('Query Position (attending from)')

        # 坐标轴标签
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels([f'{i}' for i in range(seq_len)])
        ax.set_yticklabels([f'{i}' for i in range(seq_len)])

        # 在格子中添加数值
        for i in range(seq_len):
            for j in range(seq_len):
                val = attn[h, i, j].item()
                if val > 0.05:  # 只显示显著的值
                    ax.text(j, i, f'{val:.2f}',
                           ha='center', va='center', fontsize=5, color='#333')

        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Multi-Head Attention Patterns\n(Each head learns different attention behavior)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/ch09/NN09_attention_pattern.png', dpi=150)
    print(f"\n✅ 注意力模式图已保存")

    # 统计信息
    print(f"\n注意力权重统计:")
    for h in range(num_heads):
        w = attn[h].detach().numpy()
        diag_mean = np.mean([w[i, i] for i in range(seq_len)])
        off_diag_mean = (np.sum(w) - np.sum([w[i, i] for i in range(seq_len)])) / (seq_len**2 - seq_len)
        print(f"  Head {h}: 对角线均值={diag_mean:.3f}, 非对角线均值={off_diag_mean:.3f}"
              f"  → {'偏自关注' if diag_mean > off_diag_mean * 2 else '偏全局'}")
    print(f"\n{'='*60}")
    print(f"结论：多头注意力的每个 head 可以关注不同的信息子空间")
    print(f"      — 这就是 Transformer 表达力强的原因！")
    print(f"{'='*60}")


if __name__ == '__main__':
    visualize_patterns()
