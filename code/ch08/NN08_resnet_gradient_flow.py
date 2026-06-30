"""
ResNet vs Plain 梯度流对比实验 ⭐

目标：直观展示残差连接如何让梯度「直达」浅层。
对比 10 层 Plain 网络和 10 层 ResNet 的梯度幅值随层数的变化。

关键观察：
- Plain 网络：随层数增加，梯度指数级衰减
- ResNet：梯度幅值在各层保持稳定（恒等项 1 的作用）

作者：Kaiming He 等 (Deep Residual Learning, 2015)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PlainBlock(nn.Module):
    """无残差连接的 Plain 块"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out)


class ResBlock(nn.Module):
    """带残差连接的 ResNet 块"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual          # 残差连接 ⭐
        return torch.relu(out)


class PlainNet(nn.Module):
    """N 层 Plain 网络"""
    def __init__(self, num_blocks=10, channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(*[PlainBlock(channels) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


class ResNet(nn.Module):
    """N 层 ResNet"""
    def __init__(self, num_blocks=10, channels=16):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


def compute_gradient_flow(model, num_blocks=10):
    """
    计算各层的梯度幅值（L2 norm），用于对比 Plain vs ResNet
    返回 list: 每个 block 的梯度幅值
    """
    model.train()
    x = torch.randn(8, 3, 32, 32)
    y = model(x)
    loss = y.sum()

    # 反向传播
    loss.backward()

    # 收集每个 block 的梯度幅值
    grad_mags = []
    for i, block in enumerate(model.blocks):
        grad_norm = 0.0
        count = 0
        for param in block.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
                count += 1
        if count > 0:
            grad_mags.append(np.sqrt(grad_norm / count))
        else:
            grad_mags.append(0.0)

    return grad_mags


def main():
    # 使用 15 层进行对比，效果更明显
    num_blocks = 15

    print("=" * 60)
    print("ResNet vs Plain 梯度流对比实验")
    print("=" * 60)

    # 创建模型
    plain_net = PlainNet(num_blocks=num_blocks)
    res_net = ResNet(num_blocks=num_blocks)

    # 计算梯度流
    plain_grads = compute_gradient_flow(plain_net, num_blocks)
    res_grads = compute_gradient_flow(res_net, num_blocks)

    # 打印数值
    print(f"\n各层梯度幅值（从浅层到深层）:")
    print(f"{'层号':>4} | {'Plain 梯度':>12} | {'ResNet 梯度':>12} | {'Ratio':>8}")
    print("-" * 45)
    for i in range(num_blocks):
        ratio = res_grads[i] / (plain_grads[i] + 1e-10)
        print(f"{i+1:>4} | {plain_grads[i]:>12.4e} | {res_grads[i]:>12.4e} | {ratio:>8.2f}")

    # 可视化
    plt.figure(figsize=(10, 5))
    layers = range(1, num_blocks + 1)

    # 归一化到 [0,1] 以便直观对比
    plain_norm = np.array(plain_grads) / (max(plain_grads) + 1e-10)
    res_norm = np.array(res_grads) / (max(res_grads) + 1e-10)

    plt.plot(layers, plain_norm, 'r-o', label='Plain Network', linewidth=2, markersize=6)
    plt.plot(layers, res_norm, 'b-s', label='ResNet (with shortcut)', linewidth=2, markersize=6)

    # 添加参考线
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5)

    plt.xlabel('Layer Index (from input)', fontsize=12)
    plt.ylabel('Normalized Gradient Magnitude', fontsize=12)
    plt.title('Gradient Flow: Plain Network vs ResNet', fontsize=13)
    plt.text(0.5, -0.15,
             'ResNet: gradients stay stable (identity shortcut preserves gradient)\n'
             'Plain: gradients decay rapidly with depth',
             transform=plt.gca().transAxes, fontsize=10,
             ha='center', va='top')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/ch08/NN08_gradient_flow.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ 梯度流对比图已保存")

    # 数值总结
    print(f"\n{'='*60}")
    print(f"数值总结:")
    print(f"  Plain 网络 - 第1层梯度: {plain_grads[0]:.4e}, 最后1层: {plain_grads[-1]:.4e}")
    print(f"           - 衰减倍数: {plain_grads[0]/(plain_grads[-1]+1e-10):.1f} 倍")
    print(f"  ResNet    - 第1层梯度: {res_grads[0]:.4e}, 最后1层: {res_grads[-1]:.4e}")
    print(f"           - 衰减倍数: {res_grads[0]/(res_grads[-1]+1e-10):.1f} 倍")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
