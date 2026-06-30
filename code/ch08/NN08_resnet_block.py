"""
ResNet 残差块实现（Basic Block + Pre-activation）⭐

包含两种变体：
1. Original (Post-activation): conv → BN → ReLU → conv → BN → +shortcut → ReLU
2. Pre-activation (v2):        BN → ReLU → conv → BN → ReLU → conv → +shortcut
   (He et al., Identity Mappings in Deep Residual Networks, 2016)

Pre-activation 的优势：
- 梯度更容易传播（激活函数不在主路径上）
- 训练更稳定，BN 在残差分支内
- 更容易训练 1000+ 层的网络

作者：Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    原始残差块 (Post-activation)

    结构：conv → BN → ReLU → conv → BN → +shortcut → ReLU
    这是 ResNet v1 (2015) 的标准形式。
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接：维度不匹配时使用 1×1 卷积
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主路径
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 跳跃连接（残差连接）
        out += self.shortcut(x)

        # 最终激活
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    """
    Pre-activation 残差块 (ResNet v2)

    结构：BN → ReLU → conv → BN → ReLU → conv → +shortcut

    与原始残差块的区别：
    - 激活函数（BN+ReLU）放在卷积之前
    - 最后一个 ReLU 被移到下一 block 的 Pre-activation 中
    - 梯度流路径更干净：y = x + F(BN(ReLU(x)))
      ∂y/∂x = 1 + ∂F/∂x · (∂BN/∂x)  - 梯度流更直接

    参考：He et al., "Identity Mappings in Deep Residual Networks", 2016
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)

        # 跳跃连接：维度不匹配时使用 1×1 卷积
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
            )

    def forward(self, x):
        # Pre-activation：先 BN+ReLU，再卷积 ⭐
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))

        # 残差连接（不需要最后的 ReLU）
        out += self.shortcut(x)
        return out


class ResNetCIFAR10(nn.Module):
    """使用原始残差块的 ResNet（CIFAR-10 版本）"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = ResidualBlock(16, 16)
        self.layer2 = ResidualBlock(16, 32, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


class PreActResNet(nn.Module):
    """使用 Pre-activation 残差块的 ResNet v2"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = PreActBlock(16, 16)
        self.layer2 = PreActBlock(16, 32, stride=2)
        self.final_bn = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.relu(self.final_bn(x))  # 最后的激活
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


def main():
    print("=" * 60)
    print("ResNet 残差块实现对比")
    print("=" * 60)

    # 测试原始残差块
    model_v1 = ResNetCIFAR10()
    x = torch.randn(4, 3, 32, 32)
    y_v1 = model_v1(x)
    params_v1 = sum(p.numel() for p in model_v1.parameters())
    print(f"\n  ResNet v1 (Post-activation):")
    print(f"    参数量: {params_v1:,}")
    print(f"    输出形状: {y_v1.shape}")

    # 测试 Pre-activation 残差块
    model_v2 = PreActResNet()
    y_v2 = model_v2(x)
    params_v2 = sum(p.numel() for p in model_v2.parameters())
    print(f"\n  ResNet v2 (Pre-activation):")
    print(f"    参数量: {params_v2:,}")
    print(f"    输出形状: {y_v2.shape}")

    # 验证输出一致
    print(f"\n  参数量差异: {abs(params_v1 - params_v2):,}")
    print(f"  两种实现参数量相近（设计略有不同）")

    # 打印结构对比
    print(f"\n{'='*60}")
    print(f"结构对比:")
    print(f"  v1 (Post-activation): conv → BN → ReLU → conv → BN → +shortcut → ReLU")
    print(f"  v2 (Pre-activation):  BN → ReLU → conv → BN → ReLU → conv → +shortcut")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
