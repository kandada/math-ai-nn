"""
ResNet Bottleneck 设计：参数量对比 ⭐

对比 Basic Block (2个3×3卷积) 和 Bottleneck (1×1→3×3→1×1) 的参数量。

Bottleneck 的核心创新：
  256-d → 1×1, 64 → 3×3, 64 → 1×1, 256
  先降维（256→64）→ 卷积 → 再升维（64→256）

参数量对比（以 256→256 为例）：
  - Basic Block:  2 × (3×3×256×256) = 1,179,648
  - Bottleneck:   1×1×256×64 + 3×3×64×64 + 1×1×64×256 = 114,688
                  约为 Basic Block 的 1/10！

作者：Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
      (Deep Residual Learning for Image Recognition, CVPR 2016)
"""

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic Block: 两个 3×3 卷积"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Bottleneck(nn.Module):
    """
    Bottleneck: 1×1 降维 → 3×3 卷积 → 1×1 升维

    设计哲学：
    - 1×1 降维：将高维特征压缩到低维空间，减少计算量
    - 3×3 卷积：在低维空间中进行空间特征提取
    - 1×1 升维：将特征恢复到高维空间，保持维度一致
    """
    expansion = 4  # 输出通道数是 planes 的 4 倍

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        # 1×1 降维
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3×3 卷积（在低维空间中）
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1×1 升维
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))        # 降维
        out = F.relu(self.bn2(self.conv2(out)))      # 卷积
        out = self.bn3(self.conv3(out))              # 升维
        out += self.shortcut(x)                      # 残差连接
        return F.relu(out)


def count_params(module):
    """计算模块的参数量"""
    return sum(p.numel() for p in module.parameters())


def main():
    print("=" * 65)
    print("ResNet Basic Block vs Bottleneck 参数量对比")
    print("=" * 65)

    # 测试不同通道数下的参数量
    # 公平对比：两种 Block 输出相同的通道数
    # Bottleneck 输出 = planes * expansion(4)，所以用更小的 planes 达到相同输出
    # 例如 BasicBlock(256,256) 输出 256 通道 vs Bottleneck(256,64) 输出 64*4=256 通道
    test_cases = [
        # (in_ch, basic_out, bottle_planes)
        (64, 64, 16),     # 输出 64 通道：Basic=64, Bottleneck=16*4=64
        (256, 256, 64),   # 输出 256 通道：Basic=256, Bottleneck=64*4=256  ← ResNet-50 配置
        (512, 512, 128),  # 输出 512 通道：Basic=512, Bottleneck=128*4=512  ← ResNet-101
        (1024, 1024, 256),# 输出 1024 通道
    ]

    print(f"\n{'输入→输出通道':<20} | {'BasicBlock':>14} | {'Bottleneck':>14} | {'比例':>8}")
    print('-' * 62)

    for in_ch, basic_out, bottle_planes in test_cases:
        basic = BasicBlock(in_ch, basic_out)
        bottleneck = Bottleneck(in_ch, bottle_planes)

        basic_params = count_params(basic)
        bottle_params = count_params(bottleneck)
        out_ch = basic_out  # BasicBlock 的输出通道
        ratio = bottle_params / (basic_params + 1e-10)

        print(f"{in_ch}→{out_ch:<15} | {basic_params:>14,} | {bottle_params:>14,} | {ratio:>7.2f}x")

    print()
    print()
    print("=" * 65)
    print("关键观察：")
    print("  Bottleneck 通过「降维→卷积→升维」大幅减少参数量")
    print(f"  当输入输出维度高时（如 256→256），Bottleneck 比 Basic Block")
    print(f"  参数量少约 17 倍（约 118 万 vs 7 万），这让 ResNet-50/101/152 变得可行。")
    print("=" * 65)


if __name__ == '__main__':
    main()
