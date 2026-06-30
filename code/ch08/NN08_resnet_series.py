"""
ResNet 系列完整实现：ResNet-18/34/50/101/152 ⭐

包含：
- BasicBlock（用于 ResNet-18/34）
- Bottleneck（用于 ResNet-50/101/152）
- ResNet 通用框架
- 五种标准的网络生成函数

参考：He et al., Deep Residual Learning for Image Recognition, CVPR 2016
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic Block: 2 个 3×3 卷积，用于浅层 ResNet"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Bottleneck(nn.Module):
    """Bottleneck: 1×1 降维 → 3×3 卷积 → 1×1 升维，用于深层 ResNet"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        # 1×1 降维
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3×3 卷积（低维空间）
        self.conv2 = nn.Conv2d(planes, planes, 3,
                               stride=stride, padding=1, bias=False)
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
        out = F.relu(self.bn2(self.conv2(out)))      # 低维卷积
        out = self.bn3(self.conv3(out))              # 升维
        out += self.shortcut(x)                      # 残差连接
        return F.relu(out)


class ResNet(nn.Module):
    """通用 ResNet 框架"""

    def __init__(self, block, num_blocks, num_classes=1000):
        """
        Args:
            block: BasicBlock 或 Bottleneck
            num_blocks: 列表，每个 stage 的 block 数量，如 [3, 4, 6, 3] 对应 ResNet-50
            num_classes: ImageNet 分类数（默认 1000）
        """
        super().__init__()
        self.in_planes = 64

        # conv1: 7×7, stride 2
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # maxpool: 3×3, stride 2
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # 4 个 stage
        self.layer1 = self._make_layer(block, 64,    num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128,   num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256,   num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512,   num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ── 五种标准 ResNet 的生成函数 ──

def ResNet18(num_classes=1000):
    """ResNet-18: BasicBlock, [2, 2, 2, 2], 11.7M 参数"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def ResNet34(num_classes=1000):
    """ResNet-34: BasicBlock, [3, 4, 6, 3], 21.8M 参数"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def ResNet50(num_classes=1000):
    """ResNet-50: Bottleneck, [3, 4, 6, 3], 25.6M 参数"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes=1000):
    """ResNet-101: Bottleneck, [3, 4, 23, 3], 44.5M 参数"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNet152(num_classes=1000):
    """ResNet-152: Bottleneck, [3, 8, 36, 3], 60.2M 参数"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    print("=" * 70)
    print("ResNet 系列完整配置表")
    print("=" * 70)

    configs = [
        ("ResNet-18",  BasicBlock, [2, 2, 2, 2]),
        ("ResNet-34",  BasicBlock, [3, 4, 6, 3]),
        ("ResNet-50",  Bottleneck, [3, 4, 6, 3]),
        ("ResNet-101", Bottleneck, [3, 4, 23, 3]),
        ("ResNet-152", Bottleneck, [3, 8, 36, 3]),
    ]

    print(f"\n{'模型':<12} | {'Block 类型':<15} | {'num_blocks':<20} | {'参数量':>12}")
    print("-" * 65)

    for name, block, num_blocks in configs:
        model = ResNet(block, num_blocks)
        params = count_params(model)
        block_name = "BasicBlock" if block == BasicBlock else "Bottleneck"
        block_str = str(num_blocks)
        print(f"{name:<12} | {block_name:<15} | {block_str:<20} | {params:>12,}")

    # ── 前向测试 ──
    print(f"\n{'='*70}")
    print("前向传播测试（小批量）")
    print(f"{'='*70}")

    x = torch.randn(2, 3, 224, 224)  # ImageNet 尺寸
    for name, block, num_blocks in configs[:3]:  # 测试前 3 个（避免太慢）
        model = ResNet(block, num_blocks)
        # 只跑前向，不计算梯度以加速
        with torch.no_grad():
            y = model(x)
        params = count_params(model)
        print(f"  {name:<12} | 输入: {list(x.shape)} | 输出: {list(y.shape)} | 参数: {params:>10,}")

    print(f"\n{'='*70}")
    print("结构深度分析（每个 Stage 的通道变化）")
    print(f"{'='*70}")

    for name, block, num_blocks in configs[:3]:
        model = ResNet(block, num_blocks)
        print(f"\n  {name} 各层输出通道数：")
        print(f"    conv1: 64 → (输入 224×224)")
        print(f"    layer1: 64 → ×{num_blocks[0]} blocks → 56×56")
        print(f"    layer2: 128 → ×{num_blocks[1]} blocks → 28×28")
        print(f"    layer3: 256 → ×{num_blocks[2]} blocks → 14×14")
        print(f"    layer4: 512 → ×{num_blocks[3]} blocks → 7×7")
        print(f"    fc: {512 * block.expansion} → num_classes")

    print(f"\n{'='*70}")
    print("✅ 所有 ResNet 系列模型定义完成")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
