# 附录B：PyTorch API 速查

> 本附录汇总了本书使用的全部 PyTorch API，按功能分类组织。

> © xiefujin · Contact: 490021684@qq.com · Licensed under CC BY-NC-SA 4.0
---

## B-1 Tensor 创建与操作

### B-1-1 创建 Tensor

| API | 说明 | 示例 |
|:----|:-----|:-----|
| `torch.tensor(data)` | 从列表/数组创建 | `torch.tensor([1, 2, 3])` |
| `torch.zeros(shape)` | 全零 Tensor | `torch.zeros(3, 4)` |
| `torch.ones(shape)` | 全一 Tensor | `torch.ones(2, 3)` |
| `torch.randn(shape)` | 标准正态随机 | `torch.randn(128, 784)` |
| `torch.rand(shape)` | 均匀分布 [0,1) | `torch.rand(3, 3)` |
| `torch.arange(start, end)` | 等差数列 | `torch.arange(0, 10)` |
| `torch.linspace(s, e, n)` | 等间距序列 | `torch.linspace(0, 1, 100)` |
| `torch.eye(n)` | 单位矩阵 | `torch.eye(5)` |

### B-1-2 Tensor 属性

| API | 说明 |
|:----|:-----|
| `t.shape` | 形状（元组） |
| `t.dtype` | 数据类型（如 `torch.float32`） |
| `t.device` | 所在设备（CPU/GPU） |
| `t.requires_grad` | 是否需要梯度 |
| `t.grad` | 梯度（调用 backward 后） |

### B-1-3 形状操作

| API | 说明 |
|:----|:-----|
| `t.view(shape)` | 重塑形状（不改变内存布局) |
| `t.reshape(shape)` | 重塑形状（自动处理不连续) |
| `t.transpose(dim1, dim2)` | 交换两个维度 |
| `t.permute(dims)` | 任意维度重排 |
| `t.squeeze()` | 删除长度为 1 的维度 |
| `t.unsqueeze(dim)` | 在指定位置增加维度 |
| `t.flatten()` | 展平为一维 |

---

## B-2 自动微分（Autograd）

| API | 说明 |
|:----|:-----|
| `t.backward()` | 反向传播（标量） |
| `t.backward(gradient)` | 反向传播（非标量需传入梯度) |
| `torch.no_grad()` | 上下文管理器，禁用梯度追踪 |
| `torch.inference_mode()` | 推理模式（更快） |
| `t.detach()` | 从计算图中分离 |
| `t.zero_()` | 梯度清零（原地） |
| `t.retain_grad()` | 保留非叶节点的梯度 |

---

## B-3 神经网络模块（nn.Module）

| API | 说明 |
|:----|:-----|
| `nn.Linear(in, out)` | 全连接层 |
| `nn.Conv2d(C_in, C_out, K)` | 2D 卷积层 |
| `nn.MaxPool2d(K)` | 最大池化层 |
| `nn.AvgPool2d(K)` | 平均池化层 |
| `nn.Flatten()` | 展平层 |
| `nn.ReLU()` | ReLU 激活函数 |
| `nn.Sigmoid()` | Sigmoid 激活函数 |
| `nn.Tanh()` | Tanh 激活函数 |
| `nn.Dropout(p)` | Dropout 正则化 |
| `nn.BatchNorm1d/2d(f)` | 批归一化 |
| `nn.Sequential(layers)` | 顺序容器 |
| `nn.ModuleList(modules)` | 模块列表容器 |

---

## B-4 损失函数

| API | 说明 |
|:----|:-----|
| `nn.MSELoss()` | 均方误差（回归） |
| `nn.L1Loss()` | 平均绝对误差（回归） |
| `nn.CrossEntropyLoss()` | 交叉熵（分类，自带 Softmax） |
| `nn.BCELoss()` | 二分类交叉熵 |
| `nn.BCEWithLogitsLoss()` | 二分类交叉熵 + Sigmoid |
| `nn.KLDivLoss()` | KL 散度 |

---

## B-5 优化器（torch.optim）

| API | 说明 |
|:----|:-----|
| `optim.SGD(params, lr)` | 随机梯度下降 |
| `optim.SGD(..., momentum=0.9)` | SGD + Momentum |
| `optim.Adagrad(params, lr)` | AdaGrad 自适应学习率 |
| `optim.RMSprop(params, lr)` | RMSprop |
| `optim.Adam(params, lr)` | Adam（推荐默认优化器） |
| `optim.AdamW(params, lr)` | Adam + 解耦权重衰减 |

### 学习率调度器

| API | 说明 |
|:----|:-----|
| `StepLR(step_size, gamma)` | 阶梯衰减 |
| `CosineAnnealingLR(T_max)` | 余弦退火 |
| `ReduceLROnPlateau(patience)` | 自适应衰减 |
| `OneCycleLR(max_lr, steps)` | 单周期策略 |

---

## B-6 数据加载

| API | 说明 |
|:----|:-----|
| `Dataset` | 数据集基类（需实现 `__len__` 和 `__getitem__`) |
| `DataLoader(dataset, batch_size)` | 批量数据加载器 |
| `TensorDataset(*tensors)` | 包装 Tensor 为 Dataset |
| `random_split(dataset, lengths)` | 随机分割数据集 |

### torchvision 工具

| API | 说明 |
|:----|:-----|
| `datasets.MNIST(root, train)` | MNIST 手写数字 |
| `datasets.CIFAR10(root, train)` | CIFAR-10 彩色图片 |
| `transforms.ToTensor()` | PIL→Tensor 转换 |
| `transforms.Normalize(mean, std)` | 标准化 |
| `transforms.Compose(transforms)` | 组合多个变换 |
