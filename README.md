# 📘 《AI神经网络的数学——用Python、PyTorch讲解》书稿

> 🌐 **语言切换**：[**English**](README.en.md) | [**中文版**](README.md)


> **作者**: xiefujin · **邮箱**: 490021684@qq.com
> 这是一本基于作者过往的讲课课件、笔记等整理而成的人工智能及相关数学的学习书籍


## 目录结构

```
manuscript/
├── README.md                          # 本文件：书籍概览
├── 00-前言.md                         # 前言 & 使用说明
├── 01-第1章-神经网络的思想.md           # 概念引入
├── 02-第2章-神经网络的数学基础.md       # 数学基础（线性代数、微积分）
├── 03-第3章-PyTorch基础-Tensor与自动微分.md  # 工具准备
├── 04-第4章-神经网络的训练与优化.md         # 梯度下降与损失函数
├── 05-第5章-误差反向传播法.md   # ⭐ 全书核心
├── 06-第6章-卷积神经网络.md     # CNN
├── 07-第7章-训练技术-优化器-正则化与损失函数.md  # 训练技术
├── 08-第8章-现代架构-从ResNet到Transformer.md   # 现代架构
├── 09-第9章-大语言模型-训练-采样与推理.md # 大语言模型
└── README.md
```

## 章节概览

| 章节 | 核心内容 | 代码数量 | 插图数量 |
|:----|:---------|:--------:|:--------:|
| 前言 | 定位、特色、阅读建议 | - | - |
| 第 1 章 | M-P 神经元、激活函数、前向传播 | 5 | 3 |
| 第 2 章 | 函数、向量、矩阵、导数、链式法则、梯度下降 | 8 | 8 |
| 第 3 章 | Tensor、Autograd、nn.Module、DataLoader | 7 | 2 |
| 第 4 章 | 损失函数、Softmax、NumPy→PyTorch 过渡 | 7 | 3 |
| 第 5 章 ⭐ | δ 递推、反向传播推导、手动实现、MNIST | 6 | 1 |
| 第 6 章 | 卷积、池化、特征图可视化、PyTorch CNN | 6 | 2 |
| 第 7 章 | 优化器、正则化、BatchNorm、Softmax 深度理解 | 5 | 4 |
| 第 8 章 | ResNet、RNN、Attention、Transformer 最小实现 | 4 | 2 |
| 第 9 章 | 自回归生成、KV Cache、LoRA、量化 | 3 | 1 |

## 写作规范

本书遵循 [Markdown 写作组件规范](../AI神经网络的数学-用Python、PyTorch讲解-Markdown写作规范.md)：

- **数学公式**：行内 `$...$`，行间 `$$...$$`
- **代码块**：标注语言（python / output / bash）
- **提示块**：`> **核心洞察**`、`> **提示**`、`> **警告**`、`> **注意**`
- **小精灵**：`> **小精灵说**`
- **插图**：`![图 n-m：描述](相对路径)`

## 核心理念

1. **数学直觉比公式推导更重要**——每个公式都有「人话解释」
2. **代码是用来验证直觉的，不是主角**——先直觉，后公式，再代码
3. **理解 = 知道「为什么」> 知道「是什么」**——每个子节回答「为什么需要这个」

## 代码与图片

- 代码文件：`code/chNN/NN{章号}_{功能}.py`
- 插图：`images/chNN/`
- Jupyter Notebook：`notebooks/chNN/`

---

## 相关论文

基于本书内容整理撰写的学术论文已公开发表预印本。

> **From M-P Neuron to Transformer: A Minimal Mathematical Path with Intuition-First Pedagogy for Deep Learning Education**
>
> **DOI：** [https://doi.org/10.6084/m9.figshare.32692401](https://doi.org/10.6084/m9.figshare.32692401)

## 书籍 DOI

> **《AI神经网络的数学——用Python、PyTorch讲解》**
>
> **DOI：** [https://doi.org/10.5281/zenodo.20729963](https://doi.org/10.5281/zenodo.20729963)

## 许可

本作品遵循 [LICENSE](LICENSE) 文件中的完整许可条款。

版权所有 © 2026 xiefujin · 490021684@qq.com
本作品采用 **CC BY-NC-SA 4.0** 国际许可协议（[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)）。

### 您可以自由地：

- **共享** — 在任何媒介以任何形式复制、发行本作品
- **演绎** — 修改、转换或以本作品为基础进行创作

惟须遵守下列条件：

- **署名** — 您必须给出适当的署名，提供指向本许可协议的链接
- **非商业性使用** — 您不得将本作品用于商业目的
- **相同方式共享** — 如果您再混合、转换或基于本作品进行创作，您必须以与本许可相同或兼容的许可协议分发您的贡献

### 禁止行为

- 未经授权商用、出版、二次售卖及抢先发表
- 删除或修改版权声明

> 完整的法律文本请参阅 [LICENSE](LICENSE) 文件。
