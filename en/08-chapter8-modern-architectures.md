# Chapter 8: Modern Architectures — From ResNet to Transformer

> **Goal**: Understand the key architectural innovations that enable modern deep learning — skip connections, attention mechanisms, and the Transformer.

> © xiefujin · Contact: 490021684@qq.com · Licensed under CC BY-NC-SA 4.0
>
> **Code**: `../code/ch08/` (5 files)

---

## 📋 Chapter Learning Objectives

- [ ] Understand ResNet and why skip connections solve vanishing gradients
- [ ] Understand RNN basics and the vanishing gradient problem in sequences
- [ ] Understand LSTM/GRU gating mechanisms and how they mitigate gradient vanishing
- [ ] Master the attention mechanism: Query, Key, Value
- [ ] Understand the complete Transformer architecture
- [ ] Be able to implement a minimal Transformer block

---

## 8-1 Residual Networks (ResNet) ⭐

> **Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)
> **Paper**: *Deep Residual Learning for Image Recognition*, CVPR 2016 **Best Paper Award**
> **Key Insight**: Skip connections allow gradients to flow directly to shallow layers, enabling 100+ layer networks.

---

### 8-1-1 The Degradation Problem

#### Experimental Observation

A 56-layer plain network has **higher training error** than a 20-layer network — and this is not overfitting, because the training error itself is worse. This is the **degradation problem**: deeper networks become harder to optimize.

![Figure 8-1: Training error for networks of different depths. The 56-layer network performs worse than the 20-layer network — the degradation problem.](../images/ch08/NN08_degradation.png)
*Figure 8-1: Network degradation — deeper networks become harder to optimize.*

> **Key Insight**: Degradation is not overfitting — training error itself is worse. The root cause is **optimization difficulty**, not overfitting.

#### Gradient Vanishing: Mathematical Analysis

For an $L$-layer network, the gradient at layer 1 must pass through $L-1$ activation function derivatives:

$$\frac{\partial L}{\partial W^{(1)}} = \underbrace{f'(u^{(L)}) \cdots f'(u^{(1)})}_{L \text{ derivatives}} \times \cdots$$

The chain multiplication effect:
- **Sigmoid**: derivative range $(0, 0.25]$ → $L=10$ gives $0.25^{10} \approx 10^{-6}$
- **ReLU**: derivative range $\{0, 1\}$ → alleviates but doesn't solve

#### Degradation vs Gradient Vanishing

| Problem | Symptom | Root Cause |
|:--------|:--------|:-----------|
| **Gradient Vanishing** | Shallow layers get tiny gradients → can't learn | Activation derivative product < 1 |
| **Degradation** | **Deeper is worse than shallower** | Identity mapping is hard to learn (layers struggle to approximate $F(x) \approx x$) |

> **Analogy**: Imagine a 50-layer network. Ideally, if the first 20 layers already learned good features, the remaining 30 layers should "do nothing" — maintain identity mapping. But it turns out that learning $F(x) \approx x$ through multiple nonlinear layers is surprisingly difficult! This is the mathematical essence of degradation — **stacked nonlinearities struggle to approximate identity mappings**.

---

### 8-1-2 The Mathematics of Skip Connections ⭐

> **Intuition**: Skip connections create a "VIP channel" for information flow. Previously, information had to pass through layer after layer ($F(x)$), easily getting lost. Now with $y = F(x) + x$, gradients can reach shallow layers directly — like an employee reporting directly to the CEO without going through layers of management! This is why ResNet can train 1000+ layer networks.

#### Core Idea

Give the gradient a "highway" to shallow layers:

$$y = F(x, \{W_i\}) + x$$

- $F(x)$ is the **residual mapping** to be learned (typically 2-3 conv layers)
- $x$ is the **identity mapping** passed directly via shortcut

> **The Beauty**: If identity is optimal, the network only needs to drive $F(x) \to 0$, which is much easier than fitting an identity mapping through multiple nonlinear layers.

#### Backpropagation Mathematics

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}
= \frac{\partial L}{\partial y} \cdot \left(1 + \frac{\partial F}{\partial x}\right)$$

> **Key Insight**: The gradient contains an identity term $1$! Even when $\frac{\partial F}{\partial x} \to 0$, gradients can still propagate directly to shallow layers through the $1$ term. This is the "gradient highway."

#### Gradient Flow: Plain vs ResNet

![Figure 8-2: Gradient flow comparison — Plain network vs ResNet. ResNet gradients stay stable across layers, while Plain network gradients decay exponentially.](../images/ch08/NN08_gradient_flow.png)
*Figure 8-2: Gradient flow — Plain network gradients decay ~30x (layer 1 to layer 15), while ResNet decays only ~1.7x. This is the power of the identity term $1$!*

Run `code/ch08/NN08_resnet_gradient_flow.py` to reproduce:

```bash
python3 code/ch08/NN08_resnet_gradient_flow.py
```

#### PyTorch Implementation

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut: 1×1 conv when dimensions don't match
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection! ⭐
        out = torch.relu(out)
        return out
```

---

### 8-1-3 Bottleneck Design 🆕

Bottleneck is the **engineering key** that enables ResNet-50/101/152. It uses "reduction → convolution → expansion" to drastically reduce parameters:

$$\underbrace{256}_{\text{high-dim}} \xrightarrow{1\times1} \underbrace{64}_{\text{low-dim}} \xrightarrow{3\times3} \underbrace{64}_{\text{low-dim}} \xrightarrow{1\times1} \underbrace{256}_{\text{high-dim}}$$

#### Design Philosophy

| Step | Operation | Dimension Change | Purpose |
|:-----|:----------|:-----------------|:--------|
| ① Reduce | $1\times1$ conv | $256 \to 64$ | Compress high-dim features, reduce computation |
| ② Convolve | $3\times3$ conv | $64 \to 64$ | Extract spatial features in low-dim space |
| ③ Expand | $1\times1$ conv | $64 \to 256$ | Restore features to high dimension |

> **Analogy**: Bottleneck is like a package sorting center — large packages (256-dim) are compressed into a smaller space (64-dim) for processing, then restored to original size. The task is accomplished at a fraction of the cost!

#### Parameter Comparison

When outputting 256 channels:

| Block Type | Structure | Parameters | Relative Size |
|:-----------|:----------|:-----------|:--------------|
| **Basic Block** | $3\times3 \to 3\times3$ | $\approx 1.18$M | $1\times$ |
| **Bottleneck** | $1\times1 \to 3\times3 \to 1\times1$ | $\approx 70$K | **$1/17$** |

Run `code/ch08/NN08_resnet_bottleneck.py` for the full comparison:

```bash
python3 code/ch08/NN08_resnet_bottleneck.py
```

```python
class Bottleneck(nn.Module):
    expansion = 4  # output channels = planes * 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        # ① 1×1 reduction
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # ② 3×3 convolution (in low-dim space)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # ③ 1×1 expansion
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
        out = F.relu(self.bn1(self.conv1(x)))   # reduction
        out = F.relu(self.bn2(self.conv2(out))) # convolution
        out = self.bn3(self.conv3(out))         # expansion
        out += self.shortcut(x)                 # skip connection
        return F.relu(out)
```

#### When to Use Which Block?

| ResNet Version | Block Type | Depth | When to Use |
|:---------------|:-----------|:------|:------------|
| **ResNet-18/34** | Basic Block | Shallow | Small datasets, quick experiments |
| **ResNet-50/101/152** | Bottleneck | Deep | ImageNet, large-scale data |

> **Key Insight**: The $1\times1$ convolution acts as an "information funnel" — it compresses feature dimensions before the expensive $3\times3$ convolution. Without this design, ResNet-152's parameter count would be prohibitively large.

---

### 8-1-4 Pre-activation ResNet (v2) 🆕

ResNet v2 (He et al., *Identity Mappings in Deep Residual Networks*, 2016) improved the **activation order**:

| Version | Structure | Gradient Path |
|:--------|:----------|:--------------|
| **v1 (Post-activation)** | conv → BN → ReLU → conv → BN → +shortcut → ReLU | Must pass through ReLU |
| **v2 (Pre-activation)** | BN → ReLU → conv → BN → ReLU → conv → +shortcut | **Direct identity path** |

#### Why Pre-activation is Better

v2 moves **BN + ReLU before the convolution**. Benefits:

1. **Cleaner gradient path**: $y = x + F(BN(ReLU(x)))$, gradients flow directly through $x$
2. **More stable training**: BN stays within the residual branch, doesn't pollute the identity path
3. **Easier ultra-deep training**: 1000+ layer ResNet v2 trains more stably

```python
class PreActBlock(nn.Module):
    """Pre-activation: BN → ReLU → conv → BN → ReLU → conv"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
            )

    def forward(self, x):
        # Pre-activation: BN+ReLU first, then conv ⭐
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)  # No final ReLU needed
        return out
```

> **Key Insight**: v1's path is $y = \text{ReLU}(F(x) + x)$ — the ReLU on the main path blocks negative gradient propagation. v2 moves ReLU into the residual branch: $y = x + F(\text{BN}(\text{ReLU}(x)))$, leaving the identity path completely unobstructed. This small change has a huge impact on ultra-deep networks (100+ layers)!

Run `code/ch08/NN08_resnet_block.py` to see both implementations.

---

### 8-1-5 Experiments: Plain vs ResNet 🆕

### 8-1-6 Ablation Studies and Design Choices

Key design decisions in ResNet:
- **Pre-activation** (v2): BN → ReLU → Conv improves gradient flow
- **Bottleneck**: $1\times1 \to 3\times3 \to 1\times1$ reduces parameters
- **Stride placement**: Using stride in $3\times3$ vs $1\times1$ affects information flow
- **Projection shortcuts**: $1\times1$ conv for dimension matching

### 8-1-7 Post-ResNet Architecture Evolution

- **DenseNet**: Dense connections (each layer connects to all subsequent)
- **SENet**: Squeeze-and-Excitation blocks (channel attention)
- **EfficientNet**: Neural Architecture Search (NAS) optimized
- **ConvNeXt**: Modernized ResNet with Transformer-inspired design

### 8-1-8 ResNet Training Experiments

#### Gradient Flow: Plain vs ResNet

Compare gradient magnitudes across 15 layers of Plain vs ResNet:

| Layer | Plain Gradient | ResNet Gradient | Ratio |
|:-----:|:--------------:|:---------------:|:----:|
| 1 | 1.39e+02 | 1.40e+01 | 0.10 |
| 5 | 3.29e+01 | 8.38e+00 | 0.25 |
| 10 | 4.94e+00 | 8.23e+00 | 1.67 |
| 15 | 4.60e+00 | 7.78e+00 | 1.69 |

Plain  — Layer 1 to 15: **30x decay** → gradients vanish.
ResNet — Layer 1 to 15: **only 1.7x decay** → gradients flow!

**Conclusion**: Skip connections reduce gradient decay from 30x to just 1.7x — the power of the identity term $1$!

#### Training Comparison

Train an 8-layer PlainNet vs 8-layer ResNet on synthetic data:

Run `code/ch08/NN08_resnet_training.py` to reproduce:

```bash
python3 code/ch08/NN08_resnet_training.py
```

| Network | Final Loss |
|:--------|----------:|
| PlainNet | 1.0348 |
| ResNet | 0.1685 |

ResNet achieves **6.14x lower loss** than PlainNet!

![Figure 8-3: PlainNet vs ResNet training curves. ResNet converges faster with lower loss — the power of skip connections!](../images/ch08/NN08_training_curves.png)
*Figure 8-3: PlainNet vs ResNet training loss — ResNet achieves ~6x lower loss through skip connections.*

#### Summary

| Metric | PlainNet (8-layer) | ResNet (8-layer) | Improvement |
|:-------|:-------------------|:-----------------|:------------|
| Final Training Loss | $\approx 1.03$ | $\approx 0.17$ | **6.1$\times$** |
| Gradient Decay | $\approx 30\times$ | $\approx 1.7\times$ | **17.6$\times$** |
| Convergence Speed | Slow | **Fast** | ✅ |

> **Takeaway**: These two experiments prove the effect of skip connections from both the "gradient perspective" and the "training perspective." The gradient flow experiment tells you *why* — the identity term $1$ lets gradients reach shallow layers directly. The training experiment tells you *the result* — ResNet actually learns better!

---

## 8-2 Recurrent Neural Networks & Sequence Modeling

### The RNN Cell

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$

### Vanishing Gradient in RNNs

RNNs also suffer from vanishing gradients — gradients must flow through many time steps.

---

## 8-3 From RNN to LSTM/GRU: Gating Mechanisms Explained

> The gradient vanishing problem from 8-2 has a solution: **gated architectures**. LSTM (1997) and GRU (2014) introduced gating mechanisms that allow gradients to flow across time unscathed — much like the skip connections in ResNet, but through time.

### 8-3-1 From Gradient Problems to Gating

As analyzed in 8-2, the RNN's vanishing gradient arises from the double decay in BPTT: the $\tanh'$ scaling ($(0, 1]$) multiplied by the eigenvalues of $\mathbf{W}_{hh}$ (exponential decay when $|\lambda_{\max}| < 1$). When eigenvalues are > 1, gradients explode; < 1, they vanish — making plain RNNs incapable of learning long-range dependencies. The solution is the **gating mechanism** introduced below.

> **Analogy**: Imagine shouting into a canyon — each second the echo travels, it fades. After 10 seconds, it's barely audible — that's gradient vanishing! LSTM is like giving the echo a "relay amplifier" at every point — the signal stays strong across time!

### 8-3-2 LSTM — Long Short-Term Memory

LSTM's core innovation is the **gating mechanism** — three gates (forget, input, output) and a memory cell (Cell State):

$$\mathbf{f}_t = \sigma(\mathbf{W}_f[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(forget gate)}$$

$$\mathbf{i}_t = \sigma(\mathbf{W}_i[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(input gate)}$$

$$\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_C[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C) \quad \text{(candidate memory)}$$

$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t \quad \text{(cell state update)}$$

$$\mathbf{o}_t = \sigma(\mathbf{W}_o[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(output gate)}$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t) \quad \text{(hidden state)}$$

| Gate | Role | Range | Analogy |
|:-----|:-----|:------|:--------|
| Forget gate $\mathbf{f}_t$ | How much old memory to discard | [0, 1] | Selective forgetting |
| Input gate $\mathbf{i}_t$ | How much new information to store | [0, 1] | Selective remembering |
| Output gate $\mathbf{o}_t$ | How much to reveal | [0, 1] | Selective expression |

#### Why LSTM Mitigates Vanishing Gradients

The key is that the cell state update is **additive** rather than multiplicative:

$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$

Taking the derivative with respect to $\mathbf{C}_{t-1}$:

$$\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} = \text{diag}(\mathbf{f}_t) + \cdots$$

There is an **identity pathway**! When the forget gate $\mathbf{f}_t \approx \mathbf{1}$ (the model decides to "remember"), $\partial \mathbf{C}_t / \partial \mathbf{C}_{t-1} \approx \mathbf{I}$. Gradients can propagate across arbitrarily many time steps without decay:

$$\frac{\partial \mathbf{C}_T}{\partial \mathbf{C}_0} \approx \prod_{t=1}^T \text{diag}(\mathbf{f}_t) \approx \mathbf{I} \quad (\text{when } \mathbf{f}_t \to \mathbf{1})$$

Contrast this with RNN's $\partial h_t/\partial h_{t-1} = \tanh' \cdot \mathbf{W}_{hh}^T$ (always a sub-unity multiplication). LSTM's gradient pathway is **additive + identity mapping** — this is precisely the intellectual precursor to ResNet's skip connections!

> **Analogy**: LSTM's memory cell is like a "highway" — if the forget gate is open ($\mathbf{f}_t \approx 1$), information can travel unimpeded from $t=0$ to $t=100$, and gradients flow just as freely. This is the mathematical secret behind LSTM's ability to remember "long, long ago" information.

### 8-3-3 GRU — LSTM's Simplified Version

GRU (Gated Recurrent Unit, Cho et al., 2014) simplifies LSTM's three gates into two — the **update gate** and **reset gate**:

$$\mathbf{z}_t = \sigma(\mathbf{W}_z[\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(update gate)}$$

$$\mathbf{r}_t = \sigma(\mathbf{W}_r[\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(reset gate)}$$

$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}[\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(candidate hidden state)}$$

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t \quad \text{(hidden state)}$$

#### Architecture Comparison

| Feature | RNN | LSTM | GRU |
|:--------|:----|:-----|:----|
| Number of gates | 0 | 3 (forget + input + output) | 2 (update + reset) |
| Memory unit | None | Cell State | None (hidden state only) |
| Parameter count | Lowest | Highest | Medium |
| Gradient vanishing | Severe | Greatly mitigated | Mitigated |
| Training speed | Fastest | Slowest | Medium |

> **Key Insight**: GRU is LSTM's "lean version" — comparable performance with fewer parameters. In practice, the choice depends on task and data size. LSTM's larger parameter count suits data-rich settings; GRU's lighter design is better for smaller datasets or rapid iteration.

---
## 8-4 Attention Mechanism ⭐

### The Core Idea

Attention allows the model to **focus on relevant parts** of the input.

### Query, Key, Value

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{\mathsf{T}}}{\sqrt{d_k}}\right) V
$$

| Component | Analogy | Role |
|:----------|:--------|:-----|
| **Query** | What you're looking for | Current focus |
| **Key** | What's available | What each position offers |
| **Value** | The actual content | What to extract if matched |

### Scaled Dot-Product Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    """Compute attention scores and weighted values"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)
    return output, attention_weights

# Example
batch, seq_len, d_k = 2, 4, 8
Q = torch.randn(batch, seq_len, d_k)
K = torch.randn(batch, seq_len, d_k)
V = torch.randn(batch, seq_len, d_k)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")     # (2, 4, 8)
print(f"Weights shape: {weights.shape}")    # (2, 4, 4)
```

---

## 8-5 Complete Transformer Architecture ⭐

### 8-5-1 Multi-Head Attention

Instead of one attention, use **multiple heads** in parallel:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        batch_size = Q.size(0)

        # Linear projections + split into heads
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads * self.d_head)
        return self.W_o(context)
```

### 8-5-2 Encoder Block

```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention + residual + layernorm
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN + residual + layernorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x
```

### 8-5-3 Transformer Overall Architecture

The attention matrix would show which English words align with which French words.

![Figure 8-4: Transformer attention weight heatmap. Darker cells indicate positions with higher attention — where the model "focuses".](../images/ch08/NN08_attention_visualization.png)
*Figure 8-4: Attention weight visualization — darker regions = higher attention. In translation, each output word attends to different positions in the input.*

---

## 8-6 BERT vs. GPT: Pre-training Paradigms

| Aspect | BERT | GPT |
|:-------|:-----|:----|
| Architecture | Encoder-only | Decoder-only |
| Training | Masked LM (bidirectional) | Autoregressive (left-to-right) |
| Best for | Understanding (classification, QA) | Generation (text, code) |
| Examples | BERT, RoBERTa, ALBERT | GPT-2, GPT-3, GPT-4 |

---

## 8-7 Modern Architecture Design Patterns

| Pattern | Example | Benefit |
|:--------|:--------|:--------|
| Skip connections | ResNet | Gradient flow |
| LayerNorm | Transformer | Stable training |
| Pre-norm | GPT | Better for deep models |
| GELU activation | GPT/BERT | Smooth ReLU variant |

---

## 8-8 Vision Transformer (ViT)

ViT applies the Transformer directly to image patches:

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/p, W/p)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        return x.transpose(1, 2)  # (B, num_patches, embed_dim)
```

---

## 8-9 Modern Architecture Design Principles

1. **Scale**: More data + more parameters + more compute
2. **Normalization**: Every block should normalize
3. **Residual connections**: Essential for deep models
4. **Attention is universal**: Works for text, images, audio, video

---

### 🧪 Exercises

#### Exercise 1: ResNet Skip Connection
Build a ResidualBlock, verify gradient flow by comparing with a plain block.

#### Exercise 2: Attention Visualization
Implement attention and visualize the weight matrix. Which tokens attend to which?

#### Exercise 3: LSTM Implementation
Implement a single LSTM cell from scratch. Compute the gradient through the cell state $\mathbf{C}_t$ and verify the identity pathway when $\mathbf{f}_t \to \mathbf{1}$.

#### Exercise 4: Minimal Transformer
Build a 2-layer Transformer encoder and train it on a simple sequence classification task.

#### Exercise 5: BERT vs GPT Objectives
Implement both masked LM (BERT) and autoregressive LM (GPT) objectives. Compare their training behavior.

## 📦 Chapter Code List

| File | Content | Key Concept |
|:----|:-----|:----------|
| `ch08/NN08_resnet_block.py` | Residual Block + Pre-activation implementation | Skip connections + ResNet v2 |
| `ch08/NN08_resnet_bottleneck.py` | Bottleneck parameter comparison | Bottleneck design philosophy |
| `ch08/NN08_resnet_gradient_flow.py` | Plain vs ResNet gradient flow comparison | Gradient propagation visualization |
| `ch08/NN08_resnet_training.py` | Plain vs ResNet training comparison | Degradation phenomenon reproduction |
| `ch08/NN08_resnet_series.py` | ResNet-18/34/50/101/152 full definitions | ResNet series configurations |
| `ch08/NN08_attention.py` | Attention mechanism from scratch | Attention core |
| `ch08/NN08_attention_viz.py` | Attention weight matrix visualization | Attention visualization |
| `ch08/NN08_positional_encoding.py` | Sinusoidal positional encoding | Positional encoding |
| `ch08/NN08_transformer_encoder.py` | Transformer Encoder full implementation | Transformer core |

---

## 📖 Chapter Summary

### Architecture Evolution

| Era | Architecture | Key Innovation |
|:----|:-------------|:---------------|
| 1998 | LeNet | First CNN |
| 2012 | AlexNet | ReLU + GPU + Dropout |
| 2015 | ResNet | Skip connections |
| 2017 | LSTM/GRU | Gating mechanisms for long-range memory |
| 2017 | Transformer | Self-attention |
| 2020 | ViT / GPT-3 | Transformers for vision & language |

> **One-line summary**: The evolution of modern deep learning architectures = solving vanishing gradients (ResNet) + mitigating long-range forgetting (LSTM/GRU) + capturing global dependencies (Attention) + parallelization (Transformer).

← [Chapter 7](07-chapter7-training-techniques.md) | [Table of Contents](README.md) | [Chapter 9](09-chapter9-large-language-models.md) →
