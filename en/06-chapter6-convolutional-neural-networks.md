# Chapter 6: Convolutional Neural Networks

> **Goal**: **Understand why CNNs are better than fully connected networks for images** — the mathematical intuition behind convolution and pooling, and how they solve the problems of fully connected networks.

> © xiefujin · Contact: 490021684@qq.com · Licensed under CC BY-NC-SA 4.0
>
> **Code**: `code/ch06/` (6 files)

> **Figures**: `images/ch06/` (5 images)


> **One-line summary**: CNN = Convolution (local connectivity + weight sharing) → Pooling (dimensionality reduction + translation invariance) → Fully Connected (classification), using a data-driven approach to automatically learn hierarchical features.

---

## 📋 Chapter Learning Objectives

- [ ] Understand the three major limitations of fully connected networks for images
- [ ] Understand the mathematical essence of the convolution operation (sliding dot product)
- [ ] Master Padding, Stride, and multi-channel convolution
- [ ] Understand the role of pooling layers and their backpropagation
- [ ] Be able to build a CNN in PyTorch
- [ ] Understand the principle of convolution layer backpropagation
- [ ] Learn to visualize CNN feature maps

---

## 6-1 Why Do We Need CNNs?

### 6-1-1 Three Limitations of Fully Connected Networks for Images

#### Limitation 1: Loss of Spatial Structure

A $32 \times 32$ color image has $32 \times 32 \times 3 = 3072$ pixel values. A fully connected network flattens it into a 3072-dimensional vector — **the positional relationships between adjacent pixels are completely lost**.

| Before Flattening (spatial structure preserved) | After Flattening (1D vector) |
|:--|:--|
| Pixel matrix, adjacency relationships intact | A long string of numbers, adjacency information lost |

#### Limitation 2: Parameter Explosion

$32 \times 32$ color image → hidden layer of 1024 neurons → roughly **3.14 million parameters**.

If the image is $224 \times 224$ (the ImageNet standard), just one layer has **~150 million parameters**.

$$
224 \times 224 \times 3 \times 1024 = 154,140,672 \text{ parameters}
$$

That's **154 million parameters** for just the first layer!

#### Limitation 3: No Translation Invariance

If the cat in an image shifts 10 pixels to the left, the fully connected network must **relearn all its weights** — because it treats every pixel position as an independent input feature.

### 6-1-2 Three Intuitions for Image Processing

1. **Locality**: adjacent pixels are correlated; distant pixels are not
2. **Translation invariance**: a cat on the left or right side of the image is still a cat
3. **Hierarchy**: edges → textures → shapes → objects

### 6-1-3 Three Key Ideas of CNNs

| Idea | Problem Solved | How |
|:----|:------------|:-------|
| **Local connectivity** | Parameter explosion | Each neuron only looks at a local window (receptive field) |
| **Weight sharing** | No translation invariance | The same filter slides across the entire image |
| **Pooling downsampling** | Too many parameters | Reduce resolution, retain key information |

| Layer Type | Parameters (224×224×3 → 64 filters) |
|:-----------|:-------------------------------------|
| Fully connected | ~150M |
| Convolutional (3×3) | ~1,700 |

That's **~100,000× fewer parameters**!

![Figure 6-1: Fully connected vs CNN connection pattern comparison. In FC each output connects to all inputs; in CNN each output connects only to a local region.](../images/ch06/NN06_fc_vs_cnn.png)

*Figure 6-1: Fully connected (left) vs CNN (right) connection patterns. CNN has far fewer parameters than fully connected.*

---

## 6-2 Understanding CNN with the "Little Genius" Analogy

### 6-2-1 The Little Geniuses' New Job

- Each little genius is responsible for one **filter (kernel)**
- The little genius holds up the filter and slides it across the image
- At each position, it computes the **dot product** between the filter and the local region

```
Convolution steps:
1. The filter (3×3) slides across the input image
2. At each position: element-wise multiplication of the filter with the local image region, then sum (dot product)
3. The result is written to the corresponding position in the output feature map

Sobel vertical edge detection example: filter [1,0,-1; 2,0,-2; 1,0,-1] detects vertical edges

```

### 6-2-2 The Mathematical Essence of Convolution

The convolution of input image $I$ with filter $K$:

$$
(I * K)(i, j) = \sum_{m=-a}^{a} \sum_{n=-b}^{b} I(i+m, j+n) \cdot K(m, n)
$$

where $K$ is a $(2a+1) \times (2b+1)$ filter.

> **Core Insight**: Convolution = filter **element-wise multiplication with a local image region, then sum** = dot product. So in essence, convolution is about **detecting the similarity between a local region and the filter**.

### CNN vs. Fully Connected

| Aspect | Fully Connected | CNN |
|:-------|:----------------|:----|
| Parameters (first layer) | ~150M | ~1,700 |
| Spatial info | Lost (flattened) | Preserved |
| Translation invariance | Must be learned | Built-in (pooling) |

---

## 6-3 Mathematical Details of Convolution Layers

### 6-3-1 The Role of Filters

#### One Filter = One Feature Detector

A filter (kernel) is essentially a small matrix that slides across the input image performing **dot-product operations**. Different filter numeric distributions detect different visual features:

```
Vertical edges (Sobel-x):  Horizontal edges (Sobel-y):  Texture detection (Laplacian):
[ 1  0 -1]                [ 1  2  1]                   [ 0 -1  0]
[ 2  0 -2]                [ 0  0  0]                   [-1  5 -1]
[ 1  0 -1]                [-1 -2 -1]                   [ 0 -1  0]
```

- **Vertical edge filter**: the middle column is 0, left and right are symmetric — it responds to vertical intensity changes
- **Horizontal edge filter**: the middle row is 0, top and bottom are symmetric — it responds to horizontal intensity changes
- **Texture filter**: center is high, surroundings are low — it detects isolated points or blobs

#### CNN Filters Are "Learned"

In traditional image processing, filter values are **hand-crafted** (e.g., Sobel operator for edge detection). But CNN filters are **not manually designed** — they are **automatically learned from data** during training:

- At initialization: random values, not detecting any useful features
- After training: automatically evolve into edge detectors, texture detectors, corner detectors, etc.
- In deep networks: lower-layer filters detect simple features (edges/colors), higher-layer filters compose them into complex features (eyes/wheels)

> **Core Insight**: CNN convolution layers are essentially **learnable filter banks** — instead of programmers hand-crafting features, data drives the filters to automatically discover useful feature patterns.

### 6-3-2 Padding and Stride

#### Padding

- **Same Padding**: pad with zeros around the edges, output size = input size (commonly used)
- **Valid Padding**: no padding, output size shrinks

**Formula**: $n_{\text{out}} = \left\lfloor \frac{n_{\text{in}} + 2p - k}{s} + 1 \right\rfloor$

#### Stride

- **Stride = 1**: move 1 pixel at a time (most common)
- **Stride = 2**: move 2 pixels at a time (downsampling, can replace pooling layers)

### Code Verification: Padding and Stride Experiment

```python
import torch
import torch.nn.functional as F

# Create a single-channel 5×5 image
x = torch.randn(1, 1, 5, 5)

# Compare different configurations
configs = [
    ('Same, s=1', 3, 1, 1),   # Output 5×5
    ('Valid, s=1', 3, 0, 1),  # Output 3×3
    ('Same, s=2', 3, 1, 2),   # Output 3×3
]

for name, k, p, s in configs:
    conv = torch.nn.Conv2d(1, 1, k, padding=p, stride=s, bias=False)
    out = conv(x)
    print(f"{name}: Input 5×5 → conv(k={k},p={p},s={s}) → Output {out.shape[2]}×{out.shape[3]}")
```


| Mode | Stride | Padding | Input | Output |
|:----|:------:|:------:|:----:|:----:|
| Same | 1 | 1 | 5×5 | 5×5 |
| Valid | 1 | 0 | 5×5 | 3×3 |
| Same | 2 | 1 | 5×5 | 3×3 |


> **Core Insight**: Same padding + stride=1 preserves dimensions — this is the most common combination in CNNs. When stride > 1, the output shrinks — this is itself a form of downsampling.


### 6-3-3 Output Size Formula

$$
O = \frac{W - K + 2P}{S} + 1
$$

| Symbol | Meaning |
|:----|:-----|
| $W$ | Input size |
| $K$ | Filter size |
| $P$ | Padding size |
| $S$ | Stride |
| $O$ | Output size |

**Example**: Input $32 \times 32$, filter $3 \times 3$, Stride=1, Padding=1

$$
O = \frac{32 - 3 + 2 \times 1}{1} + 1 = 32
$$

### 6-3-4 Multi-Channel Convolution

Multi-channel convolution: $C_{in}$ input channels × $C_{out}$ filter groups → $C_{out}$ output channels.

**Parameter count**: $K_h \times K_w \times C_{in} \times C_{out}$

### 6-3-5 Manual 2D Convolution in Python

```python
import numpy as np

def conv2d_manual(image, kernel, padding=0, stride=1):
    """Pure NumPy implementation of 2D convolution"""
    H, W = image.shape
    K = kernel.shape[0]

    # Padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant')

    # Output dimensions
    out_h = (image.shape[0] - K) // stride + 1
    out_w = (image.shape[1] - K) // stride + 1
    output = np.zeros((out_h, out_w))

    # Sliding window
    for i in range(out_h):
        for j in range(out_w):
            region = image[i*stride:i*stride+K, j*stride:j*stride+K]
            output[i, j] = np.sum(region * kernel)

    return output

# Test: using Sobel edge detection filter
image = np.random.randn(28, 28)
sobel_x = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])

output = conv2d_manual(image, sobel_x)
print(f"Input shape: {image.shape}")
print(f"Output shape: {output.shape}")
```

```output
Input shape: (28, 28)
Output shape: (26, 26)
```

---

## 6-4 Pooling Layers and Fully Connected Layers

### 6-4-1 Max Pooling

#### The Idea

Take the maximum value within the window as output, ignoring all other values. Typically uses a $2 \times 2$ window with Stride=2, halving the output dimensions.
| Input region | Max value | Output |
|:--------|:-----:|:----:|
| [3,1; 7,2] | 7 | 7 |
| [5,2; 4,1] | 5 | 5 |
| [2,8; 1,4] | 8 | 8 |
| [3,6; 5,2] | 6 | 6 |
| **Output 2×2** | | **[[7,5],[8,6]]** |

#### Why Max Instead of Average?

Taking the maximum means preserving the **strongest feature response** — if a filter detects an edge at a certain position, max pooling retains this detection and passes it to the next layer. This gives CNNs **local translation invariance**: if an object shifts by a few pixels, max pooling can still detect it.

#### Python Implementation

```python
def max_pool2d(image, pool_size=2, stride=2):
    """Manual max pooling implementation"""
    H, W = image.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            # Extract window region
            region = image[i*stride:i*stride+pool_size,
                          j*stride:j*stride+pool_size]
            output[i, j] = np.max(region)  # Take maximum
    return output

# Test
image = np.array([[3, 1, 5, 2],
                  [7, 2, 4, 1],
                  [2, 8, 3, 6],
                  [1, 4, 5, 2]])
print(max_pool2d(image))
# Output: [[7, 5],
#          [8, 6]]
```

### 6-4-2 Average Pooling

#### Difference from Max Pooling

Average pooling takes the **average** within the window instead of the maximum:

```
Average pooling example: Input [[3,1],[7,2]] → Output (3+1+7+2)/4 = 3.25

Takes the mean within the window, smoothing the feature map.

```

#### Comparison: Max Pooling vs Average Pooling

| Property | Max Pooling | Average Pooling |
|:----|:--------|:--------|
| Output | Maximum within window | Mean within window |
| Effect | Preserves strongest feature response | Smooths feature map |
| Translation invariance | Strong | Weak |
| Gradient | Only through max position | Evenly distributed to all positions |
| Common use case | **Mainstream choice** (CNN default) | Global Average Pooling (classification head) |

> **Tip**: Modern CNNs almost universally default to max pooling. Average pooling is mainly used in **Global Average Pooling** — compressing an entire feature map into a single value, replacing fully connected layers in classification tasks.

### 6-4-3 Flatten: From Feature Maps to Classifier

#### Why Is Flatten Needed?

The output of convolution and pooling layers is a **multi-dimensional feature map** (e.g., $64 \times 7 \times 7$), but fully connected layers can only handle **1D vectors**. Flatten's job is to "unroll" the multi-dimensional feature map into a 1D vector as input to the fully connected classifier.

```python
import torch.nn as nn

class CNNWithFlatten(nn.Module):
    """CNN + Flatten + Fully connected classifier"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3)       # (1,28,28) → (32,26,26)
        self.pool = nn.MaxPool2d(2)            # (32,26,26) → (32,13,13)
        self.flatten = nn.Flatten()            # (32,13,13) → (5408,)
        self.fc = nn.Linear(32 * 13 * 13, 10)  # 5408 → 10 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = self.flatten(x)    # Shape transformation: unroll
        x = self.fc(x)
        return x
```

#### Shape Transformation Process

| Stage | Operation | Output Shape | Description |
|:----|:----|:--------|:----|
| Input image | — | (1, 28, 28) | 1-channel grayscale |
| Conv + Pool | Conv + MaxPool | (32, 13, 13) | 32 feature maps |
| Flatten | Unroll | (5408,) | 32 × 13 × 13 |
| Fully connected | Classification output | (10,) | 10 class scores |

> **Core Insight**: Convolution layers do **feature extraction** (preserving spatial structure), fully connected layers do **classification decisions** (requiring 1D vectors). Flatten is the bridge between the two.

## 6-5 Experiencing CNN with Python

### 6-5-1 Building a Simple CNN

#### Manually Implementing a Complete CNN Forward Pass

Below is a manual NumPy implementation of a simple CNN: 1 convolution layer → 1 pooling layer → fully connected layer. This implementation **does not depend on any deep learning framework**, letting you see the complete mathematical process of CNN forward propagation.

```python
import numpy as np

class SimpleCNN:
    """Manually implemented simple CNN (NumPy version, forward pass only)"""

    def __init__(self):
        # Convolution layer: 1 input channel, 4 filters, 3×3
        self.conv1_filters = np.random.randn(4, 3, 3) * 0.1
        # Fully connected layer: feature map size depends on input (assume 28×28 input)
        self.fc = np.random.randn(4*13*13, 10) * 0.1
        self.b = np.zeros(10)

    def conv2d_manual(self, x, filters):
        """Manual 2D convolution"""
        C_out, k_size, _ = filters.shape
        h, w = x.shape[1] - k_size + 1, x.shape[2] - k_size + 1  # Valid convolution
        out = np.zeros((C_out, h, w))
        for c in range(C_out):
            for i in range(h):
                for j in range(w):
                    # Element-wise product of window and filter, then sum
                    region = x[0, i:i+k_size, j:j+k_size]
                    out[c, i, j] = np.sum(region * filters[c])
        return out

    def forward(self, x):
        """Forward pass"""
        # Conv + ReLU
        conv_out = self.conv2d_manual(x, self.conv1_filters)
        conv_out = np.maximum(conv_out, 0)  # ReLU

        # 2×2 Max Pooling (Stride=2)
        C, h, w = conv_out.shape
        pooled = np.zeros((C, h//2, w//2))
        for c in range(C):
            for i in range(h//2):
                for j in range(w//2):
                    pooled[c, i, j] = np.max(conv_out[c, i*2:i*2+2, j*2:j*2+2])

        # Flatten + Fully connected
        flat = pooled.flatten()
        out = flat @ self.fc + self.b
        return out

# Test: input a random 28×28 image
x = np.random.randn(1, 28, 28)
cnn = SimpleCNN()
output = cnn.forward(x)
print(f"CNN output shape: {output.shape}")  # (10,) ← scores for 10 classes
```

Although slow (three nested Python loops), this implementation clearly demonstrates the three-step CNN pipeline: **Convolution → Pooling → Fully Connected**.

![Figure 6-2: Visualization of the convolution operation. The yellow region shows the current coverage area of the convolution kernel on the input; blue is the output feature map.](../images/ch06/NN06_convolution_demo.png)
*Figure 6-2: Convolution operation — kernel (yellow) slides over input, producing output feature map (blue).*

## 6-6 CNN Backpropagation

### 6-6-1 Convolution Layer Backpropagation

**Key insight**: Backpropagation of convolution **is still convolution**.

#### Filter Gradient

$$
\frac{\partial L}{\partial K} = I * \delta
$$

The input $I$ convolved with the error signal $\delta$ gives the filter gradient.

#### Input Gradient

$$
\frac{\partial L}{\partial I} = \delta * \text{rot180}(K)
$$

The error $\delta$ convolved with the filter rotated 180° gives the input gradient.

### 6-6-2 Pooling Layer Backpropagation

#### Max Pooling

During the forward pass, **record the positions of maximum values**. During backpropagation, the gradient is **only propagated back to those positions**:

```python
def max_pool_backward(dout, cache):
    """Max pooling backward pass"""
    dx = np.zeros_like(cache['x'])
    for i in range(dout.shape[0]):
        for j in range(dout.shape[1]):
            # Find the max position
            (max_i, max_j) = cache['max_positions'][i, j]
            dx[max_i, max_j] = dout[i, j]  # Gradient only back to max position
    return dx
```

#### Average Pooling

The gradient is evenly distributed to all positions in the window.

> **Core Insight**: CNN backpropagation follows the same framework as fully connected networks — save intermediate values during the forward pass, apply the chain rule during backpropagation. The only difference is that the "local gradient" computation for convolution layers becomes the convolution operation itself.

---

## 6-7 PyTorch CNN Implementation

### 6-7-1 PyTorch CNN Layers

PyTorch provides ready-to-use CNN layers — no need to manually implement convolution and pooling:

```python
import torch.nn as nn

# Convolution layer: input channels=1, output channels=32, kernel=3×3
conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
# Input: (batch, 1, 28, 28) → Output: (batch, 32, 28, 28)

# Pooling layer: 2×2 window, Stride=2
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# Input: (batch, 32, 28, 28) → Output: (batch, 32, 14, 14)

# Average pooling
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

# Flatten layer
flatten = nn.Flatten()
# Input: (batch, 32, 14, 14) → Output: (batch, 32*14*14) = (batch, 6272)
```

#### nn.Conv2d Parameters Explained

| Parameter | Meaning | Example value |
|:----|:-----|:------|
| `in_channels` | Number of input channels | 1 (grayscale) / 3 (RGB) |
| `out_channels` | Number of output channels (= number of filters) | 32, 64, 128 |
| `kernel_size` | Convolution kernel size | 3, 5 |
| `stride` | Step size | 1 (default) |
| `padding` | Padding | 0 (Valid) / 1 (Same) |

### 6-7-2 Complete CNN Model

#### CNN for MNIST

```python
class CNN(nn.Module):
    """CNN for MNIST (PyTorch)"""
    def __init__(self):
        super().__init__()
        # Conv layer 1: 1→32 channels, 3×3 conv, padding to preserve size
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)   # 28×28 → 28×28
        # Conv layer 2: 32→64 channels
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 28×28 → 28×28
        # Pooling: 2×2, Stride=2
        self.pool = nn.MaxPool2d(2, 2)                  # 28×28 → 14×14 → 7×7
        # Fully connected classifier
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # 28→14
        x = self.pool(torch.relu(self.conv2(x)))   # 14→7
        x = x.view(x.size(0), -1)                  # Flatten
        x = torch.relu(self.fc1(x))                # Hidden layer
        x = self.fc2(x)                            # Output layer (logits)
        return x

model = CNN()
print(model)
print(f"Parameter count: {sum(p.numel() for p in model.parameters()):,}")
```


| Layer | Type | Output Size | Parameters |
|:--|:----|:--------|-----:|
| conv1 | Conv2d(1→32, 3×3) | 28×28 | 320 |
| conv2 | Conv2d(32→64, 3×3) | 14×14 | 18,496 |
| pool | MaxPool2d(2×2, s=2) | 7×7 | 0 |
| fc1 | Linear(3136→128) | 128 | 401,536 |
| fc2 | Linear(128→10) | 10 | 1,290 |
| **Total** | | | **~1.2M** |


#### Training Results

Training this CNN on MNIST for 5 epochs can achieve **98%+ accuracy** — far higher than the 90% from the earlier 2-layer fully connected network. This is CNN's advantage in image tasks.

## 6-8 CNN Black Box Demystified: Visualization Techniques

### 6-8-1 Filter Visualization

#### What Do Trained Convolution Kernels Look Like?

After training, convolution kernels are no longer random noise — they have evolved interpretable patterns. Here's how to inspect trained convolution kernels:

```python
import matplotlib.pyplot as plt

# Assume model is a trained CNN
# model.conv1.weight.shape = (32, 1, 3, 3)  # 32 filters of 3×3

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < 32:
        # Display the i-th convolution kernel
        kernel = model.conv1.weight[i, 0].detach().numpy()
        ax.imshow(kernel, cmap='viridis')
        ax.axis('off')
plt.suptitle('First Layer Convolution Kernels (After Training)')
plt.show()
```

You don't really need to run this code to understand the key point: **the first layer of convolution kernels typically learns edge and texture detectors; the second layer learns shapes and patterns; the deeper you go, the more abstract the features**.

### 6-8-2 Feature Map Visualization

#### What Does Each Layer See?

Plotting the output (feature maps) at each layer shows how the CNN progressively extracts features:

```python
def visualize_feature_maps(model, image):
    """Visualize feature maps at each CNN layer"""
    # Register hooks to capture intermediate outputs
    activations = {}
    def get_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks on conv1 and conv2
    hooks = [
        model.conv1.register_forward_hook(get_hook('conv1')),
        model.conv2.register_forward_hook(get_hook('conv2')),
    ]

    # Forward pass
    model(image.unsqueeze(0))

    # Display conv1 feature maps (first 8 channels)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        fm = activations['conv1'][0, i].numpy()
        ax.imshow(fm, cmap='gray')
        ax.axis('off')
    plt.suptitle('Conv1 Feature Maps (Edge Detection)')
    plt.show()

    for hook in hooks:
        hook.remove()
```

> **Core Insight**: Feature map visualization reveals the CNN's "processing pipeline" — lower layers detect simple features (edges/colors), higher layers compose them into complex patterns (eyes/wheels). This is also why CNNs are called "black boxes": although we can see the output of each layer, it's hard to intuitively understand the specific meaning of high-level features.

### 6-8-3 Grad-CAM: Class Activation Heatmaps

Grad-CAM uses the gradients of the last convolution layer to generate heatmaps that show us **where the network is looking**.

![Figure 6-3: Grad-CAM class activation heatmap. Red regions indicate where the network focuses its attention.](../images/ch06/NN06_gradcam.png)

*Figure 6-3: Grad-CAM visualization — the network focuses on the contours and key parts of the target object.*


---

## 6-9 Classic CNN Architecture Evolution

### 6-9-1 From LeNet to ResNet

| Architecture | Year | Key Innovation | Layers | ImageNet Accuracy |
|:----|:----|:--------|:---:|:-------------:|
| **LeNet-5** | 1998 | First CNN, handwritten digit recognition | 5 | — |
| **AlexNet** | 2012 | ReLU + GPU + Dropout + Data augmentation | 8 | Top-5: 15.3% |
| **VGG-16** | 2014 | Stacking small convolution kernels (3×3) | 16 | Top-5: 7.3% |
| **GoogLeNet** | 2014 | Inception module (multi-scale convolution) | 22 | Top-5: 6.7% |
| **ResNet** | 2015 | Residual connections (breaking the depth bottleneck) | 152 | Top-5: 3.57% |

### 6-9-2 VGG: The Power of Small Convolution Kernels

VGG demonstrated that **stacking multiple small convolution kernels can replace one large kernel**:

- 2 stacked 3×3 convolutions = receptive field of 1 5×5 convolution
- 3 stacked 3×3 convolutions = receptive field of 1 7×7 convolution

**Advantage**: Small kernels have fewer parameters and stronger nonlinearity.

**Parameter comparison**: $3 \times 3 \times C \times C \times 3 \ll 7 \times 7 \times C \times C$

### 6-9-3 Transfer Learning: Standing on the Shoulders of Giants

In practice, few people train CNNs from scratch — the more common approach is **transfer learning**:

1. Load a model pre-trained on ImageNet (e.g., ResNet-50)
2. Freeze most layers (retain learned feature extraction capabilities)
3. Replace the final fully connected layer (adapt to your own task)
4. Fine-tune the last few layers with small amounts of data

```python
import torchvision.models as models

# Load a pre-trained ResNet
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer (adapt to 10-class task)
model.fc = torch.nn.Linear(2048, 10)

# Only train the newly added final layer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

> **Little Genius says**: Transfer learning is like "apprenticeship"! A model that has already "seen it all" on ImageNet has already learned general features like edges, textures, and shapes. You only need to teach it your specific task (like distinguishing cats from dogs), without it having to learn from scratch — just like having a master painter learn a new style is far faster than teaching a complete beginner!


---

## 6-10 Receptive Field and Dilated Convolution

### 6-10-1 What Is the Receptive Field?

The receptive field is one of the most important concepts in CNNs — it defines **how large a region in the input image a single pixel in the output feature map corresponds to**.

$$
\operatorname{RF}_l = \operatorname{RF}_{l-1} + (k_l - 1) \times \prod_{i=1}^{l-1} s_i
$$

where $\text{RF}_l$ is the receptive field at layer $l$, $k_l$ is the kernel size at layer $l$, and $s_i$ is the stride at layer $i$.

```python
def receptive_field(layers):
    """Calculate the receptive field at each CNN layer"""
    rf = 1
    stride_product = 1
    for i, (k, s) in enumerate(layers):
        rf = rf + (k - 1) * stride_product
        stride_product *= s
        print(f"Layer {i+1}: kernel={k}, stride={s} → receptive field={rf}")
    return rf

# Receptive field for the first few layers of VGG-16
vgg_front = [(3,1), (3,1), (2,2), (3,1), (3,1), (2,2)]
rf = receptive_field(vgg_front)
print(f"Total receptive field for VGG's first 6 layers = {rf}")
```


| Layer | Kernel | Stride | Receptive Field | Description |
|:--:|:------:|:------:|:------:|:-----|
| 1 | 3 | 1 | 3 | Base layer |
| 2 | 3 | 1 | 5 | Two 3×3 = one 5×5 |
| 3 | 2 | 2 | 10 | Pooling expands receptive field |
| 4 | 3 | 1 | 12 | |
| 5 | 3 | 1 | 14 | |
| 6 | 2 | 2 | 28 | |
| **Total** | | | **28** | VGG first 6 layers total receptive field |


> **Core Insight**: **The larger the receptive field, the broader the "view" of the network.** Shallow CNNs can only see local textures (small receptive field), while deep CNNs can see entire objects (large receptive field). This is the origin of CNNs' hierarchical features — from edges to shapes to objects.

### 6-10-2 Dilated Convolution

Dilated convolution expands the receptive field **exponentially without increasing the parameter count** by inserting "holes" between kernel elements:

$$\operatorname{RF}_{\text{dilated}} = (k - 1) \times d + 1$$

where $d$ is the dilation rate.

```python
import torch.nn as nn

# Standard 3×3 conv vs Dilated 3×3 conv
standard_conv = nn.Conv2d(64, 128, 3, padding=1, dilation=1)
dilated_conv  = nn.Conv2d(64, 128, 3, padding=2, dilation=2)  # RF = 5×5
large_dilated = nn.Conv2d(64, 128, 3, padding=4, dilation=4)  # RF = 9×9

# Parameter counts are exactly the same!
print(f"Standard conv parameters: {sum(p.numel() for p in standard_conv.parameters())}")
print(f"Dilated conv parameters: {sum(p.numel() for p in dilated_conv.parameters())}")
```

```output
Standard conv parameters: 73856
Dilated conv parameters: 73856  ← Exactly the same!
```

| Dilation rate d | Receptive Field | Parameters | Use case |
|:--------|:-----|:------|:--------|
| d=1 (standard) | 3×3 | 9C² | Basic feature extraction |
| d=2 | **5×5** | 9C² | Expand RF, no extra parameters |
| d=4 | **9×9** | 9C² | Large-scale context |
| d=8 | **17×17** | 9C² | Global information capture |

> **Little Genius says**: Dilated convolution lets the little geniuses "hold hands across gaps"! Originally each little genius could only communicate with adjacent 3×3 regions. With dilated convolution, the little geniuses can skip over the ones in the middle and directly communicate with more distant little geniuses — the parameter count stays the same, but the "social circle" expands!

### 6-10-3 Depthwise Separable Convolution

This is the core of lightweight networks like MobileNet — decomposing standard convolution into two steps:

**Step 1: Depthwise convolution** — each channel convolves independently (no cross-channel)
**Step 2: Pointwise convolution** — 1×1 convolution for cross-channel fusion

**Parameter comparison**: $k^2 C_{\text{in}} C_{\text{out}} \gg k^2 C_{\text{in}} + C_{\text{in}} C_{\text{out}}$

```python
# Standard convolution: 3×3, 32 channels → 64 channels
standard = nn.Conv2d(32, 64, 3, padding=1)
# Parameters = 3*3*32*64 + 64 = 18,496

# Depthwise separable convolution
from torch.nn import Conv2d, Sequential
depthwise = Conv2d(32, 32, 3, padding=1, groups=32)  # Each group = 1 channel
pointwise = Conv2d(32, 64, 1)                         # 1×1 fusion
sep_conv = Sequential(depthwise, pointwise)
# Parameters = 3*3*32*1 + 32*64*1 + 64 = 2,400
# Savings: 18,496 / 2,400 = 7.7×!
```

| Convolution type | Parameters (32→64, 3×3) | Ratio |
|:--------|:------------------:|:----:|
| Standard convolution | 18,496 | 1× |
| Depthwise separable convolution | **2,400** | **7.7× fewer** |
| Actual speedup | — | ~3–4× compute speedup |

---

## 6-11 Data Augmentation: Training Better Models with Limited Data

### 6-11-1 Why Do We Need Data Augmentation?

Data augmentation applies random transformations to original data, **generating infinite training samples from limited labeled data**. It is one of the most effective means of preventing overfitting.

```python
import torchvision.transforms as T

# Typical data augmentation pipeline
train_transform = T.Compose([
    T.RandomResizedCrop(224),        # Random crop
    T.RandomHorizontalFlip(),        # Random horizontal flip
    T.ColorJitter(0.2, 0.2, 0.2),   # Color jitter
    T.RandomRotation(15),            # Random rotation ±15 degrees
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                 std=[0.229, 0.224, 0.225]),
])
```

### 6-11-2 Common Data Augmentation Methods

| Method | Applicable scenario | Effect |
|:----|:--------|:----|
| **Random crop** | Object detection, classification | Improves translation invariance |
| **Horizontal flip** | General (non-text) | Improves symmetry |
| **Color jitter** | Classification | Improves illumination invariance |
| **Rotation** | Symmetric objects | Improves rotation invariance |
| **CutOut / Random Erasing** | Classification | Prevents overfitting |
| **MixUp** | Classification | Smooths decision boundaries |
| **CutMix** | Classification | Combines CutOut and MixUp |

### 6-11-3 Data Augmentation in PyTorch

```python
# Custom CutOut augmentation
class CutOut:
    def __init__(self, size=16):
        self.size = size
    
    def __call__(self, img):
        h, w = img.shape[1:]
        y = np.random.randint(h - self.size)
        x = np.random.randint(w - self.size)
        img[:, y:y+self.size, x:x+self.size] = 0
        return img

# Apply augmentation to DataLoader
augmented_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, transform=train_transform
)
train_loader = DataLoader(augmented_dataset, batch_size=64, shuffle=True)
```

> **Core Insight**: The essence of data augmentation is injecting **prior knowledge** into the training process — we tell the model: "An image is still the same category even if cropped, flipped, or color-shifted." This forces the model to learn truly robust features rather than "memorizing" the training data.

---

## 📦 Chapter Code List

| File | Content | Key Knowledge Point |
|:----|:-----|:----------|
| `ch06/NN06_convolution_demo.py` | Convolution operation from scratch with demo | Convolution principles |
| `ch06/NN06_edge_detection.py` | Sobel operator edge detection | Convolution applications |
| `ch06/NN06_cnn_forward.py` | Complete CNN forward pass implementation | CNN forward computation |
| `ch06/NN06_pooling_strides.py` | Pooling operations & stride experiments | Pooling & stride |
| `ch06/NN06_feature_vis.py` | Feature map & filter visualization | Visualization analysis |
| `ch06/NN06_cifar10_cnn.py` | Full CIFAR-10 CNN training | Complete training pipeline |

![Figure 6-4: Edge detection using the Sobel operator. The original image is processed through horizontal and vertical Sobel convolution kernels to extract edge features.](../images/ch06/NN06_edge_detection.png)

*Figure 6-4: Sobel edge detection results.*

![Figure 6-5: CNN feature map visualization. Lower layers learn edges and textures; deeper layers learn semantic features.](../images/ch06/NN06_feature_maps.png)

*Figure 6-5: Feature map visualization across layers — from edges to semantics.*

---

## 📖 Chapter Summary

### 🧪 Exercises

#### Exercise 1: Manual Convolution Calculation

Input image X = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], convolution kernel K = [[1,0],[-1,1]], stride s=1, no padding. Manually compute the output feature map values.

#### Exercise 2: Implement Edge Detection

Use Sobel kernels to convolve the same image and compare horizontal vs vertical edge detection results:

```python
import torch
import torch.nn.functional as F

sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)

# Use skimage to load an image as input
# Apply both kernels using F.conv2d
# Visualize the results
```

#### Exercise 3: Calculate Convolution Output Size

For a 224×224 input image, using a 7×7 convolution kernel, padding p=3, stride s=2, calculate the output feature map size. Formula: Output size = floor((n + 2p - k)/s + 1).

#### Exercise 4: Pooling Operation Experiment

For a 4×4 input matrix, apply 2×2 max pooling and average pooling (stride 2) respectively. Write out the inputs and outputs to appreciate the downsampling effect of pooling.

#### Exercise 5: Parameter Count Calculation

A typical convolution block in VGG-16: input 256 channels — 3×3 convolution (256 channels, padding=1) — ReLU — 2×2 max pooling. Calculate the parameter count for this convolution layer.

#### Exercise 6 (Challenge): Implement LeNet-5 in PyTorch

LeNet-5 structure: Conv(6@5×5) — AvgPool(2×2) — Conv(16@5×5) — AvgPool(2×2) — FC(120) — FC(84) — FC(10). Implement it using nn.Module and train on MNIST; final test accuracy should exceed 98%.

#### Exercise 7: Ablation Study
Remove the pooling layers from your CNN. How does accuracy change? Parameter count?


### Core Concepts Review

Starting from the limitations of fully connected networks for images, this chapter progressively introduced the core ideas of CNNs:

1. **Why CNNs are needed**: Fully connected networks have three major problems when processing images — parameter explosion (a 256×256 image already has ~20M parameters for one layer), loss of spatial structure (flattening destroys positional relationships between pixels), and lack of translation invariance (a cat shifted 1 pixel left is treated as a completely different input)
2. **Convolution operation**: Using learnable filters sliding across the image to detect local features. The key innovations are **local connectivity** (each neuron only connects to a local region) and **weight sharing** (the same filter uses the same weights at all positions)
3. **Pooling**: Downsampling feature maps, providing translation invariance. Max pooling preserves the strongest response; average pooling smooths the feature map
4. **CNN hierarchical structure**: Lower layers detect edges → middle layers detect shapes → higher layers detect complete objects

| Stage | Technique | Problem Solved |
|:----|:----|:---------|
| FC limitations | — | Spatial loss, parameter explosion, no translation invariance |
| Convolution | Local connectivity + weight sharing | Parameter explosion + translation invariance |
| Pooling | MaxPool / AvgPool | Dimensionality reduction + translation invariance |
| Complete CNN | Conv + Pool + FC | End-to-end feature extraction + classification |
| Visualization | Feature maps + Grad-CAM | Understanding the black box |

> **One-line summary**: CNN = Convolution (local connectivity + weight sharing) → Pooling (dimensionality reduction + translation invariance) → Fully Connected (classification), using a data-driven approach to automatically learn hierarchical features.

---


### Core Formula Quick Reference

| Formula | Description | Use case |
|:----|:-----|:--------|
| $(I * K)_{ij} = \sum_{m=0}^{k_h-1}\sum_{n=0}^{k_w-1} I_{i+m,j+n} \cdot K_{m,n}$ | 2D convolution definition | **Convolution layer core** |
| $n_{\text{out}} = \left\lfloor \frac{n_{\text{in}} + 2p - k}{s} + 1 \right\rfloor$ | Output feature map size | CNN architecture design |
| $\mathbf{Y} = \max_{p,q \in \text{window}} \mathbf{X}_{p,q}$ | Max pooling | Downsampling |
| $\mathbf{Y} = \frac{1}{k_h k_w} \sum_{p,q \in \text{window}} \mathbf{X}_{p,q}$ | Average pooling | Smooth downsampling |
| $\text{params} = (k_h \times k_w \times C_{\text{in}}) \times C_{\text{out}} + C_{\text{out}}$ | Conv layer parameter count | Model complexity analysis |
| $\operatorname{RF} = k + (k-1)(d-1)$ | Dilated convolution receptive field | Multi-scale feature extraction |


← [Chapter 5: Backpropagation](05-chapter5-backpropagation.md) | [Table of Contents](README.md) | [Chapter 7: Training Techniques](07-chapter7-training-techniques.md) →
