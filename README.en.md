# 📘 The Mathematics of AI Neural Networks — English Edition

> **Author**: xiefujin · **Contact**: 490021684@qq.com
> *An intuition-first journey through the math behind neural networks, with Python and PyTorch*

## 🌐 Language

- [**中文版 (Chinese)**](README.md)

## Overview

This English edition is a direct translation and adaptation of the Chinese book *《AI神经网络的数学——用Python、PyTorch讲解》*. It covers the mathematical foundations of neural networks — from the M-P neuron to Transformers and LLMs — with a strong emphasis on **mathematical intuition**, **derivation clarity**, and **practical PyTorch implementation**.

## Directory Structure

```
en/
├── README.md                              # English edition README
├── 00-foreword.md                         # Foreword & reading guide
├── 01-chapter1-neural-network-ideas.md    # Neural network concepts
├── 02-chapter2-mathematical-foundations.md     # Math fundamentals
├── 03-chapter3-pytorch-basics-tensor-autograd.md  # PyTorch basics
├── 04-chapter4-optimization.md            # Optimization
├── 05-chapter5-backpropagation.md         # ⭐ Backpropagation (core)
├── 06-chapter6-convolutional-neural-networks.md  # CNN
├── 07-chapter7-training-techniques.md     # Training techniques
├── 08-chapter8-modern-architectures.md    # Modern architectures
├── 09-chapter9-large-language-models.md   # Large language models
└── appendix/
    ├── A-mathematical-review.md            # Mathematical review
    ├── B-pytorch-api.md                    # PyTorch API reference
    ├── C-notation-conventions.md           # Notation conventions
    ├── D-common-functions.md               # Common functions
    ├── E-exercise-answers.md               # Exercise answers
    └── F-recommended-reading.md            # Recommended reading
```

## Chapter Overview

| Chapter | Core Content | Status |
|:--------|:-------------|:------:|
| Foreword | Positioning, features, reading guide | ✅ Outline |
| Ch 1 | M-P neuron, activation functions, forward propagation | 📝 Translating |
| Ch 2 | Functions, vectors, matrices, derivatives, chain rule | 📝 Translating |
| Ch 3 | Tensor, Autograd, nn.Module, DataLoader | 📝 Translating |
| Ch 4 | Loss functions, Softmax, NumPy → PyTorch | 📝 Translating |
| Ch 5 ⭐ | δ recursion, backprop derivation, MNIST | 📝 Translating |
| Ch 6 | Convolution, pooling, feature maps, PyTorch CNN | 📝 Translating |
| Ch 7 | Optimizers, regularization, BatchNorm | 📝 Translating |
| Ch 8 | ResNet, RNN, Attention, Transformer | 📝 Translating |
| Ch 9 | Autoregressive generation, KV Cache, LoRA | 📝 Translating |

## Shared Resources

This English edition shares all **images** and **code** with the Chinese edition:

- **Images**: `images/chNN/` (e.g., `images/ch01/NN01_network_structure.png`)
- **Code**: `code/chNN/` (e.g., `code/ch01/NN01_mp_neuron.py`)
- **Notebooks**: `notebooks/chNN/`

## Build

To generate an EPUB or PDF of the English edition:

```bash
cd "$BOOK"
python3 tools/build_epub.py --lang en
# or
python3 tools/build_pdf.py --lang en
```

## Related Paper

A preprint paper based on this book has been published:

> **From M-P Neuron to Transformer: A Minimal Mathematical Path with Intuition-First Pedagogy for Deep Learning Education**
>
> **DOI:** [https://doi.org/10.6084/m9.figshare.32692401](https://doi.org/10.6084/m9.figshare.32692401)

## License

This work is licensed under **CC BY-NC-SA 4.0**. See [LICENSE](LICENSE) for details.
