# ProblemStatement_TredenceAnalytics
Self-Pruning Neural Network

# 🧠 Self-Pruning Neural Network (Dynamic Model Compression)

## 📌 Overview

In modern deep learning applications, deploying large neural networks is often limited by **memory constraints** and **computational costs**. Traditional pruning techniques remove unnecessary weights *after training*, which adds extra steps and may not yield optimal architectures.

This project introduces a **Self-Pruning Neural Network**, which learns to prune its own connections **during training** using a novel gating mechanism.

---

## 🚀 Key Idea

Each weight in the network is paired with a **learnable gate parameter**:

- Gate value ∈ [0, 1]
- If gate → 0 → weight is effectively removed
- If gate → 1 → weight is fully active

The model learns:
- **What to keep**
- **What to remove**

👉 This results in a **smaller, faster, and efficient model**

---

## 🏗️ Architecture

The model consists of:

- CNN layers (for feature extraction)
- Custom **PrunableLinear layers** (for dynamic pruning)

### 🔹 Prunable Linear Layer

Each layer contains:
- `weight` (learnable)
- `bias` (learnable)
- `gate_scores` (learnable)

Forward pass:
---

## ⚙️ Loss Function

The total loss is a combination of:

### 🔹 Components

1. **Classification Loss**
   - CrossEntropyLoss
   - Ensures prediction accuracy

2. **Sparsity Loss**
   - L1 norm of gate values
   - Encourages gates → 0

---

## 🎯 Objective

- Achieve **high accuracy**
- Maximize **sparsity (pruned weights)**
- Learn an **efficient architecture automatically**

---

## 📊 Evaluation Metrics

| Metric | Description |
|------|-------------|
| Accuracy | Model performance on test set |
| Sparsity (%) | % of weights pruned |
| Trade-off | Accuracy vs Compression |

---

## 📂 Dataset

- **CIFAR-10**
- Automatically downloaded via `torchvision`

```python
torchvision.datasets.CIFAR10(download=True)

