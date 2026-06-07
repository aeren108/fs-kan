# FS-KAN: Function-Sharing Kolmogorov–Arnold Networks for Point Cloud Classification

This repository contains an implementation of FS-KAN (Function-Sharing Kolmogorov–Arnold Networks) applied to 3D point cloud classification on the ModelNet40 dataset.

## 1. Introduction

### 1.1 Kolmogorov–Arnold Networks (KAN)

Kolmogorov–Arnold Networks (KANs) are a class of neural networks inspired by the Kolmogorov–Arnold theorem, which states that any continuous function can be decomposed into a finite summation of continuous functions. Unlike standard MLPs that use fixed activation functions on nodes, KANs place learnable activation functions (parameterized as B-splines) on the edges (weights) of the network.

### 1.2 FS-KAN

FS-KAN (Function-Sharing KAN) extends KANs to operate on set data, such as point clouds where order of inputs does not matter. It respects permutation symmetry. The key idea is function sharing: instead of assigning independent learnable functions to every edge, FS-KAN shares KAN functions across set elements to achieve equivariance. Concretely, an equivariant FS-KA layer computes the output for each point $q$ as:

$$\Phi(\mathbf{x})_q = \phi_1(\mathbf{x}_q) + \sum_{p \neq q} \phi_2(\mathbf{x}_p)$$

where $\phi_1$ and $\phi_2$ are learnable KAN functions shared across all points. This construction guarantees that permuting the input points results in the same permutation of the outputs (equivariance). An invariant layer then pools across points to produce a global representation. FS-KAN achieves strong performance on point cloud tasks with significantly fewer parameters than standart KAN baselines.

## 2. Implementation Details

### 2.1 Architecture

The model consists of a stack of 2 equivariant layers followed by an invariant layer and a linear classification head.

Three model variants are implemented:

| Model | Description |
|---|---|
| **FS-KAN Standard** (`fskan_std`) | Uses the standard equivariant layer where $\phi_2$ is evaluated per-point and aggregated via sum-minus-self. |
| **FS-KAN Efficient** (`fskan_eff`) | Uses the efficient equivariant layer where the input is summed before applying $\phi_2$, reducing the computation. |
| **KAN Standard** (`kan_std`) | Non-equivariant baseline that flattens the point cloud to $N \times 3$ and feeds it through a standard KAN with hidden layers $[16, 16, 16]$. |

### 2.2 Hyperparameters

| Hyperparameter | Value |
|---|---|
| Number of equivariant layers | 2 |
| Pooling method | Sum |
| Optimizer | AdamW |
| Learning rate | 0.01 |
| Epochs | 200 |
| Batch size | 32 |

### 2.3 Dataset

I used the **ModelNet40** dataset loaded via PyTorch Geometric. Points are uniformly sampled from mesh surfaces using `SamplePoints` and normalized to a unit sphere with `NormalizeScale`. Experiments are conducted across a grid of:

- **Number of points per object (N):** 64, 128, 256, 512, 1024
- **Training set size:** 200, 400, 600, 800, 1000 (balanced sampling across 40 classes)

## 3. Results

### 3.1 FS-KAN Standard — Best Test Accuracy (%)

| N \ Train Size | 200 | 400 | 600 | 800 | 1000 |
|---|---|---|---|---|---|
| **64** | 44.21 | 52.03 | 54.54 | 58.71 | 62.44 |
| **128** | 46.15 | 53.73 | 57.62 | 59.20 | 61.51 |
| **256** | 48.18 | 55.27 | 58.18 | 60.66 | 62.72 |
| **512** | 42.02 | 54.38 | 58.67 | 61.91 | 63.17 |
| **1024** | 48.54 | 54.74 | 56.52 | 61.51 | 64.55 |

### 3.2 FS-KAN Efficient — Best Test Accuracy (%)

| N \ Train Size | 200 | 400 | 600 | 800 | 1000 |
|---|---|---|---|---|---|
| **64** | 50.16 | 60.25 | 64.06 | 65.24 | 66.21 |
| **128** | 51.78 | 60.78 | 65.76 | 67.50 | 69.33 |
| **256** | 50.24 | 62.32 | 68.15 | 70.87 | 71.27 |
| **512** | 47.45 | 60.86 | 66.37 | 69.89 | 70.38 |
| **1024** | 49.15 | 56.48 | 63.21 | 68.40 | **72.08** |

### 3.3 KAN Standard (Baseline) — Best Test Accuracy (%)

| N \ Train Size | 200 | 400 | 600 | 800 | 1000 |
|---|---|---|---|---|---|
| **64** | 37.36 | 43.80 | 46.60 | 48.58 | 49.64 |
| **128** | 40.56 | 45.50 | 47.61 | 47.97 | 48.58 |
| **256** | 38.61 | 44.53 | 46.88 | 47.73 | 47.97 |
| **512** | 35.53 | 42.30 | 48.54 | 45.62 | 45.46 |
| **1024** | 41.77 | 44.53 | 46.96 | 43.52 | 39.79 |

### 3.4 Discussion

todo


