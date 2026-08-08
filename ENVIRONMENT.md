# LLM 学习环境

## 当前环境

```text
Conda environment: llm
Python: 3.11.15
PyTorch: 2.12.1+cu126
PyTorch CUDA runtime: 12.6
NumPy: 2.4.6
GPU detected by PyTorch: NVIDIA GeForce RTX 2060 SUPER
```

## 使用

```powershell
conda activate llm
python exercises/0000_verify_pytorch.py
```

首次创建环境：

```powershell
conda env create -f environment.yml
conda activate llm
python -m pip install -r requirements-pytorch-cu126.txt
```

CUDA 12.6 wheel 适用于当前 Windows/NVIDIA 环境。另一台机器如果没有 NVIDIA GPU或驱动不兼容，应根据 PyTorch 官方安装页选择对应 wheel；本机差异不得改写共享学习进度。

## 安装决策

PyTorch 2.6 起不再发布新的官方 Conda channel 二进制包。因此使用 Conda 管理隔离环境，并在环境内安装 PyTorch 官方 CUDA wheel；NumPy 等通用数值依赖由 Conda 管理。

这比安装到裸系统 Python 更适合本课程：版本可复现，不污染 Miniforge base 或 `slidelingua`，也便于后续分别处理 Windows PyTorch 学习环境与 LiteRT-Torch 的 Linux 环境要求。
