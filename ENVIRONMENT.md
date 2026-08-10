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

以上是课程主要 Windows/NVIDIA 环境。2026-08-11 另在 Linux/ARM64 用户级隔离环境实测：Python 3.11.15、PyTorch 2.13.0+cpu、NumPy 2.4.6、CUDA unavailable；CPU matmul、autograd 和第 0006 课 attention reference 均通过。虚拟环境位于仓库之外，不提交其目录内容。

## 使用

```powershell
conda activate llm
python exercises/0000_verify_pytorch.py
```

`0000_verify_pytorch.py` 默认选择当前环境实际可用的 CUDA 或 CPU；如果某项实验明确要求 NVIDIA CUDA，使用：

```text
python exercises/0000_verify_pytorch.py --require-cuda
```

首次创建环境：

```powershell
conda env create -f environment.yml
conda activate llm
python -m pip install -r requirements-pytorch-cu126.txt
```

CUDA 12.6 wheel 适用于当前 Windows/NVIDIA 环境。另一台机器如果没有 NVIDIA GPU或驱动不兼容，应根据 PyTorch 官方安装页选择对应 wheel；本机差异不得改写共享学习进度。

第 0002 课提交的运行输出显示该次练习使用 CPU；这只证明该次 PyTorch 进程的 device，不据此推断主机硬件。0002 探索与作业脚本现按 `torch.cuda.is_available()` 选择 CUDA 或 CPU，使 layout 结论不依赖某一种设备。环境验证脚本同样区分“PyTorch 不可导入”和“PyTorch 可用但没有 CUDA”，不能把后者误报成环境不可用。

## 安装决策

PyTorch 2.6 起不再发布新的官方 Conda channel 二进制包。因此使用 Conda 管理隔离环境，并在环境内安装 PyTorch 官方 CUDA wheel；NumPy 等通用数值依赖由 Conda 管理。

这比安装到裸系统 Python 更适合本课程：版本可复现，不污染 Miniforge base 或 `slidelingua`，也便于后续分别处理 Windows PyTorch 学习环境与 LiteRT-Torch 的 Linux 环境要求。
