# On-device LLM Systems Resources

## Knowledge

- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
  官方入门系列。使用范围：Tensors、Build Model、Autograd、Optimization；跳过 Datasets、DataLoaders、Transforms 和完整 FashionMNIST 项目。
- [PyTorch Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
  用于补齐 shape、indexing、reshape、transpose、dtype、device 和广播；阶段 1 的主资料。
- [PyTorch Tensor Attributes](https://docs.pytorch.org/docs/stable/tensor_attributes.html)
  `torch.dtype`、`torch.device` 与 `torch.layout` 的官方定义；第 0003 课只使用 dtype/device 部分。
- [PyTorch Numerical Accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
  用于理解浮点精度、跨设备差异和容差比较；不把逐 bit 相同当作数值对齐标准。
- [PyTorch Build the Neural Network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
  用于 `nn.Module`、子模块、参数与 `forward`；只读与最小 decoder 相关部分。
- [PyTorch Autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
  用于计算图、参数梯度和推理模式；完成一次微型训练后停止深入。
- [PyTorch Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
  用于核实 `zero_grad -> forward/loss -> backward -> step` 的训练步骤与梯度累积行为。
- [PyTorch Autograd Mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
  用于区分 grad mode、no-grad、inference mode 和 `Module.eval()`；第 0004 课只读局部关闭梯度部分。
- [PyTorch RMSNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html)
  第 0005 课的官方 reference；用于核对 last-axis 归一化、逐特征 weight 和输出 shape。
- [PyTorch SiLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html)
  第 0005 课用于理解 gated FFN 中的激活函数 `SiLU(x)=x*sigmoid(x)`。
- [Hugging Face Qwen3 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)
  用于把最小 gated FFN 映射到真实的 `down_proj(act(gate_proj(x))*up_proj(x))` 实现。
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
  gated FFN/GLU 变体的原始研究背景；第 0005 课只需理解门控结构，不要求阅读实验细节。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  Transformer 原始论文。只精读 3.1、3.2、3.4、3.5；训练实验、翻译指标和大部分附录暂时跳过。
- [PyTorch matmul](https://docs.pytorch.org/docs/stable/generated/torch.matmul.html)
  第 0006 课用于核对 batched QK 点积和 attention weight 与 V 的矩阵乘法 shape。
- [PyTorch softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
  第 0006 课用于核对沿 key axis 归一化，每个 query 的权重和为 1。
- [PyTorch masked_fill](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html)
  第 0006 课用于在 softmax 前把未来 key score 设为负无穷。
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
  GQA 原始论文。用于理解 query heads 与 KV heads 的折中及 Cache 收益；不学习 uptraining 实验细节。
- [Hugging Face Transformers: Caching](https://huggingface.co/docs/transformers/main/cache_explanation)
  KV Cache 的官方解释与当前 cache API。用于实现 prefill/decode parity 和理解标准 cache shape。
- [PyTorch `torch.export` API](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html)
  用于理解 AOT 图、example inputs、shape assumptions 与导出结果；阶段 4 才进入。
- [Qwen3.5-4B official model card](https://huggingface.co/Qwen/Qwen3.5-4B)
  模型结构与配置的权威入口。用于确认层数、hidden layout、Gated DeltaNet/Gated Attention 比例；视觉路径前期跳过。
- [Hugging Face Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
  真实 reference 实现。先读 text decoder、Attention、MLP、RMSNorm 与 cache；不要直接修改自动生成文件，应追踪其 modular source。
- [Gated Delta Networks paper](https://arxiv.org/abs/2412.06464)
  Gated DeltaNet 的原始论文。标准 attention 与 cache 通过验收后，只学习 recurrence、state update 和 inference form；并行训练算法暂缓。
- [Official GatedDeltaNet implementation](https://github.com/NVlabs/GatedDeltaNet)
  论文官方 PyTorch 代码。用于把公式映射到 reference kernel 和 recurrent step；不把训练复现作为目标。
- [LiteRT Torch repository](https://github.com/google-ai-edge/litert-torch)
  PyTorch 到 LiteRT 的官方转换与 Generative API 源码。用于 converter、Core ATen coverage、LLM authoring 与 quantization。
- [LiteRT Generative Torch guide](https://ai.google.dev/edge/litert/conversion/pytorch/genai?hl=zh-cn)
  官方端侧生成模型转换入口。用于确定 LiteRT-Torch 与 LiteRT-LM 的职责边界。
- [LiteRT-LM repository](https://github.com/google-ai-edge/LiteRT-LM)
  Android/桌面 LLM Runtime 的主源码与文档。用于 Engine、Session/Conversation、tokenizer、backend、benchmark 和贡献流程。
- [LiteRT-LM build and benchmark guide](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/getting-started/build-and-run.md)
  用于运行 CPU/GPU baseline、固定 CPU 核、测量 prefill/decode 与 peak memory。
- [LiteRT Community Qwen models](https://huggingface.co/litert-community/collections)
  官方生态中的可运行 Qwen2.5/Qwen3 `.litertlm` 基线。优先使用 Qwen3-0.6B，设备空间允许时再用 Qwen3-4B。
- [LiteRT repository](https://github.com/google-ai-edge/litert)
  底层 runtime、Compiled Model API、delegate 和 kernel 的官方实现。只在具体图或 backend 问题出现时按调用链阅读。

## Wisdom (Communities)

- [LiteRT-LM GitHub Issues](https://github.com/google-ai-edge/LiteRT-LM/issues)
  用于搜索已知问题、提交最小复现和验证维护者期望；提交前必须附版本、模型、backend、设备与复现步骤。
- [LiteRT Torch GitHub Issues](https://github.com/google-ai-edge/litert-torch/issues)
  用于 converter、unsupported op、quantization 和 Generative API 问题。
- [Hugging Face Transformers GitHub Issues](https://github.com/huggingface/transformers/issues)
  用于 Qwen3.5 reference、cache 与模型实现问题；只在已完成最小复现和版本核对后发帖。

## Gaps

- 尚未确认 Qwen3.5-4B 是否已有公开、稳定、可直接下载的 LiteRT-LM artifact；当前公开生态可见 Qwen3/Qwen2.5 基线以及 Qwen3.5-9B artifact。Qwen3.5-4B 因而保留为移植与支持缺口验证任务。
- Android 设备的 SoC、GPU、RAM 和系统版本尚未记录，具体 backend 与 4B 可行性需要在阶段 3 基线时实测。
