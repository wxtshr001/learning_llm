# 自适应课程地图

这是一张能力地图，不是固定日历。每一课只有通过门禁才解锁下一课；出现局部断层时会插入 `R` 编号的补强课。

## 模块 A：Tensor 与最小 PyTorch

| 课次 | 能力成果 | 门禁证据 |
|---|---|---|
| 0001 | 从 shape 和索引实现 Linear | 手算、纯 Python 实现、迁移到 Q/K/V 投影 |
| 0002 | 掌握 reshape、transpose 与 contiguous layout | `[B,S,H] -> [B,N,S,D]` 全程推导和代码测试 |
| 0003 | 使用 PyTorch Tensor、dtype、device | 同一计算在 CPU/CUDA 对齐并解释 dtype 差异 |
| 0004 | 理解 Module、Parameter、autograd、optimizer | 小模型梯度与 finite difference 对齐 |
| 0005 | 实现 RMSNorm 与 gated FFN | 与 PyTorch reference 数值对齐 |

可能补强：Python 索引、循环/列表、浮点误差、广播、链式求导。

## 模块 B：Transformer inference

本模块不是六个独立算子练习，而是逐步组装同一条 decoder-only Causal LM 数据流：

```text
token ids → embedding → repeated decoder layers → final norm → LM Head/logits
                         ├─ RMSNorm → Attention → residual
                         └─ RMSNorm → FFN       → residual
```

每课都必须说明新部件的上游、下游、Parameter/activation/state 边界，以及 full-sequence、prefill、decode 三种执行视角；后续课可复用已证明的基础，但不能省略系统连接。

| 课次 | 新增能力 | 嵌入整体的位置 | 门禁证据 |
|---|---|---|---|
| 0006 | 单头 causal attention | hidden states 经 Q/K/V projection 后的跨 token 信息路由内核 | 手算、代码、因果行为与 decoder 数据流解释一致 |
| 0007 | MHA 与 GQA | 并行多个 attention 子空间，合并后经 output projection 回 residual stream | 正确映射 Q heads/KV heads，并计算 Cache 节省 |
| 0008 | RoPE | score 前作用于 Q/K，把 position 写入匹配关系，不直接旋转 V | 实现旋转并解释 position 对 Q/K 和 score 的影响 |
| 0009 | Tiny Decoder Layer | 组装两段 pre-norm 子层及 residual，形成可堆叠 layer | RMSNorm、Attention、FFN、residual 全部 shape 与执行顺序正确 |
| 0010 | Prefill、decode 与 KV Cache | 把 full-sequence layer 改为请求内有状态的增量执行 | full forward 与 incremental logits parity，能解释 Cache 生命周期 |
| 0011 | Tiny Causal LM | layer 前加 embedding，layer 后加 final norm/LM Head/生成循环 | 完成从 token ids 到 logits 再到 greedy next token 的闭环 |

只有 0010 通过后才进入 Qwen3.5 state。

## 模块 C：真实源码与 LiteRT 基线

| 课次 | 能力成果 | 门禁证据 |
|---|---|---|
| 0012 | 阅读 Hugging Face Qwen3 模型 | 画出调用链并记录关键 tensor |
| 0013 | 跑通官方 Qwen LiteRT-LM | 桌面与 Android 固定 prompt 可复现 |
| 0014 | 建立 Android benchmark | TTFT、prefill、decode、RSS/PSS、thermal 数据表 |
| 0015 | 追踪 LiteRT-LM Runtime | Engine、Session、tokenizer、sampler、backend 调用链 |

## 模块 D：Export、Lowering 与正确性

| 课次 | 能力成果 | 门禁证据 |
|---|---|---|
| 0016 | `torch.export` 最小模型 | 解释 ExportedProgram 和关键 ATen op |
| 0017 | Tiny Decoder export | eager/export 输出对齐 |
| 0018 | 故障分层与最小复现 | 正确区分 authoring/export/converter/backend/runtime |
| 0019 | 数值对齐工具 | op/layer/logits/state 多级比较报告 |

## 模块 E：Qwen3.5 主项目

| 课次 | 能力成果 | 门禁证据 |
|---|---|---|
| 0020 | Qwen3.5 layer map | 识别 Gated DeltaNet 与 Gated Attention 排列 |
| 0021 | Gated DeltaNet 单步 recurrence | 单层单步与 reference 对齐 |
| 0022 | convolution/recurrent hybrid state | 10/100/500 step state drift 曲线 |
| 0023 | 最小 Qwen3.5 LiteRT authoring | random weights、text-only、CPU single-step |
| 0024 | Qwen3.5-4B bring-up | 完整运行或可复现支持边界 |
| 0025 | FP16/INT8/INT4 correctness | 精度阶梯与误差预算 |
| 0026 | Android CPU/GPU 优化 | 性能、内存、同步、fallback、thermal 报告 |
| 0027 | Repo 与 upstream contribution | 第三方可复现，形成 issue/PR/技术缺口报告 |

## 动态调整原则

- 如果 Tensor/PyTorch 进展快，不重复基础题，提前进入 decoder。
- 如果数学正确但代码受 Python 阻碍，插入 Python 适应课，不降低模型问题难度。
- 如果 LiteRT 官方 baseline 尚未跑通，不进入 Qwen3.5 converter 调试。
- 如果设备无法承载 4B，改用缩小 config 做正确性，并把真实设备边界写成可验证成果。
