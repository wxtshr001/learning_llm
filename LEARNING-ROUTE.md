# Qwen3.5-4B -> LiteRT-LM -> Android 学习路线

## 结论

推荐采用 **24 周平衡路线**，每周至少 10 小时，总投入约 240～280 小时。第 3 周开始接触真实 Qwen 代码，第 8 周进入 LiteRT-LM，第 13 周正式进入 Qwen3.5；基础学习与主项目不是前后分离，而是逐层合并。

路线的核心循环：

```text
理解一个计算
  -> 手推 shape
  -> 写最小 PyTorch
  -> 与 reference 对齐
  -> export 并检查图
  -> 在 LiteRT/Android 验证
  -> 记录结果和失败边界
```

## 一、诊断驱动的优先级

### P0：进入主项目前必须补齐

1. Tensor layout：batch、seq、heads、head_dim、reshape、transpose、broadcast。
2. 最小 PyTorch：Tensor、`nn.Module`、Parameter、autograd、dtype/device、inference mode。
3. Decoder inference：RMSNorm、GQA、RoPE、FFN、residual、LM Head。
4. Prefill/decode、KV Cache layout、cache update 和 logits parity。
5. eager、export、converter、backend、runtime 的故障分层。

### P1：边做项目边补

- ATen op、decomposition、StableHLO/MLIR 表示。
- 数值误差指标与逐层比较。
- FP16、INT8、INT4 与 state/cache 精度。
- Android CPU/GPU delegate、内存、同步、温度和持续性能。

### P2：主项目后段再学

- Gated DeltaNet recurrence、convolution state、gating 和 hybrid state。
- 自定义 lowering、backend 修复与 upstream contribution。

### 暂时不学

- 完整概率论、完整线性代数证明、CNN/SVM/传统 NLP。
- 从零大规模训练 Transformer。
- SFT/QLoRA：仅在主项目正确性与 Runtime 已闭环后作为可选扩展。
- 分布式训练、RLHF、复杂 RAG、Agent 框架和通用 CUDA 训练栈。

## 二、知识依赖树

```text
Qwen3.5-4B Android Runtime
├─ Model
│  ├─ Tensor shape / dtype / device
│  ├─ PyTorch Module / Parameter / forward
│  ├─ Decoder Transformer
│  │  ├─ RMSNorm / residual / gated FFN
│  │  ├─ RoPE
│  │  └─ MHA -> GQA
│  └─ Qwen3.5 hybrid architecture
│     ├─ Gated Attention -> KV Cache
│     └─ Gated DeltaNet -> recurrent + convolution state
├─ Correctness
│  ├─ full forward == prefill + decode
│  ├─ op/layer/logits/state parity
│  └─ FP32 -> FP16 -> INT8/INT4 error budget
├─ Conversion
│  ├─ torch.export / ExportedProgram
│  ├─ Core ATen graph / decomposition
│  ├─ StableHLO/MLIR / LiteRT graph
│  └─ unsupported op / static state representation
└─ Runtime & Hardware
   ├─ LiteRT-LM Engine / Session / tokenizer / sampler
   ├─ CPU/GPU backend and fallback
   ├─ buffer/state lifetime and memory planning
   └─ TTFT / prefill / decode / RSS/PSS / thermal
```

已有 C++、并发、ARM Linux、内存和 profiling 能力直接挂在 Runtime & Hardware 分支；需要新建的是上方 Model、Correctness 与 Conversion 三个分支。

## 三、三条候选路线

| 维度 | A：快速项目，16 周 | B：平衡路线，24 周 | C：稳固基础，32 周 |
|---|---:|---:|---:|
| 每周投入 | 10～12h | 10～12h | 10～12h |
| PyTorch/Transformer 基础 | 3 周 | 6～7 周，随项目复习 | 10～12 周 |
| 开始真实 Qwen | 第 3 周 | 第 3 周阅读，第 6 周实验 | 第 8 周 |
| 开始 LiteRT-LM | 第 5 周 | 第 8 周 | 第 13 周 |
| 正式进入 Qwen3.5 | 第 8 周 | 第 13 周 | 第 19 周 |
| 风险 | 概念债务高，容易只跑通不理解 | 可控，成果与基础平衡 | 产出慢，可能失去求职节奏 |
| 简历成果速度 | 最快 | 第 10～12 周已有 baseline | 最慢 |
| 面试覆盖 | 偏项目故事 | 模型、图、Runtime 较均衡 | 理论更稳但项目较晚 |

### 推荐 B 的原因

- 诊断显示你不能直接跳到 Qwen3.5，但也没有必要先学几个月通用 ML。
- 10 小时/周足以在 6 周内建立最小 decoder 实现，同时从第 3 周开始读真实源码。
- 第 8 周就进入 Android baseline，能尽早利用你的系统优势并验证设备现实约束。
- 保留约 8 周处理 Qwen3.5 hybrid state、转换失败与数值误差，这些才是主项目最可能产生高价值 issue/PR 的地方。

## 四、推荐路线：24 周

### 阶段 0：建立可复现实验骨架（第 1 周，约 10h）

目标：让后续每个实验都能重复、比较和记录。

实验：

- 建立 `tiny-models/`、`tests/parity/`、`benchmarks/`、`notes/`。
- 固定随机种子，输出 PyTorch、Transformers、LiteRT-Torch 和设备信息。
- 写一个 tensor shape tracer：记录 name、shape、dtype、device、min/max/mean。
- 写第一个 `Linear` forward 与手算对齐测试。

验收：

- 同一 commit、同一 seed 可重复得到相同结果。
- 能解释 `Y[i,j]` 的完整求和公式以及 bias 的索引。
- 测试失败时能看到具体 tensor 名和最大误差。

面试证明：可复现实验设计、测试意识、数值调试基础。

### 阶段 1：最小 PyTorch 与训练机制（第 2～3 周，约 20h）

只学：Tensor 操作、Module、Parameter、forward、autograd、optimizer、dtype/device、`no_grad`。

实验：

- 实现并测试 `Linear`、RMSNorm 和 gated FFN。
- 用一个参数或一个小 MLP 完成 20～100 步训练，打印 loss、gradient、parameter update。
- 对 `[B,S,H] -> [B,S,N,D] -> [B,N,S,D]` 做 shape 单元测试。
- 用 finite difference 检查一个参数的 autograd 梯度。

验收：

- 不看答案写出 `nn.Module`，列出所有 parameter。
- 给定任意合法的 B/S/H/head 数，能推导 reshape/transpose 后的 shape。
- 能区分 forward 计算、backward 求梯度和 optimizer 更新。
- 能说明推理为何关闭 autograd。

面试证明：能阅读和编写模型组件，不只是调用 `pipeline()`。

### 阶段 2：Tiny Decoder 与 KV Cache（第 4～7 周，约 40h）

实现一个：1 layer、hidden 128、4 Q heads、2 KV heads、RMSNorm、RoPE、gated FFN、causal mask、LM Head 的小模型。

实验顺序：

1. 单头 causal attention，与手算小矩阵对齐。
2. MHA，再改为 GQA，验证 query/KV head 映射。
3. 加 RMSNorm、RoPE、FFN、residual。
4. 实现 full forward。
5. 实现 prefill + 单 token decode + KV Cache。
6. 比较每个位置 logits，以及连续 100 token 的 greedy generation。

验收：

- 不参考实现写出 causal GQA，并解释全部 tensor shape。
- 对每层 Cache 明确 `[B,KV_heads,seq,head_dim]` 或代码采用的等价 layout。
- full forward 与 prefill+decode 的 logits 在设定容差内一致。
- 能计算不同 context length、dtype、layer 数下的 KV Cache 内存。

面试证明：Transformer inference、GQA、KV Cache、prefill/decode。

### 阶段 3：真实 Qwen + LiteRT-LM Android 基线（第 8～10 周，约 30h）

先使用官方生态已有的 Qwen3-0.6B `.litertlm`；若设备资源允许，再增加 Qwen3-4B。不要先转换 Qwen3.5。

实验：

- 在桌面和 Android 跑通同一模型与固定 prompt。
- 记录模型加载、TTFT、prefill tok/s、decode tok/s、RSS/PSS。
- 比较 CPU/GPU；记录完整 delegate/fallback 信息。
- 运行 5 分钟和 15 分钟，采集温度、频率和 tok/s 曲线。
- 从 LiteRT-LM C++ 入口追踪 Engine、Conversation/Session、tokenizer、sampler 和 backend。

验收：

- 一条命令可复现实验，并生成 CSV/Markdown 表。
- 能解释 prompt 从文本到 token、prefill、decode、sampling、文本输出的生命周期。
- 能区分“GPU 算子更快”与“整条生成链路更快”。
- 能说明至少一个 CPU/GPU 差异的证据，而非只给猜测。

面试证明：Android LLM 部署、CPU/GPU benchmark、thermal/performance analysis。

### 阶段 4：`torch.export` 与转换链路（第 11～13 周，约 30h）

从 Tiny Decoder 开始，不从 4B 模型开始。

实验：

- 依次 export `Linear -> RMSNorm -> RoPE -> Attention -> Cache Update`。
- 每加入一个组件，保存 eager 输出、ExportedProgram、转换结果和 LiteRT 输出。
- 人为制造 data-dependent control flow、动态 shape 或 unsupported op。
- 建立故障分类表：model authoring / export / decomposition / legalization / backend / runtime。

验收：

- 能查看 export graph，把关键 ATen op 映射回模型计算。
- 转换失败后能指出最先失败的层，并提供最小复现。
- eager、exported program 与 LiteRT 的输出在容差内一致。
- 不要求系统学习完整 MLIR；只需读懂当前失败附近的表示。

面试证明：AOT export、图调试、unsupported op 定位、最小复现。

### 阶段 5：Qwen3.5 hybrid state（第 14～17 周，约 40h）

先走 text-only、单 layer、单 step、FP32/FP16 CPU；视觉编码器暂不进入。

实验：

- 从 model config 生成 layer map，确认 Gated DeltaNet 与 Gated Attention 的排列。
- hook 每层 input/output、attention KV、recurrent state、convolution state。
- 分别实现或抽取：单步 Gated Attention reference、单步 Gated DeltaNet reference。
- 比较 HF full sequence 与 recurrent single-step。
- 先用缩小 config/随机权重，再加载真实 4B 权重。

验收：

- 能解释 Attention layer 为什么使用 KV Cache，DeltaNet layer 为什么使用另一类 state。
- 单层单步输出与 HF reference 对齐。
- 10、100、500 step 时能画出 state/output error 曲线。
- 能定位误差首次出现在哪一层、哪个 state、哪个时间步。

面试证明：混合模型架构、递归状态、reference-driven debugging。

### 阶段 6：Qwen3.5-4B LiteRT bring-up（第 18～21 周，约 40h）

推进顺序固定为：

```text
最小 config + random weights
-> text-only
-> FP32/FP16 CPU
-> single decode
-> prefill
-> greedy generation
-> 真实 4B weights
-> GPU
```

实验：

- 明确 Qwen3.5 reference 与 LiteRT authoring representation 的差异。
- 为 hybrid state 设计静态签名、更新策略和导出测试。
- 建立逐层/逐 state 对比工具，而不是只比较最终文本。
- 每个转换失败形成一个最小模型和归属层判断。

验收：

- 最低成功线：最小 Qwen3.5 text decoder 的 FP16 CPU single-step parity。
- 标准成功线：Qwen3.5-4B text-only CPU 完整生成正确。
- 进阶成功线：GPU 跑通或形成证据充分的 backend/converter upstream issue/PR。
- 若设备装不下或公开 API 暂不支持，必须用可复现数据证明边界；这仍是合格项目成果。

面试证明：真实模型移植、跨框架数值调试、状态建模、工程边界判断。

### 阶段 7：量化、性能与项目收束（第 22～24 周，约 30h）

实验：

- 建立 FP32/BF16 reference -> FP16 -> INT8 -> INT4 的 correctness ladder。
- 指标至少包括 max abs error、cosine similarity、top-k overlap、top-1 match、state drift。
- Android 测量 TTFT、prefill/decode、RSS/PSS、512/2048 context、5/15 分钟热稳定性。
- 检查 allocation、memcpy、CPU/GPU sync、fallback、thread affinity 和 state residence。

验收：

- benchmark 包含环境、版本、模型、量化、context、backend、warmup 和重复次数。
- correctness 报告能说明容差为何合理，并展示误差随生成长度的变化。
- Repo 可由第三方根据 README 重现至少一条 CPU 和一条 GPU/失败边界结果。
- 形成一份 upstream issue、PR 或经过充分检索后确认的技术缺口报告。

面试证明：量化误差、端侧性能工程、可复现实验、开源协作。

## 五、每周固定节奏

按 10 小时最低投入：

| 时间 | 内容 |
|---:|---|
| 2h | 官方文档/论文，只读当前实验需要的部分 |
| 4.5h | 编码与 debug |
| 1.5h | parity、单元测试和错误注入 |
| 1h | benchmark 或源码调用链追踪 |
| 0.5h | 闭卷 shape/概念检索练习 |
| 0.5h | 更新实验记录、图表和下周问题 |

每周必须产生一种可观察结果：通过的测试、失败的最小复现、性能数据表、调用链图或数值误差曲线。

## 六、阶段门禁

- 没通过 Tiny Decoder parity：不进入 DeltaNet。
- 没跑通官方 `.litertlm` baseline：不把 Android 问题归咎于 Qwen3.5。
- 没有 FP32/FP16 reference parity：不进入 INT4。
- 没有单层/单步对齐：不跑 4B 完整生成。
- 没有版本、设备和最小复现：不提交 upstream issue。

门禁不是要求完美；它负责防止多个未知问题叠在一起。

## 七、主项目沉淀结构

```text
README.md
docs/
  architecture.md
  qwen35-porting.md
  hybrid-state.md
  conversion-pipeline.md
  quantization.md
  android-performance.md
tiny-models/
tests/
  tensor-shapes/
  numerical-parity/
  state-consistency/
tools/
  trace-tensors/
  compare-models/
  benchmark/
converter/
runtime/
android/
benchmarks/
  cpu-gpu.md
  fp16-int4.md
  thermal.md
```

每个阶段先把代码和原始数据提交，再写结论；报告中的数字必须能回溯到命令、commit、设备和原始输出。

## 八、求职与面试节点

- 第 7 周：可证明自己能实现 Transformer inference 与 KV Cache。
- 第 10 周：简历加入 LiteRT-LM Android CPU/GPU baseline 与 benchmark。
- 第 13 周：可讨论 `torch.export`、ATen graph 和转换错误分层；开始少量试投。
- 第 17 周：可讲 Qwen3.5 hybrid architecture、两类 state 和逐层 parity。
- 第 21 周：把 Qwen3.5 bring-up 或明确支持缺口作为主项目。
- 第 24 周：正式投递 On-device LLM、ML Runtime、Edge AI Runtime、Inference Engineer 岗位。

不等待项目“100% 完美”再面试；但简历中的每项声明都必须由代码、测试、数据或公开 issue/PR 支撑。

## 九、路线完成标准

路线不以 24 周自然结束为完成，而以以下证据结束：

- 独立实现 Tiny Decoder，full forward 与 cache decode 对齐。
- 在 Android 上复现一个官方 Qwen LiteRT-LM baseline。
- 能从 eager 一直追踪到 export、LiteRT graph 和 backend。
- 对 Qwen3.5 至少完成最小 hybrid-state FP16 CPU parity。
- 对 Qwen3.5-4B 完成运行、修复、PR，或形成可复现的支持边界报告之一。
- Repo、benchmark 和技术报告能够被第三方复核。
