# On-device LLM 学习起点已完成诊断

学习者具备成熟的 C++/ARM Linux/嵌入式系统能力，以及基础矩阵、简单求导、Attention 和 KV Cache 直觉；但 Tensor layout、PyTorch 实现、完整 decoder inference、计算图转换和 ML Runtime 分层尚未达到独立修改 Qwen/LiteRT 源码的水平。这意味着后续应从“最小 PyTorch + decoder inference”开始，而不是完整 ML 课程，也不能直接跳入 Gated DeltaNet 移植。

## Evidence

- 能正确计算 matmul、标量梯度、单头 attention score 和 causal mask。
- 能解释新 token 只需新 Q、历史 K/V 应缓存。
- 尚不能完成 reshape/transpose、GQA、KV Cache 完整 shape 和 Runtime 故障分层。

## Implications

- 先建立普通 Transformer 的实现与 parity 能力。
- LiteRT 基线应在早期进入，但 Qwen3.5 特有状态推迟到标准 KV Cache 掌握之后。
