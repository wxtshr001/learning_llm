# Mission: On-device LLM Systems

## Why
利用已有的 C++、ARM Linux、嵌入式系统和性能分析经验，转向 On-device LLM / ML Runtime 工作。核心载体是完成 Qwen3.5-4B 到 LiteRT-Torch / LiteRT-LM、再到 Android ARM CPU/GPU 的可验证移植。

## Success looks like
- 能独立解释并实现 decoder inference、prefill/decode、GQA 和 KV Cache。
- 能阅读 Qwen3.5 与 LiteRT 相关源码，定位 model、export、lowering、backend 或 runtime 层的问题。
- 在 Android 上建立 FP16/量化模型的正确性、性能、内存和温度基线。
- 形成可复现 Repo、技术报告，以及至少一个高质量 upstream issue 或 PR。

## Constraints
- 每周稳定投入至少 10 小时。
- 可使用 NVIDIA GPU 和 Android 测试设备；当前 PyTorch 实测 GPU 为 RTX 2060 SUPER，显存规格后续以设备查询结果为准。
- 以项目产出和可验证能力为完成标准，不以看完课程为标准。
- 当前 PyTorch 与模型实现基础较薄，系统工程经验较强。

## Out of scope
- 完整的传统机器学习、CV 或 NLP 课程。
- 大规模预训练、分布式训练和完整 RLHF 流程。
- LangChain、复杂 RAG、Multi-Agent 和通用 Prompt Engineering。
- 在掌握标准 decoder inference 前深入 Gated DeltaNet 训练算法。
