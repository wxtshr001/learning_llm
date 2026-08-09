# 学习者画像

这是课程难度与讲解方式的唯一入口。任何 Agent 在生成、重写或评分课程前都必须阅读本文件。能力结论只依据答卷、代码、解释和测试输出更新，不依据“看过某资料”或主观自评自动升级。

## 学习目标

利用已有的 C++、ARM Linux、嵌入式系统、部署和性能分析经验，建立能够阅读、修改、验证并部署 LLM 推理代码的能力；重点是 PyTorch tensor、decoder inference、KV Cache、模型导出、lowering 和移动端 runtime。

## 已有优势（可以作为类比基础）

- 多年 C++、ARM Linux 与嵌入式开发经验。
- 熟悉连续内存、数组索引、指针、资源管理、并发、性能分析和故障定位。
- 已通过 Linear shape、逐元素计算、输出索引 `b[j]` 与 bias broadcasting 的复测。
- 已能计算 `D=H/N`、检查 reshape 元素总数、解释 Sequence/Head 轴交换，并实现 head split/merge 主路径。
- 已通过 0002R 三个关键概念题：直接使用 `n*D+d` 映射 hidden 索引、追踪 transpose 后具体数值，并从 Q/K/V projection 宽度推出不同 head layout。
- 第 0002 课已正式通过；一次 rank 检查顺序问题由学习者明确归类为未留意代码逻辑，不作为概念能力缺口。
- 对单头 Attention、causal mask、增量解码和 KV Cache 有初步直觉，但尚不能据此跳过 tensor shape 推导。

## 当前必须从基础解释的内容

- PyTorch tensor 的 rank、shape、axis/dim、stride、view、contiguous 和 broadcasting 规则。
- Python/PyTorch API 的语义、报错边界和调试方法。
- GQA、完整 KV Cache layout、decoder forward、计算图导出与 runtime 分层。

## 强制教学适配

1. 中文讲解；代码、API 和 tensor 变量名保留英文。
2. 首次出现的符号立即解释。例如不能只写 `[B,N,S,D]`，必须说明 `B=Batch size`、`N=Number of heads`、`S=Sequence length`、`D=Head dimension`。
3. 先使用小而具体的数值 tensor，让每个元素可手工追踪，再推广到字母公式。
4. 优先借助 C++ 行主序数组、线性存储和索引计算建立直觉，同时指出 PyTorch stride/view 与普通 C++ 数组的不同。
5. 一课只引入一个紧密目标；新术语不能依赖另一个尚未教授的新术语。
6. 先运行完整探索示例，再做独立 TODO，最后闭卷迁移到不同数字和 shape。
7. 代码运行成功不代表理解通过；必须同时能解释轴含义、元素映射和为什么这样写。
8. 学习者说“没填的都不会”时，把空白项记为未掌握证据，不得当作遗漏或默认已会。

## 当前最近发展区

第 0002 课已经完成。当前进行第 0003 课：PyTorch Tensor 的 dtype、device、数据字节数以及 CPU/CUDA、float32/float16 结果对齐。可以复用已掌握的 Linear 和 shape；浮点格式、设备迁移与容差必须从零解释。

## 更新规则

- 通过课程门禁后，把新证明的能力写入 `learning-records/`，并同步更新本文件的“已有优势”和“当前必须从基础解释的内容”。
- 局部答错只降低对应知识点的抽象层级，不重复已经证明掌握的全部内容。
- 如果课程因符号未解释、前置知识假设错误或资料失效而无法学习，这是课程设计缺陷；先修课，不把它计为学习者失败。
