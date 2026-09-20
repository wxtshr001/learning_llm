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
- 第 0003 课代码已证明会计算 dtype 字节数、迁移 X/W/b 到统一 device/dtype，并在 CUDA 上比较最大绝对误差。
- 第 0003 课最终 92/100 通过；能解释 dtype 精度损失、显式 device 迁移、内存成本与容差门限。
- 第 0004 课最终 97/100 通过；能解释 Parameter 注册、训练四步职责、梯度累积、finite difference 与推理模式。
- 0005 独立实现已证明 RMSNorm/gated FFN 主数据流、parity、梯度和 token-local decoder 关系；RMS 手算与门控数值正确。
- 第 0005 课已正式通过；能区分 RMSNorm statistic/Parameter/output，以及 Linear 构造、`[out,in]` storage 与运行 tensor。
- 第 0006 课已正式通过；能完成单头 Attention 手算、shape、softmax key axis、causal 行为、scale、两个求和轴与 PyTorch 实现，并能把 Attention/output projection 放回 decoder residual stream。
- 已能处理 attention mask 的 device 与 positive-D 代码 contract；对增量解码和完整 KV Cache 仍只具初步直觉。
- 第 0007 课首次提交已证明 MHA/GQA/MQA 边界、Q→KV 连续分组、逐 head 数值、GQA attention 代码、KV Cache 主公式和 decode 带宽意义。
- 第 0007 课已在 0007R 后正式通过；能稳定区分 raw/split/logical/merge tensor、显式 head axis 和 concat/average。
- 第 0008 课 95/100 正式通过；能实现并解释 Qwen half-split RoPE、position/frequency/cos/sin、相对位置 score、GQA 广播以及旋转后 K 的 Cache 位置。

## 当前必须从基础解释的内容

- PyTorch tensor 的 rank、shape、axis/dim、stride、view、contiguous 和 broadcasting 规则。
- Python/PyTorch API 的语义、报错边界和调试方法。
- 完整 KV Cache layout、完整 decoder forward、计算图导出与 runtime 分层。

## 强制教学适配

1. 中文讲解；代码、API 和 tensor 变量名保留英文。
2. 首次出现的符号立即解释。例如不能只写 `[B,N,S,D]`，必须说明 `B=Batch size`、`N=Number of heads`、`S=Sequence length`、`D=Head dimension`。
3. 先使用小而具体的数值 tensor，让每个元素可手工追踪，再推广到字母公式。
4. 优先借助 C++ 行主序数组、线性存储和索引计算建立直觉，同时指出 PyTorch stride/view 与普通 C++ 数组的不同。
5. 一课只引入一个紧密目标；新术语不能依赖另一个尚未教授的新术语。
6. 先运行完整探索示例，再做独立 TODO，最后闭卷迁移到不同数字和 shape。
7. 代码运行成功不代表理解通过；必须同时能解释轴含义、元素映射和为什么这样写。
8. 学习者说“没填的都不会”时，把空白项记为未掌握证据，不得当作遗漏或默认已会。
9. 对 RoPE、位置编码等数学概念，不能因给过一个数字例子就立即跳到公式；必须先补角度/弧度/cos/sin 直觉，再用逐 axis 表和一个 token 的完整数据流建立可操作模型，最后才引入符号公式。
10. 对数学与 PyTorch 紧密结合的课程，优先提供 Jupyter 形式：一个知识点后紧跟一个可执行代码单元与 assert/shape 输出；静态 HTML 可保留为备用阅读版，但不能替代交互验证。
11. 对完整网络、模块组合和多条 residual 数据依赖，必须同时提供四类视觉表示：组件/UML 图说明“谁包含谁”，数据流图说明“按什么顺序执行”，shape 图说明“每一步张量如何变化”，token 依赖图说明“是否跨 token 读取”。ASCII 箭头或只有节点名称的简图不能作为主讲解。

## 当前最近发展区

下一步进入第 0009 课：在已分别掌握 RMSNorm、GQA Attention、RoPE、gated FFN 和 residual 的基础上，组装一个完整 Tiny Decoder Layer，并追踪两条 pre-norm residual 路径。

## 更新规则

- 通过课程门禁后，把新证明的能力写入 `learning-records/`，并同步更新本文件的“已有优势”和“当前必须从基础解释的内容”。
- 局部答错只降低对应知识点的抽象层级，不重复已经证明掌握的全部内容。
- 如果课程因符号未解释、前置知识假设错误或资料失效而无法学习，这是课程设计缺陷；先修课，不把它计为学习者失败。
