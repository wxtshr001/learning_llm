# 0006 课程设计反馈与修订裁决

## 结论

本次不是学习者答题错误，也不评分。0006 初版没有满足仓库已经声明的“从零解释”与 Transformer 系统上下文要求，判定为课程设计缺陷；第 0006 课保持 `in_progress`，门禁成绩仍为空。

## 学习者指出的问题

1. 正文没有真正从零解释 Q/K/V 角色、score、scale、causal mask、softmax axis 和 value 加权。
2. 课程属于 Transformer inference 模块，却没有说明 Transformer 整体、各组件作用、结构嵌入位置与端到端数据流。
3. 仅按课程计划罗列概念无法覆盖学习者尚不知道要问什么的隐含知识，不能服务最终的 Qwen/LiteRT/Android 推理目标。

## 修订内容

- 从 token ids、embedding、decoder layers、final norm、LM Head/logits 建立整体 Causal LM 数据流。
- 解释 pre-norm decoder layer、两条 residual、Attention 与 FFN 的分工，以及 Attention 输出如何回到 `[B,S,H]` 主路径。
- 从 hidden states 的三组 Linear projection 解释 Q/K/V 的来源、角色与类比边界。
- 使用 D=2 和 D=1 的逐元素数字，完整推导 score、`sqrt(D)` scale、causal mask、softmax key axis 与 V 加权。
- 增加 Parameter、activation、KV Cache 的生命周期区别，以及 full-sequence、prefill、decode 三种执行视角。
- 区分 causal mask 与 padding mask，说明 Attention 是动态信息路由而不是笼统的“模型记忆”。
- 引导脚本现在显式输出 raw scores、future mask、masked scores、weights、行和与 output。
- 速查表和闭卷题同步加入 Transformer 结构位置与完整数据流检查。
- `AGENTS.md`、`COURSE-PROTOCOL.md` 与 `CURRICULUM.md` 加入从最终目标反向审查隐含前置知识的规则。

## 门禁影响

- 不记录任何学习者知识失败。
- 不生成 0007。
- 学习者从修订版 0006 重新开始；提交前已明确把 Q/K/V 角色与 Transformer 数据流纳入关键题，总分门禁仍为 80。

## 修订后运行验证

- 初次仅调用系统 `python`，错误报告“宿主没有 PyTorch”；学习者指出用户级虚拟环境后已更正。
- 用户级环境实测 Python 3.11.15、PyTorch 2.13.0+cpu，可正常运行 tensor 计算。
- 使用 PyTorch reference 重新计算 0006：weights 为 `[1,0,0]`、`[1/2,1/2,0]`、`[1/3,1/3,1/3]`，output 为 `[1,2,3]`，未来 V 不影响过去输出的测试通过。
- 独立作业仍保留给学习者完成的 TODO；reference 通过不代替独立作业门禁。

## PyTorch 引导脚本一致性修订

- 学习者指出：既然 PyTorch 环境已经可用，第 0006 课引导脚本不应继续用 `list[list[float]]` 模拟矩阵。
- 裁决：这是课程实现与环境事实不一致，不是学习者知识问题。
- `0006_explore_single_head_attention.py` 已改为真正的 `torch.Tensor` 计算，直接使用 batched matmul、`transpose`、布尔 mask、`masked_fill`、`F.softmax(dim=-1)` 和 weights@V。
- 脚本仍打印全部中间 tensor，并保留 D=2 手算对照、weights 不受 V 改变、未来 V 不影响过去输出三类验证。
- `0006_single_head_causal_attention.py` 继续保留 TODO，确保引导示范与独立作业分离。

---

## 2026-08-12 学习提交评分

### 结论

**87/100，未通过关键项门禁，进入 0006R 针对性复测。**

两个脚本在当前 PyTorch 2.13.0+cpu 环境均能运行，单头 causal attention 的主计算、精确手算、shape 主链、softmax key axis、causal 行为和 scale 已有正确证据。本次不重学这些内容。

未通过的关键项是：独立代码的 device/positive-D contract、第 4 题的完整公式与两个求和轴、第 6 题的 decoder block 完整顺序。根据提交前公布的门禁，不能仅凭总分进入 0007。

### 独立代码：主计算正确，关键 contract 未通过

已复跑原有 12 项测试，全部通过：weights、output、每行和、未来权重和修改未来 V 的因果行为都正确。

仍有两个题面已明确要求的边界问题：

1. `make_causal_mask(sequence_length, torch.device("cpu"))` 忽略了输入 tensor 的 device。CPU 测试会通过，但 CUDA 输入会出现 device mismatch；应使用 `query.device`。
2. 题面要求 `D` 为正数，代码写成 `dim_head < 0`。Tensor shape 不可能为负，所以该判断永远无法拦住 `D=0`。Agent 额外运行 `[1,3,0]` 输入后，代码接受了输入并产生非有限 weights；正确条件是 `dim_head <= 0`。

这两处不否定 Attention 数学主链，但独立代码是关键项，因此需要在 0006R 修正并复跑边界测试。

### 第 1 题：25/25，关键题通过

- 四行 weights 全部正确。
- outputs `[2,3,14/3,15/2]` 全部正确。
- 正确解释 Q/K 相同导致允许位置 score 相同，并由 causal mask 造成每行允许 token 数不同。

### 第 2 题：22/25，关键题通过

- Q/K/V、K 转置、scores、weights 和 output 的 shape 正确。
- 正确说明 scores 的行是 query、列是 key，softmax 沿 key 轴。
- 本课 `make_causal_mask()` 返回共享的 `[S,S]=[5,5]`，再广播到 `[B,S,S]`；答卷写成 `[2,5,5]`。批量 mask 也可以实现相同语义，但不是题目所指当前函数的数字 shape，因此局部扣 3 分，不升级为 shape 概念失败。

### 第 3 题：19/20，关键题通过

- 修改 `V[:,4,:]` 时，在本题 `S=5` 范围内只有 q=4 可能改变，q=0..3 必须不变；“q4 和以后”方向正确，但本题不存在 q=5 以后的位置。
- 修改 `K[:,2,:]` 时 q=2..4 可能改变、q=0..1 必须不变，正确。
- 正确解释 softmax 后再清零会让剩余 key 权重和不再为 1。

### 第 4 题：5/10，关键题未通过

- Q/K 决定匹配、V 提供被读取内容的角色解释正确。
- score 只写成 `Qi*Kj^T`，漏掉题目要求的逐元素 `sum_d` 和 `/sqrt(D)`；也没有说明 score 对 feature 轴 d 求和。
- output 公式方向正确，但没有明确说明它对 key position 轴 j/k 求和。

完整形式应在 0006R 由学习者重新填写，本反馈不代替作答。

### 第 5 题：10/10

正确指出点积累加 D 项造成尺度增长，`sqrt(D)` 用于避免 softmax 过度饱和；缩放不改变 shape；`sqrt(S)` 混淆了 key 数量和点积累加维度。

### 第 6 题：6/10，关键题未通过

- 四类检查证据有效：shape、小数字手算、weights 行和、修改未来 V 的因果行为。
- 正确指出本课 output 尚未映射到词表 logits。
- block 路径写成 `RMSNorm -> QKV projection -> residual`，漏掉题面明确要求的 causal attention 内核和 output projection。Residual 必须加在 Attention 内核与 output projection 之后；第二段 RMSNorm/FFN/residual 顺序正确。

### 0006R 只复测什么

1. 用一个带数字 axis 的公式同时写清 score 对 d 求和、output 对 k 求和。
2. 把 `RMSNorm -> Q/K/V projection -> attention -> output projection -> residual` 与 FFN 子层接成完整路径。
3. 修正 `query.device` 与 `D > 0` 两个代码 contract，并通过对应测试。

通过 0006R 后，第 0006 课正式完成；不再重复第 1、2、3、5 题。
