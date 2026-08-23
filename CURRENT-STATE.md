# Current Learning State

更新时间：2026-08-23

## 当前门禁

**第 0007 课异议复核后为 89/100；关键题 2 未通过，关键题 6 已更正为通过，当前进入只补 shape/axis 的 0007R。**

0004 独立脚本已由 Agent 复跑，全部检查通过。两处公式抄写/变量名笔误未改变完整数值链与概念结论，因此仅轻微扣分，不安排重复补强。当前进入 RMSNorm 与 gated FFN。

0005R 已证明能够区分 RMSNorm 统计量/Parameter/输出，以及 Linear 构造/weight storage/运行 tensor。当前进入单头 causal attention。

## 已完成

- 初始能力诊断。
- 第 0001 课：Linear shape、元素索引、bias broadcasting、Q projection 迁移。
- 0001R：纠正 `b[i]`/`b[j]` 输出索引误区。
- 0001R2：二维 Linear 逐元素手算复测通过。
- 0002 已证明部分：能计算 `D=H/N`、保持 reshape 元素总数、解释 Sequence/Head 轴交换，并实现 split/merge 主路径。
- 0002R 书面题 95/100：直接索引、具体数值、Q/K/V projection 输出宽度与 head layout 三个关键题全部通过。
- 第 0002 课与 0002R 门禁正式完成。
- 第 0003 课最终 92/100：dtype 转换、字节计算、三输入迁移、CUDA Linear、最大绝对误差和容差判断均通过。
- 第 0004 课 97/100：Parameter 注册、训练四步、梯度累积、finite difference 与推理模式均通过。
- 0005 已证明部分：RMSNorm 手算、last-axis 实现、gated FFN 数据流、数值 parity、梯度与 decoder residual 关系。
- 0005R 90/100：参数与运行 tensor shape 门禁通过，第 0005 课正式完成。
- 第 0006 课首次提交 87/100；手算、shape、causal 行为、scale 与 PyTorch 主计算通过。
- 0006R 95/100：两个求和轴、完整 decoder block 路径与代码 contract 通过，第 0006 课正式完成。
- 第 0007 课异议复核后 89/100：head 映射、逐 head 数值、GQA 代码、KV Cache 主计算、decode 带宽意义与题面内 inference 迁移已证明；只剩 raw/split/merge tensor 阶段待补强。

## 第 0006 课最终能力状态

- 已证明 scaled score 对 feature d 求和，output 对 key k 求和。
- 已证明 Q/K/V projection → causal attention → output projection → residual 的完整 decoder block 顺序。
- 已修正并证明 mask 跟随 `query.device`，且严格拒绝 `D=0`。
- 公式上界和括号有局部书写不规范，但轴语义、代码和运行证据一致，不作为概念缺口。

## 第 0007 课首次验收

- 代码和关键题 1、5、6 通过；关键题 2 未通过，因此第 0007 课尚未完成。
- `k/v raw` 应为 `[B,S,Nkv*D]`；首次答案把它写成 `[B,S,H]`，但在 projection 参数题中又正确算出了 output width，说明问题集中在阶段对应。
- merge 与 `o_proj` 是 rank-3 tensor，没有显式 head axis；`Nq*D` 只是扁平后的特征宽度。
- concat/merge 保留各 head 特征供 `o_proj` 学习组合；平均会不可逆丢失 head 身份和宽度。
- 原 Q6.4 要求完整 GQA+RoPE+Cache 路径，但课程只做了一句话预告，教学覆盖不足；该小题已撤销，不扣分、不作门禁。完整路径与 RoPE 不进入 0007R。

## 课程设计修订

- 学习者指出 0006 初版虽然声明“必须从零解释”，正文却直接给结论性例子与公式；同时没有交代 Transformer 整体、组件嵌入位置和数据流。
- 该问题按 `AGENTS.md` 判定为课程设计缺陷，不记为学习者答错或知识缺口。
- 0006 已改为：整体 Causal LM → decoder layer/residual stream → Q/K/V 来源与分工 → 逐元素 score/scale/mask/softmax/value → PyTorch → full-sequence/prefill/decode 边界。
- 课程协议和模块 B 地图已加入“未知的未知”反向审查，防止后续课程继续孤立罗列概念。
- 学习者指出 PyTorch 位于仓库外的用户级 venv 后，已用 PyTorch 2.13.0+cpu 重新验证 0006 reference：精确 weights/output 与因果测试通过；独立 TODO 仍等待学习者完成。
- 学习者继续指出 0006 引导脚本仍用 Python `list` 模拟矩阵，与已验证的 PyTorch 环境和本课目标不一致；引导脚本已改为完整 `torch.Tensor` 数据流并实际复跑。

## 非阻塞工程提醒

- `split_heads()` 当前先解包 shape 再检查 rank；以后修改生产代码时应先验证再解包，但不将此记录为知识概念缺口。
- `repeat_kv_for_query_heads()` 同样先读取 `shape[1]` 再验证 rank，且 `Nkv=0` 会先除零；按既有裁决作为非阻塞工程提醒，不作为 0007 概念门禁。
- 0003 有一次 0.0010/0.0001 算术笔误，但最大误差与门限结论正确，不作为 dtype/device 概念缺口。
- 0004 的 `training_step()` 类型标注为 `dict[str, float]`，但 prediction/loss 返回标量 Tensor；以后生产代码应使用 `.item()` 满足接口，不阻塞课程门禁。

## 学习者下一步

1. 阅读 `lessons/0007R-gqa-tensor-stages-and-head-axis.html`。
2. 运行 `exercises/0007R_explore_gqa_stages.py`，核对 raw/split/logical/merge 的 rank 与 axis。
3. 完成 `exercises/0007R_gqa_shape_contract.py` 的 TODO 并运行测试。
4. 闭卷完成 `assessments/0007R-gqa-tensor-stages-and-head-axis.md`，填写 `submissions/0007R.md`。

## Agent 下一步

- 收到 0007R 前不生成 0008。
- 0007R 独立脚本与闭卷第 1 题通过，且总分至少 80，才正式完成第 0007 课并进入 RoPE。

## 最近证据

- `submissions/0007-feedback.md`
- `submissions/0007.md`
- `learning-records/0011-mha-gqa-partial-remediation-required.md`

环境配置见 `ENVIRONMENT.md`。硬件信息只代表记录时主机，另一平台必须运行 `exercises/0000_verify_pytorch.py` 自行验证。
