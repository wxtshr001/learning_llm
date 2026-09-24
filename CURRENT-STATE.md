# Current Learning State

更新时间：2026-09-25

## 当前门禁

**0008 95/100，通过；第 0009 课 Tiny Decoder Layer 已收敛为四模块可视化教程，当前等待学习者提交。**

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
- 0007R 98/100：raw/split/logical/merge、显式 head axis 与 concat/average 门禁通过，第 0007 课正式完成。
- 第 0008 课 95/100：独立 RoPE 代码与关键题 1、2、3 全部通过；二维旋转、half-split、相对 score、GQA 广播、prefill/decode position 和 Cache 位置已证明。

## 第 0008 课最终能力状态

- 已证明 Qwen half-split 的逐 axis 旋转、默认频率构造和 Q/K 数值实现。
- 已证明相同相对位置差的 score 效应，并区分 RoPE 与 causal mask。
- 已证明不同 Nq/Nkv 下 cos/sin 的 head-axis 广播以及 RoPE → Cache → attention 执行顺序。
- 独立脚本在 CUDA 复跑通过；额外 CPU、不同 shape、默认频率和非法输入测试通过。
- 范数平方项顺序写反但总和与结论正确，属于局部抄写。
- `position_ids` 应为运行输入，不是请求 state；请求 state 是历史 K_rope 与 V。该错误位于非关键题，不触发补强。

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

## 第 0007 课最终能力状态

- 已证明 MHA/GQA/MQA 边界、连续 Q→KV 分组、projection width 和逐 head attention 数值。
- 已证明 q/k/v raw、split、logical K/V、scores/weights、head output、merge 与 `o_proj` 的 rank/axis。
- 已证明 merge 后没有显式 head axis，以及 concat 不能替换为 average。
- 已证明 Cache 保存未展开的 `Nkv` heads、容量公式与 decode 带宽意义。
- 0007R 把 `(1,6,12)` 称作“2D 矩阵”是局部术语笔误；完整 shape 和 axis 结论正确。

## 第 0008 课范围

- 从二维向量旋转开始解释 RoPE，不依赖此前一句式预告。
- 实现 Qwen/Hugging Face half-split `rotate_half`，明确 D=4 时配对为 `(0,2)`、`(1,3)`。
- 从 `position_ids [B,S]` 构造 `inv_freq [D/2]`、angles、cos/sin `[B,S,D]`。
- 通过 `[B,1,S,D]` 广播同时旋转 `Q [B,Nq,S,D]` 与 `K [B,Nkv,S,D]`，V 不旋转。
- 用手算点积证明相同相对位置差产生相同位置效应，并区分 RoPE 与 causal mask。
- 明确 full-sequence/prefill/decode 的 position_ids 以及旋转后 K 在 Cache 路径中的位置；不提前实现完整 Cache。

## 课程设计修订

- 2026-09-25：学习者指出 0009 闭卷题缺少明确前提，尤其第 4.1 题没有说明“哪个 layer 的 Parameter”。已为整卷补充本层结构、输入、无 bias 和 full-sequence 边界；逐题澄清 `Nq*D=H` 时 `o_proj` 的作用、shape 阶段、固定输入条件、residual 引用及 Cache 假设。分值和关键门禁未变；这是题面设计修订，不记录为学习者知识缺口。修订细节见 `submissions/0009-feedback.md`。
- 2026-09-20：学习者指出 0009 第二版互动形式过多、信息增量不足。已删除组件聊天、逐步消息动画和重复测验，将五模块收敛为四模块；主教程现明确覆盖 q/k/v raw → split → scores → head output → merge、全部符号、cos/sin 分类以及 full-sequence/prefill/decode 边界。课程设计规则改为“按问题选择最少有效图”，不再要求同一结论使用多种视觉形式重复表达。
- 2026-09-20（已被上条修订取代）：学习者指出 0009 初版过于简略，曾增加五模块互动教程；随后确认该版形式过多、信息不足，因此不再作为当前设计。
- 2026-09-05：按学习者要求，0008 增加 Jupyter 主教程 `notebooks/0008-rope-from-rotation-to-gqa.ipynb`；35 个单元按“知识点 → 先预测 → Python/PyTorch 代码 → assert 验证”组织，14 个代码单元已在 `llm` 环境顺序执行通过。原 HTML 保留为备用阅读版，作业与门禁不变。
- 学习者指出 0008 初版虽有二维数字例子，仍过快跳到 `inv_freq`、旋转矩阵和相对位置公式，缺少可操作的前置直觉；判定为课程设计缺陷，不记为学习者知识缺口。
- 0008 已重写为：角度/弧度/cos/sin 表 → `[3,4]` 旋转 90° → D=4 half-split 逐 axis 表 → 完整逐元素乘加 → position_ids → 简化 inv_freq → 默认公式 → 单 token Q/K → GQA broadcast → prefill/decode → 代码与调试不变量。
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

1. 打开 `lessons/0009-tiny-decoder-layer-course/index.html`，依次完成四个可视化模块。
2. 从上到下运行 `notebooks/0009-tiny-decoder-layer.ipynb`，每个预测题先回答再执行。
3. 运行 `exercises/0009_explore_decoder_layer.py`，核对完整 shape 与两次 residual 数值。
4. 完成 `exercises/0009_tiny_decoder_layer.py` 的组装 TODO 并运行测试。
5. 查看 `reference/0009-tiny-decoder-layer-cheatsheet.html` 后关闭资料。
6. 闭卷完成 `assessments/0009-tiny-decoder-layer.md`，填写 `submissions/0009.md`。

## Agent 下一步

- 收到 0009 前不生成 0010。
- 0009 独立代码与闭卷第 1、2、3 题通过，且总分至少 80，才进入 KV Cache 增量执行。

## 最近证据

- `submissions/0008-feedback.md`
- `submissions/0008.md`
- `learning-records/0013-rope-proven.md`
- `submissions/0007R-feedback.md`
- `submissions/0007R.md`
- `learning-records/0012-mha-gqa-proven.md`

环境配置见 `ENVIRONMENT.md`。硬件信息只代表记录时主机，另一平台必须运行 `exercises/0000_verify_pytorch.py` 自行验证。
