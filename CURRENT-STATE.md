# Current Learning State

更新时间：2026-08-09

## 当前门禁

**第 0004 课 97/100，通过；第 0005 课进行中。**

0004 独立脚本已由 Agent 复跑，全部检查通过。两处公式抄写/变量名笔误未改变完整数值链与概念结论，因此仅轻微扣分，不安排重复补强。当前进入 RMSNorm 与 gated FFN。

0005 初版不只 gated FFN 章节过于简略，其他章节也存在“结论先行”。现已完成全课审查与重写：补充进入条件、rank/axis、RMSNorm 动机与完整数据流、广播索引、dtype 边界、Linear 逐元素映射、Module/梯度路径、token 独立性、parity 流程、Qwen 映射、练习证据和指定阅读范围。

学习者仍无法从“RMSNorm/FFN 不混合 token”理解二者关系，说明第 7 章缺少 decoder block 上下文。现已按 Qwen3DecoderLayer 的真实顺序补充两段 pre-norm 子层、residual connection、RMSNorm→FFN 数据流、双 token 数值流，以及“FFN 不主动混合但可加工 Attention 已汇集信息”的关键边界。

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

## 非阻塞工程提醒

- `split_heads()` 当前先解包 shape 再检查 rank；以后修改生产代码时应先验证再解包，但不将此记录为知识概念缺口。
- 0003 有一次 0.0010/0.0001 算术笔误，但最大误差与门限结论正确，不作为 dtype/device 概念缺口。
- 0004 的 `training_step()` 类型标注为 `dict[str, float]`，但 prediction/loss 返回标量 Tensor；以后生产代码应使用 `.item()` 满足接口，不阻塞课程门禁。

## 学习者下一步

1. 学习 `lessons/0005-rmsnorm-gated-ffn.html`。
2. 运行 `exercises/0005_explore_rmsnorm_gated_ffn.py`。
3. 完成 `exercises/0005_rmsnorm_gated_ffn.py` 中的 TODO 并运行测试。
4. 查看 `reference/0005-rmsnorm-gated-ffn-cheatsheet.html` 后关闭资料。
5. 闭卷完成 `assessments/0005-rmsnorm-gated-ffn.md`，并填写 `submissions/0005.md`。

## Agent 下一步

- 评分不得增加题面未声明条件；单次算术笔误须结合完整证据判断。
- 0005 独立作业、第 1 题和第 3 题均通过且总分至少 80，才进入第 0006 课。

## 最近证据

- `submissions/0004-feedback.md`
- `learning-records/0007-module-autograd-optimizer-proven.md`
- `lessons/0005-rmsnorm-gated-ffn.html`

环境配置见 `ENVIRONMENT.md`。硬件信息只代表记录时主机，另一平台必须运行 `exercises/0000_verify_pytorch.py` 自行验证。
