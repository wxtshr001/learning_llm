# Current Learning State

更新时间：2026-08-11

## 当前门禁

**0005R 90/100，通过；第 0006 课进行中。**

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

## 当前知识缺口

- 单头 Q/K/V score、scale、causal mask、softmax key axis 与 value 加权尚未验证。

## 非阻塞工程提醒

- `split_heads()` 当前先解包 shape 再检查 rank；以后修改生产代码时应先验证再解包，但不将此记录为知识概念缺口。
- 0003 有一次 0.0010/0.0001 算术笔误，但最大误差与门限结论正确，不作为 dtype/device 概念缺口。
- 0004 的 `training_step()` 类型标注为 `dict[str, float]`，但 prediction/loss 返回标量 Tensor；以后生产代码应使用 `.item()` 满足接口，不阻塞课程门禁。

## 学习者下一步

1. 学习 `lessons/0006-single-head-causal-attention.html`。
2. 运行 `exercises/0006_explore_single_head_attention.py`。
3. 完成 `exercises/0006_single_head_causal_attention.py` 的 TODO 并运行测试。
4. 闭卷完成 `assessments/0006-single-head-causal-attention.md`，填写 `submissions/0006.md`。

## Agent 下一步

- 收到 0006 前，不生成第 0007 课。
- 独立作业、第 1、2、3 题通过且总分至少 80，才进入 MHA/GQA。

## 最近证据

- `submissions/0005R-feedback.md`
- `learning-records/0009-rmsnorm-gated-ffn-proven.md`

环境配置见 `ENVIRONMENT.md`。硬件信息只代表记录时主机，另一平台必须运行 `exercises/0000_verify_pytorch.py` 自行验证。
