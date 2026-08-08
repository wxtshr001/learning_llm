# Current Learning State

更新时间：2026-08-09

## 当前门禁

**第 0002 课首次提交未通过，当前进入 0002R。**

补强主题：直接元素索引映射、Q/K/V projection 到 head layout 的迁移，以及代码输入边界。

## 已完成

- 初始能力诊断。
- 第 0001 课：Linear shape、元素索引、bias broadcasting、Q projection 迁移。
- 0001R：纠正 `b[i]`/`b[j]` 输出索引误区。
- 0001R2：二维 Linear 逐元素手算复测通过。
- 0002 已证明部分：能计算 `D=H/N`、保持 reshape 元素总数、解释 Sequence/Head 轴交换，并实现 split/merge 主路径。

## 当前知识缺口

- 未把 stride 推导收敛为直接关系 `heads[b,n,s,d] = x[b,s,n*D+d]`，指定数值索引未完成。
- 第 7 题留空：尚未理解 Q/K/V projection 输出宽度与各自 head layout 的连接。
- `torch.arange` 数值追踪最终答案与自己算出的 offset 矛盾。
- `split_heads()` 尚未主动检查错误 rank 与非正 `num_heads`。

## 学习者下一步

1. 学习 `lessons/0002R-index-and-qkv-layout.html`。
2. 运行 `exercises/0002R_trace_qkv_layout.py` 并解释输出宽度的来源。
3. 在 `exercises/0002_head_layout.py` 补上 rank 与 `num_heads > 0` 检查。
4. 闭卷完成 `assessments/0002R-index-and-qkv-layout.md`，写入 `submissions/0002R.md`。

## Agent 下一步

- 收到 0002R 答卷前，不生成第 0003 课。
- 代码边界检查与 0002R 第 1、2、3 题均通过且总分至少 80% 才能结束第 0002 课。
- 若仍无法迁移 Q/K/V，只继续降低 projection 输出宽度的抽象层级，不重复已通过的 reshape shape 推导。

## 最近证据

- `submissions/0002-feedback.md`
- `learning-records/0004-head-layout-partial-remediation-required.md`

环境配置见 `ENVIRONMENT.md`。硬件信息只代表记录时主机，另一平台必须运行 `exercises/0000_verify_pytorch.py` 自行验证。
