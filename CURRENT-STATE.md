# Current Learning State

更新时间：2026-08-09

## 当前门禁

**0002R 书面复测已通过，等待最后一项代码边界修正。**

当前只剩：把 rank 检查移动到 shape 解包之前，并提供主动异常检查的运行证据。

## 已完成

- 初始能力诊断。
- 第 0001 课：Linear shape、元素索引、bias broadcasting、Q projection 迁移。
- 0001R：纠正 `b[i]`/`b[j]` 输出索引误区。
- 0001R2：二维 Linear 逐元素手算复测通过。
- 0002 已证明部分：能计算 `D=H/N`、保持 reshape 元素总数、解释 Sequence/Head 轴交换，并实现 split/merge 主路径。
- 0002R 书面题 95/100：直接索引、具体数值、Q/K/V projection 输出宽度与 head layout 三个关键题全部通过。

## 当前知识缺口

- `split_heads()` 先执行 `B,S,H = x.shape`，再检查 rank；错误 rank 会在解包时提前抛异常，主动检查不可达。

## 学习者下一步

1. 在 `exercises/0002_head_layout.py` 中先检查 `x.ndim` 和 `num_heads`，再读取 `B,S,H`。
2. 重新运行原有检查，并分别运行错误 rank、`num_heads=0` 两个异常路径。
3. 只提交三段代码输出；无需重答 0002R 第 1～4 题。

## Agent 下一步

- 只复核 rank 与 `num_heads=0` 主动检查是否在 shape/取模操作之前执行。
- 通过后将 0002 与 0002R 标记完成；遵循学习者本轮明确要求，只保存进度，不生成第 0003 课。

## 最近证据

- `submissions/0002R-feedback.md`
- `learning-records/0005-head-index-and-qkv-layout-proven.md`

环境配置见 `ENVIRONMENT.md`。硬件信息只代表记录时主机，另一平台必须运行 `exercises/0000_verify_pytorch.py` 自行验证。
