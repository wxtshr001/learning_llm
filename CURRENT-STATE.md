# Current Learning State

更新时间：2026-08-09

## 当前门禁

**第 0002 课已正式通过；当前暂停，不生成新课程。**

0002R 最终成绩 95/100。学习者将 rank 检查顺序认定为非概念性疏忽；结合已通过的关键概念题和运行证据，该问题不再阻塞门禁。

## 已完成

- 初始能力诊断。
- 第 0001 课：Linear shape、元素索引、bias broadcasting、Q projection 迁移。
- 0001R：纠正 `b[i]`/`b[j]` 输出索引误区。
- 0001R2：二维 Linear 逐元素手算复测通过。
- 0002 已证明部分：能计算 `D=H/N`、保持 reshape 元素总数、解释 Sequence/Head 轴交换，并实现 split/merge 主路径。
- 0002R 书面题 95/100：直接索引、具体数值、Q/K/V projection 输出宽度与 head layout 三个关键题全部通过。
- 第 0002 课与 0002R 门禁正式完成。

## 非阻塞工程提醒

- `split_heads()` 当前先解包 shape 再检查 rank；以后修改生产代码时应先验证再解包，但不将此记录为知识概念缺口。

## 学习者下一步

当前无需提交。等待学习者明确要求开始下一课。

## Agent 下一步

- 不自动生成第 0003 课。
- 学习者明确要求继续时，再依据已保存证据生成下一课。

## 最近证据

- `submissions/0002R-feedback.md`
- `learning-records/0005-head-index-and-qkv-layout-proven.md`

环境配置见 `ENVIRONMENT.md`。硬件信息只代表记录时主机，另一平台必须运行 `exercises/0000_verify_pytorch.py` 自行验证。
