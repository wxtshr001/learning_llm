# Current Learning State

更新时间：2026-08-09

## 当前门禁

**第 0003 课首次提交 75/100；正在进行 0003R 局部补强。**

独立作业已在 `cuda:0` 复跑并全部通过。关键题 3 的书面执行路径不完整：只迁移 X，没有迁移 W、b 和 dtype；candidate 也没有明确统一为 CPU/float32。另有一次 0.0010/0.0001 小数位错误。进入 20～30 分钟 0003R，不重做代码主作业。

## 已完成

- 初始能力诊断。
- 第 0001 课：Linear shape、元素索引、bias broadcasting、Q projection 迁移。
- 0001R：纠正 `b[i]`/`b[j]` 输出索引误区。
- 0001R2：二维 Linear 逐元素手算复测通过。
- 0002 已证明部分：能计算 `D=H/N`、保持 reshape 元素总数、解释 Sequence/Head 轴交换，并实现 split/merge 主路径。
- 0002R 书面题 95/100：直接索引、具体数值、Q/K/V projection 输出宽度与 head layout 三个关键题全部通过。
- 第 0002 课与 0002R 门禁正式完成。
- 0003 已证明部分：dtype 转换、字节计算、三输入迁移、CUDA Linear 和最大绝对误差的代码实现均通过。

## 非阻塞工程提醒

- `split_heads()` 当前先解包 shape 再检查 rank；以后修改生产代码时应先验证再解包，但不将此记录为知识概念缺口。

## 学习者下一步

1. 学习 `lessons/0003R-device-path-and-error.html`。
2. 运行 `exercises/0003R_trace_device_path.py`。
3. 闭卷完成 `assessments/0003R-device-path-and-error.md`。
4. 将输出和答案写入 `submissions/0003R.md` 或直接提交给当前 Agent。

## Agent 下一步

- 收到 0003R 答卷前不生成第 0004 课。
- 复测只检查完整 device/dtype 路径、误差小数位和 token id 索引语义。
- 0003R 关键题 1、2 均通过且总分至少 80，才将第 0003 课记为正式通过。

## 最近证据

- `submissions/0002R-feedback.md`
- `learning-records/0005-head-index-and-qkv-layout-proven.md`
- `lessons/0003-tensor-dtype-device.html`
- `submissions/0003-feedback.md`
- `learning-records/0006-dtype-device-partial-remediation-required.md`

环境配置见 `ENVIRONMENT.md`。硬件信息只代表记录时主机，另一平台必须运行 `exercises/0000_verify_pytorch.py` 自行验证。
