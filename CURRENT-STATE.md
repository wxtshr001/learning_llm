# Current Learning State

更新时间：2026-08-09

## 当前门禁

**第 0003 课最终裁决 92/100，通过；第 0004 课进行中。**

学习者反对初次 75 分裁决后，复核确认题中 W、b 已在 CUDA，且未规定 dtype；只迁移 X 的答案正确。单次小数位笔误没有破坏最大误差与门限结论，不构成概念缺口。0003R 已撤销。当前进入最小 Module、Parameter、autograd 和 optimizer。

## 已完成

- 初始能力诊断。
- 第 0001 课：Linear shape、元素索引、bias broadcasting、Q projection 迁移。
- 0001R：纠正 `b[i]`/`b[j]` 输出索引误区。
- 0001R2：二维 Linear 逐元素手算复测通过。
- 0002 已证明部分：能计算 `D=H/N`、保持 reshape 元素总数、解释 Sequence/Head 轴交换，并实现 split/merge 主路径。
- 0002R 书面题 95/100：直接索引、具体数值、Q/K/V projection 输出宽度与 head layout 三个关键题全部通过。
- 第 0002 课与 0002R 门禁正式完成。
- 第 0003 课最终 92/100：dtype 转换、字节计算、三输入迁移、CUDA Linear、最大绝对误差和容差判断均通过。

## 非阻塞工程提醒

- `split_heads()` 当前先解包 shape 再检查 rank；以后修改生产代码时应先验证再解包，但不将此记录为知识概念缺口。
- 0003 有一次 0.0010/0.0001 算术笔误，但最大误差与门限结论正确，不作为 dtype/device 概念缺口。

## 学习者下一步

1. 学习 `lessons/0004-module-autograd-optimizer.html`。
2. 运行 `exercises/0004_explore_training_step.py`。
3. 完成 `exercises/0004_module_autograd_optimizer.py` 中的 TODO 并运行测试。
4. 闭卷完成 `assessments/0004-module-autograd-optimizer.md`。
5. 将输出和答案写入 `submissions/0004.md` 或直接提交给当前 Agent。

## Agent 下一步

- 收到 0004 答卷前不生成第 0005 课。
- 评分不得增加题面未声明条件；单次算术笔误须结合完整证据判断。
- 0004 代码与关键题 2、3 均通过且总分至少 80，才进入第 0005 课。

## 最近证据

- `submissions/0003-feedback.md`
- `learning-records/0006-tensor-dtype-device-proven.md`
- `lessons/0004-module-autograd-optimizer.html`

环境配置见 `ENVIRONMENT.md`。硬件信息只代表记录时主机，另一平台必须运行 `exercises/0000_verify_pytorch.py` 自行验证。
