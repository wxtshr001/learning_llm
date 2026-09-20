# 第 0008 课正式通过：RoPE 从二维旋转到 GQA

0008 最终 95/100，独立代码与关键题 1、2、3 全部通过，第 0008 课正式完成。

## 已证明

- 能把 Qwen half-split 的 D 维向量配成 `(i,i+D/2)`，逐元素计算 `rotate_half` 与 RoPE。
- 能从 `position_ids [B,S]`、`inv_freq [D/2]` 构造 angles 与 cos/sin `[B,S,D]`。
- 能解释 RoPE 旋转 Q/K、不旋转 V，并区分位置 score 与 causal mask 的职责。
- 能通过具体点积说明相同相对位置差产生相同位置效应。
- 能把 `[B,1,S,D]` cos/sin 广播到不同 head 数的 Q 和 K，不需要在 RoPE 前 repeat K。
- 能说明 prefill/decode 的 position 不能重置，以及 Cache 保存旋转后的 K 和未旋转的 V。
- 能实现 rank、shape、正偶数 D、device 与 dtype 契约，并通过 CUDA、CPU 和非法输入测试。

## 非阻塞提醒

- Q1 旋转后范数的平方项沿用了旋转前顺序，但总和、范数结论、旋转数值和代码都正确，按局部抄写处理。
- `position_ids` 应归类为运行输入；跨 decode step 保存的请求 state 是历史 `K_rope` 与 V。答卷已在执行路径题正确识别 Cache 内容，因此不安排补强课。
- Notebook 文件只保存了前 4/14 个代码单元的执行计数；这不是公布的门禁，且更强的代码与闭卷证据已通过。

## Evidence

- `exercises/0008_rope.py`
- `submissions/0008.md`
- `submissions/0008-feedback.md`
- `notebooks/0008-rope-from-rotation-to-gqa.ipynb`
