# 0008 反馈

## 结论

**95/100，通过。第 0008 课 RoPE 正式完成，解锁第 0009 课 Tiny Decoder Layer。**

公布的关键门禁全部通过：独立代码测试通过；第 1 题 half-split 旋转、第 2 题相对位置 score、第 3 题 GQA shape/广播/执行位置均已证明。局部算术抄写和非关键对象分类错误不安排重学 RoPE。

## 代码：关键项通过

使用课程 Conda 环境重新运行 `exercises/0008_rope.py`：

```text
Lesson 0008: all RoPE checks passed.
device: cuda:0
cos/sin shape: (1, 3, 4) (1, 3, 4)
Q/K rotated shape: (1, 4, 3, 4) (1, 2, 3, 4)
manual rotated Q: tensor([-3.,  2.,  1.,  4.], device='cuda:0')
V unchanged: True
```

另加 CPU、`B=2,Nq=8,Nkv=2,D=6`、默认频率和非法输入测试，全部通过：

- `rotate_half()` 正确拒绝标量、空 D 和奇数 D。
- `build_rope_cos_sin()` 正确拒绝错误 rank、浮点 position、奇数 D 和非正 base。
- `apply_rope()` 正确拒绝 rank、shape 和 dtype 不匹配。
- Q/K 旋转前后范数保持，position 0 恒等。

## 闭卷题：95/100

### 1. Qwen half-split：24/25，关键题通过

配对 `(0,2)`、`(1,3)`、`rotate_half=[-5,-7,2,3]` 和 `x_rope=[-5,3,2,7]` 全部正确。

局部抄写：旋转后平方项应写成 `25+9+4+49`，答卷仍写成原向量的 `4+9+25+49`。两者总和同为 87，且范数保持的结论、旋转结果和实现均正确，因此只扣 1 分，不记为概念缺口。

### 2. 相对位置与 score：25/25，关键题通过

三个位置向量、三个点积均正确；能够解释相同位置差产生相同位置效应，并明确 RoPE 不能替代把未来 score 设为负无穷的 causal mask。

### 3. GQA shape、广播与执行位置：30/30，关键题通过

`position_ids`、`inv_freq`、angles、cos/sin、Q/K/V、广播后 cos/sin 与旋转后 Q/K 的 shape 全部正确。能够解释长度 1 的 head axis 如何分别广播到 `Nq=8` 和 `Nkv=2`，并正确给出 projection → split → RoPE → Cache → GQA attention → merge → `o_proj` 的顺序。decode 新 token 的 position 7 也正确。

### 4. 实现与系统边界：16/20

二维旋转展开、三个实现不变量以及“不需要在 RoPE 前 repeat K”的解释正确。

需要纠正一处分类：`position_ids` 是本次 forward/decode step 的**运行输入**，不是跨 step 保存的请求 state。请求 state 是历史 `K_rope` 与 V 组成的 KV Cache。答卷在第 3 题已经正确写出 Cache 保存 `K_rope` 与 V，因此这是局部分类错误，不升级为 RoPE 或 Cache 路径失败。

## Notebook 证据

工作区 notebook 当前保存了 14 个代码单元中的前 4 个执行计数，且无 error output。课程门禁没有把“保存所有 notebook execution_count”列为关键项；独立代码、引导输出和闭卷迁移题已经提供更强证据，因此不据此扣分或阻塞通过。

## 最终裁决

第 0008 课现已证明：二维旋转、Qwen half-split、position → frequency → cos/sin、相对位置 score、GQA 广播、RoPE/causal mask 边界、prefill/decode position、旋转后 K 的 Cache 位置以及代码输入契约。

下一课进入 Tiny Decoder Layer，把已经分别掌握的 RMSNorm、GQA Attention、RoPE、gated FFN 和 residual 组装为一个可堆叠 layer。
