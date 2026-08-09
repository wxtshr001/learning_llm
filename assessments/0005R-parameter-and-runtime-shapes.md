# 0005R 闭卷复测：参数与运行 tensor shape

运行引导脚本后关闭课程与源码作答。

## 提交证据

粘贴 `0005R_trace_shapes.py` 的完整输出。

## 1. RMSNorm 三类 shape（关键题，40 分）

输入 `x.shape=[B=3,S=2,H=6]`。

1. `mean_square = x.pow(2).mean(-1,keepdim=True)` 的数字 shape；
2. RMSNorm 可训练 `weight` 的数字 shape；
3. 最终 output 的数字 shape；
4. 补全逐元素公式中的索引：`output[b,s,h] = x[b,s,h] * rsqrt(mean_square[?,?,?]+eps) * weight[?]`；
5. 分别说明 mean_square 和 weight 怎样广播，以及二者为什么不是同一类对象。

## 2. Gated FFN 的三种写法（关键题，50 分）

令 `B=4,S=5,H=6,I=14`。对 gate_proj、up_proj、down_proj 分别写出：

1. `nn.Linear(in_features,out_features)` 的数字构造参数；
2. PyTorch `.weight` 的数字 shape；
3. 输入和输出运行 tensor 的数字 shape。

再用一句逐元素公式解释：为什么 `.weight` 的第一轴是 out、第二轴是 in？

## 3. 分类检查（10 分）

对下面对象分别标记“统计量”“Parameter”或“运行 tensor”：

- `mean_square [B,S,1]`
- `norm.weight [H]`
- `gate_raw [B,S,I]`
- `gate_proj.weight [I,H]`

## 通过标准

- 总分至少 80。
- 第 1、2 题均通过。
- 通过后第 0005 课正式完成，才可生成第 0006 课。
