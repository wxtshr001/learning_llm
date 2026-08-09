# 0005 反馈

## 结论

**82/100，未通过；进入 0005R 短补强。**

独立作业、RMSNorm 手算和 gated FFN 数值均通过。第 3 题是预先公布的关键题，三个 PyTorch Linear weight shape 全部按 `[in,out]` 写反；第 2 题又把 RMSNorm 的 `weight [H]` 写成 `mean_square [B,S,1]`。同一种“运行 tensor shape、归约统计量 shape 与参数 storage shape 混淆”在两题重复出现，不能作为单次抄写失误处理。

0005R 只补这一处 shape 分类，不重复已经证明的代码、手算、parity、梯度或 decoder 数据流。

## 逐题评分

### 1. RMSNorm 手算：25/25，关键题通过

- `x^2=[9,16]`；
- mean square 为 `12.5`；
- RMS 为 `sqrt(12.5)=3.5355`；
- `x/RMS=[0.8485,1.1314]`；
- 乘 `gamma=[2,0.5]` 后为 `[1.6970,0.5657]`。

计算完整、正确。

### 2. RMSNorm axis 与广播：8/15

- `mean_square.shape=[2,3,1]`，正确。
- 每个值来自固定 `b,s` 的四个 H 特征平方平均，正确。
- `weight` 应为 `[H]=[4]`，并按 `weight[h]` 乘到所有 `b,s`；答卷写成 `[2,3,1]`，混入了 mean_square 的 shape，而且把 weight 的乘法写成“相除”。
- 最终输出 shape `[2,3,4]` 未填写。
- mean square 不减均值，而 variance 先围绕均值计算离差，方向正确。

### 3. Gated FFN shape：17/25，关键题未通过

题面给出 `H=8,I=20`。PyTorch `Linear(in_features,out_features).weight` 按 `[out,in]` 存储：

```text
gate_proj.weight = [20,8]
up_proj.weight   = [20,8]
down_proj.weight = [8,20]
```

答卷写成 `[8,20] [8,20] [20,8]`，三项均反转。另一方面，gate_raw、SiLU(gate_raw)、up、mixed 的 `[2,5,20]`，最终 output 的 `[2,5,8]`，以及逐元素乘法的判断均正确。

代码中的构造也正确：`Linear(H,I)`、`Linear(H,I)`、`Linear(I,H)`。这证明数据流方向已经会用，但闭卷时尚未稳定区分构造参数、weight storage 和运行 tensor。

### 4. 门控数值：15/15

`[0,-2.1933,-1.0756]` 正确。这里需要同位置相乘而不是对 I 轴求和，矩阵乘法不能替代。

### 5. Module、梯度与 parity：8/10

三个 Parameter 名称正确；知道 output 同时依赖 gate 与 up 两条路径；知道“无异常”不等于数值正确。Parity 的控制变量表达较笼统，但已经指出输入、weight、eps 都属于必须保持一致的计算条件。

### 6. decoder 迁移：9/10

`RMSNorm -> GatedFFN -> residual add` 顺序、Attention 已混入上下文、I 必须降回 H 才能残差相加、门控分支的能力与成本均正确。FFN 不只是“H 维放缩”，而是 Linear 在最后特征轴上做加权求和，但“不沿 S 轴混合 token”的核心结论正确。

## 代码证据

独立实现通过，且与 reference 数值完全对齐：

```text
Lesson 0005: all checks passed.
RMSNorm max absolute error: 0.0
Gated FFN max absolute error: 0.0
changing one token leaves other token outputs unchanged: True
all FFN parameter gradients finite: True
```

## 下一步

完成 `lessons/0005R-parameter-and-runtime-shapes.html`，运行 `exercises/0005R_trace_shapes.py`，再闭卷填写 `submissions/0005R.md`。关键题通过且总分至少 80 后，才结束第 0005 课。
