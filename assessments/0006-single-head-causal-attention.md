# 0006 闭卷测试：单头 Causal Attention

完成探索与独立作业后，关闭课程、代码和速查表作答。

## 提交证据

粘贴两个脚本的完整输出。

## 1. 精确手算（关键题，25 分）

`B=1,S=4,D=1`，Q、K 的所有元素均为 1，`V=[2,4,8,16]`。

1. 写出四行 causal attention weights；
2. 写出四个 output 数值，其中分数可保留为分数；
3. 解释为什么这些权重是均匀的，但每一行允许参与的 token 数不同。

## 2. Shape 与 axis（关键题，25 分）

令 `B=2,S=5,D=4`，写出 Q、K、V，`K.transpose(-2,-1)`，scores，mask，weights 和 output 的数字 shape。说明 scores 的两个长度 5 分别是什么轴，以及 softmax 沿哪个轴。

## 3. Causal 行为（关键题，20 分）

1. 只修改 `V[:,4,:]` 时，哪些 query 位置可能改变，哪些必须不变？
2. 只修改 `K[:,2,:]` 时，哪些 query 位置可能改变，哪些必须不变？
3. 为什么 mask 应在 softmax 之前应用？

## 4. 两个求和轴（10 分）

分别补全 score 与 output 的逐元素公式，并说明 score 对哪个轴求和、output 对哪个轴求和。

## 5. Scale 与数值稳定（10 分）

为什么 score 除以 `sqrt(D)`？它是否改变 tensor shape？如果错误地除以 `sqrt(S)`，概念上错在哪里？

## 6. Parity 与错误定位（10 分）

代码没有异常为什么不足以证明 attention 正确？至少列出四项你会检查的证据，其中必须包含一项 causal 行为证据。

## 通过标准

- 独立作业、第 1、2、3 题通过。
- 总分至少 80。
- 未通过时只补 hand calculation、shape/axis 或 causal mask 中的实际缺口。
