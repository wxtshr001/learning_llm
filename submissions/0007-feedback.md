# 0007 反馈

## 结论

**85/100，总分达标，但关键题 2、6 未通过；进入 0007R 针对性补强，暂不进入 0008。**

本次不是重学整节 0007。Head 分组映射、逐 head 数值、KV Cache 字节主计算、decode 带宽意义和独立代码主路径都已经通过。需要补的是：不同 tensor 阶段的 shape/axis 分类，以及 GQA 子层从 `hidden_states` 到 residual add 的完整路径。

## 代码：关键项通过

使用课程 Conda 环境在 CUDA 上重新运行：

```text
Lesson 0007 guided GQA exploration passed.
Lesson 0007: all GQA checks passed.
device: cuda:0
Q head -> KV head mapping: [0, 0, 1, 1]
MHA/GQA KV Cache MiB: 512 128
```

实现正确覆盖了连续 Q→KV 分组、causal mask、key axis softmax、V 聚合、head merge 和 Cache 字节公式。

非阻塞工程提醒：`repeat_kv_for_query_heads()` 在读取 `shape[1]` 后才检查 rank，并且 `Nkv=0` 时会先发生除零；因此某些非法输入得到 `IndexError`/`ZeroDivisionError`，而不是统一的 `ValueError`。这与此前已经裁决过的 rank 检查顺序同类，不作为本次概念门禁，但以后写生产代码时应先验证 rank 和正 head 数，再读取或参与除法。

## 1. Head 映射与边界：20/20，关键题通过

- `group_size=12/3=4` 正确。
- 映射 `0～3→KV0`、`4～7→KV1`、`8～11→KV2` 正确。
- MHA 的 `Nkv=12`、MQA 的 `Nkv=1` 正确。
- 能解释 GQA 是一组 Q heads 共享一个 KV head。

## 2. 完整 shape 数据流：18/25，关键题未通过

正确部分：

- `hidden_states=[2,7,24]`、`q raw=[2,7,24]`。
- Q/K/V 经 reshape 与 transpose 后分别得到 `[2,6,7,4]`、`[2,2,7,4]`、`[2,2,7,4]`。
- K/V 逻辑映射、scores/weights、per-head output、merge 和 `o_proj` 的最终数字 shape 正确。
- 能正确指出 Cache 持久保存未展开的 `[B,Nkv,S,D]` K/V。

需要补强：

1. `k_proj/v_proj raw` 应是 `[B,S,Nkv*D]=[2,7,8]`，不是 `[2,7,24]`。第 4 题已经正确算出 K/V projection 的 output width 是 8，所以这里属于阶段对应不稳定，而不是完全不会 projection。
2. `merge=[B,S,Nq*D]` 和 `o_proj output=[B,S,H]` 都是 rank-3 tensor，没有独立的 head axis；不能因为宽度中包含 `Nq*D`，就说它们的 head axis 是 `Nq`。
3. 题目要求解释为什么不能平均 6 个 head output，此项漏答。merge/concat 保留每个 head 的 `D` 个特征，得到 `Nq*D` 个特征交给可学习的 `o_proj`；平均会把 head 身份和大量特征压成 `D`，信息已经不可逆丢失。

## 3. 逐 head 数值迁移：15/15

causal weights、四个 Q head 的输出和 merge 后具体数值全部正确。

## 4. Projection 参数：8/10

q/k/v 的构造、weight shape 与参数量正确，并正确判断只有 K/V projection 变小。漏写了 `o_proj: Linear(24,24)`、weight `[24,24]`、576 个参数。

## 5. KV Cache 字节：18/20，关键题通过

公式、MHA 384 MiB、GQA 96 MiB、节省 288 MiB以及保存 repeat 前 K/V 的原因都正确。

局部错误是把“节省比例”写成了“GQA/MHA 容量比例”：

```text
GQA/MHA = 96/384 = 25%
节省比例 = (384-96)/384 = 75%
```

主公式、bytes、MiB 和存储结论均正确，因此按单点比例混淆扣分，不升级为关键概念失败。

## 6. Transformer inference 迁移：6/10，关键题未通过

前 3 小题正确：单步 decode 需要反复读取历史 K/V；端到端性能还有其他计算和系统瓶颈；attention weights 的 head axis 由 `Nq` 决定。

第 4 小题没有作答。你的质疑需要分开看：

- **RoPE 的旋转公式和实现**确实要到 0008 才验收，本题不能要求你计算 RoPE。
- 但本题只要求指出连接位置。0007 正文第 1、10 节以及速查表已经给出：`split → RoPE(Q,K) → update/read KV Cache → GQA causal attention`。因此“定位”属于本课已教范围，不是超纲。

应能写出的完整子层路径是：

```text
hidden_states → RMSNorm → q/k/v projections → split heads
→ RoPE 作用于 Q/K → 更新或读取未展开的 K/V Cache
→ GQA causal attention → merge heads → o_proj → residual add
```

## 门禁裁决

- 代码：通过。
- 关键题 1：通过。
- 关键题 2：未通过。
- 关键题 5：通过。
- 关键题 6：未通过。
- 总分：85/100。

按预先公布的门禁，关键题必须全部通过，因此进入 0007R。复测不再考 head 映射、attention 数值和代码主计算。
