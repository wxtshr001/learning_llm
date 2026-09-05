"""Build the self-contained lesson 0008 Jupyter notebook with no extra packages."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "0008-rope-from-rotation-to-gqa.ipynb"
cells: list[dict[str, object]] = []


def markdown(source: str) -> None:
    cells.append(
        {
            "cell_type": "markdown",
            "id": f"markdown-{len(cells):02d}",
            "metadata": {},
            "source": dedent(source).strip() + "\n",
        }
    )


def code(source: str) -> None:
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "id": f"code-{len(cells):02d}",
            "metadata": {},
            "outputs": [],
            "source": dedent(source).strip() + "\n",
        }
    )


markdown(
    r"""
    # 0008：用可执行代码理解 RoPE

    **RoPE**（Rotary Position Embedding，旋转位置编码）做一件核心事情：

    > token 所在位置不同，就把这个 token 的 Q 和 K 朝不同方向旋转。后续计算 Q·K 时，分数既包含内容匹配，也会受到两个 token 相对位置的影响。

    本 notebook 的顺序固定为：**具体数字 → Python/PyTorch 输出 → shape/axis → 公式 → GQA 与推理路径**。

    ## 使用方式

    1. 从上到下逐个运行代码单元，不要一开始选择“全部运行”。
    2. 每个“先预测”问题先在纸上写答案，再运行下面的代码。
    3. 看到 shape 时，把每个数字对应的 axis 说出来。
    4. notebook 是带答案的引导实验；最后仍需独立完成 `../exercises/0008_rope.py` 和闭卷题。

    **本课可以依赖：**你已经掌握 Q/K/V 分工、Attention 点积、GQA 以及 `[B,N,S,D]` 的 axis。

    **本课从零解释：**角度、弧度、cos、sin、half-split、position frequency、RoPE 广播和 decode position。
    """
)

markdown(
    """
    ## 0. 先把所有字母说清楚

    | 字母 | 英文 | 中文含义 | 具体例子 |
    |---|---|---|---|
    | `B` | Batch size | 同时处理的序列数量 | `B=2` 表示两条序列 |
    | `Nq` | Number of query heads | Q 的 head 数 | `Nq=6` |
    | `Nkv` | Number of key/value heads | K/V 的 head 数 | `Nkv=2` |
    | `S` | Sequence length | 每条序列的 token 数 | `S=5` |
    | `D` | Head dimension | 每个 head 的特征数 | `D=4` |
    | `i` | Pair index | 第几个二维特征对 | D=4 时 `i=0,1` |
    | `d` | Feature-axis index | head 内第几个特征 | D=4 时 `d=0..3` |

    Q 的 shape `[B,Nq,S,D]` 是 rank 4 tensor：axis 0 是 batch，axis 1 是 Q head，axis 2 是 token，axis 3 是 head 内特征。
    """
)

code(
    """
    import math
    import torch

    torch.set_printoptions(precision=4, sci_mode=False)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    """
)

markdown(
    r"""
    ## 1. 角度、弧度、cos 和 sin

    人通常用角度描述方向，PyTorch 的 `torch.sin()` 和 `torch.cos()` 接收**弧度**。

    - 0° = 0 弧度
    - 90° = π/2 弧度
    - 180° = π 弧度
    - 270° = 3π/2 弧度

    `cos(θ)` 可以先理解为旋转后在 x 方向留下多少，`sin(θ)` 可以先理解为旋转后转到 y 方向多少。这里的 `θ`（theta）就是旋转角。
    """
)

code(
    """
    degrees = torch.tensor([0.0, 90.0, 180.0, 270.0])
    radians = torch.deg2rad(degrees)

    print("角度      弧度       cos       sin")
    for degree, radian in zip(degrees, radians):
        print(
            f"{degree.item():>3.0f}°   "
            f"{radian.item():>7.4f}   "
            f"{torch.cos(radian).item():>7.4f}   "
            f"{torch.sin(radian).item():>7.4f}"
        )
    """
)

markdown(
    r"""
    ## 2. 先只旋转一个二维向量

    二维向量 `[a,b]` 可以看作从原点出发的箭头：向右 `a`，向上 `b`。

    旋转 `θ` 后：

    $$
    new\_x=a\cos(θ)-b\sin(θ)
    $$

    $$
    new\_y=a\sin(θ)+b\cos(θ)
    $$

    **先预测：** `[3,4]` 逆时针旋转 90° 后是什么？长度是否改变？
    """
)

code(
    """
    def rotate_2d(vector: torch.Tensor, angle_radians: float) -> torch.Tensor:
        # Rotate one [x,y] vector counter-clockwise.
        a, b = vector[0], vector[1]
        angle = torch.tensor(angle_radians, dtype=vector.dtype)
        new_x = a * torch.cos(angle) - b * torch.sin(angle)
        new_y = a * torch.sin(angle) + b * torch.cos(angle)
        return torch.stack((new_x, new_y))


    vector_2d = torch.tensor([3.0, 4.0])
    rotated_2d = rotate_2d(vector_2d, math.pi / 2)

    print("旋转前:", vector_2d)
    print("旋转后:", rotated_2d.round(decimals=4))
    print("旋转前长度:", torch.linalg.vector_norm(vector_2d).item())
    print("旋转后长度:", torch.linalg.vector_norm(rotated_2d).item())

    torch.testing.assert_close(rotated_2d, torch.tensor([-4.0, 3.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        torch.linalg.vector_norm(rotated_2d),
        torch.linalg.vector_norm(vector_2d),
    )
    """
)

markdown(
    """
    ## 3. D=4 不是四个数一起旋转，而是两个二维平面

    `D=4` 表示每个 Attention head 有 4 个特征。RoPE 每两个特征组成一个二维 pair。

    Qwen/Hugging Face 的 **half-split**（前后半段配对）布局是：

    ```text
    x = [x0, x1, x2, x3]

    pair 0: (axis 0, axis 2) = [x0, x2]
    pair 1: (axis 1, axis 3) = [x1, x3]
    ```

    这和论文中常见的相邻配对 `(0,1)、(2,3)` 内存排列不同。本课和独立作业统一使用 Qwen half-split。

    C++ 类比：把连续数组分成长度相同的前半段和后半段，再让 `first[i]` 与 `second[i]` 配对。类比只解释索引布局；PyTorch 的 tensor 还拥有 dtype、device 和 stride。
    """
)

code(
    """
    def rotate_half_demo(x: torch.Tensor) -> torch.Tensor:
        # Qwen layout: [first_half, second_half] -> [-second_half, first_half].
        head_dim = x.shape[-1]
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("D must be a positive even number")
        half = head_dim // 2
        first_half = x[..., :half]
        second_half = x[..., half:]
        return torch.cat((-second_half, first_half), dim=-1)


    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    helper = rotate_half_demo(x)

    print("x:              ", x)
    print("first half:     ", x[:2])
    print("second half:    ", x[2:])
    print("-second + first:", helper)
    print("pair 0 [x0,x2] -> helper axes [0,2]:", x[[0, 2]], "->", helper[[0, 2]])
    print("pair 1 [x1,x3] -> helper axes [1,3]:", x[[1, 3]], "->", helper[[1, 3]])

    torch.testing.assert_close(helper, torch.tensor([-3.0, -4.0, 1.0, 2.0]))
    """
)

markdown(
    r"""
    ## 4. 完整 RoPE 公式也只是一组逐元素乘加

    $$x_{rope}=x\cdot cos+rotate\_half(x)\cdot sin$$

    `·` 在这里是**逐元素乘法**，不是 dot product。

    设 pair 0 `(axis 0,2)` 转 90°，pair 1 `(axis 1,3)` 转 0°：

    ```text
    cos = [0,1,0,1]
    sin = [1,0,1,0]
    ```

    **先预测：** `x=[1,2,3,4]` 的四个输出元素分别是多少？
    """
)

code(
    """
    cos_manual = torch.tensor([0.0, 1.0, 0.0, 1.0])
    sin_manual = torch.tensor([1.0, 0.0, 1.0, 0.0])
    x_rope = x * cos_manual + helper * sin_manual

    print("d | x[d]*cos[d] | helper[d]*sin[d] | output")
    for d in range(x.shape[-1]):
        first_term = x[d] * cos_manual[d]
        second_term = helper[d] * sin_manual[d]
        print(f"{d} | {first_term.item():>11.1f} | {second_term.item():>16.1f} | {(first_term + second_term).item():>6.1f}")

    print("x_rope:", x_rope)
    torch.testing.assert_close(x_rope, torch.tensor([-3.0, 2.0, 1.0, 4.0]))
    """
)

markdown(
    """
    ## 5. `position_ids`：先把它理解成 token 编号

    ```text
    token:           我    喜    欢
    position:        0     1     2
    position_ids = [[0,    1,    2]]
    ```

    `position_ids` 的 shape 是 `[B,S]`，rank 为 2：

    - axis 0：batch；这里 `B=1`，只有一条序列。
    - axis 1：token position；这里 `S=3`，有三个 token。
    - 它没有 head axis，也没有 feature axis。
    """
)

code(
    """
    position_ids = torch.tensor([[0, 1, 2]])
    print("position_ids:", position_ids)
    print("rank:", position_ids.ndim)
    print("shape [B,S]:", tuple(position_ids.shape))
    print("dtype:", position_ids.dtype, "（整数 token 编号）")
    """
)

markdown(
    r"""
    ## 6. `inv_freq`：每前进一个 token，各 pair 转多快

    `inv_freq` 是 inverse frequency。本课直接把它理解成**每个二维 pair 的基础旋转速度**：

    $$angle=position\times inv\_freq$$

    先使用容易手算的教学频率：

    - pair 0 每前进一个 token 转 90°，即 π/2 弧度。
    - pair 1 每前进一个 token 转 0°。

    `position_ids [B,S]` 末尾增加一个长度为 1 的 axis 后，和 `inv_freq [D/2]` 广播相乘，得到 `angles [B,S,D/2]`。
    """
)

code(
    """
    teaching_inv_freq = torch.tensor([math.pi / 2, 0.0])
    angles = position_ids[..., None].to(torch.float32) * teaching_inv_freq

    print("position_ids[...,None] shape [B,S,1]:", tuple(position_ids[..., None].shape))
    print("inv_freq shape [D/2]:", tuple(teaching_inv_freq.shape))
    print("angles shape [B,S,D/2]:", tuple(angles.shape))
    print("angles（弧度）:\\n", angles)
    print("angles（角度）:\\n", torch.rad2deg(angles))
    """
)

markdown(
    r"""
    ## 7. 真实默认频率：先代入 D=4，再看通式

    默认 RoPE 常用：

    $$inv\_freq[i]=\frac{1}{base^{2i/D}}$$

    - `i`：pair index，D=4 时是 0、1。
    - `D`：head dimension，这里是 4。
    - `base`：频率基数，这里使用 10000。

    代入后：

    ```text
    i=0: 1 / 10000^(0/4) = 1
    i=1: 1 / 10000^(2/4) = 0.01
    ```

    所以 D=4 时 `inv_freq=[1,0.01]`。教学频率用于手算，独立作业使用这个真实默认公式。
    """
)

code(
    """
    head_dim = 4
    base = 10_000.0
    pair_indices = torch.arange(head_dim // 2, dtype=torch.float32)
    default_inv_freq = 1.0 / (base ** (2 * pair_indices / head_dim))

    print("pair indices i:", pair_indices)
    print("default inv_freq:", default_inv_freq)
    torch.testing.assert_close(default_inv_freq, torch.tensor([1.0, 0.01]))
    """
)

markdown(
    """
    ## 8. 从 `[B,S,D/2]` 复制到 `[B,S,D]`

    每个 pair 只有一个角度，但它有两个 axis。half-split 布局必须让 axis 0 与 2 使用同一角度，axis 1 与 3 使用同一角度：

    ```text
    [pair0_angle, pair1_angle]
           复制一次
    [pair0_angle, pair1_angle, pair0_angle, pair1_angle]
    ```

    然后分别取 cos 和 sin。
    """
)

code(
    """
    angle_embedding = torch.cat((angles, angles), dim=-1)
    cos_table = angle_embedding.cos()
    sin_table = angle_embedding.sin()

    print("angles [B,S,D/2]:", tuple(angles.shape))
    print("embedding [B,S,D]:", tuple(angle_embedding.shape))
    print("cos [B,S,D]:\\n", cos_table.round(decimals=4))
    print("sin [B,S,D]:\\n", sin_table.round(decimals=4))
    """
)

markdown(
    """
    ## 9. 一个 token 的 Q 和 K 完整走一遍

    当前 token 位于 position 1，所以从上面的表取 `cos_table[0,1]` 和 `sin_table[0,1]`。

    Q 与 K 使用相同位置对应的 cos/sin，但 Q/K 的原始内容不同，因此旋转结果也不同。V 不参与 Q·K 匹配，不旋转。
    """
)

code(
    """
    one_q = torch.tensor([1.0, 2.0, 3.0, 4.0])
    one_k = torch.tensor([10.0, 20.0, 30.0, 40.0])
    cos_position_1 = cos_table[0, 1]
    sin_position_1 = sin_table[0, 1]

    one_q_rope = one_q * cos_position_1 + rotate_half_demo(one_q) * sin_position_1
    one_k_rope = one_k * cos_position_1 + rotate_half_demo(one_k) * sin_position_1

    print("position 1 cos:", cos_position_1.round(decimals=4))
    print("position 1 sin:", sin_position_1.round(decimals=4))
    print("Q before -> after:", one_q, "->", one_q_rope.round(decimals=4))
    print("K before -> after:", one_k, "->", one_k_rope.round(decimals=4))

    torch.testing.assert_close(one_q_rope, torch.tensor([-3.0, 2.0, 1.0, 4.0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(one_k_rope, torch.tensor([-30.0, 20.0, 10.0, 40.0]), atol=1e-5, rtol=0)
    """
)

markdown(
    """
    ## 10. 为什么点积能感受到相对位置

    使用最小二维例子，原始 `q=k=[1,0]`，每前进一个 position 逆时针旋转 90°：

    - position 0：`[1,0]`
    - position 1：`[0,1]`
    - position 2：`[-1,0]`

    **先预测：** `score(0,1)`、`score(1,2)`、`score(0,2)` 分别是多少？这里的 score 是两个向量的 dot product。
    """
)

code(
    """
    vectors_by_position = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ])

    score_0_1 = torch.dot(vectors_by_position[0], vectors_by_position[1])
    score_1_2 = torch.dot(vectors_by_position[1], vectors_by_position[2])
    score_0_2 = torch.dot(vectors_by_position[0], vectors_by_position[2])

    print("score(0,1):", score_0_1.item())
    print("score(1,2):", score_1_2.item())
    print("score(0,2):", score_0_2.item())
    print("相对距离 1 的两个 score 相同:", score_0_1.item() == score_1_2.item())
    """
)

markdown(
    r"""
    上面先用数字证明：相对距离同为 1 的 `(0,1)` 与 `(1,2)` 得到相同的位置效应。现在才压缩成公式：

    $$(R(mθ)q)\cdot(R(nθ)k)=q\cdot R((n-m)θ)k$$

    - `R(angle)`：旋转操作。
    - `m`：Q 的 token position。
    - `n`：K 的 token position。
    - 最终只剩 `(n-m)`，也就是相对位置差。

    不要求证明这个等式，只要求能用上一段数字解释它。

    ### RoPE 不等于 causal mask

    - RoPE：改变 Q/K，让 score 带有位置关系。
    - causal mask：把未来 token 的 score 设为负无穷，禁止看未来。

    RoPE 不能阻止 position 0 与 position 10 计算有限 score，因此不能替代 causal mask。
    """
)

markdown(
    """
    ## 11. 接入 GQA：为什么 Q/K head 数不同仍可直接旋转

    具体配置：

    ```text
    B=2    两条序列
    Nq=6   六个 Q heads
    Nkv=2  两个 K/V heads
    S=5    每条五个 tokens
    D=4    每个 head 四个特征
    ```

    cos/sin 是 `[B,S,D]=[2,5,4]`，没有 head axis。`unsqueeze(1)` 在 axis 1 插入长度 1，得到 `[2,1,5,4]`。长度 1 的 head axis 可以分别广播到 Q 的 6 个 heads 和 K 的 2 个 heads。
    """
)

code(
    """
    B, Nq, Nkv, S, D = 2, 6, 2, 5, 4
    query = torch.arange(B * Nq * S * D, dtype=torch.float32).reshape(B, Nq, S, D)
    key = torch.arange(B * Nkv * S * D, dtype=torch.float32).reshape(B, Nkv, S, D)
    value = key + 1000.0

    gqa_position_ids = torch.arange(S).repeat(B, 1)
    gqa_rates = torch.tensor([math.pi / 2, 0.0])
    gqa_angles = gqa_position_ids[..., None].to(torch.float32) * gqa_rates
    gqa_embedding = torch.cat((gqa_angles, gqa_angles), dim=-1)
    gqa_cos = gqa_embedding.cos()
    gqa_sin = gqa_embedding.sin()

    cos_with_head_axis = gqa_cos.unsqueeze(1)
    sin_with_head_axis = gqa_sin.unsqueeze(1)
    query_rope = query * cos_with_head_axis + rotate_half_demo(query) * sin_with_head_axis
    key_rope = key * cos_with_head_axis + rotate_half_demo(key) * sin_with_head_axis

    print("Q [B,Nq,S,D]:       ", tuple(query.shape))
    print("K/V [B,Nkv,S,D]:    ", tuple(key.shape))
    print("cos [B,S,D]:         ", tuple(gqa_cos.shape))
    print("cos [B,1,S,D]:       ", tuple(cos_with_head_axis.shape))
    print("Q_rope shape:        ", tuple(query_rope.shape))
    print("K_rope shape:        ", tuple(key_rope.shape))
    print("V unchanged:         ", torch.equal(value, key + 1000.0))
    print("position 0 identity: ", torch.equal(query_rope[:, :, 0], query[:, :, 0]))

    assert query_rope.shape == query.shape
    assert key_rope.shape == key.shape
    """
)

markdown(
    """
    ## 12. 放回完整 Decoder 数据流

    ```text
    token ids
      → embedding → hidden_states [B,S,H]
      → decoder layer
          → RMSNorm
          → q_proj / k_proj / v_proj
          → split + transpose
              Q [B,Nq,S,D]
              K [B,Nkv,S,D]
              V [B,Nkv,S,D]
          → position_ids 生成 cos/sin [B,S,D]
          → RoPE(Q,K)，V 不变
          → Cache 保存/追加旋转后的 K 与原始 V
          → GQA attention（Q·K、mask、softmax、weights·V）
          → merge heads → o_proj → residual add
      → 更多 decoder layers
      → final norm → LM Head → logits → 选择 next token
    ```

    哪些步骤混合 token：

    - projection：不混合，每个 token 单独做 Linear。
    - RoPE：不混合，只按各 token 自己的 position 旋转。
    - Q·K、softmax、weights·V：跨 token 比较和汇集。

    C++ 类比：Q/K 像查找键与比较规则，V 像命中后取回的 payload。这个类比解释“为什么 V 不旋转”，但 Attention 实际是连续 tensor 上的批量矩阵运算，不是哈希表查询。
    """
)

markdown(
    """
    ## 13. Prefill 与单步 decode 的 position

    - **Prefill**：一次处理整个 prompt，例如 5 个 token 使用 positions `0..4`。
    - **Decode**：Cache 已有 5 个历史 token，新 token 的全局 position 是 5，不能重新使用 0。

    如果 decode 每次重置为 0，新 Q/K 会被当成序列开头，和历史 K 的相对位置全部错位。
    """
)

code(
    """
    prompt_length = 5
    prefill_position_ids = torch.arange(prompt_length).unsqueeze(0)
    decode_position_id = torch.tensor([[prompt_length]])

    print("prefill positions [B,S]:", prefill_position_ids)
    print("decode position [B,1]:  ", decode_position_id)
    assert decode_position_id.item() == 5
    """
)

markdown(
    """
    ## 14. Parameter、Activation、Buffer 与请求 State

    | 对象 | 类别 | 生命周期 |
    |---|---|---|
    | q/k projection weights | Parameter（训练参数） | 训练学习，保存在 checkpoint，多请求共享 |
    | `position_ids` | 运行输入 | 每次请求/step 按实际位置产生 |
    | `inv_freq` | 配置或 buffer | 由 base 与 D 确定，默认不是训练参数 |
    | angles、cos、sin、Q_rope、K_rope | Activation（中间 tensor） | forward 中产生和消费 |
    | 历史 K_rope 与 V | 请求 State | 跨 decode steps 保留在 KV Cache |

    RoPE 本身通常没有需要梯度下降学习的位置参数表。
    """
)

markdown(
    """
    ## 15. 独立作业的输入契约

    这些是**题目明确要求的接口条件**，不是让你自行猜测：

    | 函数 | 必须检查 |
    |---|---|
    | `rotate_half(x)` | 最后一维 `D>0` 且为偶数 |
    | `build_rope_cos_sin(position_ids,D,base)` | position 是 rank-2 整数 tensor；D 为正偶数；base>0 |
    | `apply_rope(Q,K,cos,sin)` | Q/K 均为 rank 4；B/S/D 相同；D 为正偶数 |
    | 同上 | Q、K、cos、sin 的 device 与 dtype 一致 |
    | 同上 | cos/sin 的 shape 为 `[B,S,D]` |

    Q/K 的 head 数可以不同：Q 的 axis 1 是 `Nq`，K 的 axis 1 是 `Nkv`。cos/sin 插入的 head axis 长度为 1，可以广播给两者，因此 RoPE 前不需要 repeat K。
    """
)

markdown(
    """
    ## 16. 用不变量验证实现，而不是只看“能运行”

    RoPE 的四个核心不变量：

    1. Q/K 旋转前后 shape 不变。
    2. position 0 的 cos=1、sin=0，因此输出等于输入。
    3. 每个二维 pair 及整个 head vector 的 L2 长度不变。
    4. V 的数值和 shape 不变。

    这些不变量将来可用于 PyTorch eager、模型 export、converter、backend kernel 与 runtime 之间的数值对齐。
    """
)

code(
    """
    assert query_rope.shape == query.shape
    assert key_rope.shape == key.shape
    torch.testing.assert_close(query_rope[:, :, 0], query[:, :, 0])
    torch.testing.assert_close(key_rope[:, :, 0], key[:, :, 0])
    torch.testing.assert_close(
        torch.linalg.vector_norm(query_rope, dim=-1),
        torch.linalg.vector_norm(query, dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        torch.linalg.vector_norm(key_rope, dim=-1),
        torch.linalg.vector_norm(key, dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.equal(value, key + 1000.0)
    print("四类 RoPE 不变量全部通过。")
    """
)

markdown(
    """
    ## 17. 运行前先回答的检索题

    请不要回看上文，先口头回答：

    1. D=6 的 Qwen half-split 配对轴是什么？
    2. `position_ids [2,7]`、`D=6` 时，`inv_freq`、angles、复制后 cos/sin 的 shape 分别是什么？
    3. 为什么 RoPE 旋转 Q/K 而不旋转 V？
    4. 为什么 RoPE 不能替代 causal mask？
    5. Cache 已有 7 个 token 时，新 token 的 position_id 是多少？

    这些题不在 notebook 中显示答案；它们与正式闭卷题使用不同组织方式，目的是检查你能否从运行结果迁移到新 shape。
    """
)

markdown(
    """
    ## 18. 接下来怎么完成 0008

    1. 再运行一次本 notebook，确保能解释每个输出而不是只看到绿色对勾。
    2. 独立完成 [`../exercises/0008_rope.py`](../exercises/0008_rope.py) 的 TODO。
    3. 运行独立脚本并保存完整输出。
    4. 查看 [`../reference/0008-rope-cheatsheet.html`](../reference/0008-rope-cheatsheet.html) 后关闭资料。
    5. 闭卷完成 [`../assessments/0008-rope.md`](../assessments/0008-rope.md)，填写 [`../submissions/0008.md`](../submissions/0008.md)。

    指定的一手资料只用于核实与延伸，notebook 本身足够完成作业：

    - [RoFormer 原论文](https://arxiv.org/abs/2104.09864)：第 3.1、3.2、3.4.2 节。
    - [Hugging Face RoPE utilities](https://github.com/huggingface/transformers/blob/main/docs/source/en/internal/rope_utils.md)：Overview。
    - [Qwen3 modeling source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)：`rotate_half`、`apply_rotary_pos_emb` 与 Attention forward。

    **门禁不变：**独立代码和闭卷第 1、2、3 题全部通过，且总分至少 80，才进入 Tiny Decoder Layer。

    如果任何代码输出和文字解释对不上，直接告诉 Agent 具体单元标题与输出，不要靠背公式继续。
    """
)


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (llm)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT.relative_to(ROOT)} with {len(cells)} cells")
