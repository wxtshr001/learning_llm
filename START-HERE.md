# 从这里开始

当前课程状态：**第 0001～0008 课已通过；正在进行第 0009 课 Tiny Decoder Layer。**

先打开 [第 0009 课可视化教程：组装 Tiny Decoder Layer](./lessons/0009-tiny-decoder-layer-course/index.html)。它用四个紧凑模块讲清完整 layer、Attention shape、FFN residual、因果性与生命周期；只保留承担解释任务的数据流图、shape 图和 token 依赖图。随后使用 [Jupyter Notebook](./notebooks/0009-tiny-decoder-layer.ipynb) 运行代码；[旧 HTML 阅读版](./lessons/0009-tiny-decoder-layer.html) 仅作备用。

## 今天要做的事

1. 学完 0009 可视化教程的四个模块，并完成末尾三道迁移题。
2. 从上到下运行 0009 notebook；每个“先预测”先回答，再执行代码。
3. 运行 [0009 引导脚本](./exercises/0009_explore_decoder_layer.py)，逐行核对两条 residual。
4. 完成 [0009 独立作业](./exercises/0009_tiny_decoder_layer.py) 中的四个 TODO。
5. 查看 [0009 速查表](./reference/0009-tiny-decoder-layer-cheatsheet.html)，然后关闭资料。
6. 完成 [0009 闭卷测试](./assessments/0009-tiny-decoder-layer.md)，填写 [0009 提交模板](./submissions/0009.md)。

## 我会如何判定

- **通过**：独立代码与关键题 1、2、3 正确且总分至少 80，进入 KV Cache 增量执行。
- **针对性补强**：只补 residual 基线、完整 shape 或 token mixing 中的实际缺口。
- **评分约束**：不增加题面未声明条件，单次算术笔误结合完整证据判断。

第 0009 课预计 100～130 分钟；不要求重写已通过组件，不实现 KV Cache，不考 backward 或多 layer 堆叠。

## 其他文件是什么

- [课程地图](./CURRICULUM.md)：课程顺序、门禁和可能的分支。
- [完整路线](./LEARNING-ROUTE.md)：24 周项目全景，不是每天从头照读的教材。
- [资料清单](./RESOURCES.md)：经过筛选的官方资料库。
- [学习使命](./MISSION.md)：所有课程为什么要学、哪些内容不学。
- `learning-records/`：记录已经真正证明掌握的能力。
- `reference/`：以后反复查阅的速查表。

## 环境说明

已建立独立 Conda 环境 `llm`：Python 3.11.15、PyTorch 2.12.1、CUDA 12.6 和 NumPy 2.4.6。

```powershell
conda activate llm
python exercises/0000_verify_pytorch.py
```

启动当前 notebook：

```powershell
conda activate llm
jupyter lab notebooks/0009-tiny-decoder-layer.ipynb
```

kernel 选择 `Python 3 (llm)`。
