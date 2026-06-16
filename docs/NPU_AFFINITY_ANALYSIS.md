# MANIQA NPU 亲和性分析

生成日期：2026-06-16

## 绑定口径

本报告按 Ascend NPU 亲和性分析流程重刷，结论只绑定下表环境，不外推到其他 NPU 架构、其他 dtype 或多卡配置。

| 项目 | 口径 |
| --- | --- |
| 目标平台 | Ascend950PR |
| NPU 架构版本 | 3510，分离式 AIC/AIV，AIC:AIV=1:2，支持 Regbase AIV、SIMT、NDDMA 等 950 系列能力 |
| 运行栈 | CANN 9.0.0，`torch 2.10.0+cpu`，`torch_npu 2.10.0`，PyTorch eager |
| 实测 dtype | FP32 eager 为主；BF16 autocast 只做 smoke，未形成精度/性能结论 |
| 任务阶段 | 图像推理与训练 smoke；MANIQA 不是 token 模型，prefill/decode 口径不适用 |
| 设备约束 | 只使用物理后四张 NPU；进程内可见 `npu:0..3`，主验证路径使用逻辑 `npu:0` |
| GPU 对比 | 本版不做 NVIDIA GPU 横向定量对比 |

## 分析结论

MANIQA 在 Ascend950PR、CANN 9.0.0、`torch_npu 2.10.0`、FP32 eager 环境下具备较好的单 NPU 亲和性。模型主干由 ViT patch embedding、Transformer MLP、TAB 通道注意力、Swin window attention、LayerNorm、Softmax、GELU、Conv2d 和 Linear 组成，重计算集中在 MatMul/AddMM/BatchMatMul/Conv2d/Softmax/LayerNorm/GELU 这类 Ascend NPU 可有效承载的 Cube 与 Vector 路径上。

但这个项目不能简单写成“全量 NPU 原生高性能”。MANIQA 的主干矩阵计算贴合 Cube 路径，Vector 路径也能承载 Softmax、LayerNorm、GELU 和 reduce；真正限制来自五类边界：CPU 图像数据管线、score head Python loop、小窗口 attention 和大量小 kernel 带来的 host/head 开销、PIPAL22/checkpoint 工程契约、多卡 HCCL 和 TorchAir 图模式环境阻塞。当前结论应定位为：单 NPU eager 可复现，完整训练吞吐、多卡训练、图模式优化和真实数据集指标仍需专项验证。

## 五路径拆解

| 路径 | 压力判断 | 关键证据 | 亲和性判断 | 流程依据 |
| --- | --- | --- | --- | --- |
| Cube | 高。ViT dense、TAB 的 `q @ k.T` / `attn @ v`、Swin projection、Conv2d 是主计算路径。 | profiler 中 `aclnnAddmm` 90 次约 `2806 us`、`BatchMatMul` 40 次约 `818 us`、`Conv2d` 7 次约 `690 us`、`MatMul` 14 次约 `637 us`。 | 主干亲和较好；大矩阵段适合 Cube，但小窗口 attention 的 tile 利用率需单独测。 | 原流程实测。 |
| Vector | 中到高。Softmax、LayerNorm、GELU、ReLU/Sigmoid、score reduce、dropout/elementwise 都走 Vector 或相关路径。 | profiler 中 Softmax 24 次约 `241 us`、LayerNorm 41 次约 `156 us`、GELU 20 次约 `136 us`，并有 Add/Mul 等 elementwise kernel。 | 功能亲和可用；Vector 不是瓶颈结论前需要 `aiv_vec_time` 和 GM bytes。 | 原流程实测。 |
| MTE/FixPipe | 中。模型内部存在 transpose/rearrange、patch/window layout 变化和 H2D 输入搬运；PyTorch eager 下未确认 NDDMA/layout 折叠。 | profiler 中 `TransposeAiCore` 30 次约 `330 us`；OpenCV/NumPy 预处理在 CPU，进入 NPU 前有 host 到 device 传输。 | 可运行但有搬运和小包风险；layout 是否能折叠为地址生成或持久 layout 属于待测项。 | 原流程实测，layout 折叠为重写待测。 |
| communication | 单 NPU 路径无压力；多卡路径阻塞。 | `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` 可正确隔离设备；`nn.DataParallel` 和两进程 HCCL `all_reduce` 当前失败。 | 单卡可用；多卡不可声明可用，需先修 HCCL 最小 collective。 | 原流程实测。 |
| host/head | 高。PyTorch eager、20-crop 串行预测、score head Python loop 和大量小 kernel 会放大 launch/head 开销。 | batch 1 约 `13.78 ms/image`，batch 8 降到约 `2.12 ms/image`；score head 逐样本 `torch.cat`；README 单图预测是 20 次 crop 串行 forward。 | 当前最大工程优化面；优先 batch 化和向量化，再考虑图模式。 | 原流程实测；batch 化/向量化为等价重写建议。 |

## 理论量化与 roofline 初判

本节只做粗 roofline 初判，不把理论值写成承诺性能。时间模型按：

```text
T_segment ≈ T_head + max(T_GM, T_Cube_or_Vector)
```

矩阵路径应和 Cube 平衡点比较，向量/reduce/layout 路径应和 Vector 或 GM 搬运比较，不能混用。按本报告采用的亲和性分析参考框架，950PR 的 FP16 Cube 平衡点约为 `270 FLOP/Byte`；本轮实测主口径是 FP32 eager，缺少可核验的 950PR FP32 Cube/Vector 峰值与实际 GM bytes，因此 FP32 精确平衡点标为 `待测`。下表中的算术密度是基于 MANIQA 形状的数量级估算，不含 cache、tile、mask、layout 和 runtime overlap 效率。

| 子段 | 形状/工作量口径 | 主导路径 | 算术密度 vs 平衡点 | 初判 |
| --- | --- | --- | --- | --- |
| ViT patch embedding 与 dense/MLP | 输入 `224x224`，patch size 8，token 数 `28x28=784`，hidden dim 768 | Cube + MTE | dense/MLP 具备较高复用；按典型 `[784,768] x [768,*]` 估算为百级 FLOP/Byte，接近或低于 FP16 Cube 平衡点，FP32 平衡点待测 | 主干可贴 Cube，但仍可能受权重/激活 GM 搬运影响。 |
| TAB stage 1 | concat 4 层 ViT token 后约 `[B,3072,784]`，`q@k.T` 形成 `[B,3072,3072]` | Cube + Vector Softmax | `q@k.T` 按 FP32 读写粗估约 `260 FLOP/Byte`，接近 950PR FP16 Cube 平衡点；Softmax 是 Vector/GM 路径 | 矩阵段亲和好，Softmax 与大 attention map 物化是待测风险。 |
| TAB stage 2 | 约 `[B,768,784]`，`q@k.T` 形成 `[B,768,768]` | Cube + Vector Softmax | 粗估百级 FLOP/Byte，低于 stage 1；tile 更小，head 占比更高 | 中等亲和；需要 profiler GM bytes 和 tile 利用率确认。 |
| Swin window attention | window size 4，单 window 16 token，窗口数约 49 | 小 Cube + Vector/head | attention 矩阵很小，理论 FLOPs 不大；小 GEMM tile 利用率与 launch/head 可能主导 | 亲和中等；linear projection 比 window attention 本体更贴 Cube。 |
| score head | `[B,784,C]` 上 `fc_score/fc_weight` 后 reduce；当前逐样本 Python loop | Vector + host/head | small MLP 和 reduce 算术密度不高；反复 `torch.cat` 放大小包和 head 开销 | 原流程亲和差；应做等价向量化。 |
| 图像预处理与 H2D | OpenCV/NumPy 读图、resize/crop/normalize，再传 NPU | host + MTE | NPU 算术密度为 0；真实训练吞吐取决于 CPU worker、H2D 和 batch | 不是 NPU kernel 问题，但会限制端到端吞吐。 |

模型级推理可分为：

```text
T_image ≈ T_preprocess_CPU + T_H2D + T_head_eager + max(T_GM_layout, T_Cube主干 + T_Vector后处理)
```

当前实测 batch 变大后 `ms/image` 明显下降，说明 `T_head_eager` 和小 kernel 开销不可忽略。真实训练还要额外叠加 DataLoader、CPU 增强、反向传播、optimizer 和 CPU 侧 SRCC/PLCC 统计；这些不能从单 forward profiler 直接外推。

## NPU 特有亲和项自检

| 项目 | 当前观察 | 风险与待测 |
| --- | --- | --- |
| tile 驻留与 double buffer | ViT/TAB 大矩阵段理论上具备 tile 复用；Swin window attention tile 小。 | 缺 L0/L1/UB tile 利用率和 double buffer 效率数据，需用 CANN profiler 或 microbench 补证。 |
| 32B/512B 对齐与小包 | 3510 口径下 UB 32B、L0 分形 512B；PyTorch eager 隐藏了底层 copy 切分。 | 不能套用 A2 的 GM 512B 规则；950 L2 cacheline/sector 行为和实际小包效率待测。 |
| repeat/mask 密度 | Softmax/LayerNorm/GELU 可运行；score reduce 和 window attention 存在小向量/小矩阵。 | 需要 `aiv_vec_time`、mask 利用率和小 shape microbench 判断 Vector 是否低效。 |
| layout 折叠 | 代码中大量 `einops.rearrange`、window partition、transpose；部分可能是 view，部分触发真实 transpose kernel。 | 已观察 `TransposeAiCore`；哪些 layout 能折叠为地址生成、stride copy、NDDMA 或持久 layout 需要逐段确认。 |
| reduce/state layout | score head 对 patch 维度做加权 reduce；训练指标 SRCC/PLCC 在 CPU 侧统计。 | score reduce 可向量化；SRCC/PLCC 若要 NPU 化，需要单独设计排序/统计路径。 |
| 同步边界 | PyTorch eager 每个 kernel 有 launch/head；20-crop 预测是 Python 串行循环。 | 跨 kernel 硬同步无法通过“少一次 view”消除；收益主要来自 batch 化、向量化和图模式。 |
| 通信边界 | 单卡无通信；多卡 HCCL 当前失败。 | DDP 前必须先让最小 `all_reduce` 通过，否则 MANIQA 上层没有可靠多卡结论。 |

## 已验证的 NPU 计算路径

| 能力范围 | 实测结果 |
| --- | --- |
| NPU 环境与设备隔离 | `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` 下 `torch.npu.device_count() == 4`，默认计算设备为 `npu:0`。 |
| 单图 checkpoint 预测 | `python predict_one_image.py` 在 NPU 上成功运行，`kunkun.png` 输出约 `0.3407`，与 README 中 `0.3398` 接近。 |
| README 五张示例图 | 使用 Koniq10k checkpoint、20-crop 路径，在 NPU 上得到 `kunkun 0.340658`、`bird 0.261935`、`dog 0.308199`、`ball 0.372101`、`people 0.358600`，与 README 示例数值一致。 |
| 模型 forward | 使用项目默认 MANIQA 配置，随机 `1x3x224x224` NPU 输入可得到 `torch.Size([1])` 输出。 |
| batch 推理吞吐 | 随机权重 forward 计时：batch 1 约 `13.78 ms/image`，batch 4 约 `3.93 ms/image`，batch 8 约 `2.12 ms/image`。 |
| 训练/验证 epoch | 使用 Koniq 合成 DataLoader 和随机权重模型，`train_epoch`、`eval_epoch` 在 NPU 上完成 forward/backward/Adam/scheduler 和 SRCC/PLCC 计算。 |
| BF16 autocast | BF16 autocast smoke 通过，但最终 score tensor 仍为 `float32`，尚未做精度收益结论。 |
| NPU profiler | `torch_npu.profiler` 可生成 CANN trace，top kernel 与 ViT/TAB/Swin 结构匹配。 |
| 多卡和图模式 | `DataParallel`、HCCL `all_reduce`、TorchAir graph mode 当前均未通过。 |

## 不亲和点与等价重写

| 原流程问题 | 数学等价/工程重写 | 预期收益 | 验证要求 |
| --- | --- | --- | --- |
| score head 逐样本 loop + `torch.cat` | 将 `fc_score(x)`、`fc_weight(x)` 保持为 `[B,N,1]`，直接按 patch 维 reduce：`score = (f*w).sum(dim=(1,2)) / w.sum(dim=(1,2))` | 减少 Python loop、小 kernel、cat 物化和 host/head 开销 | 用 README 五图和 checkpoint 单图对齐，误差应仅为浮点舍入级。 |
| 20-crop 串行 forward | 将 20 个 crop stack 成 batch，一次或分批送入模型后平均 | 摊薄 launch/head，提升端到端单图预测吞吐 | 与串行 20-crop 结果逐图对齐，记录显存峰值。 |
| PIPAL22 推理 checkpoint 形态不统一 | 增加 state dict 与完整模型对象两种加载分支 | 提升 challenge 推理脚本开箱可用性 | 用实际 PIPAL22 checkpoint 文件验证。 |
| layout transpose/rearrange 多 | 能保持 view 的保持 view；真实 transpose 尝试持久 layout 或在图模式中折叠 | 降低 MTE/FixPipe 和 transpose kernel 开销 | 需要 CANN profiler 的 GM bytes 与 transpose kernel 对比。 |
| CPU 图像预处理 | DataLoader worker 亲和后四卡 CPU range；批量 H2D；必要时评估 DVPP/AscendCL 预处理 | 避免真实训练时 CPU pipeline 吞吐不足 | 用真实数据集跑端到端吞吐，不用合成样例外推。 |

基于等价重写后的亲和性刷新：score head 向量化和 20-crop batch 化都不改变 MANIQA 数学语义，属于优先级最高的安全改造；TorchAir 图模式和 layout 持久化需要先解决环境或 profiler 证据，不应先写成确定收益。

## 阻塞项分析

阻塞项按 NPU 复现链路归纳如下。这里的“阻塞”既包括明确运行失败，也包括当前环境缺少真实数据或 checkpoint 形态不明确，导致不能把 smoke 结果外推为完整训练、完整推理或多卡能力。

| 阻塞的复现链路 | 实测现象 | 影响范围 | 判断 |
| --- | --- | --- | --- |
| 真实 Koniq10k/KADID-10K/PIPAL 数据集指标复现 | 当前复现环境未提供 README 对应真实数据目录；本轮只能用合成样例验证 Dataset、transform、训练循环和推理入口 | 官方 SRCC/PLCC、完整训练 epoch、真实验证集推理 | 属于数据可用性阻塞，不能把合成样例 smoke 写成官方指标复现。 |
| NPU 多卡训练 | `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` 可正确限制可见设备，但 `nn.DataParallel` 在小 Linear 模型上失败；两进程 HCCL `all_reduce` 在 `HCCLUtils.cpp:140` 报 error code `4` | DataParallel、DDP、跨卡梯度同步、分布式训练吞吐 | 属于底层 collective/运行栈阻塞；在 HCCL 最小 all-reduce 通过前，不应迁移 MANIQA 到多卡训练。 |
| TorchAir 图模式 | `torchair.get_npu_backend()` 可 import，但 tiny model 和 MANIQA 的 `torch.compile` 都在 GE/TBE 初始化阶段失败 | 静态 shape 图编译、host launch 开销优化、图模式性能结论 | 属于图编译环境阻塞；当前只能以 eager NPU 作为可复现路径。 |
| PIPAL22 推理 checkpoint 形态 | `predict_one_image.py` 使用 state dict 构建模型；`inference.py` 当前按完整模型对象 `torch.load(config.model_path)` 使用 | PIPAL22 challenge 推理脚本的开箱可用性 | 属于 checkpoint 格式契约不明确；如果实际 checkpoint 是 state dict，需要补模型构建和 `load_state_dict` 分支。 |
| 固定输入尺寸与真实图像预处理 | ViT patch embed 要求 `224x224` 输入；PIPAL22 原图需要通过 five-point crop 后进入模型 | 任意尺寸图片推理、PIPAL22 批量推理、服务化输入校验 | 属于前处理契约阻塞；推理入口必须显式 resize/crop 或拒绝不合规输入。 |

## 待测清单

| 待测项 | 目的 | 建议方法 |
| --- | --- | --- |
| FP32/BF16/FP16 roofline microbench | 确认 950PR 在当前 torch_npu 栈下的实际 Cube/Vector 平衡点 | 对 MatMul、BatchMatMul、Softmax、LayerNorm、GELU 分 dtype 做 microbench，记录 FLOPs、GM bytes、AIC/AIV 时间。 |
| profiler GM bytes 与 tile 利用率 | 验证上文 roofline 初判 | 用 CANN profiler 展开 kernel 级 GM 读写、AIC/AIV 时间、transpose bytes。 |
| score head 向量化 | 降低 host/head 和小 kernel 开销 | 改写后跑 README 五图、batch 1/4/8 profiler 对比。 |
| 20-crop batch 化 | 提升单图预测吞吐 | 对比串行 20 次 forward 和 batch forward 的分数、时延、峰值显存。 |
| HCCL 最小 all_reduce | 解锁 DDP 判断 | 先跑两进程 NPU tensor `all_reduce`，通过后再测 MANIQA DDP。 |
| TorchAir tiny compile | 解锁图模式判断 | 先让 tiny Linear/MLP compile 通过，再评估 MANIQA 静态 shape。 |
| 真实数据 DataLoader affinity | 判断端到端训练瓶颈 | 在后四卡 CPU affinity 范围内扫 `num_workers`、batch size、prefetch。 |

## 资料来源

- Ascend 950 NPU 架构白皮书
- Ascend Extension for PyTorch: https://github.com/Ascend/pytorch
- torch-npu package metadata: https://pypi.org/project/torch-npu/
- TorchAir guide: https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00003.html
- MANIQA checkpoint release: https://github.com/IIGROUP/MANIQA/releases/tag/Koniq10k
