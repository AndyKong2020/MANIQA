# MANIQA NPU 亲和性分析

生成日期：2026-06-12

## 分析结论

MANIQA 在 Ascend950PR、CANN 9.0.0、`torch_npu 2.10.0` 环境下具备较好的单 NPU eager 亲和性。模型主干由 ViT patch embedding、Transformer MLP、TAB 通道注意力、Swin window attention、LayerNorm、Softmax、GELU、Conv2d 和 Linear 组成，重计算集中在 MatMul/AddMM/BatchMatMul/Conv2d/Softmax/LayerNorm/GELU 这类 Ascend NPU 友好的 tensor 和 vector 算子上。实测 checkpoint 预测、README 五图预测、随机 forward、训练 backward、PIPAL22 推理入口和 profiler 均能在后四卡隔离环境下工作。

但这个项目不能简单归类为“全量 NPU 原生高性能”。当前风险集中在四类路径：第一类是 CPU-bound 图像数据管线，OpenCV/NumPy 读取、resize、随机裁剪和 normalize 仍在 CPU 执行；第二类是模型 score head 仍包含 Python batch loop 和逐样本 `torch.cat`，功能正确但不利于大 batch 和图模式；第三类是多卡和图模式环境尚未打通，`DataParallel`、HCCL `all_reduce` 和 TorchAir 编译都失败；第四类是复现工程问题，包括真实数据路径硬编码、训练入口缺少 `tensorboard` 依赖声明、NumPy 2.x 对 vendored timm 的潜在兼容风险，以及 224 固定输入尺寸对推理图片的约束。

因此，本次结论是：MANIQA 的核心视觉 Transformer 推理和单卡训练计算对 Ascend950PR 适配度高，适合作为单 NPU eager 复现目标；完整训练吞吐、多卡训练、TorchAir 图优化和真实数据集指标复现仍需要专项处理后才能作为稳定交付。

## 已验证的 NPU 计算路径

本次实测的 NPU 友好路径比较清晰。`predict_one_image.py` 使用 Koniq10k checkpoint 在 NPU 上完成 20-crop 单图预测，README 中五张示例图也完成复现：`kunkun.png 0.340658`、`bird.jpg 0.261935`、`dog.jpg 0.308199`、`ball.jpg 0.372101`、`people.jpg 0.358600`。这些分数与 README 中 `0.3398/0.2612/0.3078/0.3716/0.3581` 接近，说明 checkpoint 加载、图像预处理、模型主干和 score head 在 NPU 上可以给出合理结果。

模型 forward 覆盖了完整 MANIQA 结构。ViT backbone 通过 forward hook 取第 6 到第 9 个 Transformer block 的 patch token，随后进入 TABlock 的 `q @ k.transpose`、Softmax 和 `attn @ v`，再进入两级 SwinTransformer 局部窗口建模，最后通过 `fc_score` 与 `fc_weight` 做 patch 加权质量分。随机输入、checkpoint 输入和训练输入都验证了这个路径可以在 `npu:0` 上执行。

训练路径也已覆盖到 autograd 和优化器。使用合成 Koniq10k DataLoader 跑 `train_epoch`，NPU 上完成 forward、MSE loss、backward、Adam step、CosineAnnealingLR step 和 CPU 侧 SRCC/PLCC 统计；随后 `eval_epoch` 也完成 five-point crop、forward 和 loss/相关系数统计。这个结果说明训练入口的 NPU 设备迁移是可用的，但不能替代真实 Koniq/KADID/PIPAL 训练指标。

数据集路径覆盖了项目主要功能面。`Koniq10k`、`Kadid10k`、`PIPAL` 三个训练 Dataset 均能从合成图片和标签生成 `3x224x224 float32` tensor；`PIPAL22` 推理 Dataset 能读取目录图片并保留原始尺寸，真实进入模型前需要通过 `inference.eval_epoch` 的 five-point crop 转成 `224x224`。这点很重要，因为 ViT patch embedding 固定检查输入高度和宽度必须等于 224。

Profiler 结果与模型结构匹配。一次 NPU forward 中耗时靠前的 kernel 为：

| Kernel | 调用次数 | 总耗时 |
| --- | ---: | ---: |
| `aclnnAddmm_MatMulV3Common_MatMulV3` | 90 | 约 `2806 us` |
| `aclnnMatmul_BatchMatMulV3Nd_BatchMatMulV3` | 40 | 约 `818 us` |
| `aclnnConvolution_Conv2dWithFlag_Conv2DV2` | 7 | 约 `690 us` |
| `aclnnMatmul_MatMulV3Common_MatMulV3` | 14 | 约 `637 us` |
| `aclnnMuls_MulAiCore_Mul` | 28 | 约 `366 us` |
| `aclnnAdd_AddAiCore_Add` | 67 | 约 `355 us` |
| `aclnnMatmul_TransposeAiCore_Transpose` | 30 | 约 `330 us` |
| `aclnnSoftmax_SoftmaxAiCore_SoftmaxV2` | 24 | 约 `241 us` |
| `aclnnLayerNormWithImplMode_LayerNormV4` | 41 | 约 `156 us` |
| `aclnnGelu_Gelu_Gelu` | 20 | 约 `136 us` |

这说明 MANIQA 的重算子主要落在 NPU 的矩阵计算、卷积和向量激活归一化路径上。结合 Ascend 950 NPU 架构白皮书，950PR 的 Cube Core 面向矩阵和 tensor 计算，Vector Core 面向 Softmax/GELU 等向量算子，并支持 BF16/FP16/TF32/FP8 等低精度格式；MANIQA 的主干算子与这些能力匹配。

## fallback 与性能风险

实测中有几类风险不会阻断单卡功能正确性，但会影响“纯 NPU”和性能判断。

| 路径 | 观察 | 影响 |
| --- | --- | --- |
| 图像预处理 | Dataset 和单图预测使用 OpenCV/NumPy 做读取、resize、随机 crop、normalize，再将 tensor 送 NPU。 | 真实训练和批量推理可能被 CPU 数据管线限制，不能只看模型 forward 时间。 |
| score head | `models/maniqa.py` 逐样本循环计算 `fc_score/fc_weight`，并反复 `torch.cat`。 | 功能正确，但会制造小 kernel 和 host dispatch；batch 越大越不利。 |
| eager launch | batch 1 约 `13.78 ms/image`，batch 8 降到约 `2.12 ms/image`。 | 说明单图 eager 模式 launch 占比明显，服务化推理应优先 batch 化或后续再评估图模式。 |
| BF16 autocast | BF16 autocast smoke 通过，但最终 score tensor 仍为 `float32`。 | 不能据此声称 BF16 已带来性能或显存收益，需要逐层 dtype/profiler 复查。 |
| NumPy 2.x | 当前 venv 为 `numpy 2.4.6`，vendored `timm/data/mixup.py` 仍有 `np.bool`。 | 默认 MANIQA 路径未触发，但如果启用 timm mixup/augment 路径会有兼容风险。 |
| TensorBoard | 训练入口 import `torch.utils.tensorboard.SummaryWriter`。 | 未安装 `tensorboard` 时训练脚本 import 即失败；已在当前环境补装，并补充到 `requirements.txt`。 |
| 固定输入尺寸 | ViT patch embed 要求输入为 `224x224`。 | PIPAL22 原图或任意小图不能直接进模型，必须 resize/crop；`predict_one_image.Image` 对高度或宽度刚好等于 224 的图片也存在 `np.random.randint(0, 0)` 风险。 |
| 输出排序 | 原 `sort_file` 写 `./output.txt`。 | 已修复为原地排序指定文件；否则 PIPAL22 复现产物会落错目录。 |

还观察到两个 warning。`torch.meshgrid` 未来需要显式传入 `indexing` 参数，这来自 Swin/timm 路径，当前不影响运行。NPU backward 中出现 `Cannot create tensor with interal format... base format`，当前未造成失败，但说明部分 tensor factory 未走内部优化格式，后续做性能分析时应保留这个线索。

## 阻塞项分析

阻塞项按 NPU 复现链路归纳如下。这里的“阻塞”既包括明确运行失败，也包括当前环境缺少真实数据或 checkpoint 形态不明确，导致不能把 smoke 结果外推为完整训练、完整推理或多卡能力。

| 阻塞的复现链路 | 实测现象 | 影响范围 | 判断 |
| --- | --- | --- | --- |
| 真实 Koniq10k/KADID-10K/PIPAL 数据集指标复现 | 当前复现环境未提供 README 对应真实数据目录；本轮只能用合成样例验证 Dataset、transform、训练循环和推理入口 | 官方 SRCC/PLCC、完整训练 epoch、真实验证集推理 | 属于数据可用性阻塞，不能把合成样例 smoke 写成官方指标复现。 |
| NPU 多卡训练 | `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` 可正确限制可见设备，但 `nn.DataParallel` 在小 Linear 模型上失败；两进程 HCCL `all_reduce` 在 `HCCLUtils.cpp:140` 报 error code `4` | DataParallel、DDP、跨卡梯度同步、分布式训练吞吐 | 属于底层 collective/运行栈阻塞；在 HCCL 最小 all-reduce 通过前，不应迁移 MANIQA 到多卡训练。 |
| TorchAir 图模式 | `torchair.get_npu_backend()` 可 import，但 tiny model 和 MANIQA 的 `torch.compile` 都在 GE/TBE 初始化阶段失败 | 静态 shape 图编译、host launch 开销优化、图模式性能结论 | 属于图编译环境阻塞；当前只能以 eager NPU 作为可复现路径。 |
| PIPAL22 推理 checkpoint 形态 | `predict_one_image.py` 使用 state dict 构建模型；`inference.py` 当前按完整模型对象 `torch.load(config.model_path)` 使用 | PIPAL22 challenge 推理脚本的开箱可用性 | 属于 checkpoint 格式契约不明确；如果实际 checkpoint 是 state dict，需要补模型构建和 `load_state_dict` 分支。 |
| 固定输入尺寸与真实图像预处理 | ViT patch embed 要求 `224x224` 输入；PIPAL22 原图需要通过 five-point crop 后进入模型 | 任意尺寸图片推理、PIPAL22 批量推理、服务化输入校验 | 属于前处理契约阻塞；推理入口必须显式 resize/crop 或拒绝不合规输入。 |

真实数据集指标复现仍被数据路径阻塞。`train_maniqa.py` 和 `inference.py` 默认使用固定数据目录，这些目录在当前复现环境中不可用。本轮只能验证 Dataset 解析、transform、训练循环和推理入口的功能正确性，不能声称已经复现 README 中 Koniq10k、KADID-10K 或 PIPAL22 的 SRCC/PLCC。

多卡训练当前被底层运行栈阻塞。`ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` 能正确限制可见设备，但 `nn.DataParallel` 在 NPU 上失败；进一步用两进程 HCCL 做最小 `all_reduce` 也在 `HCCLUtils.cpp:140` 报 error code `4`。因此在 HCCL 独立 all-reduce 修复前，不应把 MANIQA 训练迁移到 DDP，也不能把失败归因到 MANIQA 模型本身。

TorchAir 图模式当前不可用。`torchair` 已安装，`torchair.get_npu_backend()` 可调用，但 tiny model 和 MANIQA 的 `torch.compile` 都在 GE/TBE 初始化阶段失败，报错涉及 `InitCannKB`、`tbe-custom`、`GEInitializeV2` 和 `ERR03005 GRAPH internal error`。这说明当前容器环境还不满足 TorchAir 编译运行条件，不能把图模式作为本轮复现基线。

PIPAL22 推理 checkpoint 格式也需要明确。`predict_one_image.py` 加载的是 state dict，并手动构建 MANIQA；`inference.py` 当前逻辑是 `torch.load(config.model_path)` 后直接把结果当完整模型对象使用。本轮用随机完整模型对象验证了 `inference.eval_epoch` 和输出文件路径，但如果实际 PIPAL22 checkpoint 是 state dict，`inference.py` 还需要补一个与 `predict_one_image.py` 一致的模型构建和 `load_state_dict` 分支。

固定输入尺寸是推理链路的显式前处理约束。MANIQA 的 ViT patch embed 会检查输入高度和宽度必须为 `224x224`，因此 PIPAL22 原图或任意尺寸用户图片不能直接进入模型。本轮 PIPAL22 入口通过 `inference.eval_epoch` 的 five-point crop 验证可行，但后续如果要做通用推理服务，应在入口处明确 resize/crop 策略和小图拒绝逻辑。

## 适配建议

后续如果要把这套复现推进到可交付的 NPU 适配，优先级应放在能直接提升可用面和可解释性的点上。

第一，向量化 `MANIQA.forward` 的 score aggregation，避免逐样本 Python loop 和反复 `torch.cat`。等价实现可以把 `fc_score(x)` 和 `fc_weight(x)` 保持为 `[B, N, 1]`，再按 `dim=(1, 2)` 求和得到每个 batch 样本的质量分。改完后需要用五张 README 示例图回归分数。

第二，把路径、checkpoint 格式和运行模式从源码常量变成 CLI/config。至少应覆盖 `image_path`、`ckpt_path`、`dataset_name`、三类数据目录、PIPAL22 `test_dis_path`、`model_path`、`num_workers`、`batch_size`、`num_crops` 和 `vit_pretrained`。这能避免为了换数据集或 checkpoint 反复改源码。

第三，继续收敛依赖兼容性。当前训练入口实际需要的 `tensorboard` 已补充到 `requirements.txt`；当前环境的 `numpy 2.4.6` 对旧 timm 代码仍存在风险，建议要么固定 `numpy<2`，要么修掉 vendored timm 中的 `np.bool` 等旧别名。

第四，真实训练先保持单 NPU。后四卡 CPU affinity 在 topology 中对应 `64-127,192-255`，真实数据训练时应在这个 CPU 范围内测试 DataLoader worker 数量、prefetch 和 batch size。只有当最小 HCCL `all_reduce` 通过后，再考虑 DDP。

第五，TorchAir 作为独立优化阶段处理。先用 tiny model 证明 `torch.compile(..., backend=torchair.get_npu_backend())` 能工作，再评估 MANIQA 的静态 shape 推理。不要在 tiny compile 失败时直接调 MANIQA 图模式。

第六，补 NPU smoke 脚本，作为后续每次同步后的最小验收。脚本应覆盖后四卡 device guard、checkpoint 单图预测、随机 batch forward、训练一步、PIPAL22 two-image inference、profiler 可启动，并在多卡/HCCL/TorchAir 失败时输出明确的环境阻塞原因。

## 资料来源

- Ascend 950 NPU 架构白皮书
- Ascend Extension for PyTorch: https://github.com/Ascend/pytorch
- torch-npu package metadata: https://pypi.org/project/torch-npu/
- TorchAir guide: https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00003.html
- MANIQA checkpoint release: https://github.com/IIGROUP/MANIQA/releases/tag/Koniq10k
