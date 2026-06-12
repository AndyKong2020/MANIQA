# MANIQA NPU 部署复现报告

生成日期：2026-06-12

## 任务概述

本次部署任务是在 MANIQA 代码基础上，将项目同步到 Ascend NPU 容器环境内，建立可复用 Python 运行环境，并完成一轮覆盖模型预测、训练入口、数据集加载、PIPAL22 推理入口、checkpoint 加载、NPU profiler 和后四卡隔离的复现验证。

MANIQA 是面向 No-Reference Image Quality Assessment 的 PyTorch 项目，核心能力是对单张失真图像预测质量分数。模型结构由 ViT 特征提取、Transposed Attention Block、Scale Swin Transformer Block 和 patch 加权打分头组成。仓库 README 中提供了 Koniq10k、KADID-10K、PIPAL22 三类训练/推理路径，以及单图质量分预测脚本。

代码版本锚点为 commit `f573d8624012`。远端复现运行在 Ascend NPU 容器内，容器镜像为 `quay.io/ascend/cann:9.0.0-950-openeuler24.03-py3.12-devel`，虚拟环境使用 Python 3.11。

硬件环境为 8 张 Ascend950PR，`npu-smi` 版本 `25.7.rc1`，单卡 HBM 约 `114688 MB`。本次按任务约束只使用物理 NPU `4,5,6,7`，运行时通过 `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` 隔离设备；隔离后进程内可见 `npu:0..3`，其中逻辑 `npu:0` 映射到物理 NPU 4。测试结束后 `npu-smi info` 未发现残留运行进程。

## 部署过程

代码通过同步工具复制到远端工作目录。同步后的代码基于 `f573d8624012`，并包含本次为了 NPU 复现所做的适配：新增 `utils/accelerator.py`，将训练、预测和推理入口从硬编码 `.cuda()` 改为自动选择 `npu/cuda/cpu`，让 `MANIQA` 支持 `vit_pretrained=False` 以避免离线环境构建模型时下载权重，并把 score tensor 创建从 `torch.tensor([]).cuda()` 改为跟随输入设备。

补充覆盖时发现并修复了两个 PIPAL22 复现阻塞点。第一，`inference.py` 原来导入 `data.pipal22_test`，但仓库实际文件路径是 `data/PIPAL22/pipal22_test.py`；现已改为正确导入。第二，`utils/inference_process.sort_file` 原来总是把排序结果写到当前目录 `./output.txt`，而不是 `inference.py` 生成的 `config.valid_path/output.txt`；现已改为默认原地排序指定文件，避免输出落错位置。

依赖安装采用 Python 3.11.6 虚拟环境。当前 Ascend 950 / CANN 9.0.0 运行栈使用 `torch 2.10.0+cpu` 与 `torch_npu 2.10.0`。补充训练入口覆盖时发现 `train_maniqa.py` import `torch.utils.tensorboard.SummaryWriter`，因此运行训练入口需要安装 `tensorboard`；已在复现环境中补装 `tensorboard 2.20.0`，并补充到 `requirements.txt`。

最终关键包版本如下：

| 组件 | 版本 |
| --- | --- |
| Python | `3.11.6` |
| CANN | `9.0.0` |
| torch | `2.10.0+cpu` |
| torch_npu | `2.10.0` |
| torchvision | `0.25.0+cpu` |
| numpy | `2.4.6` |
| opencv-python-headless | `4.13.0.92` |
| scipy | `1.17.1` |
| tensorboard | `2.20.0` |
| tensorboardX | `2.6.5` |
| einops | `0.8.2` |
| torchsummary | `1.5.1` |

官方 Koniq10k checkpoint 已放置为项目根目录下的 `ckpt_koniq10k.pt`。文件大小为 `543335435` bytes，SHA256 为 `a207f8ab57322e6be38ff5c8d019301dc032b454bef21c3c9f9dbf7974eebff6`。

## 功能验证结果

本次验证不是只跑最小 smoke，而是按 MANIQA 的实际能力面做了分层覆盖：先确认 NPU 运行栈和后四卡隔离，再覆盖模型 forward、checkpoint 单图预测、README 示例图、三类训练数据集、PIPAL22 推理入口、训练/验证 epoch、autograd/optimizer、profiler、BF16 smoke、TorchAir 和多卡边界。

| 能力范围 | 实测结果 |
| --- | --- |
| NPU 环境与设备隔离 | `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` 下 `torch.npu.device_count() == 4`，默认计算设备为 `npu:0`。 |
| 单图 checkpoint 预测 | `python predict_one_image.py` 在 NPU 上成功运行，`kunkun.png` 输出约 `0.3407`，与 README 中 `0.3398` 接近。 |
| README 五张示例图 | 使用 Koniq10k checkpoint、20-crop 路径，在 NPU 上得到 `kunkun 0.340658`、`bird 0.261935`、`dog 0.308199`、`ball 0.372101`、`people 0.358600`，与 README 示例数值一致。 |
| 模型 forward | 使用项目默认 MANIQA 配置，随机 `1x3x224x224` NPU 输入可得到 `torch.Size([1])` 输出。 |
| batch 推理吞吐 | 随机权重 forward 计时：batch 1 约 `13.78 ms/image`，batch 4 约 `3.93 ms/image`，batch 8 约 `2.12 ms/image`。batch 化能明显摊薄 eager launch 开销。 |
| Koniq10k 数据集路径 | 合成标签和图片样例可完成 `Koniq10k` Dataset 加载、resize、normalize、`ToTensor`，输出 `3x224x224 float32`。 |
| KADID-10K 数据集路径 | 合成标签和图片样例可完成 `Kadid10k` Dataset 加载、normalize、crop、`ToTensor`，输出 `3x224x224 float32`。 |
| PIPAL 训练数据集路径 | 合成标签和图片样例可完成 `PIPAL` Dataset 加载、normalize、crop、`ToTensor`，输出 `3x224x224 float32`。 |
| PIPAL22 推理数据集路径 | `PIPAL22` Dataset 可读取目录图片，原始输出为 `3x256x256 float32`；通过 `inference.eval_epoch` 的 five-point crop 后可进入固定 224 输入模型。 |
| 训练/验证 epoch | 使用 Koniq 合成 DataLoader 和随机权重模型，`train_epoch`、`eval_epoch` 在 NPU 上完成 forward/backward/Adam/scheduler 和 SRCC/PLCC 计算。样例 `train_loss=0.4549516`、`eval_loss=0.2429362`。 |
| PIPAL22 输出文件 | `inference.eval_epoch` 可生成推理输出文件，`sort_file` 修复后原地排序，未再生成根目录 `output.txt`。 |
| autograd 与 optimizer | 单步 MSE backward 和 Adam 更新通过。 |
| BF16 autocast | `torch.npu.amp.autocast(dtype=torch.bfloat16)` smoke 通过，但最终 score tensor 仍为 `float32`，尚未做精度收益结论。 |
| NPU profiler | `torch_npu.profiler` 可生成 CANN trace，输出位于项目 `output/npu_affinity/profiler_forward/<run_dir>` 下。 |
| 后四卡逐卡可见性 | 设备 guard 生效；本次功能验证主路径使用逻辑 `npu:0`，即物理 NPU 4。 |
| `nn.DataParallel` | 当前 NPU 环境下失败，甚至小 Linear 模型也在参数复制阶段报错，不建议使用 PyTorch `DataParallel`。 |
| HCCL/DDP | 两进程 HCCL `all_reduce` 在 `HCCLUtils.cpp:140` 报 error code `4`，因此当前不能声明多进程 NPU 训练同步可用。 |
| TorchAir graph mode | `torchair.get_npu_backend()` 可 import，但 tiny model 和 MANIQA 的 `torch.compile` 均因 GE/TBE 初始化失败，当前容器只应按 eager NPU 路线复现。 |

复现过程中观察到两类重要 warning。第一，Swin/timm 路径触发 `torch.meshgrid` 未来需要显式 `indexing` 参数的 warning，不影响当前结果。第二，NPU autograd 触发 `Cannot create tensor with interal format while allow_internel_format=False` warning，当前未导致功能失败，但说明部分 tensor 创建没有走 Ascend 内部优化格式。

## 部署结论

MANIQA 已在 Ascend NPU 容器内完成部署，后四张 Ascend950PR 的设备隔离、单图 checkpoint 预测、README 示例图复现、三类训练 Dataset、PIPAL22 推理入口、训练/验证 epoch、NPU autograd、随机 batch 推理和 profiler 均已验证。对当前目标来说，单 NPU eager 路线已经具备可复现基础。

当前不建议直接宣称全量训练和多卡复现已经完成。主要限制是：真实 Koniq10k/KADID-10K/PIPAL22 数据目录在服务器上缺失，无法验证官方 SRCC/PLCC；HCCL collective 当前失败，阻塞 DDP；TorchAir 图模式初始化失败，不能作为本轮性能路线；数据预处理仍主要在 CPU OpenCV/NumPy 上完成，真实训练吞吐需要另行测量。

## 复现命令

进入已配置好的 Ascend NPU 容器环境后：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source <venv>/bin/activate
cd <MANIQA_WORKDIR>
```

限制只使用后四张卡：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
```

单图预测：

```bash
python predict_one_image.py
```

PIPAL22 推理入口需要先准备 `config.test_dis_path` 和 `config.model_path` 指向真实数据和模型，再运行：

```bash
python inference.py
```

## 资料来源

- Ascend 950 NPU 架构白皮书
- Ascend Extension for PyTorch: https://github.com/Ascend/pytorch
- torch-npu package metadata: https://pypi.org/project/torch-npu/
- TorchAir guide: https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00003.html
- MANIQA checkpoint release: https://github.com/IIGROUP/MANIQA/releases/tag/Koniq10k
