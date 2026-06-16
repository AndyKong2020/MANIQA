# MANIQA — 图像质量评估 NPU 部署及亲和性报告

| 项 | 内容 |
|---|---|
| 任务编号 | MANIQA |
| 任务用途 | 无参考图像质量评估(NR-IQA),对单张失真图像预测质量分数 |
| 仓库 | https://github.com/IIGROUP/MANIQA |
| 版本 / commit | f573d8624012 |
| 报告人 | - |
| 日期 | 2026-06-16 |
| 硬件 | Ascend 950PR ×8 / CANN 9.0.0 |
| 软件 | torch 2.10.0+cpu / torch_npu 2.10.0 / torchvision 0.25.0+cpu / Python 3.11.6 |

---

## 1. 技术栈梳理
- 主语言:Python。
- ML 框架:PyTorch eager,通过 torch_npu 接入 Ascend NPU。
- CUDA 依赖:原仓库训练、预测、推理脚本硬编码 `.cuda()` 与 `CUDA_VISIBLE_DEVICES`;本轮已改为 `get_device()` / `.to(device)`。无必须 CUDA 自定义核。
- 自定义核(.cu / C++ 扩展):无。仓库内主要是 PyTorch、torchvision、OpenCV、NumPy、timm vendored 代码。
- 第三方库:torchvision、opencv-python-headless、scipy、pandas、einops、tensorboardX、tensorboard、tqdm、torchsummary。
- 模型权重 / 来源:官方 Koniq10k checkpoint,文件 `ckpt_koniq10k.pt`,大小 `543335435` bytes,SHA256 `a207f8ab57322e6be38ff5c8d019301dc032b454bef21c3c9f9dbf7974eebff6`。
- 模型结构:ViT patch8 backbone + TAB 通道注意力 + Scale Swin Transformer Block + patch 加权质量分头。

## 2. 部署步骤
- [x] 依赖安装:创建 Python 3.11 虚拟环境,安装 torch/torch_npu/torchvision 及项目依赖;补 `tensorboard`,否则 `train_maniqa.py` import `SummaryWriter` 失败。
- [x] 编译 / 构建:无编译步骤;纯 Python/PyTorch 运行。
- [x] 权重获取:下载并放置官方 Koniq10k checkpoint 为 `ckpt_koniq10k.pt`。
- [x] NPU 适配改动(device、torch_npu、禁用 CUDA 核等):新增 `utils/accelerator.py`;训练、预测、推理入口移除硬编码 CUDA;checkpoint 采用 `map_location='cpu'` 后再迁移设备;`MANIQA` 支持 `vit_pretrained=False`;修复 PIPAL22 数据集导入路径;`sort_file` 改为原地排序指定输出文件。
- 命令:
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source <venv>/bin/activate
cd <MANIQA_WORKDIR>
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
python predict_one_image.py
```

## 3. 验证用例
- 输入数据:仓库 README 示例图 `image/kunkun.png`、`bird.jpg`、`dog.jpg`、`ball.jpg`、`people.jpg`;合成 Koniq/KADID/PIPAL/PIPAL22 小样例用于 Dataset 与训练/推理入口 smoke。
- 运行命令:
```bash
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 python predict_one_image.py
```
- 期望输出:单图质量分可在 NPU 上输出,与 README 示例数值接近。
- 实测输出:`Image kunkun.png score: tensor([0.3407], device='npu:0')`。
- 与 CPU/GPU 基准对比(误差/一致性):未做 CPU/GPU 逐算子数值基准;README 五图 NPU 20-crop 结果为 `kunkun 0.340658`、`bird 0.261935`、`dog 0.308199`、`ball 0.372101`、`people 0.358600`,与 README 示例 `0.3398/0.2612/0.3078/0.3716/0.3581` 接近。
- 训练 smoke:合成 Koniq DataLoader 跑通 `train_epoch` 与 `eval_epoch`,覆盖 forward、MSE loss、backward、Adam、CosineAnnealingLR、SRCC/PLCC 统计;样例 `train_loss=0.4549516`,`eval_loss=0.2429362`。
- PIPAL22 smoke:`PIPAL22` Dataset 可读目录图片;入口需 five-point crop 到 `224x224`;输出文件可原地排序,未再生成根目录 `output.txt`。

## 4. NPU 亲和性

| 指标 | 数值 |
|---|---|
| 能否在 NPU 跑通 | 是。单图 checkpoint 预测、随机 forward、训练/验证 smoke、PIPAL22 推理入口均跑通 |
| NPU 利用率 (npu-smi) | 未做稳定采样;短 forward 结束快,报告不填写伪利用率 |
| HBM 占用 | 共享环境当前存在其他进程占用,未做稳定峰值采样;MANIQA 峰值需 profiler 或长任务采样,当前标待测 |
| 关键算子是否回退 CPU | 主干未观察到硬 CPU 回退;CPU 路径主要是 OpenCV/NumPy 预处理与 SRCC/PLCC 统计 |
| 性能(吞吐/时延) | 随机权重 forward:batch1 `13.78 ms/image`,batch4 `3.93 ms/image`,batch8 `2.12 ms/image`;checkpoint 单图 20-crop 输出正常 |

- 算子回退清单:未发现阻断主干的 NPU 算子缺失;`DataParallel`、HCCL、TorchAir 属运行栈/图模式阻塞,不是 MANIQA 主干 eager 算子缺失。
- profiler 摘要:一次 NPU forward 的热点为 `Addmm/MatMulV3`、`BatchMatMulV3`、`Conv2DV2`、`Softmax`、`LayerNorm`、`GELU`、`Transpose`。top kernel 包括 `aclnnAddmm` 90 次约 `2806us`,`BatchMatMul` 40 次约 `818us`,`Conv2d` 7 次约 `690us`,`MatMul` 14 次约 `637us`,`Softmax` 24 次约 `241us`。

**各计算单元逐条判定**:

| 单元 | 压力 | 判定 | 证据 |
|---|---|---|---|
| 算力(Cube,矩阵卷积) | 高。ViT dense、TAB `q@k.T/attn@v`、Swin projection、Conv2d 是主计算 | 亲和 | profiler 中 Addmm/MatMul/BatchMatMul/Conv2d 为主要耗时;模型主干为规则 dense tensor 计算 |
| 向量(Vector,归一激活) | 中到高。Softmax、LayerNorm、GELU、Add/Mul、score reduce | 可用,需继续看 Vector 时间 | Softmax/LayerNorm/GELU 均在 NPU 跑通;缺 `aiv_vec_time` 和 GM bytes,不写成最终瓶颈 |
| 搬运(MTE/FixPipe) | 中。rearrange/transpose/window layout 与 H2D 输入搬运 | 有优化空间 | profiler 观察到 Transpose kernel;OpenCV/NumPy 预处理在 CPU,layout 折叠/NDDMA 未验证 |
| 通信(communication) | 单卡无通信;多卡阻塞 | 单卡可用,多卡不可声明 | 后四卡隔离有效;`nn.DataParallel` 小 Linear 失败,HCCL `all_reduce` 报 `HCCLUtils.cpp:140` error code `4` |
| 调度(host/head) | 高。PyTorch eager、小 kernel、20-crop 串行、score head Python loop | 当前主要工程风险 | batch 越大 `ms/image` 越低,说明 launch/head 被摊薄;score head 仍逐样本 `torch.cat` |

**roofline 初判**:
- 口径:Ascend950PR,架构 3510,当前实测 FP32 eager;FP32 Cube/Vector 实际平衡点和 GM bytes 未采集,标 `待测`。
- FP16 Cube 平衡点参考约 `270 FLOP/Byte`;MANIQA 的 ViT/TAB 大矩阵段具备百级 FLOP/Byte 的复用,主干比图像预处理和 score head 更贴 NPU。
- Swin window attention 的 window size 为 4,单 window 16 token,小矩阵 tile 利用率与 kernel head 可能主导,不能只按总 FLOPs 判断。
- 端到端可写为 `T_image ≈ T_preprocess_CPU + T_H2D + T_head_eager + max(T_GM_layout, T_Cube主干 + T_Vector后处理)`;当前 batch 变大后单图耗时下降,说明 `T_head_eager` 不可忽略。

## 5. 阻塞项

| 阻塞点 | 原因 | 是否硬阻塞 | CANN/AscendC 替代方案 | 兜底 |
|---|---|---|---|---|
| 真实数据集指标复现 | 当前环境未提供 README 对应 Koniq/KADID/PIPAL 真实数据目录 | 是,阻塞官方 SRCC/PLCC 复现 | 无;属于数据可用性问题 | 仅报告 Dataset/训练循环 smoke,不声称官方指标 |
| NPU 多卡训练 | `DataParallel` 参数复制失败;HCCL `all_reduce` 失败 | 是,阻塞 DDP 结论 | 先修 HCCL 最小 collective;通过后再测 DDP | 单 NPU eager 复现 |
| TorchAir 图模式 | tiny model 与 MANIQA `torch.compile` 均在 GE/TBE 初始化阶段失败 | 否,不阻塞 eager 功能 | 先修 tiny compile;再评估静态 shape 图模式 | PyTorch eager |
| score head Python loop | 逐样本 `fc_score/fc_weight` + `torch.cat`,放大小 kernel/head 开销 | 否,功能已跑通 | 数学等价向量化:`(f*w).sum(dim=(1,2))/w.sum(dim=(1,2))` | 保留原实现,牺牲性能 |
| 20-crop 串行预测 | README 单图预测逐 crop forward,head 开销高 | 否 | 20 个 crop stack 成 batch 后一次或分批 forward | 保留串行,结果稳定但吞吐差 |
| PIPAL22 checkpoint 形态 | `predict_one_image.py` 加载 state dict,`inference.py` 当前按完整模型对象加载 | 否,但影响开箱推理 | 增加 state dict / 完整模型双分支加载 | 明确要求用户提供完整模型对象 |
| 固定输入尺寸 | ViT patch embed 要求 `224x224`;任意尺寸原图不能直接进模型 | 否 | 入口显式 resize/crop;小图拒绝或 padding | five-point crop |
| NumPy 2.x 兼容 | vendored timm 仍有 `np.bool`;默认路径未触发 | 否 | 固定 `numpy<2` 或修旧别名 | 避免启用相关 mixup/augment 路径 |

## 6. 结论
- 运行方案(NPU / NPU+CPU / CPU):NPU+CPU。MANIQA 主干和训练/预测核心 tensor 路径跑在 NPU;图像读取/预处理、相关系数统计等仍在 CPU。
- 单 NPU eager 可复现:checkpoint 单图预测、README 五图、随机 forward、训练/验证 smoke、PIPAL22 推理入口均通过。
- NPU 亲和判断:主干 ViT/TAB/Swin dense 计算亲和 NPU;Vector 路径可用;MTE/layout 和 host/head 是主要优化面。
- 待办 / 风险:真实数据集指标、HCCL 多卡、TorchAir 图模式、score head 向量化、20-crop batch 化、profiler GM bytes/tile 利用率均待补。
- 不建议表述:不能宣称全量训练、多卡 DDP、TorchAir 图模式或官方 SRCC/PLCC 已完成复现。
