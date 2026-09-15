# 复用 MiniMax-H3 的历史 NPU 适配经验

适用于用户允许参考既有 MiniMax-H3 工作的模型族适配。以下经验来自历史
Ascend 910B1、CANN 8.2.RC1、PyTorch/torch_npu 2.6.0 实测记录，是排查线索和
测试设计，不是当前 ActionWM 的已通过结论。当前源码、算子、形状和环境必须重验。
实验应将本页加入 manifest，记录允许读取的历史资料；不能声称未接触该模型族经验。
其他模型不必为了此案例引入 MiniMax/SGLang 依赖。

## 先识别可迁移的部分

历史 MiniMax-H3 主推理路径使用固定 SGLang、完整 FL2VA 权重和 8 卡 TP/SP。
当前模型可能使用 DiffSynth、ControlNet、不同音频布局及单卡训练。优先复用
算子选择、设备分支、数据边界和验证方法；逐项说明当前源码是否出现同一个问题。

| 历史观察 | 当前检查 | 不能直接外推的结论 |
|---|---|---|
| 兼容层下 NPU tensor 可出现 `.is_cuda == True` | CUDA JIT/Triton/C++ 分支检查真实 `tensor.device.type`，保留 CPU/CUDA 路径 | CUDA_HOME 错误不说明必须安装 CUDA |
| 推理 TND 拒绝某些 head 数，BNSD 分段回退通过 | 固定具体 API、heads、head_dim、dtype、布局与段边界再微测 | 不能概括为“NPU 不支持 TND” |
| 训练 attention 通过 7-head 反向 | 验证输出和 dq/dk/dv，覆盖当前真实形状 | 推理成功不证明可反传；历史局部 shape 不证明单卡完整 shape 成功 |
| VAE 也会误选 CUDA activation 快路径 | 首帧编码和完整视频解码都执行，检查低层设备判断 | DiT 成功不证明 VAE 成功 |
| offload 降低权重常驻，但解码仍有显存峰值 | 分开测编码、DiT/ControlNet 前后向、optimizer 和解码 | 减少去噪步数或开启 offload 不保证消除工作集 OOM |
| 多卡推理曾在指定 TP/SP 下通过 | 单卡重新预算权重、梯度、优化器和激活，实测 CPU offload | 8 卡结果不证明单卡训练可行，也不能替换单卡验收 |

## Attention：分清推理与训练

历史推理的 `npu_fused_infer_attention_score` 在当时 CANN 8.2 组合下不接受
7/14 heads、D=128 的 TND 输入。按独立真实 segment 执行 BNSD 的回退完成过
9 个 BF16 对照 case，覆盖非连续 QKV、段间隔离及 padding。默认推理 TND 对照
使用另一个可用组合（8 heads、D=192）。这是特定 API/版本/形状的结果。

历史训练使用 **`torch_npu.npu_fusion_attention`**，完成了 `[2368,7,128]`
BF16 TND 的前向和 dq/dk/dv 反向，并与 CPU FP32 参考比较。packed bounds 为
`[0,2327,2368]`，API 接收的累计结束位置为 `[2327,2368]`；改变 padding 段 V 后
真实段输出变化为零。不要把这些历史常量写入新模型。

当前应分别验证真实 heads/head_dim/序列长度、dtype、scale、causal/mask/dropout；
明确 API 需要累计结束位置还是其他长度格式；覆盖实际 QKV view/stride；比较
前向和梯度误差；干预另一段或 padding 检查段间隔离，最后验证完整训练图。

冻结 backbone 参数不等于其前向可以放进 `no_grad()`：若训练模块的信号穿过
冻结 backbone 才到 loss，仍需输入梯度。attention、激活检查点和 offload 都必须
保留这条图。历史多卡训练还遇过推理 collective 缺少 autograd；只有当前任务确实
走多卡训练时才检查相应通信反向，单卡任务不用照搬 TP 修复。

## 设备、加载与内存

在设备检测框架决策前初始化 torch_npu；这不等于尽早启用所有全局 CUDA 兼容
monkeypatch。历史 SGLang 的平台识别与兼容层顺序曾相互影响，当前框架应在冷
进程测试自己的导入顺序。

低层引入上层 platform 包可能循环导入。限制 CUDA 专用路径时，使用已有无环
helper 或真实 tensor device 判断，并保留原 dtype/shape/stride 条件。fallback
需要数值验证，不能靠删除编译参数或伪造 CUDA_HOME 启动未经验证的内核。

历史还有“本地目录名未匹配模型注册，回退到错误通用 pipeline”的加载失败。
检查实际模型类型、配置、分片列表、参数量、严格加载结果和设备驻留位置。
旧 SGLang 参数名不适用于 DiffSynth，不能直接复制启动命令。单卡 offload 应
分开记录参数常驻、瞬时工作集、保存激活、梯度和优化器峰值，不以权重大小代替预算。

## 环境、资产与媒体

原始资产可按来源、revision、文件集合和哈希复用。旧环境、site-packages 修改、
模型专属缓存和适配源码需单独声明，不能当成原始依赖。使用新环境覆盖层，固定
torch/torch_npu/CANN，其他依赖按当前源码推导，不盲目复制旧整套环境。

发行包名不等于 import 名，例如 Pillow/PIL、opencv-python/cv2。分别记录分发包
是否存在、版本、import 结果和异常；传递依赖 ImportError 不等于未安装。历史
镜像曾有 simplejson 连带破坏其他库导入，应定位当前导入来源，做有边界的依赖
修复并重验，不把旧进程的 JSON monkeypatch 复制成全局修复。

缓存绑定当前数据、预处理、prompt、音频/视觉布局、权重及形状。旧音频布局不能
与当前 video-only 混用。RGB 首帧和动作控制视频应来自同一窗口，分别检查几何
标定、时间采样和分辨率。

历史系统 FFmpeg 曾缺少 libx264，PyAV 自动时间戳也曾产生错误帧率。当前输出
应验证编码器、帧数、尺寸、fps/time_base、有限像素、首帧条件和可解码性。CPU
编码失败时保留已生成帧，从该阶段修复，避免重复去噪。旧 SGLang 的 sigma 点数
与实际 denoiser evaluation 数不同；当前框架以自己的 scheduler 和运行轨迹为准。

## 引用记录与验收

台账建立对应表：`历史经验 → 当前代码位置/条件 → 采用或不适用理由 → 修改
→ 实测证据`。旧报告的停止点、SSH 端口、IP、占用和审批是当时上下文，以当前
契约及实时探测为准；旧示例 broad ps/env 诊断改用本工具包的安全字段。

历史完整推理、真实反向与当前 optimizer 更新、strict reload、真正 resume 是
不同门禁。保存权重不等于保存优化器/调度器/步数；恢复验收要求新进程恢复完整
训练状态后再完成真实更新。

本页不含 ActionWM 成品补丁。用户若授权复用完整补丁，应记录范围、冻结并回归，
区分“从经验完成适配”和“部署已有适配”。历史报告/补丁的路径及哈希保存在
受授权的实验输入清单，不把私有源码和服务器资料复制进公开工具包。
