# 官方资料索引

链接用于解决当前具体未知项，不要求逐项阅读。2026-09-08 核查了下列入口。`master`、默认版和 stable 页面会变化；实际决策必须记录目标安装版本、访问日期、适用章节和必要的 commit/tag。文档描述的 API 语义不等于该 API 已在目标 NPU 上通过。

## 昇腾版本、算子与通信

| 官方入口 | 何时查、要确认什么 |
|---|---|
| [torch_npu 项目](https://github.com/Ascend/pytorch) | 安装/设备注册/功能支持；按目标 release 查源码，沿官方链接进入当前维护仓库 |
| [中文版本配套矩阵](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md) | 更换或诊断 torch_npu、PyTorch、CANN、Python、固件/驱动组合；不能直接安装页面最新推荐来覆盖平台环境 |
| [昇腾 PyTorch API 文档](https://ascend.github.io/docs/sources/pytorch/api_doc.html) | 查 NPU 设备、算子、dtype、布局等接口；同时确认该页和目标 torch_npu 版本是否一致 |
| [昇腾官方文档中心](https://www.hiascend.com/document) | 选择当前硬件/CANN 版本后查算子约束、HCCL 通信、环境变量、Profiler、错误码和自定义算子开发；不要照抄 CUDA/NCCL 参数代替 HCCL |

文档中心是多产品入口，不能把其首页作为某个具体算子“支持”的唯一引用。做实际决定时继续进入对应版本章节，并把最终页面 URL 写进该项目的 `NPU_PORTING.md`。

## PyTorch 算法语义与训练状态

以下 2.6 页面保留为可定位的历史版本入口；新项目使用其他版本时切换到匹配版本。

| 官方入口 | 何时查、要确认什么 |
|---|---|
| [torch.distributed](https://docs.pytorch.org/docs/2.6/distributed.html) | 进程组、rank、collective 的顺序和语义；NPU 后端是否支持还要看 torch_npu/HCCL |
| [FSDP](https://docs.pytorch.org/docs/2.6/fsdp.html) | 分片、参数包装、CPU offload、state dict 等约束；不是默认要求所有模型采用 FSDP |
| [自动混合精度 AMP](https://docs.pytorch.org/docs/2.6/amp.html) | autocast、梯度缩放、非有限梯度和跳步语义；具体 NPU dtype 支持另核实 |
| [scaled_dot_product_attention](https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html) | mask、dropout、causal、shape 和 attention fallback 的语义；不能只验证输出尺寸 |

## 使用了对应框架时再查

| 官方入口 | 何时查、要确认什么 |
|---|---|
| [Accelerate 梯度累积](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation) | 项目使用 Accelerate 时查 `accumulate`、同步边界与 optimizer 行为；选匹配安装版本 |
| [Accelerate v1.12.0 DeepSpeed 包装源码](https://github.com/huggingface/accelerate/blob/v1.12.0/src/accelerate/utils/deepspeed.py) | 本次历史版本中确认 `DeepSpeedEngineWrapper.backward`、engine 更新和 optimizer 包装调用时点；其他版本重新核实 |
| [DeepSpeed checkpoint 文档](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html) | 确认各 rank 参与保存、分片状态及恢复限制；latest 会变化，按已安装版本查对应源码 |
| [GLM 官方 skills](https://github.com/zai-org/GLM-skills) | 了解目录式 skill 的组织；宿主实际文件/命令能力仍须单独验证 |

模型自身官方仓库、实际依赖的 Transformers/Diffusers/DeepSpeed 等文档、任务数据说明和 checkpoint 说明，由本次项目契约动态补充。不要预先把某个框架设为所有模型的前提。

## 引用与离线使用

每个影响实现的判断记录：问题、目标版本、官方 URL/章节、对应源码位置、采用的方案和运行证据。网络不可达时可读已安装包源码或用户提供的官方文档副本，并记录版本/哈希；未核实的能力标为未知。社区经验只作为诊断线索，不替代官方约束或本机实测。
