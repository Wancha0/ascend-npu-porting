---
name: ascend-npu-porting
description: 将 PyTorch 模型适配到昇腾 NPU，供 GPT 和 GLM 编码智能体使用。适用于 torch_npu/CANN、真实训练与推理、多卡 HCCL、性能调优，以及 Notebook 到 ModelArts 正式训练和已授权的 OBS 资产交付。
---

# 昇腾 NPU 适配

将用户指定的真实模型在目标 NPU 上跑通，交付可重建的代码、启动配置和实测记录。GPT 与 GLM 使用同一套流程；文件、Shell、SSH 和云平台访问能力由宿主提供。只有文本接口时交付方案和补丁，不声称已执行。

## 执行流程

1. **接续状态。** 先看源码、台账和最新日志，核对活跃任务，避免重复启动。用户最新要求和已有授权优先。
2. **建立契约。** 记录源码版本、实际入口、运行时组合、模型及可训练模块、数据/特征身份、精度、节点/卡数、global batch、学习率、更新步数、保存与评估要求。
3. **完成适配。** 保留 CPU/CUDA 路径，增加必要的 NPU 分支；环境与依赖补丁必须可重建。
4. **验证真实任务。** 从当前证据缺口选择测试，直到本次要求的前向、有效更新、保存/恢复或评估完成。条件未变的通过记录直接复用，不默认全量哈希或重复逐卡测试。
5. **正式训练。** 推荐 Notebook 开发 → 固定代码/镜像/配置 → 有界生产试运行 → 正式训练与周期备份 → 最终交付确认 → 已授权评估。流程不能依赖电脑、SSH 或智能体持续在线。
6. **记录结果。** 分开报告训练、备份、恢复和效果；保存实际退出状态、指标、产物位置及未测项。

## 必须保持的约束

- 不静默修改模型、模态、冻结范围、数据划分、精度或有效 batch；参数调整按用户契约执行。
- 优先保留平台 torch/torch_npu/CANN 组合；按真实设备类型路由，在框架判断设备前加载 torch_npu。
- 有效更新需真实数据、loss、反向、同步及参数变化；调用 optimizer.step 或生成文件不足以证明成功。
- 首个因果错误出现后先定位，再做最小修复；重试使用独立 attempt，保留失败记录。
- 新增长训练或大规模提取前，按备份参考列出具体资产、目标和频率，让用户选择 OBS 范围；已有适用选择直接沿用，不将本地保存称为备份。
- 同一资产/目标只有一个传输负责人。全部 checkpoint 分片交付后才发布完成清单；生产作业启动前确定备份失败及退出策略。
- 只操作本次授权资源；不把 skill 的示例或历史参数当成新作业默认值。

## 按需读取

| 当前工作 | 参考 |
|---|---|
| 陌生模型、追踪实际入口 | [模型契约](references/model-independent.md) |
| 尽快打通真实训练、复用资产 | [快速训练](references/fast-training.md) |
| 设备、算子、依赖、offload、多卡 | [兼容与修复](references/compatibility.md) |
| 真实更新、保存恢复、结果验收 | [模型验收](references/validation.md) |
| SSH、进程、退出码、停止与重试 | [任务生命周期](references/job-lifecycle.md) |
| Notebook 开发后提交生产作业 | [ModelArts 正式训练](references/modelarts-production.md) |
| 保存范围、OBS 选择及校验强度 | [产物保存](references/artifact-backup.md) |
| MoXing 拉取/上传、缓存和完整快照交付 | [OBS 与 MoXing](references/obs-moxing.md) |
| GPT/GLM 宿主、断线接续、离线交接 | [智能体交接](references/agent-handoff.md) |
| 脚本及官方资料 | [工具说明](references/tools.md)、[官方索引](references/official-links.md) |

## 交付与边界

通常交付源码补丁、固定环境与命令、中文 NPU_PORTING.md、简短 ledger、指标和资产回执。正文不包含权重、数据、环境目录或凭证。

区分“已准备”“真实更新已验证”“保存恢复已验证”“生产试运行通过”和“效果已验证”。Notebook 或本地模拟测试不能证明生产训练已通过；缺失退出码记为未知。

整包首次复制到新宿主时可运行一次 `python3 scripts/self_check.py`。它仅检查工具包，不证明 NPU 或模型可用；不要每轮重复执行。
