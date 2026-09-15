# ModelArts：Notebook 开发 → 生产训练作业

用于把已适配的昇腾项目从交互开发迁移到平台管理的训练作业。配合[任务生命周期](job-lifecycle.md)和[产物保存与 OBS 选择](artifact-backup.md)使用；本参考不是作业提交器，也不扩大传输或算力使用授权。

## 证据边界

资料核对日期：2026-09-15。下面的平台能力来自链接中的官方文档及迁移实践；工程建议是工具包建议，不代表本项目已完成生产部署。找回的项目记录能证明 SSH 调试模式训练及多节点准备，**尚不能证明生产作业的无人值守启动、产物持久化和新作业恢复均已通过**。

先确认区域、控制台版本、资源池及启动模式。旧版“自定义算法/自定义镜像”和新版“自定义作业”的表单不同；不要照搬历史字段限制、路径、GPU 版本或 DLS_* 变量。

## 推荐流程

1. **Notebook 开发。** 适配真实模型，记录运行时组合，把 notebook 的隐含状态收敛成训练入口和配置。复用已有真实更新/保存证据，将手工依赖补丁变成可重建的交付物。
2. **准备全新训练作业。** 固定源码、镜像、环境初始化、输入、输出映射和训练参数。Notebook 文件、挂载及已安装依赖不会自动跟随新作业。验证非交互入口，显式初始化环境，不依赖终端里已激活的 conda。
3. **在授权范围内跑有界生产试运行。** 使用实际提交入口、真实模型/数据及相关目标拓扑，确认有效更新、checkpoint 交付、launcher 退出和平台状态。声称生产恢复已验证之前，还要有新作业恢复证据。复用未变的底层测试，这次只补部署差异。
4. **正式提交。** 沿用通过试运行的包和用户指定训练预算，通过平台状态及持久日志监控。训练、保存及已授权的评估不依赖 Notebook、电脑或智能体持续在线。

该路线与华为[本地 PyTorch 迁移实践](https://support.huaweicloud.com/usermanual-standard-modelarts/develop-modelarts-0148.html)一致。其中旧 CUDA/镜像示例不能作为 NPU 版本推荐。

## 提交前形成具体作业包

用户已经决定的参数直接沿用，不重复询问。

| 项目 | 需要落实的内容 |
|---|---|
| 执行 | 区域/项目/工作空间、资源池/规格、节点/每节点卡数、启动模式、重试/时间预算 |
| 环境 | SWR 镜像及可得的不可变 digest、架构、Python/torch/torch_npu/CANN、已测初始化和依赖补丁 |
| 源码 | commit 和随包改动、代码来源/实际落点、工作目录、启动命令、解析后的配置 |
| 输入 | 模型/processor 版本、数据划分、特征契约；持久源到本地路径映射；每个读取节点的准备完成证据 |
| 输出 | 普通/EMA/恢复 checkpoint、指标/评估路径；持久目标、用户备份选择、保存/同步间隔、校验和失败策略 |
| 观测 | 按选择设置永久日志位置、run/job ID、结构化进度、外部实验看板断网时行为 |
| 结束 | 目标 optimizer 更新数、评估权重及可选评估顺序、最终交付、真实退出状态 |

训练代码必须写入表单映射的本地输出目录。永久日志与模型产物是两项配置；不能认为 /cache 中的所有内容都被平台收集。核对当前[自定义作业表单](https://support.huaweicloud.com/develop-modelarts/develop-modelarts-0006.html)。

优先准备依赖齐全、可重建的训练兼容镜像。保存了 Notebook 镜像，不等于挂载代码/数据也在镜像里，更不等于满足训练入口约束。按[训练镜像要求](https://support.huaweicloud.com/usermanual-standard-modelarts/docker-modelarts_0017.html)和[当前自定义镜像配置](https://support.huaweicloud.com/docker-modelarts/docker-modelarts_0118.html)核对。大数据/权重与经常变化的代码分开；长训练不能依赖公网临时下载、交互登录或电脑代理。确需启动时准备资产，按已授权策略实现有界、可恢复的准备阶段。

## 多卡进程只能由一层负责启动

- **平台负责拉起进程：** 部分预置引擎逐卡运行启动文件，或通过 MA_RUN_METHOD 使用平台 TorchRun。提供该引擎需要的 worker 入口，不能让每个 worker 再执行完整 torchrun/Accelerate launcher。
- **自定义镜像自行拉起：** 确认平台在每节点执行一次命令后，由项目 launcher 在该节点启动本地进程。不能假设所有模式都如此，也不要从 worker 0 用 SSH 拉起其他节点。
- 先读所选引擎的[启动语义](https://support.huaweicloud.com/intl/zh-cn/usermanual-standard-modelarts/develop-modelarts-1415.html)，不能仅根据某个 rank 变量判断。

自定义适配层应核对 MA_NUM_HOSTS、VC_TASK_INDEX、VC_WORKER_HOSTS 与分配资源，再映射节点数、节点 rank、master。MA_NUM_GPUS 虽名字含 GPU，文档含义是加速卡数量；仍须核对本作业 NPU 实际值及可见卡。普通一进程一卡 DDP 的 world size = 节点数 × 每节点进程数。HCCL 接口来自实际分配，不能继承旧作业 IP、端口、rank table 或 SSH 设置。

[环境变量说明](https://support.huaweicloud.com/develop-modelarts/develop-modelarts-0104.html)给出了 MA_JOB_DIR 及分布式字段。不要自创以 MA_ 开头的私有变量，文档规定的平台参数除外。日志只输出非敏感拓扑/版本白名单，不转储全量环境。

## 前台运行，完整交付后结束

OBS 准备、MoXing 示例、checkpoint 提交及容器退出时备份失败的处理，见 [OBS 与 MoXing](obs-moxing.md)。同一资产/目标只选一个传输负责人。平台退出后收集输出需要外部核验；应用自行上传则在前台监督器报告持久交付完成前结束上传。

生产入口以前台方式运行。不能把后台式 run_recorded.py start、nohup、tmux 或 sleep infinity 作为生产训练命令，否则可能只有提交成功而没有平台托管的真实训练。包装器要等待所属 worker、转发停止信号、保留训练失败状态，并等待选定的最终交付。若交付由 trainer 内部或已验证的平台机制完成，可用 exec；若后面还有必要上传/评估，则不能让 exec 跳过这些步骤。

以下是工具包工程建议：
- 数据/输出路径参数化；在昂贵的模型初始化前检查缺失输入或不兼容特征。
- 所有 checkpoint 分片写完才发布完成标记，共享元数据由指定进程发布；不能由 rank 0 提前宣布分片模型完整。
- 周期性保存可恢复状态。只在结束时上传或依赖终止 trap，无法应对强制杀进程和节点丢失。对象存储挂载不应直接假设具有本地 POSIX rename/fsync 语义。
- 恢复状态包含模型、优化器、调度器/scaler、已使用的 EMA、更新计数、RNG 及框架支持的数据/采样进度；明确无法精确恢复的部分。
- 区分训练失败和上传失败，按用户选择的备份失败策略执行；不可用无条件成功标记掩盖交付失败。
- 已授权训练后评估时，预先固定数据集顺序、checkpoint、依赖及输出。可同作业分阶段执行，也可通过持久交接启动后续作业；只写计划不等于已经调度。

备份范围及授权仍由原有备份参考管理。日常校验优先使用资产身份、数量/大小及已有可靠回执/校验和，不每轮全量哈希或回读大体积 OBS 对象；遇到不匹配、证据缺失或用户要求更强完整性再升级。

## 监控、失败与恢复

分别记录排队、初始化/准备数据、有效更新、checkpoint 已交付、训练结束、评估结束。ETA 区分排队/拉镜像/准备数据和稳定训练耗时。通常只读小型状态及增量日志。

官方支持[在专属资源池中用 Cloud Shell 进入运行中的生产作业](https://support.huaweicloud.com/develop-modelarts/develop-modelarts-0119.html)，还受 modelarts:trainJob:exec 等权限限制。这是有条件的排障能力，流程不能依赖它。官方 sleep/保留现场示例用于调试；不要照搬成掩盖原始退出码、无限占用资源的生产入口。

失败时保留首个因果堆栈、job ID/配置及最后持久 checkpoint。提交返回不确定时先查询状态，避免重复提交。重试有上限，每次使用独立 attempt 输出和明确恢复源；不能静默从零开始或修改 batch/精度。已结束作业不假定可原地重启，按当前[复制/停止管理说明](https://support.huaweicloud.com/intl/en-us/usermanual-standard-modelarts/develop-modelarts-0017.html)处理。核对平台自动恢复和应用恢复策略，避免互相叠加造成重复训练。

## 交付时说明实际验证级别

分别表述：Notebook 训练已验证、生产作业包已准备、生产试运行通过、生产恢复已验证、完整训练已完成。生产测试证据应含 job ID、模式、源码/镜像/配置、拓扑、实际更新、checkpoint 目标和校验、最终平台/进程状态。没有这些记录不能把本参考称为生产实测经验。

本参考不提供假定跨版本通用的提交配置。按当前控制台/API 或已安装 CLI 生成实际作业文件。[ma-cli 作业说明](https://support.huaweicloud.com/intl/zh-cn/usermanual-standard-modelarts/devtool-modelarts_0320.html)提供提交/查询/日志能力，使用前检查已安装版本的帮助。更新 skill 本身不启动试运行。
