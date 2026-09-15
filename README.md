# Ascend NPU Porting

## GLM 中文使用入口

先读 **[中文 skill 的 README](skills/ascend-npu-porting-zh/README.md)**，再按引导读取 [SKILL.md](skills/ascend-npu-porting-zh/SKILL.md)。README 包含可直接交给 GLM 的启动提示、输入清单、快速训练模式和断线续接方法；整目录可复制到任何具备文件、Shell 和目标机访问能力的编码宿主。

本分支合并了此前本地技能改进，并新增通用中文包。中文包补充了已有资产优先、按用户要求跳过资产 hash、框架下的真实参数更新测量，以及 FastWAM 实测经验。只有文档和通用工具，没有模型权重、数据、环境或 FastWAM 适配代码。

获取本次发布分支：

```bash
git clone --branch codex/glm-zh-training-first --single-branch https://github.com/Wancha0/ascend-npu-porting.git
cd ascend-npu-porting
```

给 GLM 的最短提示：

```text
请读取当前仓库 skills/ascend-npu-porting-zh/README.md，并依照其中的
SKILL.md 和启动模板，为我指定的模型执行昇腾 NPU 适配。
```

选择中文包时只需复制 `skills/ascend-npu-porting-zh/`，它不依赖根目录的英文包；不要把两套说明当成需要重复执行的检查清单。

## Original English toolkit

An agent-independent, code-first workflow for adapting and validating PyTorch
projects on Huawei Ascend NPU. Codex is not required: GLM-hosted coding agents,
other coding agents, and human operators can use the Markdown instructions and
Python standard-library helpers.

- Start on a new computer with [PORTABLE_AGENT_GUIDE.md](PORTABLE_AGENT_GUIDE.md).
- Give an agent [SKILL.md](SKILL.md) as the authoritative decision guide.
- For GLM, also use [references/glm-agent.md](references/glm-agent.md) to verify
  the host's actual capabilities and persistence behavior.
- Verify a copied toolkit with `python3 scripts/self_check.py` before use.
- For remote Bash, use the bundled SSH helper described in
  [references/ssh-execution.md](references/ssh-execution.md).
- To test independent execution by GLM, use
  [references/agent-validation.md](references/agent-validation.md) for a recorded
  development run, a fresh replay, and a second-model transfer test.

Quick start:

```bash
git clone https://github.com/Wancha0/ascend-npu-porting.git
cd ascend-npu-porting
python3 scripts/self_check.py
```

The workflow covers source compatibility, real model/training/serving gates,
HCCL/DDP, checkpoint and resume evidence, offline handoff, and optional
post-port profiling and performance tuning. It also covers hash-guarded
multi-library patch delivery and scheduler-independent training-job lifecycle
contracts. It includes local artifact availability contracts but deliberately
excludes OBS and data-transfer operations.

Public delivery contains instructions, small source patches, project-owned
overlays, launch/config files, tests, and hash manifests—not complete dependency
trees, accelerator runtimes, virtual environments, weights, datasets, or
caches. See
[references/dependency-patch-delivery.md](references/dependency-patch-delivery.md)
for reconstructing changes that span installed libraries.
