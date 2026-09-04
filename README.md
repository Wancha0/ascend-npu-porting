# Ascend NPU Porting

An agent-independent, code-first workflow for adapting and validating PyTorch
projects on Huawei Ascend NPU. Codex is not required: GLM-hosted coding agents,
other coding agents, and human operators can use the Markdown instructions and
Python standard-library helpers.

- Start on a new computer with [PORTABLE_AGENT_GUIDE.md](PORTABLE_AGENT_GUIDE.md).
- Give an agent [SKILL.md](SKILL.md) as the authoritative decision guide.
- For GLM, also use [references/glm-agent.md](references/glm-agent.md) to verify
  the host's actual capabilities and persistence behavior.
- Verify a copied toolkit with `python3 scripts/self_check.py` before use.

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
