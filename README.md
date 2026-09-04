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

The workflow covers source compatibility, real model/training/serving gates,
HCCL/DDP, checkpoint and resume evidence, offline handoff, and optional
post-port profiling and performance tuning. It includes local artifact
availability contracts but deliberately excludes OBS and data-transfer
operations.
