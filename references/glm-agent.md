# GLM coding-agent compatibility

Use this reference when GLM is the model executing the port. The NPU workflow
and evidence gates do not change; this file only makes the agent-host contract
explicit.

## Distinguish the model from the host

GLM can run behind Claude Code, OpenCode, OpenClaw, Cline, Roo Code, Kilo Code,
Cursor, and other coding tools. Z.AI documents these as separate integrations,
and its official [GLM skills repository](https://github.com/zai-org/GLM-skills)
uses a directory containing `SKILL.md`. Therefore never infer file, shell, Git,
SSH, skill-discovery, or persistence capabilities from the word “GLM.” Inspect
the host first.

Official starting points:

- [Z.AI Coding Plan quick start](https://docs.z.ai/devpack/quick-start)
- [Z.AI coding-tool integrations](https://docs.z.ai/devpack/tool/others)
- [Official GLM skills repository](https://github.com/zai-org/GLM-skills)

## Host capability gate

Before delegating a port, prove that the host can:

1. read the complete toolkit directory, not just one pasted prompt;
2. recursively inspect the project and preserve exact line-level edits;
3. execute shell commands with an explicit working directory and return the
   exit code plus untruncated stdout/stderr;
4. inspect Git revisions and dirty state without resetting user changes;
5. keep a file-backed run ledger across context compaction or restarts;
6. reach the Ascend target for direct mode, or create and validate artifacts
   for handoff mode.

If 1–4 are absent, GLM can review text but cannot perform a reliable code port.
If only 6 is absent, use the offline handoff protocol and stop at **prepared**.
API-only GLM is not a coding agent until an orchestrator supplies these tools.

Skill auto-discovery is host-specific. Hosts compatible with directory skills
may discover `SKILL.md` after the whole folder is installed, but still verify
that discovery. For all other hosts, give GLM the toolkit's absolute path and
explicitly instruct it to read `SKILL.md`, this file, and every routed reference
in full. Do not translate the workflow into host-only tool names or assume MCP,
plugins, or Codex metadata.

## Keep long work resumable

GLM must store progress in the run directory rather than relying on chat
memory. Maintain a small ledger containing:

- goal and strongest proven readiness level;
- source graph, immutable revisions, and dirty states;
- runtime/topology and local artifact availability;
- executed commands, exit codes, positive gates, and artifact hashes;
- changed files and why each change exists;
- current failure, next smallest gate, and rollback.

After any context reset, re-read `SKILL.md`, the routed references, and this
ledger before acting. Re-probe mutable target state such as free devices,
processes, ports, IP addresses, and local paths. Do not repeat a mutation merely
because the preceding chat is missing.

## Composite-source rule

Before the risk scan, trace the user's requested command to its actual sources.
Training projects commonly combine a library checkout, a launcher from another
branch, a patch or overlay, and generated artifacts. Record these as a graph:

```text
launcher@revision -> imports/patches -> library@revision
                 -> reads          -> local artifact contract
                 -> writes         -> unique run directory
```

Freeze every source node independently and preserve the composition order.
Never overwrite one branch with another to make a synthetic checkout. Scan only
the nodes and paths reachable from the requested entrypoint; repository-wide
CUDA matches are an inventory, not a to-do list.

## ActionWM PAC ControlNet test case

The ActionWM example demonstrates why this gate is mandatory:

- `diffsynth-pac-controlnet` at audit revision
  `8998f3746c51637feaef2f490765773d17cd8fdc` is the modified DiffSynth library.
  Its README explicitly says data preparation, training launch, and evaluation
  are elsewhere.
- `minimax-h3-adaln` at audit revision
  `139aaf26acb0669edde43e2bfcd3a0535933fe96` contains
  `tools/robotwin_h3_controlnet.sh`, data preparation, the time-embedding-table
  builder, and evaluation tools.
- The library already contains NPU device helpers and optional TorchNPU
  dependencies, but that does not prove this PAC training path.

A capable GLM host should build and validate the composed project, then triage
these first-order blockers before broad edits:

1. The recommended H20 path loads a pre-quantized bitsandbytes NF4 host. Treat
   it as CUDA-specific until a representative NPU load and forward proves
   otherwise. Select an NPU-supported precision/memory plan from measurements;
   do not silently dequantize a 30B-class host and hope it fits.
2. The training launcher exports `CUDA_VISIBLE_DEVICES` and uses a GPU-oriented
   Accelerate command. Add target-aware visibility, device selection, and HCCL
   launch configuration while retaining the CUDA branch.
3. Trace optional FlashAttention/xFormers/custom-kernel imports and offload
   synchronization on the executed MiniMax-H3 path. Route by the real device
   type and exercise the NPU fallback at representative video shapes.
4. Preserve the PAC contracts: tail-aligned control rows, keyframe semantics,
   the `MINIMAX_H3_CONTROLNET_T_TABLE` artifact, frozen host DiT, trainable
   ControlNet, silent audio behavior, and checkpoint prefix/reload semantics.
5. Prove memory with the real 3.402B trainable ControlNet, optimizer state,
   gradients, activations, and checkpointing. A tiny import or a different
   DiffSynth NPU example cannot establish trainability.
6. Validate data-cache generation separately from training, then one real
   optimizer update, save/strict reload/resume, two-rank HCCL, and only then the
   intended topology.

With repository access but no Ascend target or artifacts, GLM can produce a
reviewed **prepared** patch bundle. With a shell-capable host, both source
branches, local weights/data, and target access, the skill gives GLM enough
instructions to attempt and evidence an end-to-end port. It still cannot claim
training readiness until the target gates pass.

## Prompt template for GLM

```text
You are operating through [HOST]. First prove which file, shell, Git, context-
persistence, and target-execution capabilities this host actually exposes.
Read [TOOLKIT]/SKILL.md and [TOOLKIT]/references/glm-agent.md completely, then
follow all references routed for [OUTCOME]. Run the toolkit self-check. Map the
real command across every repository, branch, patch, and generated artifact;
freeze each revision and dirty state before editing. Keep a file-backed ledger.
Preserve CPU/CUDA behavior and the installed torch/torch_npu/CANN tuple. Treat
the static scan as a scope-limited inventory. Patch and validate one failure
category at a time through the requested gates. Never infer success from an
import, another model's NPU example, or a live PID. If target execution is
unavailable, build the manifest-verified two-round handoff and label it
prepared. If a checkpoint/input is absent locally, record its exact interface
and path as a blocker; do not start an unrequested transfer.
```

The prompt is a bootstrap, not a substitute for the toolkit files. If the host
cannot read those files completely, do not delegate an autonomous port to it.
