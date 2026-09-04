# Portable agent guide

This repository can guide an NPU port on a computer that does not have Codex.
The instructions are ordinary Markdown and the helper programs use the Python
standard library. `agents/openai.yaml` only supplies optional Codex UI metadata;
it is not part of the execution path.

## Requirements and boundaries

The controller computer needs Python 3.9 or newer and a coding agent capable of
reading files, editing a repository, and running shell commands. Git is useful
for revision and patch evidence but is not required by the helper programs.

The Ascend target still needs its project dependencies, CANN/driver, and a
compatible PyTorch/TorchNPU pair. Only `probe_ascend_runtime.py` imports
`torch`/`torch_npu`; the scanner, manifest, evidence validator, and self-check
are standard-library-only. No Codex service, MCP server, plugin, OBS access, or
internet connection is required after the toolkit, project sources, and
required local artifacts are present. Checking whether checkpoints, cached
features, and representative inputs exist locally is part of the code/runtime
contract; acquiring or moving them is a separate operation outside this
toolkit unless the user explicitly requests it.

An agent without target access can prepare and statically check a patch bundle,
but cannot truthfully claim runtime, training, serving, distributed, or
performance readiness. Those claims require returned target evidence.

“GLM” does not identify a single agent runtime. GLM can be the model behind
several coding-agent hosts, each with different skill discovery, shell access,
permissions, context limits, and remote-execution support. Read
[references/glm-agent.md](references/glm-agent.md) before delegating this
workflow to GLM.

## Bootstrap on a new computer

Clone the repository or copy the whole directory without dropping
`references/` or `scripts/`. From the toolkit root run:

```bash
python3 scripts/self_check.py
```

Continue only after `ASCEND_SKILL_SELF_CHECK_PASS`. Record the toolkit Git
revision when available; otherwise create and preserve a manifest:

```bash
python3 scripts/manifest.py create . --output /absolute/path/to/toolkit-manifest.json
python3 scripts/manifest.py verify /absolute/path/to/toolkit-manifest.json --root .
```

The self-check proves package integrity and helper behavior without NPU
hardware. It does not validate the target runtime or any model.

## Prompt for any coding agent

Give another agent the toolkit path, project path, and this prompt with the
bracketed values filled in:

```text
Read [TOOLKIT]/SKILL.md completely, then read
[TOOLKIT]/references/glm-agent.md if you are running on GLM. Follow its routing
instructions and read only the references needed for
[training/inference/serving/performance] on
[PROJECT]. First map every repository, branch, patch, and launcher used by the
requested command; freeze each revision and dirty-worktree state as immutable
evidence. The target is [ASCEND TYPE/COUNT/TOPOLOGY] with the observed
[PYTHON/PYTORCH/TORCH_NPU/CANN/DRIVER] tuple. Preserve CPU/CUDA behavior, do not
replace the platform torch pair without an official compatibility match, and
do not transfer datasets/weights. You may inventory existing local artifact
paths and record missing ones as blockers. Start with the toolkit self-check,
runtime probe, a scope-limited static risk scan, and an execution-path map. Patch one failure
category at a time; validate from imports and representative operators through
the requested full-model and distributed gates. Leave NPU_PORTING.md, exact
commands, positive gates, artifacts, hashes, rollback notes, and a validated
evidence envelope. If the target is unreachable, prepare the two-round bundle
in references/offline-handoff.md and label it prepared rather than validated.
```

Never ask the agent to “use its judgment and make it work” without the frozen
revision, target outcome, runtime tuple, topology, and checkpoint/input
contracts. Unknown values should be discovered or recorded as blockers, not
guessed from another server.

## Agent-independent execution sequence

1. Run the toolkit self-check and record its revision or manifest.
2. Freeze every source/branch/overlay revision and dirty state; draw how the
   actual launcher composes them. Then freeze target outcome, runtime, topology,
   entrypoint, checkpoint contract, input shapes/dtypes/layouts, and resource
   limits.
3. Run the target runtime probe and repository scanner using absolute paths:

   ```bash
   python3 /absolute/path/to/ascend-npu-porting/scripts/probe_ascend_runtime.py \
     --output /absolute/path/to/run/evidence/runtime.json
   python3 /absolute/path/to/ascend-npu-porting/scripts/scan_npu_risks.py \
     /absolute/path/to/project \
     --output /absolute/path/to/run/evidence/static-scan.json
   ```

4. Trace only the requested execution path across all source nodes. Treat the
   scanner as an inventory, not a mechanical replacement queue. Classify
   dependencies and patch the narrowest device, dtype, operator, import,
   optimizer, or distributed incompatibility while preserving CPU/CUDA
   branches.
5. Validate in the gate order in `SKILL.md`. Use
   `references/training-readiness.md` or
   `references/serving-readiness.md` for the requested outcome.
6. Only after correctness and topology readiness, use
   `references/training-performance.md` for performance work.
7. Produce `NPU_PORTING.md`, scoped patches, exact launch/config files, logs,
   artifacts, hashes, rollback instructions, and JSON evidence. Validate the
   returned envelope with `scripts/validate_evidence.py`.

When the editing agent and target executor are separate, follow
`references/offline-handoff.md`. Every command must carry an explicit working
directory, environment initialization, absolute inputs/outputs, expected gate,
timeout, stop condition, and cleanup target. No step may depend on remembered
chat content.

## Minimum acceptance matrix

| Requested result | Evidence required on the Ascend target |
|---|---|
| Runtime | Compatible runtime inventory plus real BF16 forward/backward |
| Inference | Strict full checkpoint load and validated real output |
| Training | Real data path, finite optimizer update, strict checkpoint reload |
| Distributed training | Every intended rank updates, saves, reloads, and resumes |
| Serving | Real request, output semantics, cleanup, persistence, cold restart |
| Performance | Frozen repeatable baseline, profile attribution, A/B gain, correctness and reload guards |

If a row cannot be proven, report the highest lower gate that passed and the
exact missing command, artifact, or target prerequisite.
