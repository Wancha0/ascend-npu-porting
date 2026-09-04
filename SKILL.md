---
name: ascend-npu-porting
description: Adapt, validate, profile, and tune PyTorch or model-serving code for Huawei Ascend NPU with torch_npu/CANN. Use for CUDA-to-NPU source changes, operator/dtype/device routing, NPU training or serving reviews, HCCL/DDP debugging, post-port training throughput or memory optimization, scaling analysis, or agent-independent offline code handoff, including GLM-hosted coding agents. Focus on code and runtime; do not use for OBS/data-transfer work or ordinary CUDA optimization without an Ascend target.
---

# Ascend NPU Porting

Produce an auditable, code-first NPU adaptation without weakening the original
CPU/CUDA paths. Treat imports, tiny tensor tests, real operators, full-model
execution, and topology probes as separate gates; none alone proves that a
model can train or serve.

Do not spend the task on OBS, downloads, dataset synchronization, or weight
distribution unless the user explicitly expands the scope. It is still valid
to inventory whether required artifacts already exist locally, record their
path/hash/interface contracts, and continue with source adaptation. Never turn
that inventory into an unrequested transfer operation.

This repository is an agent-neutral Markdown and Python toolkit. It must not
depend on Codex, MCP, chat history, or a proprietary orchestration API to carry
out the port. On another computer or with a different coding agent, begin with
[PORTABLE_AGENT_GUIDE.md](PORTABLE_AGENT_GUIDE.md) and run
`python3 scripts/self_check.py` before trusting a copied toolkit.
When the acting model is GLM, or GLM is connected through Claude Code,
OpenCode, OpenClaw, Cline, Roo Code, Cursor, or another host, also read
[references/glm-agent.md](references/glm-agent.md). The host's tools and
permissions—not the GLM model name—determine whether direct or handoff mode is
possible.

## Choose the operating mode

- **Direct mode:** The adapting agent can inspect and execute on the target. Follow the
  staged workflow in [references/porting-workflow.md](references/porting-workflow.md).
- **Handoff mode:** The adapting agent cannot reach the target, or only a human can run
  commands there. Also read
  [references/offline-handoff.md](references/offline-handoff.md) and create a
  self-contained, hash-manifested bundle instead of guessing target state.
- **Review mode:** The project claims to be adapted already. Re-run the gates
  appropriate to the requested outcome and distinguish runtime, model,
  single-rank, and distributed readiness.

For training or multi-card work, read
[references/training-readiness.md](references/training-readiness.md). For
creating, submitting, monitoring, stopping, or resuming a training job, also
read [references/training-job-lifecycle.md](references/training-job-lifecycle.md).
For maximum-batch, throughput, step-time, memory, profiling, or scaling work
after the relevant readiness gate passes, also read
[references/training-performance.md](references/training-performance.md). For
inference servers, generation pipelines, or parallel serving engines, read
[references/serving-readiness.md](references/serving-readiness.md). For
CUDA-only kernels, dtype failures, Accelerate, or optimizer memory issues, read
[references/compatibility-patterns.md](references/compatibility-patterns.md).
When selecting versions, checking API/operator support, or writing a handoff
for a human operator, read
[references/official-links.md](references/official-links.md).
When changes extend into editable dependency checkouts, `site-packages`,
framework forks, or binary extensions, read
[references/dependency-patch-delivery.md](references/dependency-patch-delivery.md)
and deliver reconstructable, hash-guarded deltas rather than copied libraries.

## Establish the contract before changing code

Record the following from evidence rather than assumption:

- immutable source revision and the current dirty-worktree state for every
  repository, submodule, overlay, patch set, or branch that participates in the
  requested execution path;
- target outcome: import, inference, preprocessing, training, evaluation, or
  serving;
- Python, PyTorch, torch_npu, CANN, driver, OS/kernel, NPU type/count, and
  single- versus multi-node topology;
- executed entrypoints, model/checkpoint contract, representative tensor
  shapes/dtypes/layouts, and distributed process model;
- optional CUDA extensions, Triton/xFormers/FlashAttention paths, compile/JIT
  paths, and their eager or portable fallbacks;
- HBM, host-memory, disk, and time constraints.

When launchers, preprocessing, model code, and evaluation live in different
repositories or branches, draw a source/execution graph before scanning or
patching. Freeze every node to an immutable revision, record how nodes are
composed, and test the composed checkout. A library branch without the launcher
that consumes it is not a trainable project contract.

For a performance target, also freeze the optimization objective, workload,
precision, effective batch, accumulation, sequence/resolution, topology, data
path, optimizer, and checkpoint cadence. Otherwise a faster number may describe
a different training job rather than an improvement.

When the target is unreachable, package the toolkit-root
`scripts/probe_ascend_runtime.py` and `scripts/scan_npu_risks.py` for the
operator to run. Do not select package versions, topology, or memory strategy
from a different server's history.

## Non-negotiable invariants

1. Import torch_npu before Accelerate or other device-discovery frameworks make
   a device decision. Do not replace an existing torch build without proving
   the torch/torch_npu/CANN pairing.
2. Route CUDA-only code using the real `tensor.device.type`, not only
   `.is_cuda`; compatibility layers may make an NPU tensor report CUDA-like
   properties.
3. Keep fast CUDA/Triton/xFormers/custom-C++ paths intact and add the narrowest
   NPU-safe fallback. Never fake CUDA_HOME or silently run an unverified kernel.
4. Preserve numerical intent. Move only unsupported cache construction to CPU
   or use an equivalent real-valued formulation; do not silently lower
   precision without a finite-value and parity check.
5. Preserve the installed torch/torch_npu/CANN tuple until the official
   compatibility table proves a replacement. Put extra dependencies in a
   separate NPU constraints file so a generic installer cannot replace torch.
   Do not publish complete dependency trees, environment directories, runtime
   binaries, weights, datasets, caches, or credentials as an adaptation.
6. Centralize device, autocast, synchronize, seed, and memory helpers. Avoid
   global monkey-patches and mechanical `cuda -> npu` replacement.
7. A code gate succeeds only when it exits zero, emits its expected positive
   gate, exercises the intended NPU path, and validates output or state.
8. Keep upstream CPU/CUDA tests runnable and add narrow NPU regression tests for
   each changed operator or dispatch branch.
9. Stop on the first new OOM, HCCL failure, non-finite value, traceback, or
   contract mismatch. Inspect state before retrying; do not mutate knobs at
   random.
10. Establish correctness and readiness before tuning. Change one logical
    variable at a time, keep a rollback, and claim a performance improvement
    only from synchronized, repeatable A/B measurements on the same contract.

## Work in escalating gates

Use the smallest gate capable of disproving the next claim:

1. static inventory, composed source graph, import graph, entrypoints, and
   dependency contract;
2. runtime probe and one BF16 forward/backward tensor operation;
3. each changed component at representative shape/dtype/layout;
4. strict checkpoint load and one real model forward;
5. real dataloader to loss to backward to optimizer update;
6. saved checkpoint strict reload and another forward;
7. two-rank collective or DDP smoke;
8. intended single-node topology, then intended multi-node topology;
9. bounded end-to-end task with actual inputs and outputs.
10. optional bounded performance profile, controlled A/B tuning, and scaling
    validation after the requested readiness level has passed.

Do not jump to a full model or all cards to diagnose an import, operator, or
single-rank memory problem. Conversely, do not call a project train-ready after
only a tiny model smoke.

## Expected project deliverables

Adapt the list to the repository, but normally leave:

- scoped source patches with upstream CUDA/CPU behavior preserved;
- for changes outside the main project, a per-library patch registry with
  upstream revision, license reference, pristine file hashes, ordered patch
  hashes, validation commands, and rollback;
- a target-aware device/runtime helper or equivalent centralized logic;
- NPU dependency constraints that do not overwrite the platform torch pair;
- per-change tests plus repeatable import, component, full-model, optimizer,
  and distributed gates;
- an NPU launcher/config with explicit topology and effective-batch math;
- when job creation is in scope, a scheduler-independent project launcher plus
  any authorized platform adapter, preflight, status, scoped stop, and resume
  contracts;
- `NPU_PORTING.md` containing a patch map, runtime tuple, exact commands,
  expected gates, official reference links, cleanup, and known limitations;
- machine-readable evidence containing source revision, patch hashes, runtime
  versions, exercised code paths, topology, metrics, artifacts, and failures.
- for performance work, a baseline-versus-final report containing the frozen
  workload, warmup/measurement windows, repeat runs, throughput/step-time/HBM,
  bottleneck attribution, accepted and rejected changes, and rollback paths.
- for a disconnected handoff, the exact toolkit revision or manifest plus all
  relevant instructions and scripts; never require unrecorded chat context.

From the toolkit root, run `scripts/manifest.py create` for an offline
code-handoff bundle, then `scripts/manifest.py verify` at the destination. Run
`scripts/validate_evidence.py` on evidence returned from a disconnected target.

## Completion language

- Say **runtime-ready** only after runtime and real component gates.
- Say **inference-ready** only after the released/full checkpoint runs the
  intended inference entry and produces a validated output.
- Say **service-ready** only after intended request behavior, monitoring,
  shutdown/resource release, persistence, and cold restart also pass.
- Say **training-engine-ready** only after a real optimizer update and strict
  checkpoint reload on the intended single-node process count.
- Say **distributed-training-ready** only after the intended HCCL/DDP topology,
  optimizer update, save, strict reload, and resume all pass.
- Say **performance-profiled** only after a bounded trace on the intended
  workload attributes the dominant time or memory costs.
- Say **performance-tuned** only after a repeatable improvement over a frozen
  baseline, finite/parity checks, stable optimizer steps on the target
  topology, and checkpoint reload. Never use this label to mean “optimal.”

If a prerequisite is missing, report the strongest proven level and the exact
remaining blocker. Never convert “likely” into a pass gate.
