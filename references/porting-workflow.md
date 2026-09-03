# Porting workflow

Use this workflow as a decision sequence, not as permission to launch an
expensive job. The user's target outcome determines the last required gate.

## 1. Freeze what is being adapted

Capture the repository URL, commit, submodule revisions, dirty status, original
launch commands, dependency lockfiles, required checkpoint interface, and
expected outputs. Read project instructions and the model paper or architecture
notes when they define tensor layouts, losses, or inference-only branches.

Separate contracts that are often accidentally conflated:

- source and dependency contract;
- checkpoint key/shape/dtype contract;
- input keys, normalization, padding, shape, layout, and sequence contract;
- runtime and operator contract;
- distributed topology/effective-batch contract;
- output, persistence, and resume contract.

Do not patch before identifying the real entry points and which optional
components are training-only, inference-only, or preprocessing-only.

## 2. Inventory the target before installing anything

Run `probe_ascend_runtime.py` on the target and preserve its JSON. Record
`npu-smi info`, host/NPU memory, environment activation, CANN/driver paths, and
whether cards are already occupied. Probe git, the Python compiler toolchain,
custom-extension compilers, and any serving/graphics runtime before relying on
them.

Run `scan_npu_risks.py` on the repository. Triage findings by execution path:
an unused CUDA benchmark is not a blocker, while a CUDA-only attention kernel
on the requested training path is.

## 3. Build a compatibility matrix

Create a small table with the observed and required versions of Python,
PyTorch, torch_npu, CANN, driver, Accelerate/DeepSpeed, compiler extensions,
and major model libraries. Prefer the existing working torch/torch_npu pair.
Put NPU-only additions in a separate constraints or requirements file so a
generic installer cannot replace the platform torch wheel.

Classify dependencies:

- portable pure PyTorch;
- portable after device/dtype routing;
- optional accelerator fast path with an eager fallback;
- CUDA-only build/import that must be made lazy or guarded;
- missing Python/source dependency that must be supplied or guarded;
- unsupported on the target stack and requiring a scoped reimplementation.

Use `official-links.md` to verify the installed version tuple and relevant API
constraints. The latest documentation is not automatically valid for an older
runtime.

## 4. Trace the executed code before patching

Map the requested entrypoint through imports, registries, factories, device
selection, model construction, loss/inference calls, and optional fast paths.
Mark each CUDA-specific branch as executed, potentially executed, or irrelevant
to the requested outcome. Use representative shapes from configs or the model
contract rather than inventing convenient toy dimensions.

Separate missing-checkpoint/input errors from code incompatibility. Record the
missing prerequisite and continue adapting import, construction, synthetic
operator, and distributed paths that can still be proven without acquiring or
moving those prerequisites.

## 5. Centralize platform behavior

Prefer one small runtime abstraction over scattered string replacements. It
usually owns:

- early torch_npu bootstrap;
- `auto -> npu -> cuda -> cpu` device resolution;
- BF16/FP32 selection and autocast;
- seed, synchronize, and memory-stat helpers;
- optional dependency routing by real device type.

Preserve CPU/CUDA behavior. Avoid global monkey-patches unless the upstream
project already depends on a compatibility layer and the affected import order
is proven.

Apply operator fixes one failure at a time. Use `compatibility-patterns.md` for
known categories, then reproduce the failing shape/dtype in a minimal gate.

## 6. Validate bottom-up

For each gate, save the command, environment, exit code, stdout/stderr, runtime
versions, NPU mapping, peak HBM, elapsed time, and an explicit gate string.

Recommended order:

1. compile/import without loading weights;
2. small BF16 tensor forward/backward on NPU;
3. the first real accelerator-sensitive component;
4. full architecture construction;
5. strict released/pretrained checkpoint load;
6. bounded real forward or serving request;
7. real dataloader/preprocessing sample;
8. one optimizer update and checkpoint reload;
9. two-rank collective/DDP;
10. target single-node topology;
11. target multi-node topology;
12. bounded end-to-end task.

If the user also requests training-performance work, begin the separate
[training-performance workflow](training-performance.md) only after the
corresponding functional and topology gate passes. Do not fold compilation
warmup, OOM search attempts, or profiler overhead into the steady-state
baseline.

A gate must fail on non-finite tensors, missing ranks, unexpected fallback,
wrong shapes, missing artifact, or stale artifact provenance—not only on a
Python exception.

## 7. Review the launch and lifecycle

Launchers must validate numeric topology settings, unique ranks, a reachable
master address, an unused port, NPU visibility, output/run IDs, and target
paths. Re-read node IPs in every new job. Set timeouts deliberately rather than
copying a historical value.

For background work, keep PID/process-group ownership, logs, and a bounded
monitor. On exit, verify all child processes, ports, and NPUs are released.
Never count an unchanged process as progress without checking logs, utilization,
and expected artifacts.

## 8. Produce a reproducible handoff

Leave project-specific commands and expected gates in `NPU_PORTING.md`. Record
what was actually tested, what was inferred, and what remains. Generate
manifests for patches, overlays, scripts, configs, references, and result
evidence. Include `official-links.md` so a disconnected operator can identify
the authoritative version/API pages later.

Do not commit, publish, or start a long formal run unless the user asked for
that state change. A clean evidence-backed working tree can still be a completed
code-adaptation deliverable.
