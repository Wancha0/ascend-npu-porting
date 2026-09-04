# Training job lifecycle

Read this reference when the requested outcome includes creating, submitting,
monitoring, stopping, or resuming an NPU training job. Keep the project training
contract separate from the scheduler or cloud adapter.

Generating launcher/config files is a normal code deliverable. Submitting a
remote or paid job, reserving accelerators, stopping another process, or
deleting outputs changes external state and requires explicit user authority.
Without that authority, prepare and validate the job package but do not submit
it.

## Freeze the job contract

Record these values before generating a launcher:

- project root, environment initialization, training entrypoint, and immutable
  source/patch/config revisions;
- local checkpoint, dataset/cache, processor, and output contracts;
- NPU type, node count, processes per node, world size, node ranks, visible
  devices, master address/port, and communication interface;
- precision, per-rank batch, accumulation, data-parallel ranks, effective global
  batch, optimizer, learning rate/scheduler, and gradient clipping;
- target optimizer steps or epochs, checkpoint cadence/retention, resume source,
  profile window, and stop lines for HBM/host memory/time;
- run ID, per-node log/evidence directories, expected gates, ownership, and
  cleanup boundaries.

Do not inherit stale IP addresses, ports, device lists, or output paths from a
previous run. Re-probe them immediately before submission. Reject a multi-node
contract unless `world_size == nodes * processes_per_node` for pure data
parallelism, or document how other parallel groups change that equation.

## Build the project launcher first

The project launcher owns model/data arguments and must work independently of a
particular scheduler. Prefer the supplied
[`torchrun_npu.sh`](../assets/training-job/torchrun_npu.sh) as a hardened
starting point when the project uses TorchRun. It expects its topology through
environment variables and accepts the training entrypoint plus arguments as
ordinary positional parameters.

For Accelerate or DeepSpeed, generate a versioned config and retain the same
contract and preflight checks. Do not assume a GPU-named configuration field is
valid or invalid on NPU from its name alone; verify the behavior of the
installed framework and TorchNPU tuple. Ensure `torch_npu` is imported before
device discovery when the stack requires it.

Before multi-node execution, prove in order:

1. launcher syntax and `--help` or configuration parsing without allocation;
2. one-rank runtime probe;
3. two-rank exact all-reduce;
4. two-rank real model optimizer update;
5. intended single-node process count;
6. one rank per intended node;
7. full topology for one bounded optimizer update.

## Add a scheduler adapter

A scheduler adapter should only translate platform allocation metadata into the
project launcher contract. Keep Slurm, ModelArts, Kubernetes/Volcano, or local
background syntax outside model code. Save the generated job specification and
the resolved non-secret parameters as evidence.

Every adapter must:

- establish one stable node-rank mapping and one master endpoint;
- initialize the same environment and source/patch manifest on every node;
- pass unique per-node run directories while sharing the intended output path;
- forward signals to the launcher and preserve its exit code;
- avoid embedding credentials, signed URLs, or private keys;
- expose a read-only status command and a scoped stop command;
- distinguish scheduler acceptance, process startup, optimizer progress, and
  successful completion.

“Job created” means the scheduler accepted a specification. It does not mean
the training process started, all ranks joined, or an optimizer step completed.

## Preflight immediately before launch

Fail closed on:

- a source/patch/config hash mismatch or unexpected dirty tree;
- missing local input/checkpoint paths or an ambiguous resume checkpoint;
- occupied NPUs, insufficient HBM/disk/host memory, or an already-owned output
  directory;
- duplicate/out-of-range ranks, inconsistent world size, loopback master
  address for a multi-node job, unreachable master port, or conflicting port;
- incompatible runtime tuple or failure of the last required lower gate.

Create the run directory atomically. Never reuse a failed run directory as if
it were a clean attempt. Record the exact submission command but redact secret
values.

## Monitor and decide

After submission, inspect both scheduler state and process evidence. Require:

- every expected hostname/rank/device mapping;
- advancing microstep and optimizer-step counters;
- finite total and component losses plus plausible learning-rate movement;
- HBM/host-memory margins and utilization consistent with the phase;
- absence of OOM, HCCL, NaN/Inf, traceback, compile, missing-operator, and
  unexpected-fallback errors;
- checkpoint creation at the promised cadence.

Stop only the exact job ID or recorded process group. On failure, preserve logs,
the last valid checkpoint, scheduler metadata, and the first causal traceback
before cleanup. Do not automatically retry with lower precision, smaller data,
fewer ranks, or a different optimizer because that changes the contract.

## Resume and completion

A resume job must pin the previous checkpoint hash and restore optimizer,
scheduler/scaler, counters, RNG, and data position when the trainer supports
them. Prove that the next optimizer step and scheduler value continue rather
than restart.

Count the job complete only after all intended ranks exit zero, the expected
checkpoint exists, strict reload and another forward pass, hashes are recorded,
child processes and ports are released, and NPUs return to the expected idle
state. Then assign the readiness label from `SKILL.md`; scheduler success alone
does not raise it.
