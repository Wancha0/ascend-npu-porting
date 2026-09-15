# ModelArts: Notebook development to production training

Use for Ascend projects moving from interactive development to a platform-managed training job. This reference supplements the job lifecycle and [backup choice](artifact-backup.md); it does not submit jobs or authorize transfers.

## Evidence and scope

Reviewed 2026-09-15. Platform facts below come from the linked official documentation and migration examples. The operational recommendations are this toolkit's engineering guidance, not a claim of a completed production deployment. Recovered project history proves SSH/debug-mode training and multi-node preparation; it does **not** yet prove a production job's unattended start, durable output and cold resume.

Record the actual region, console generation, pool and launch mode. Older “custom algorithm / custom image” and newer “custom job” screens are not identical. Do not copy historical field limits, paths, GPU versions or DLS_* variables into an Ascend job without checking the current interface.

## Preferred workflow

1. **Develop in Notebook.** Adapt the real model, freeze the runtime tuple and turn notebook state into an importable training entrypoint plus config. Reuse valid optimizer/save evidence; make manual dependency patches reconstructable.
2. **Package for a fresh job.** Pin source revision, image identity, runtime initialization, inputs, output mapping and resolved training parameters. Notebook files, mounts and installed packages do not automatically follow the job. Test the entry non-interactively; keep initialization explicit instead of relying on a terminal's activated environment.
3. **Run a bounded production pilot when authorized.** Exercise the production submission path with the intended model/data and relevant topology. Check real update, checkpoint delivery, launcher exit and platform status. A fresh-job resume is needed before claiming production resume is verified. Reuse unchanged lower-level tests; the pilot targets the deployment gap.
4. **Submit the full job.** Use the same tested package with the agreed training budget. Monitor through platform status and durable logs; training, checkpoint delivery and any authorized evaluation must not need the Notebook, laptop or chat agent to remain alive.

This sequence is consistent with Huawei's [PyTorch migration practice](https://support.huaweicloud.com/usermanual-standard-modelarts/develop-modelarts-0148.html). Its old CUDA/image examples are examples, not NPU version recommendations.

## Make a concrete job package

Prepare the following before submission; reuse choices already made by the user.

| Item | Required decision/evidence |
|---|---|
| Execution | Region/project/workspace, pool/specification, nodes, cards per node, launch mode and retry/time budget |
| Environment | SWR image with immutable digest when available, architecture, Python/torch/torch_npu/CANN tuple, tested initialization and dependency deltas |
| Source | Commit plus included patches, exact code source/local destination, working directory, entry command and resolved config |
| Inputs | Model and processor identities, dataset split, feature contract; durable source to actual job-local path mapping; staging completion on every reader node |
| Outputs | Checkpoint/resume/EMA/metrics/evaluation paths, durable destination, backup choice, cadence, verification and failure policy |
| Observability | Persistent platform log destination if selected, run/job ID, structured progress, external tracker behavior during network failure |
| Completion | Target optimizer updates, checkpoint selection, optional evaluation sequence, final delivery and real exit status |

An output field is not sufficient: the trainer must write to its mapped local directory. Permanent platform logs and model outputs are separate configurations. Match the current [custom-job form](https://support.huaweicloud.com/develop-modelarts/develop-modelarts-0006.html); do not assume every file under /cache is collected.

Prefer a reproducible training-compatible image with dependencies already present. A saved Notebook image is not proof that mounted code/data was included or that the training image entry contract is satisfied. Check the [training-image requirements](https://support.huaweicloud.com/usermanual-standard-modelarts/docker-modelarts_0017.html) and [current custom-image setup](https://support.huaweicloud.com/docker-modelarts/docker-modelarts_0118.html). Separate large data/weights from frequently changed code. Do not make a long job depend on public downloads, an interactive login or a laptop proxy. If startup staging is needed, make it bounded and restartable under the authorized storage policy.

## Select exactly one process-launch owner

- **Platform-managed processes:** Supported prebuilt engines can launch the startup file per device, or use platform TorchRun through MA_RUN_METHOD. Supply the worker entry expected by that engine; do not run another full torchrun/Accelerate launcher inside every worker.
- **Custom-image launcher:** When the platform invokes the command once per node, let the project launcher create local workers once. Verify this invocation behavior for the selected job mode. Do not SSH from worker 0 to start the other nodes.
- Read the selected engine's [startup semantics](https://support.huaweicloud.com/intl/zh-cn/usermanual-standard-modelarts/develop-modelarts-1415.html), not just the presence of a rank variable.

For a custom adapter, validate MA_NUM_HOSTS, VC_TASK_INDEX and VC_WORKER_HOSTS against the allocation before mapping them to node count/rank/master. MA_NUM_GPUS is documented as accelerator count despite its name; verify its actual NPU value and visible devices. World size for ordinary one-process-per-device DDP is nodes × processes per node. HCCL interfaces must match the actual allocation. Never inherit another job's IP, port, rank table or SSH configuration.

The [environment reference](https://support.huaweicloud.com/develop-modelarts/develop-modelarts-0104.html) documents MA_JOB_DIR and allocation variables. Do not create private variables with reserved MA_ names; documented platform knobs are distinct. Log an allowlist of non-secret topology/runtime fields, not the full environment.

## Foreground execution and durable completion

Production entrypoints run in the foreground. Do not use detached run_recorded.py start, nohup, tmux or sleep infinity as the production training command: submission/startup can otherwise finish while useful work is absent or detached. A wrapper must wait for its owned workers, forward termination, preserve training failure and wait for selected final delivery. An exec launcher is appropriate when checkpoint delivery is inside the trainer or a proven platform output mechanism; it is insufficient if required post-training upload/evaluation still lives after exec.

Toolkit recommendations:
- Parameterize data/output paths; reject missing or incompatible cached features before expensive model startup.
- Finish all required checkpoint shards before publishing a completion marker; use one designated publisher for shared metadata. Do not let rank 0 claim a sharded checkpoint is complete while other shards are pending.
- Periodically persist recoverable state. A final-only upload or termination trap cannot protect against hard kill/node loss. Do not assume object-backed mounts have local POSIX rename/fsync semantics.
- Preserve model/optimizer/scheduler/scaler, EMA when used, update counters, RNG and supported sampler/data progress. Record any resume limits.
- Keep upload errors and training errors distinct. Apply the selected backup failure policy; do not hide a failed selected delivery behind an unconditional success marker.
- If evaluation is authorized, configure its dataset order, checkpoint, environment and output before launch. It may run as a subsequent stage or separately submitted job with durable handoff; merely writing a plan is not scheduling it.

Use the existing backup reference for scope/authorization. Default routine checks to identity, count/size and available trusted receipts/checksums; do not rehash full caches or read back every large OBS object on every run. Escalate for mismatch, missing evidence or an explicit stronger integrity requirement.

## Monitor, failures and resume

Distinguish accepted/queued, initializing/staging, optimizer progress, checkpoint delivered, training finished and evaluation finished. ETA separates queue/image pull/staging from measured steady-state training. Small status/log reads are normally sufficient.

ModelArts supports [Cloud Shell for running jobs in dedicated pools](https://support.huaweicloud.com/develop-modelarts/develop-modelarts-0119.html), subject to permissions including modelarts:trainJob:exec. This is a conditional debugging option, not an unattended execution dependency. Official keep-alive/sleep examples are for diagnosis: do not copy “failure then sleep” into production in a way that masks the original exit code or indefinitely holds resources.

On failure, retain the first causal traceback, exact job ID/config and last durable checkpoint. Query state before resubmitting an uncertain submission. Use finite retry limits, unique attempt outputs and an explicit resume source; never silently restart from step zero or change batch/precision. A stopped/ended job is not assumed restartable: follow current [copy/stop management](https://support.huaweicloud.com/intl/en-us/usermanual-standard-modelarts/develop-modelarts-0017.html). Reconcile any platform automatic recovery policy with application resume to avoid duplicate attempts.

## Report the actual proven level

Separate “Notebook training verified”, “production package prepared”, “production pilot passed”, “production resume verified” and “full training complete”. For each production test record job ID, mode, source/image/config, topology, observed updates, checkpoint destination, verification and final platform/process status. Do not mark this workflow production-tested until those records exist.

No cloud-independent submit template is supplied here: generate the specification from the current console/API or installed CLI schema. Huawei's [ma-cli job reference](https://support.huaweicloud.com/intl/zh-cn/usermanual-standard-modelarts/devtool-modelarts_0320.html) provides submit/status/log operations; check the installed version's help before using it. Preparing this skill does not start a pilot.

