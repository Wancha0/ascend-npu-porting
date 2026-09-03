# Training performance after porting

Use this workflow only after the training-engine and requested topology gates
pass. Its purpose is to find and verify improvements, not to apply a fixed bag
of “fast” settings. Runtime version, graph shape, model family, topology, and
input pipeline can reverse the value of a tuning choice.

## 1. Freeze the comparison contract

Choose one primary objective: steady-state samples/frames/tokens per second,
optimizer-step time, maximum safe per-rank batch, scaling efficiency, or cost
per completed optimizer step. Treat the others as guardrails.

Keep these fields identical across an A/B comparison unless the field itself is
the tested variable:

- source, patch, config, checkpoint, dataset sample order, and random seed;
- precision policy and loss-scaling behavior;
- model inputs, sequence/action horizon, image/video resolution, and padding;
- optimizer, scheduler, gradient clipping, and checkpoint cadence;
- per-rank batch, accumulation, effective global batch, parallelism type, and
  rank topology;
- input cache state, worker policy, and storage path.

If a capacity sweep changes per-rank batch, keep effective global batch fixed
with accumulation when comparing training semantics. Report separately when
that is impossible.

## 2. Establish a trustworthy baseline

Give every attempt a unique run ID, log directory, output directory, and
evidence file. Use the real training entrypoint and a representative workload.
Run enough warmup for lazy initialization or compilation to settle, then time a
bounded sequence of completed optimizer steps. Synchronize the device before
and after each timed window.

Record at minimum:

- median and p95 optimizer-step time plus task-relevant throughput;
- warmup count, measured-step count, repeat count, and wall-clock boundaries;
- per-rank allocated/reserved and device-reported peak HBM;
- input wait, host-to-device, forward, backward, gradient communication,
  optimizer, and checkpoint time when observable;
- host CPU/RAM/I/O pressure, NPU utilization, and rank stragglers;
- finite loss/gradients, completed optimizer steps, checkpoint strict reload,
  and resume result.

Do not compare a compile/warmup step with a steady-state step, extrapolate from
one iteration, or include profiler overhead in the baseline.

## 3. Separate capacity from throughput

Sweep per-rank batch upward in bounded increments. Each point must complete the
first optimizer update and several steady-state updates. Stop at a predeclared
HBM margin below device capacity; do not repeatedly crash all ranks to locate
the exact OOM boundary. Clean failed process groups and use a fresh run
directory before the next point.

Report two results:

- **maximum-fit batch:** largest point that satisfies the HBM margin and all
  correctness/checkpoint gates;
- **best-throughput batch:** measured point with the strongest repeatable
  throughput under the frozen comparison contract.

They are often different. Larger batches increase activation, temporary, and
workspace memory and may reduce headroom without improving throughput.

## 4. Profile a narrow representative window

Probe which tools and options exist in the installed TorchNPU/CANN version.
Prefer `torch_npu.profiler` or the version-matched msProf workflow. Start with a
single rank or one representative rank and a short scheduled capture window.
Collect every rank only when diagnosing communication or rank imbalance.

Attribute time and memory to the input pipeline, copies, forward, backward,
collectives, optimizer, checkpointing, and synchronization. Inspect operator
shape/dtype/layout, implicit casts or contiguous copies, graph breaks, CPU
fallbacks, small-op launch density, temporary workspaces, and allocation peaks.
Retain the command, trace path, tool/runtime version, and trace hash.

Do not treat high or low operator utilization as the result. Use it to form one
testable bottleneck hypothesis.

## 5. Choose changes from measured bottlenecks

Use the narrowest relevant branch below. Preserve an easy rollback for every
candidate.

| Evidence | Candidate direction | Required guardrail |
|---|---|---|
| Input wait or host/I/O stalls | workers, prefetch, caching, decoding, collation, transfer overlap | identical samples and preprocessing |
| Device copy stalls | fewer redundant transfers/casts, contiguous layout at the boundary, supported asynchronous copy | tensor equality and lifetime safety |
| Generic/fallback-heavy compute | supported fused attention/norm/MLP/RoPE, fewer layout conversions, larger fused regions | parity at representative shapes/dtypes |
| Excess small operators or graph breaks | supported compile/graph mode, static regions, removal of Python-side synchronization | eager fallback and compile-cache stability |
| Activation-dominated HBM | selective activation checkpointing, sequence/resolution policy only if contract permits | throughput and parity, not memory alone |
| Optimizer/gradient peaks | supported fused/foreach policy, gradient bucket views, sharding/offload when necessary | optimizer-state and resume equivalence |
| Collective or rank-straggler time | accumulation `no_sync`, bucket size/order, overlap, HCCL interface/topology checks | every rank advances and reloads |
| Checkpoint or filesystem stalls | cadence, asynchronous or staged writes when supported | durable complete checkpoint and cold reload |

Only tune allocator settings after evidence of fragmentation. Only introduce
ZeRO/FSDP/tensor/sequence parallelism or offload when the simpler data-parallel
contract cannot meet capacity or scaling requirements; these change failure
modes and checkpoint semantics. Never copy environment variables or bucket
sizes from a different cluster without a local A/B result.

## 6. Run controlled A/B experiments

Apply one logical change at a time. Use the same revision, data order, seed,
workload, topology, warmup, and measured windows. Repeat both baseline and
candidate enough to estimate run-to-run noise. Accept a change only when the
gain exceeds that noise and all of these remain true:

- loss and expected gradients are finite;
- task-appropriate parity tolerances pass;
- optimizer/microstep counters and effective batch are unchanged as intended;
- target ranks exit zero without OOM or HCCL errors;
- a new checkpoint strictly reloads and resume completes another update.

Record rejected changes as well as accepted ones. Lower peak HBM is a capacity
improvement, not a speedup, unless throughput also improves.

## 7. Validate scaling separately

Measure one device, the intended single-node process count, one rank per node,
and the full intended topology when available. State whether the experiment is
strong scaling (fixed global workload) or weak scaling (workload grows with
rank count). For a comparable data-parallel workload, report:

```text
scaling_efficiency_N = throughput_N / (N * throughput_1)
```

Also report per-rank step-time dispersion and communication fraction. Do not
claim linear scaling from aggregate utilization or a successful HCCL smoke.

## 8. Evidence and stop conditions

Extend the common evidence envelope with `performance` containing:

- objective and frozen workload/precision/effective-batch contract;
- baseline, candidates, and final metrics with commands and repeat windows;
- runtime, topology, per-rank metrics, HBM, and host-resource observations;
- profiler command, capture range, trace path/hash, and bottleneck attribution;
- patch/config hashes, parity and checkpoint results, accepted/rejected status,
  and exact rollback instructions.

Stop the experiment on a new non-finite value, parity breach, OOM, HCCL error,
rank loss, checkpoint/reload failure, resource-health alert, or workload
contract drift. Report the strongest validated result; do not trade away a
required correctness or precision contract for a performance label.
