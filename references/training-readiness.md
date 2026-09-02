# Training readiness

Training readiness is a stack of claims. Report the highest proven layer rather
than collapsing them into a single yes/no.

## Readiness levels

| Level | Minimum evidence |
|---|---|
| Runtime | Correct torch/torch_npu/CANN pair and a BF16 NPU forward/backward |
| Model | Full architecture and intended initialization/checkpoint strictly load |
| Input integration | A representative batch satisfies keys, shape, dtype, mask, and normalization contracts |
| Training engine | Real dataloader → loss → backward → optimizer update; all values finite |
| Checkpoint | Produced checkpoint strictly reloads into a fresh model and runs a forward |
| Single-node distributed | Intended local rank count completes a DDP update and main-rank save |
| Multi-node distributed | Intended node/rank topology completes collective and DDP gates |

For a code-only port, stop at the highest layer proven by locally available
inputs. Do not acquire or transfer datasets as part of this workflow. A
synthetic batch can validate tensor plumbing but cannot prove the real Dataset,
collate, or normalization path.

## Data and preprocessing gate

Write down expected cameras/order, frame count/stride, resolution, padding,
action horizon/dimension, proprio dimension, text model/revision, normalization,
and feature teacher revisions. Load a real sample through the same Dataset and
collate path used by training. Validate every key, shape, dtype, finite value,
mask, and context reference.

If no real sample is locally available, build a synthetic batch only from the
documented input contract, label that gate synthetic, and leave the real input
integration gate pending.

## Optimizer-step semantics

The configured training step should normally mean a completed optimizer update.
With gradient accumulation, continue consuming microbatches until gradients
synchronize, then clip, update, step the scheduler, and increment the optimizer
step. Record both counters.

Test at least:

- accumulation greater than one;
- a dataloader length not divisible by accumulation;
- the configured world size and effective batch calculation;
- finite component losses and nonzero expected gradients;
- an observable parameter or optimizer-state change.

Effective batch is generally:

```text
per-rank batch × data-parallel ranks × accumulation steps
```

Account separately for tensor/sequence parallel ranks that do not replicate
independent data batches.

## Checkpoint and resume contract

A long-run checkpoint should contain or reference:

- model and optimizer state;
- scheduler/scaler state;
- completed optimizer step and microstep/accumulation position;
- exact config and source revision;
- RNG state where reproducibility matters;
- dataset/normalization and encoder provenance when available;
- topology/effective-batch information.

Perform two different tests:

1. strict model reload plus another forward;
2. resume training, complete another optimizer update, and verify counters and
   scheduler continue rather than restart.

A model-only warm start is useful but is not a true resume. If the trainer only
writes a final checkpoint, report that as an operational risk before a long job.
Use periodic checkpoints and a retention policy sized for the actual model and
optimizer state. Verify the locally written checkpoint's size/hash and strict
reload; remote persistence is outside this code-adaptation workflow.

## Distributed gates

Run in this order:

1. two-rank all-reduce with an exact expected value;
2. two-rank real DDP optimizer update;
3. intended single-node rank count;
4. one rank per node across all intended nodes;
5. intended full topology and one bounded optimizer update.

Every rank must exit zero. Capture rank/hostname/device mapping, HBM, logs, and
the main-rank checkpoint. A DDP import or HCCL initialization alone does not
prove model gradients fit.

When DDP OOMs after a successful single-card update, compare memory just before
and during optimizer step. Model-sized gradient buckets, `foreach` temporaries,
and duplicated optimizer state are common causes; validate any mitigation with
the full model.

## Launch and monitoring contract

Launchers should reject invalid counts/ranks, stale output directories, reused
ports, missing input/checkpoint paths, and ambiguous master addresses. Each run
gets a unique run ID, log directory, PID/process-group record, output directory,
and evidence file.

Immediately after launch, verify:

- all ranks exist on the intended host/device;
- HBM and host memory are below the chosen stop line;
- loss and learning rate are finite and scheduler movement is plausible;
- optimizer and microstep counters advance as intended;
- no OOM/HCCL/NaN/Inf/traceback/compile error appears.

After exit, verify checkpoint reload, child-process cleanup, port release, NPU
release, and result hashes.

## Minimum training evidence record

Record a JSON object with:

- gate and status;
- source/config/checkpoint/data revisions;
- runtime and topology;
- per-rank batch, accumulation, effective batch, target optimizer steps, and
  observed micro/optimizer steps;
- finite loss components, learning rate, elapsed time, and peak HBM;
- checkpoint path, size, hash, strict-reload result, and resume result;
- failures and known limitations.

`validate_evidence.py` validates the common envelope. Project-specific checks
should add fields rather than weakening the required envelope.
