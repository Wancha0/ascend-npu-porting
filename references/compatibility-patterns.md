# Compatibility patterns and failure triage

These are diagnostic categories, not patches to apply universally. Confirm
that the affected code is on the requested execution path and reproduce the
failure with the smallest real shape first.

## Device discovery and import order

Accelerate, DeepSpeed, Transformers, or project registries may cache the device
backend during import. Import torch_npu before those frameworks inspect torch.
If the project is a package, a small idempotent bootstrap imported from the
package entry point is preferable to adding `import torch_npu` everywhere.

Resolve a requested device to `torch.device` once. Compare `device.type`, not
string equality with a bare `"npu"`, when indexed devices are possible.

## The `.is_cuda` trap

Some compatibility layers map CUDA-facing properties to NPU. An NPU tensor can
therefore satisfy a CUDA-looking condition while still having
`tensor.device.type == "npu"`. Guard CUDA C++, NVCC, Triton-CUDA, FlashAttention
CUDA, and other CUDA-only paths with the real device type. Retain the original
shape/dtype/stride/HIP checks for actual CUDA.

Do not fix this by fabricating CUDA_HOME, installing a CUDA toolkit, or globally
disabling a compatibility layer before understanding who imports it.

## Optional kernels and extensions

Classify each custom path as import-time, build-time, or runtime optional.
Common safe shapes are:

- lazy import the optional extension only inside its eligible branch;
- keep the CUDA fast path and add an NPU pure-PyTorch or torch_npu operator
  path;
- fall back to SDPA/eager math for an unsupported shape;
- make compilation failure explicit and local rather than breaking package
  import.

Validate output shape, finite values, a tolerance against a CPU/eager oracle
when feasible, and backward gradients when training uses the operation.

## Unsupported dtypes and complex math

Ascend stacks commonly reject float64 or complex128 operations that CPU code
creates implicitly. Identify whether high precision is part of the algorithm
or only cache construction.

Examples of scoped remedies:

- construct sinusoidal/RoPE indices and complex caches on CPU, then move real
  cos/sin tensors to NPU;
- express a complex rotation with real multiply/add operations;
- force literals/ranges to float32 where the original algorithm does not
  require float64;
- keep normalization and loss reductions in a supported stable dtype.

Never change a checkpoint or model parameter dtype globally just to bypass the
first error. Test numerical parity and non-finite values.

## Autocast, seeds, synchronization, and memory

Use `torch.autocast(device_type=device.type, dtype=...)` only on supported
accelerators and a null context on CPU. Seed NPU devices explicitly when the
framework helper does not. Use the backend's synchronize and memory-stat APIs
for measurements; host timing without synchronization is misleading.

Pinning CPU memory is not automatically beneficial or supported for NPU. Leave
it disabled unless the exact transfer path was benchmarked and validated.

## Checkpoint loading

Validate the checkpoint's outer structure, required sections, keys, shapes,
dtypes, and provenance before loading. Use strict loading for released/full
checkpoints unless a documented conversion intentionally changes keys.

Some torch_npu/PyTorch combinations reject a `Path` object in mmap checkpoint
loading even though CPU accepts it. Passing `str(path)` is a narrow portable
fix. Do not disable mmap or `weights_only` globally without measuring memory and
understanding the checkpoint payload.

After training, strictly reload the produced checkpoint into a fresh model and
run another forward. A successful `torch.save` is not a resume gate.

## Optimizer memory

Large models may fit forward/backward but OOM in `optimizer.step`.

Check separately:

- parameter, gradient, and optimizer-state dtypes;
- peak versus reserved memory and the largest single temporary;
- PyTorch `foreach` or fused implementations that materialize optimizer-wide
  temporaries;
- DDP gradient buckets that duplicate a full gradient set;
- master-weight behavior in mixed precision.

For NPU, `foreach=False` can avoid optimizer-wide temporary lists. In DDP,
`gradient_as_bucket_view=True` can remove a model-sized duplicate gradient
allocation. These change memory behavior and must be proven with a real
optimizer update; they are not unconditional defaults for every model.

Prefer an explicit HBM stop line below the physical limit while exploring.
Waiting for natural OOM can leave distributed workers or the device runtime in
an unhealthy state.

## Accelerate and gradient accumulation

Verify whether a project's `steps` means microbatches or completed optimizer
updates. If the baseline used accumulation 1, changing accumulation without
changing loop semantics can silently reduce training by that factor.

Count an optimizer step only when gradients synchronize. Step the scheduler at
the same event. If the loop cycles a finite dataloader, decide deliberately
whether partial accumulation should sync at the epoch boundary; for a constant
effective batch, carry it across the boundary.

Record both `micro_steps` and `optimizer_steps` in smoke metrics. Test a dataset
length that is not divisible by the accumulation factor.

## Distributed mapping

For every rank, record world rank, local rank, hostname, visible devices, and
selected device. A zero exit from rank 0 is insufficient. Start with an
all-reduce value whose expected result is obvious, then run real model DDP.

Single-node success does not prove multi-node networking or launcher
orchestration. Verify that every node uses the same source revision, patch
manifest, runtime tuple, master address, and rank table.

## Error categories worth scanning

Scan logs case-insensitively for at least OOM, HCCL, NaN, Inf, traceback,
runtime error, assertion, compile/JIT failure, missing operator, unsupported
dtype, and unexpected fallback. Pair negative scans with positive gates and
artifact validation; an empty log is not proof of success.
