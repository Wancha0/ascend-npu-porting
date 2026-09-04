# Official and Primary References

Use these links to resolve version, operator, precision, memory, and distributed
questions. Prefer the document branch/version matching the installed runtime;
do not apply commands from the newest page to an older CANN/torch_npu stack.
Record the exact URL, branch/tag, and retrieval date in `NPU_PORTING.md`.

## Ascend runtime and API sources

- [TorchNPU repository and quick start](https://github.com/Ascend/pytorch):
  source of truth for installation entrypoints, runtime behavior, issues, and
  the implementation of `torch_npu`.
- [TorchNPU version compatibility table](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md):
  check the PyTorch, TorchNPU, CANN, Python, driver, and firmware tuple before
  installing or replacing anything.
- [TorchNPU Chinese documentation portal](https://www.hiascend.com/document/detail/zh/Pytorch/2610/index/index.html):
  navigate from the matching installed version to native API support, custom
  APIs, environment variables, model migration, troubleshooting, and release
  notes. Change `2610` only after confirming the target documentation version.
- [TorchNPU custom API reference](https://ascend.github.io/docs/sources/pytorch/api_doc.html):
  inspect supported dtypes, shapes, layouts, products, and constraints for
  NPU-specific operators.
- [`transfer_to_npu` compatibility shim source](https://github.com/Ascend/pytorch/blob/master/torch_npu/contrib/transfer_to_npu.py):
  read the actual monkey-patches before relying on CUDA-like attributes or API
  rewrites. This is especially relevant to `.is_cuda` and CUDA-JIT routing.
- [`PYTORCH_NPU_ALLOC_CONF` reference](https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/memory_management/PYTORCH_NPU_ALLOC_CONF.md):
  consult only after measuring an actual allocator or fragmentation problem.
- [HCCL process-group parameter example](https://github.com/Ascend/pytorch/blob/master/docs/zh/developer_notes/distributed/parameter_setting/setting_HCCL_communicator_parameter.md):
  reference for explicit HCCL process-group configuration. Do not copy buffer
  or timeout values without measuring the target topology.

## Profiling and performance sources

- [Ascend msProf quick start](https://www.hiascend.com/document/detail/en/mindstudio/2600/TITools/msProf/docs/en/getting_started/quick_start.md):
  official collection workflow for host and device performance traces. Use the
  documentation version matching the installed toolkit.
- [msprof-analyze quick start](https://github.com/Ascend/msprof-analyze/blob/master/docs/zh/quick_start/msprof-analyze_quick_start.md):
  official analysis-tool entrypoint for collected profiling data.
- [msprof-analyze operator MFU guidance](https://github.com/Ascend/msprof-analyze/blob/master/docs/en/advanced_features/operator_mfu_instruct.md):
  reference for interpreting operator utilization; MFU is diagnostic evidence,
  not a substitute for end-to-end throughput.
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html):
  reference for scheduled traces, activities, shapes, memory, and stack capture.
  Confirm TorchNPU support in the installed version before enabling options.

## Precision and framework semantics

- [Ascend msProbe](https://github.com/Ascend/msprobe): official tool for
  collecting and comparing CPU/GPU/NPU activations, gradients, and operator
  precision when simple parity tests cannot locate the first divergence.
- [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html):
  reference for device-aware autocast and gradient-scaling semantics.
- [PyTorch distributed documentation](https://docs.pytorch.org/docs/stable/distributed.html):
  reference for process groups, collectives, rank/world-size semantics, and
  cleanup. Use `hccl` where required by the installed TorchNPU stack.
- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html):
  reference for DDP construction, gradient buckets, unused parameters, and
  optimizer interaction.
- [PyTorch `torchrun` documentation](https://docs.pytorch.org/docs/stable/elastic/run.html):
  primary reference for standalone and multi-node launcher arguments, rank
  environment, rendezvous, and failure behavior.
- [Accelerate NPU guide](https://huggingface.co/docs/accelerate/usage_guides/npu):
  framework-maintained NPU setup guidance. Confirm its release matches the
  installed Accelerate and TorchNPU tuple before generating a job config.

## Source-project case studies

- [DreamWAM source](https://github.com/hustvl/DreamWAM) and
  [paper](https://arxiv.org/abs/2608.04996): useful for tracing a multi-encoder,
  video/action training graph with Accelerate, custom preprocessing, and strict
  checkpoint contracts.
- [FastWAM source](https://github.com/yuantianyuan01/FastWAM): useful for
  tracing Hydra configuration, DeepSpeed/torchrun entrypoints, optional compile
  paths, VideoDiT/ActionDiT coupling, and evaluation workers.

The GLM guide also records an audited ActionWM composite-source snapshot. It is
an illustrative failure pattern, not an availability dependency or an
authoritative source link; the public workflow must remain usable when that
external repository or its branches are unavailable.

## GLM coding-agent sources

- [Z.AI Coding Plan quick start](https://docs.z.ai/devpack/quick-start): official
  setup entrypoint showing that GLM is used through multiple coding-agent hosts.
- [Z.AI coding-tool integrations](https://docs.z.ai/devpack/tool/others):
  official host-specific setup pages; use them to distinguish model access from
  the host's file, shell, and editing capabilities.
- [Official GLM skills repository](https://github.com/zai-org/GLM-skills):
  primary example of portable directory skills with `SKILL.md`. Its presence
  does not prove that every GLM host auto-discovers arbitrary local skills.

These projects are architectural examples, not universal NPU patches. Re-run
static inventory and representative operator gates for every new repository and
runtime tuple.
