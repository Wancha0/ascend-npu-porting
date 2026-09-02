# Inference and Serving Readiness

Serving readiness is more than importing a framework or returning HTTP 200.
Separate backend support, model loading, request execution, output semantics,
and lifecycle/resource cleanup.

## Freeze the serving contract

Record the exact engine/framework commit, Ascend plugin or branch, model and
adapter revisions, request schema, target latency/throughput, parallel factors,
precision, maximum input/output shape, and expected output format. For media or
multimodal systems also freeze tokenizer/processor, scheduler, VAE/vocoder,
frame/audio parameters, and postprocessing tools.

Prefer a framework revision that explicitly supports the target architecture
and Ascend stack. A generic CUDA-to-NPU compatibility layer is useful for
portable PyTorch code but is not proof that CUDA JIT, Triton, custom collectives,
quantization, paged attention, or graph compilation is safe.

## Review process-global initialization

List every runtime that initializes hardware or process-global state: Ascend,
Accelerate, serving platform detection, compiler backends, EGL/OpenGL, media
libraries, and simulation engines. Their required order can differ by project.
Keep the proven order in one bootstrap entry and test it in a fresh process.

Import torch_npu before generic device discovery selects CUDA/CPU. If the
serving framework must establish its own NPU platform object before a
compatibility shim is enabled, preserve that ordering. Graphics runtimes may
also require EGL initialization before Ascend. Do not generalize one project's
order without a cold-process probe.

## Validate topology before the full model

Write down what each parallel factor partitions and verify that their product
matches the intended world size. Exercise the same collectives and layouts
with representative shapes before loading weights. Test at least the final
candidate topology and one useful diagnostic alternative; a topology that
loads may still fail at the first all-to-all or attention call.

For attention/custom operators, reproduce real head counts, packed-sequence
metadata, padding, dtype, and layout. Backend operator constraints can depend
on values that a toy square tensor does not expose. Keep an optimized path
behind an explicit eligibility check and retain a numerically checked eager or
supported-operator fallback.

## Escalating serving gates

1. import engine, plugin, model class, processor, and platform detection;
2. minimal NPU operator and target-topology collective probes;
3. static model/weight inventory and strict shard/index audit;
4. cold load all components, then verify a health/readiness endpoint;
5. smallest real request that executes the main model and decoder;
6. representative request at the intended shape/steps;
7. output semantic validation and, when relevant, media decoding;
8. bounded repeated or concurrent requests under the intended policy;
9. graceful shutdown, port/process cleanup, and NPU release;
10. restart in a fresh process from the same code/runtime and repeat the
    functional smoke.

A readiness endpoint proves only cold load and server control flow. Validate
that the request actually exercised the intended accelerator components rather
than returning a cached, placeholder, CPU-only, or truncated output.

## Output gates

Define task-specific semantics before launching. Examples include finite tensor
shape/dtype checks, non-empty generated tokens, image dimensions/colorspace,
video frame count/duration/fps/codec, audio rate/channels/duration, or response
schema and provenance. Preserve raw output before optional CPU postprocessing;
if postprocessing fails, resume from the validated raw artifact rather than
rerunning the expensive NPU stage.

Hash the request, generation config, model/adapters, raw output, final output,
and logs. File integrity does not replace these functional gates.

## Capacity and lifecycle

Measure cold-load time, first-request time, steady-state time, per-rank peak
HBM, host-memory floor, disk/cache growth, and compile-cache behavior. Establish
a stop line with margin based on the actual device capacity; do not copy a GB
threshold from another NPU type or topology.

Start with automatic retries, speculative features, compilation, caches,
quantization, and optional fast paths disabled unless they are part of the
requested contract. Enable one optimization at a time only after a baseline
functional result, with parity and capacity evidence.

Use a unique run ID, port, log directory, output directory, and compiler cache.
On failure preserve the first error and resource trace. On exit
verify all worker processes, ports, shared-memory objects where applicable, and
NPU allocations are released before another run.

## Completion claims

- **load-ready:** all required components cold-load on the intended topology.
- **inference-ready:** a real request produces a validated task output.
- **service-ready:** bounded intended request behavior, monitoring, shutdown,
  persistence, and restart have also passed.

Report framework/model revisions and the exact tested request boundary with any
claim. Do not imply arbitrary resolution, sequence length, concurrency, LoRA,
quantization, or topology support from one accepted request.
