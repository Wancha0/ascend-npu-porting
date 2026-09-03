# Offline Code Handoff Protocol

Use this protocol when Codex cannot log in to the target or when execution must
be performed by an operator. The goal is to replace an interactive debugging
session with an explicit two-round exchange and machine-verifiable evidence.

## Prefer two rounds

### Round 1: discovery

Send only non-destructive discovery tools and exact commands. Ask the operator
to return their raw JSON outputs and logs without editing them:

```bash
python3 scripts/probe_ascend_runtime.py --output evidence/runtime.json
python3 scripts/scan_npu_risks.py . --output evidence/static-scan.json
```

Also request:

- the immutable source revision and `git status --short`;
- the exact target task and intended process/node count;
- full dependency/version output and the availability of required local
  checkpoint or representative-input paths (availability only; do not transfer
  them as part of this workflow);
- the first failing traceback and the smallest command that reproduces it;
- platform launch conventions, network interface names, and allowed ports.

Do not infer a torch/torch_npu/CANN pairing, device count, compiler support, or
operator behavior from another server.

### Round 2: adaptation bundle

Use the returned discovery data to build a versioned bundle. A useful layout is:

```text
npu-handoff-<project>-<run-id>/
  bundle.json
  MANIFEST.json
  NPU_PORTING.md
  REFERENCES.md
  patches/
  overlay/
  configs/
  scripts/
  probes/
  evidence/
```

Keep `evidence/` empty except for templates. The operator fills it with actual
target results. Include only the directories needed by the project.

## Bundle contract

`bundle.json` should record:

- schema version, bundle ID, creation time, project, and target outcome;
- expected source repository and exact base revision;
- supported operating-system/runtime tuple from discovery;
- patch order and the SHA-256 of every patch;
- changed files/symbols and the CPU/CUDA/NPU dispatch behavior after each
  patch;
- required local checkpoint/input contracts without transport instructions;
- commands and positive gates for each validation stage;
- explicit stop conditions and unverified assumptions.

If performance is in scope, also include an exact baseline command, a bounded
profile command, the frozen workload contract, one-command candidate toggles,
rollback instructions, and a performance-evidence template. Never ask a remote
operator to try an unordered list of tuning flags.

Copy `references/official-links.md` into the bundle as `REFERENCES.md`. In
`NPU_PORTING.md`, add a patch map with one row per logical change:

| Change | File/symbol | Why NPU failed | NPU behavior | CPU/CUDA behavior | Gate | Revert |
|---|---|---|---|---|---|---|

This map is mandatory for a non-author operator: a raw patch alone does not
explain which branch must execute or how to recognize a silent fallback.

Before delivery:

1. Apply the patches in a clean throwaway checkout of the exact base revision.
2. Run syntax/static checks locally where possible.
3. Create `MANIFEST.json` with `scripts/manifest.py create`.
4. Verify that manifest from a copied or unpacked bundle.
5. Check the bundle for credentials, private keys, tokens, signed URLs, large
   weights, datasets, generated caches, and machine-specific absolute paths.

At the destination, the runbook must verify the base revision and dirty state
before applying patches. Use `git apply --check` before `git apply`. If the
checkout differs, stop and return evidence; do not force, reset, or overwrite
local changes.

## Make the runbook executable by a non-author

The runbook must give exact commands, not intentions. For each command record:

- working directory and required user;
- environment activation and Ascend toolkit initialization;
- input paths, output path, process count, port, and run ID;
- expected exit code, positive gate, artifact, and verification command;
- timeout/expected duration and stop conditions;
- cleanup steps that target only the unique run directory or process ID.

Use a fresh run directory, log file, master port, and output directory for each
attempt. Never call a background launch successful just because a PID exists.

If a one-shot handoff is unavoidable, make runtime detection fail closed. The
bundle may choose among documented, already-supported branches, but it must not
install arbitrary versions, rewrite the platform torch pair, guess network
interfaces, or automatically retry with weaker correctness settings.

## Evidence envelope

Each target run should produce one JSON envelope plus referenced logs and
artifacts. `scripts/validate_evidence.py` accepts this minimum form:

```json
{
  "schema_version": 1,
  "project": "example",
  "source_revision": "40-hex-commit-or-immutable-id",
  "target_outcome": "training-engine-ready",
  "status": "pass",
  "gate": "ASCEND_TRAINING_ENGINE_PASS",
  "runtime": {
    "python": "3.10.x",
    "torch": "2.x",
    "torch_npu": "2.x",
    "cann": "8.x",
    "device_count": 8
  },
  "checks": [
    {
      "name": "one_optimizer_update",
      "status": "pass",
      "command": "bash scripts/smoke_train_npu.sh",
      "exit_code": 0,
      "gate": "ASCEND_OPTIMIZER_STEP_PASS",
      "artifacts": ["logs/smoke-train.log"]
    }
  ],
  "artifacts": [
    {
      "path": "logs/smoke-train.log",
      "size_bytes": 1234,
      "sha256": "<64-lowercase-hex>"
    }
  ],
  "failures": []
}
```

For a blocked or failed run, set `status` accordingly, keep the raw failing
check, and describe failures rather than deleting them. Validate returned files
with:

```bash
python3 scripts/validate_evidence.py evidence/result.json \
  --artifact-root evidence/returned
```

A `pass` envelope is invalid if any check failed, any command exited nonzero,
the top-level gate is absent, an artifact does not match its size/hash, or
`failures` is non-empty.

## What can be claimed remotely

- A locally built patch or bundle can be called **prepared**, never validated
  on the target.
- Discovery JSON can prove only what its commands measured.
- A bundle manifest proves code-bundle integrity, not model function.
- Only returned evidence that passes validation can raise readiness level.
- A performance claim additionally requires synchronized, repeated baseline and
  candidate windows on the same workload plus finite/parity and checkpoint
  checks. A trace or reduced memory number alone does not prove a speedup.
- If no evidence returns, state the strongest locally proven result and list
  the exact operator commands still required.
