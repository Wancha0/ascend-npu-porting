# OBS and MoXing for production training

Read after [ModelArts production](modelarts-production.md) when OBS asset transfer is in scope. Keep [the user's backup choice](artifact-backup.md). Reviewed 2026-09-15 against official sources; examples below are application patterns, not an end-to-end cloud-tested adapter.

## Decide who moves each asset

| Mechanism | Contract |
|---|---|
| Platform input/output mapping | Record exact local paths, preparation/sync timing and terminal delivery status. Do not also MoXing-copy the same asset to the same destination. An exit-triggered output transfer cannot be verified from inside a process that has not exited; use an external platform/receipt check before downstream evaluation. |
| Explicit MoXing staging/upload | Job-owned preparation and upload stages control immutable versions, cadence, progress and completion. Do not assume platform collection backs up paths outside its mapping. |
| Shared persistent filesystem | Verify actual mount, lifetime, accessibility on all reader nodes and any OBS synchronization separately. A mount alone does not prove an OBS backup. |

Code, weights, data and features may use different mechanisms. Pick one delivery owner per asset/destination. Public-weight acquisition is a separate preparation task; production should consume pinned durable sources without the development laptop's relay.

## Environment, permission and a small probe

Use the already-working MoXing in the selected image. Record its installed version and interpreter; an import pass is not an OBS access pass. If absent, prepare a training-compatible image with a trusted, pinned MoXing distribution and dependencies before submission, rather than guessing a PyPI package or upgrading the platform torch tuple.

Huawei's [Framework introduction](https://support.huaweicloud.com/usermanual-standard-modelarts/modelarts_11_0001.html) distinguishes ordinary OBS object buckets from parallel filesystems and documents version-specific acceleration. Do not set old MA_MOXING_FWVER values universally, assume every custom image contains MoXing, or use this tutorial on a parallel filesystem without checking compatibility.

Use the job's supported delegated/temporary credentials. A Notebook's successful login does not establish the training job's access. Check actual source read/list and destination write permissions, bucket policy, region/endpoint and credential renewal in the job environment. Never embed AK/SK or dump credentials. Official [job permission practice](https://support.huaweicloud.com/permission-modelarts/modelarts_24_00135.html) covers ListBucket/GetObject/PutObject delegation; extra multipart operations depend on the client and policy.

Within already-authorized output scope, read a small known source object and write/read a tiny uniquely named receipt in the run prefix. Do not test by copying full weights. Do not delete objects merely for cleanup unless authorized. On 403, fix permissions; on missing input, fix the manifest; on transient timeout, retry the exact owned transfer within a configured budget.

## Copy basics

The [MoXing file guide](https://support.huaweicloud.com/intl/zh-cn/develop-modelarts/develop-moxing-0002.html) supports file and directory copies. These are caller-provided paths in the resolved job configuration, not credentials:

```python
import moxing as mox

# One file: OBS to a local staging file.
mox.file.copy(source_obs_file, local_staging_file)
# Directory: OBS to an isolated, not-yet-ready local staging directory.
mox.file.copy_parallel(source_obs_directory, local_staging_directory,
                       threads=4, is_processing=False)
# Upload a frozen directory to a new attempt-specific OBS prefix.
mox.file.copy_parallel(frozen_snapshot_directory, new_obs_prefix,
                       threads=4, is_processing=False)
```

Four threads is only an example starting point. Account for nodes × copy workers × internal multipart concurrency; tune with a small representative transfer. [BrokenPipe guidance](https://support.huaweicloud.com/trouble-modelarts/modelarts_trouble_0042.html) documents excessive concurrency and large-file settings. Do not transplant its tuning values blindly. Copying is not an atomic directory transaction, and copy_parallel must not be advertised as guaranteed byte-offset resume.

## Staging and publishing examples

The following functions demonstrate a file download and a complete-snapshot upload using size checks. They do not implement distributed coordination, a resumable transfer queue, credentials or a timeout supervisor. The caller must provide those lifecycle controls. A same-size corruption will not be detected: use a trusted digest where the selected contract requires content verification. The [official file API](https://github.com/huaweicloud/ModelArts-Lab/blob/master/docs/moxing_api_doc/MoXing_API_File.md) documents get_size/read/write.

```python
import json
import uuid
from pathlib import Path
import moxing as mox

def download_file(obs_uri, destination, expected_bytes):
    # Caller owns this destination; source identity is pinned in the job manifest.
    dst = Path(destination)
    if dst.exists():
        raise FileExistsError("Validate an existing cache before reusing it")
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp = dst.with_name(dst.name + ".part-" + uuid.uuid4().hex)
    mox.file.copy(obs_uri, str(temp))
    if temp.stat().st_size != expected_bytes:
        raise ValueError("Downloaded size mismatch; partial file is not published")
    # Same local filesystem. A single staging owner is required.
    temp.replace(dst)
    return {"source": obs_uri, "local": str(dst),
            "bytes": expected_bytes, "verification": "size_only"}

def publish_snapshot(snapshot_dir, unique_obs_prefix, identity):
    # Caller guarantees a frozen, complete snapshot and one publisher.
    # Prefix must be new and owned by this attempt, never a shared 'latest/'.
    base = Path(snapshot_dir)
    files = sorted(p for p in base.rglob("*") if p.is_file())
    if not files or any(p.is_symlink() for p in base.rglob("*")):
        raise ValueError("Need a nonempty regular-file snapshot")
    prefix = unique_obs_prefix.rstrip("/") + "/"
    if mox.file.exists(prefix):
        raise FileExistsError("Inspect a prior attempt before retrying")
    records = []
    for path in files:
        relative = path.relative_to(base).as_posix()
        if relative == "COMMITTED.json":
            raise ValueError("Reserved receipt name")
        target = prefix + relative
        size = path.stat().st_size
        mox.file.copy(str(path), target)
        if mox.file.get_size(target) != size:
            raise ValueError("Uploaded size mismatch")
        records.append({"path": relative, "bytes": size})
    receipt = {"identity": identity, "files": records,
               "verification": "size_only"}
    marker = prefix + "COMMITTED.json"
    mox.file.write(marker, json.dumps(receipt, sort_keys=True))
    if json.loads(mox.file.read(marker)) != receipt:
        raise ValueError("Receipt readback mismatch")
    return receipt
```

Call download_file for files from a pinned manifest with model/revision, processors and feature-generation identity. Do not call publish_snapshot on the directory actively written by the trainer. For FSDP/ZeRO, gather or persist every required shard and metadata first; a rank-0-only snapshot is insufficient. Supply identity with run/attempt, checkpoint step, ordinary/EMA/resume role and source/config revision.

On an upload exception no valid completion marker is promised. The example deliberately refuses an existing prefix; inspect it and either resume missing files with a manifest-aware implementation or use a new attempt prefix. Never overwrite a committed prefix. A crash during marker publication requires reading/parsing the full marker and validating its referenced objects before acceptance.

For repeated transfers, use a ledger to reuse already verified complete files. Incomplete MoXing files may need to restart from the beginning unless the installed client's resume behavior has been verified. Preserve partial state for inspection; do not turn “exists” or “same total directory bytes” into a cache-valid decision.

## Multi-node and training integration

For node-local caches, stage once **per node**, normally before that node launches workers. Global rank 0 alone cannot fill other nodes' disks. For truly shared storage, elect one writer with a real lock/lease and let all readers validate the same readiness record. A check-then-create OBS object is not a distributed lock.

Use an identity-bound local READY record only after all required files pass the selected checks. Include source manifest identity, expected paths/count, sizes and verification strength. Other readers have bounded waits and a failure signal; no indefinite barrier if staging fails. With platform-per-worker launch, use one local staging owner plus a local lock and bounded failure propagation, not eight parallel copies.

Checkpoint pipeline:
1. Trainer finalizes an immutable snapshot; keep its files until the upload acknowledges them.
2. Job-owned uploader copies to a unique run/attempt/step prefix, verifies files, and publishes COMMITTED.json last.
3. Record local step, durable step, pending bytes, last success and first error. Update any “latest” pointer only after commit; resolve it to an immutable manifest for resume.
4. Apply bounded queue/backpressure before pending snapshots fill local disk. Do not prune unacknowledged files. Deletion of older durable checkpoints follows the user's retention policy.
5. On completion, drain selected uploads within the declared finalization budget. Record training, backup and evaluation status separately.

Avoid per-step remote object reads in the training hot path unless remote streaming was deliberately benchmarked. Stage reusable features to appropriate local/shared storage when feasible. Feature manifests must preserve encoder identity, camera coverage, resolution/token shape, dtype, dataset split and preprocessing; matching filenames are not enough.

## Exit policy must be decided before a non-interactive job

A process cannot promise to retain ephemeral files after the platform removes its container. “Exit nonzero and keep /cache” is therefore **not** a retention solution.

For OBS-selected production runs, propose and record finalization timeout/retry budget and one of: a verified persistent spool for undelivered assets; a supported bounded container-retention period for recovery; or explicit acceptance of a possible unsaved tail. Do not silently choose the last option, wait forever for a human, or grant automatic resubmission authority.

If final delivery fails, preserve the training exit status plus a separate delivery failure; emit a failed overall result when durable delivery was required. Recovery is only possible from bytes that actually survived. Platform retention/retry timing must be verified for that job mode. Periodic committed checkpoints bound the loss window; they do not promise zero loss during a prolonged OBS outage.

For application-managed transfers, an overall success record requires training success and selected artifact commit, plus authorized evaluation success if evaluation is part of this job. For platform-managed post-exit transfer, record application completion first, then require external delivery confirmation before claiming overall completion or triggering consumers.

## Bounded acceptance

Reuse existing evidence and verify only the changed integration: one small input to the actual job-local path; one real checkpoint committed to the chosen destination; and a fresh-job load of that committed checkpoint. Check ordinary/EMA distinction and restore training state if continuation is claimed. Local/mock tests do not prove cloud permissions, termination retention or production recovery.
