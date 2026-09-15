# Artifact retention and OBS choice

Apply this to remote production training and substantial feature extraction,
including successor jobs. A code review or short disposable operator test does
not require a storage questionnaire. This defines retention decisions, not an
OBS client implementation or permission to transfer assets.

For authorized ModelArts transfers, use [obs-moxing.md](obs-moxing.md).
Choose a single transfer owner per asset/destination; a platform output mapping
and an explicit uploader must not unknowingly duplicate or overwrite each other.

## Prepare the choice

Before asking, inspect the actual output paths, mount/storage lifecycle and
already-authorized backup receipts. Directory names such as `/cache` and
`/home/ma-user/work`, a login banner, or a detached process do not establish
durability. Record whether storage survives process exit, pod/job deletion and
node release; mark unsupported claims unknown. Pod-local `emptyDir` can disappear
when the pod is removed. SwanLab/W&B scalar curves and a locally saved report
do not back up model weights or feature tensors.

Prepare a compact table: asset category, actual/planned source, existing verified
durable copy, proposed OBS prefix, estimated bytes (or unknown), and cadence.
Distinguish pretrained encoders, trained policy checkpoints, full resume state,
configs/source patches, metrics/evaluation/latency records, raw datasets, visual
and text caches. Reuse matching verified backups; do not reupload them merely
because a new job starts. Never infer an existing backup from an OBS-looking
path in a script, an upload plan, or historical access to a different server.

## Let the user decide

If the current project has an applicable explicit choice, reuse and record it.
Otherwise present the concrete proposal with these choices:

| Choice | Scope and consequence |
|---|---|
| Key artifacts to OBS (recommended) | Configs/source deltas, metrics, evaluation and raw latency records; periodic latest recoverable checkpoint plus final ordinary/EMA checkpoints when produced. Include optimizer/scheduler/scaler/RNG/data progress where supported. No full datasets or feature caches by default. |
| Key artifacts plus selected caches | Above, plus specifically listed pretrained weights, raw data or visual/text caches; show their estimated volume and transfer/storage cost implications before selection. |
| Server-local only | No OBS upload. State the actual storage lifetime and that artifacts on ephemeral storage can be lost on pod removal or node release. |

Propose a specific checkpoint upload cadence using the run's existing save
cadence, and name the authorized bucket/prefix if known. Ask for missing target
or cadence details together with the scope choice. Do not invent a bucket, embed
credentials, or treat silence as a choice. Explain that the question implements
this skill's storage-choice requirement, linking this reference and `SKILL.md`.
Record scope, destination, cadence, retention, verification method, and the
user's decision in the job contract/ledger for subsequent agents and controllers.

Continue independent adaptation, read-only preparation and bounded tests while
the choice is pending. Resolve it before committing to a new long production
run unless the user explicitly says to proceed with that decision pending.
Do not stop an already-authorized running job solely to retrofit this choice;
ask while it continues. This skill edit itself does not authorize an upload.

## Implement and report the selected policy

Use an existing authorized client/SDK and a job-specific OBS prefix. Upload only
atomically completed files or immutable snapshots, never a checkpoint being
written. Keep synchronization independent of SSH or the user's laptop when
available. If a relay or local process is required, disclose the dependency.
Do not automatically delete old checkpoints, caches or OBS objects; define any
retention deletion in the user's selected policy first.

Track per asset: `local_only`, `pending`, `uploading`, `uploaded_unverified`,
`verified`, `failed`, or `excluded_by_choice`. Store source/run/model identity,
step and ordinary/EMA/resume role, exact OBS URI, bytes, upload receipt/time,
verification method/result, and outstanding errors outside ephemeral storage.
An upload process or zero submission exit is not a completed transfer. Verify
the remote object against the chosen contract. For integrity archival, compare
size and trusted checksums, using full readback when required; do not assume a
multipart ETag is a content hash. Honor explicit no-hash instructions and report
metadata-only verification without claiming full content equality. Avoid full
rehashing on every monitoring tick.

For routine production delivery, use the selected identity/count/size and
receipt checks; full object readback is not the default. Record the verification
method in the completion manifest so metadata-only validation is never confused
with content hashing. Reuse trusted checksums already available.

On upload failure, preserve the local artifact, record/report the failure and
resume the same transfer after inspecting its state. Backup failures should not
silently stop training unless the user selected a backup gate. Report training
and backup completion separately, including the latest backed-up step and exact
excluded/pending assets. Before routine cleanup or releasing the source node,
finish selected backups; if impossible, explain the remaining assets before
seeking a release decision. An explicit instruction to release without backup
takes precedence. Never describe successful training or a local save as OBS
backup success.

For non-interactive production jobs, resolve the finalization timeout and
failure policy before launch. “Preserve locally” only applies while that storage
survives: exiting nonzero does not protect /cache from platform cleanup. Use the
verified persistent spool or bounded platform-retention policy in the job
contract, or record explicit acceptance of the unsaved tail. Do not wait for a
human at the end of an unattended job, loop forever, or report a required
delivery as successful after failure. See the exit policy in obs-moxing.md.
