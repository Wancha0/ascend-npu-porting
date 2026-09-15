# Portable SSH script execution

Use the bundled [scripts/ssh_script.py](../scripts/ssh_script.py) for multiline
remote Bash. It requires Python 3.9+, local Bash for syntax validation, an SSH
client for execution, and Bash on the target. It has no dependency on a user's
skill directory or agent host. Supply the verified SSH target, user, port, and
identity as appropriate for the current contract; none is inferred from an
earlier run.

From the toolkit root, this example only validates and prints metadata:

```bash
python3 scripts/ssh_script.py \
  --host example.invalid --user operator --env RUN_ID=probe-001 \
  --dry-run <<'REMOTE'
printf 'run=%s\n' "$RUN_ID"
REMOTE
```

Replace the example target and remove `--dry-run` only for the intended,
authorized execution. For a script already saved on disk, use
`python3 scripts/ssh_script.py --host TARGET < /absolute/path/to/probe.sh`.
There is no connection during dry-run: the helper runs local `bash -n` and
prints the SSH argument array, payload byte count/hash, and environment names.
It does not print the script or environment values. Syntax errors stop before
SSH and report a failure without echoing potentially sensitive source lines.

The helper passes an argument array to the local SSH client, sends the script
through stdin, and explicitly selects remote `bash -s`. Each `--env NAME=VALUE`
is exported in the payload with one shell-quoting layer; spaces, quotes,
newlines, and shell metacharacters remain literal values. It adds
`set -euo pipefail` by default; `--no-strict` omits that prefix when the script
already manages its own failure handling. Syntax validation is always applied.
`--host-key-policy` defaults to `accept-new`; select `yes` when the target must
already be in known hosts. Authentication uses SSH configuration, agent, or an
explicit existing `--identity` file. `--env` is an argument visible to local
process inspection, so do not use it to carry secrets; use the target's
authorized credential mechanism. The executed script's stdout/stderr pass
through unchanged, so its own diagnostics must also avoid secrets.

When Python generates a complex script, create the Python source file with the
host's file-editing capability (`apply_patch` is one option when available),
then have that program write the `.sh` file. Do not wrap a
Python generator containing another heredoc inside a fresh shell command
string. Python can invoke the helper directly with an argument array and file
contents as `input`, without adding a shell layer:

```python
result = subprocess.run(
    [sys.executable, str(toolkit / "scripts/ssh_script.py"),
     "--host", target_host, "--env", "RUN_ID=" + run_id],
    input=script_path.read_text(encoding="utf-8"),
    text=True, encoding="utf-8", check=False,
)
gate_exit = result.returncode
```

For an authorized source or patch transfer using an encoded payload, generate
the payload, byte count, and SHA-256 mechanically from the exact source bytes.
Derive old-file and replacement-file identities from their respective frozen
files; do not hand-transcribe base64 or reconstruct sizes/hashes from memory.
Before transfer, locally decode the final payload with the intended decoder,
compare the decoded bytes with the source byte for byte, and recompute its size
and hash. Verify the received bytes against that same generated record before
applying them.

Verify exact paths and the run identity before mutation. The helper preserves
SSH's exit status and performs no retry, job scheduling, background launch, or
artifact verification. A zero exit still needs the expected gate/artifact from
the requested operation. After a failure or uncertain disconnect, inspect the
owned process/session and artifacts before retrying. For a deployment failure,
preserve the log and use a new attempt after correcting the evidenced cause;
do not guess expected values or weaken identity checks to continue. A connection
timeout or keepalive interval is not a job runtime limit; apply the existing run
contract's bound and stop conditions separately.

`python3 scripts/self_check.py` exercises this helper using a mocked SSH call
and local fixture scripts when local Bash is available. It checks literal
environment transport, argument boundaries, dry-run, validation failures, and
exit preservation without any
network connection or NPU work. Without local Bash, the core Python self-check
still runs, but its `skipped_checks` explicitly marks SSH behavior unverified.
The SSH helper itself requires Bash and never bypasses syntax validation.
