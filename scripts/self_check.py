#!/usr/bin/env python3
"""Offline, standard-library self-check for the portable Ascend porting kit."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
from typing import Any
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
REQUIRED = (
    "README.md",
    "SKILL.md",
    "PORTABLE_AGENT_GUIDE.md",
    "assets/training-job/torchrun_npu.sh",
    "references/agent-validation.md",
    "references/compatibility-patterns.md",
    "references/dependency-patch-delivery.md",
    "references/glm-agent.md",
    "references/minimax-h3-lessons.md",
    "references/official-links.md",
    "references/offline-handoff.md",
    "references/porting-workflow.md",
    "references/serving-readiness.md",
    "references/ssh-execution.md",
    "references/training-performance.md",
    "references/training-readiness.md",
    "references/training-job-lifecycle.md",
    "scripts/manifest.py",
    "scripts/probe_ascend_runtime.py",
    "scripts/scan_npu_risks.py",
    "scripts/self_check.py",
    "scripts/ssh_script.py",
    "scripts/validate_evidence.py",
    "scripts/validate_patch_registry.py",
)


def run(argv: list[str], expected_gate: str | None = None) -> str | None:
    try:
        result = subprocess.run(
            argv,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"cannot execute {argv[1]}: {exc}"
    combined = result.stdout + result.stderr
    if result.returncode != 0:
        return f"command failed ({result.returncode}): {' '.join(argv)}\n{combined[-2000:]}"
    if expected_gate is not None and expected_gate not in combined:
        return f"command omitted {expected_gate}: {' '.join(argv)}"
    return None


def run_expected_failure(argv: list[str], expected_text: str) -> str | None:
    try:
        result = subprocess.run(
            argv,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"cannot execute expected-failure check {argv[1]}: {exc}"
    combined = result.stdout + result.stderr
    if result.returncode == 0:
        return f"command unexpectedly succeeded: {' '.join(argv)}"
    if expected_text not in combined:
        return f"failed command omitted {expected_text}: {' '.join(argv)}\n{combined[-2000:]}"
    return None


def local_link_errors() -> list[str]:
    errors: list[str] = []
    for document in sorted(ROOT.rglob("*.md")):
        if ".git" in document.parts:
            continue
        text = document.read_text(encoding="utf-8")
        for raw in LINK_RE.findall(text):
            target = raw.strip().strip("<>").split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (document.parent / target).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                errors.append(f"link escapes toolkit: {document.relative_to(ROOT)} -> {raw}")
                continue
            if not resolved.exists():
                errors.append(f"missing link: {document.relative_to(ROOT)} -> {raw}")
    return errors


def syntax_errors() -> list[str]:
    errors: list[str] = []
    for script in sorted((ROOT / "scripts").glob("*.py")):
        try:
            compile(script.read_text(encoding="utf-8"), str(script), "exec")
        except (OSError, SyntaxError) as exc:
            errors.append(f"invalid Python script {script.name}: {exc}")
    return errors


def ssh_fixture_errors(temp: Path, skipped_checks: list[str]) -> list[str]:
    """Replace SSH before exercising the real helper and local Bash parser."""
    if shutil.which("bash") is None:
        skipped_checks.append("SSH helper behavior unverified: local Bash is unavailable")
        return []
    errors: list[str] = []
    helper = runpy.run_path(str(ROOT / "scripts/ssh_script.py"))
    real_run = subprocess.run
    ssh_calls: list[tuple[list[str], dict[str, Any]]] = []
    ssh_results: list[Any] = []

    def fake_run(argv: list[str], **kwargs: Any) -> Any:
        if argv == ["bash", "-n"]:
            return real_run(argv, **kwargs)
        if argv[0] != "ssh":
            raise AssertionError("unexpected command in SSH self-check")
        ssh_calls.append((argv, kwargs))
        # Execute only the fixed local fixture body, never a network command.
        result = real_run(["bash", "-s"], capture_output=True, timeout=10, **kwargs)
        ssh_results.append(result)
        return result

    def invoke(options: list[str], body: str) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(sys, "argv", ["ssh_script.py"] + options), \
                patch.object(sys, "stdin", io.StringIO(body)), \
                patch.object(sys, "stdout", stdout), \
                patch.object(sys, "stderr", stderr), \
                patch.object(subprocess, "run", side_effect=fake_run):
            try:
                code = helper["main"]()
            except SystemExit as exc:
                code = int(exc.code)
        return code, stdout.getvalue(), stderr.getvalue()

    identity = temp / "identity with spaces"
    identity.write_text("non-key fixture\n", encoding="utf-8")
    private_marker = "private-fixture-do-not-log"
    value = private_marker + " 'quote' \"double\" $HOME $(printf expanded) `printf expanded`; a=b\n第二行"
    options = [
        "--host", "fixture.invalid", "--user", "fixture-user", "--port", "2222",
        "--identity", str(identity), "--connect-timeout", "7",
        "--server-alive-interval", "9", "--host-key-policy", "yes",
        "--env", "ASCEND_SSH_FIXTURE=" + value,
    ]
    expected_argv = [
        "ssh", "-T", "-p", "2222", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=7", "-o", "ServerAliveInterval=9",
        "-o", "StrictHostKeyChecking=yes", "-i", str(identity),
        "fixture-user@fixture.invalid", "bash -s",
    ]
    body = 'printf \'%s\' "$ASCEND_SSH_FIXTURE"\n'
    code, stdout, stderr = invoke(options + ["--dry-run"], body)
    if code != 0 or ssh_calls:
        errors.append("SSH dry-run failed or invoked SSH")
    else:
        metadata = json.loads(stdout)
        if metadata.get("argv") != expected_argv or metadata.get("syntax_checked") is not True:
            errors.append("SSH dry-run lost target/options or syntax-check evidence")
        if metadata.get("environment_names") != ["ASCEND_SSH_FIXTURE"] or private_marker in stdout + stderr:
            errors.append("SSH dry-run exposed environment values or lost their names")

    code, stdout, stderr = invoke(options, body)
    if code != 0 or len(ssh_calls) != 1:
        errors.append("SSH helper did not execute exactly once")
    elif ssh_calls[0][0] != expected_argv or ssh_results[0].stdout != value:
        errors.append("SSH argument array or literal environment round-trip changed")
    if private_marker in stdout + stderr:
        errors.append("SSH helper itself logged an environment value")

    count = len(ssh_calls)
    for syntax_options in (options, options + ["--dry-run"]):
        code, stdout, stderr = invoke(syntax_options, "if then " + private_marker + "\n")
        if code == 0 or len(ssh_calls) != count or "Bash syntax check failed" not in stderr:
            errors.append("invalid Bash reached SSH or omitted its failure")
        if private_marker in stdout + stderr:
            errors.append("SSH syntax-failure diagnostic exposed script/environment content")

    invalid_cases = (
        (["--host=-oProxyCommand=bad"], "true\n"),
        (["--host", "fixture.invalid", "--port", "0"], "true\n"),
        (["--host", "fixture.invalid", "--port", "65536"], "true\n"),
        (["--host", "fixture.invalid", "--identity", str(temp / "absent")], "true\n"),
        (["--host", "fixture.invalid", "--identity", str(temp)], "true\n"),
        (["--host", "fixture.invalid", "--env", "bad-name=" + value], "true\n"),
        (["--host", "fixture.invalid", "--env", private_marker], "true\n"),
        (["--host", "fixture.invalid", "--env", "ASCEND_SSH_FIXTURE=" + private_marker + "\0"], "true\n"),
        (["--host", "fixture.invalid"], "\n"),
        (["--host", "fixture.invalid"], "true\0\n"),
    )
    for invalid_options, invalid_body in invalid_cases:
        code, stdout, stderr = invoke(invalid_options, invalid_body)
        if code == 0 or len(ssh_calls) != count:
            errors.append("invalid SSH input was accepted or invoked SSH")
        if private_marker in stdout + stderr:
            errors.append("SSH input validation exposed an environment value")

    for failure_body, expected_code in (("exit 37\n", 37), ("false\nprintf unexpected\n", 1)):
        count = len(ssh_calls)
        code, _, _ = invoke(["--host", "fixture.invalid"], failure_body)
        if code != expected_code or len(ssh_calls) != count + 1:
            errors.append("SSH helper changed failure status or retried an execution")
        elif ssh_results[-1].stdout:
            errors.append("SSH strict mode continued after a failed command")
    return errors


def fixture_errors(skipped_checks: list[str]) -> list[str]:
    errors: list[str] = []
    for name in (
        "manifest.py",
        "probe_ascend_runtime.py",
        "scan_npu_risks.py",
        "ssh_script.py",
        "validate_evidence.py",
        "validate_patch_registry.py",
    ):
        error = run([sys.executable, str(ROOT / "scripts" / name), "--help"])
        if error:
            errors.append(error)
    bash = shutil.which("bash")
    if bash is not None:
        error = run([bash, "-n", str(ROOT / "assets/training-job/torchrun_npu.sh")])
        if error:
            errors.append(error)
    else:
        skipped_checks.append("torchrun_npu.sh syntax unverified: local Bash is unavailable")
    with tempfile.TemporaryDirectory(prefix="ascend-porting-self-check-") as raw_temp:
        temp = Path(raw_temp)
        errors.extend(ssh_fixture_errors(temp, skipped_checks))
        fixture = temp / "fixture-repo"
        fixture.mkdir()
        (fixture / "train.py").write_text(
            "import torch\nvalue = torch.zeros(1).cuda()\n", encoding="utf-8"
        )
        scan_output = temp / "scan.json"
        error = run(
            [
                sys.executable,
                str(ROOT / "scripts/scan_npu_risks.py"),
                str(fixture),
                "--output",
                str(scan_output),
            ],
            "ASCEND_STATIC_INVENTORY_COMPLETE",
        )
        if error:
            errors.append(error)
        else:
            scan = json.loads(scan_output.read_text(encoding="utf-8"))
            if scan.get("counts_by_category", {}).get("hardcoded-cuda", 0) < 1:
                errors.append("risk scanner missed the hardcoded CUDA fixture")

        payload_root = temp / "payload"
        payload_root.mkdir()
        (payload_root / "sample.txt").write_text("manifest fixture\n", encoding="utf-8")
        (payload_root / ".git").mkdir()
        (payload_root / ".git/config").write_text("private remote fixture\n", encoding="utf-8")
        (payload_root / "__pycache__").mkdir()
        (payload_root / "__pycache__/sample.pyc").write_bytes(b"cache fixture")
        (payload_root / ".DS_Store").write_bytes(b"metadata fixture")
        manifest = temp / "MANIFEST.json"
        for argv, gate in (
            (
                [
                    sys.executable,
                    str(ROOT / "scripts/manifest.py"),
                    "create",
                    str(payload_root),
                    "--output",
                    str(manifest),
                ],
                "ASCEND_MANIFEST_CREATE_PASS",
            ),
            (
                [
                    sys.executable,
                    str(ROOT / "scripts/manifest.py"),
                    "verify",
                    str(manifest),
                    "--root",
                    str(payload_root),
                ],
                "ASCEND_MANIFEST_VERIFY_PASS",
            ),
        ):
            error = run(argv, gate)
            if error:
                errors.append(error)

        if manifest.is_file():
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            manifest_paths = {item.get("path") for item in manifest_payload.get("entries", [])}
            forbidden = {".git/config", "__pycache__/sample.pyc", ".DS_Store"}
            leaked = sorted(forbidden & manifest_paths)
            if leaked:
                errors.append(f"manifest included default-excluded paths: {leaked}")

        unsafe_root = temp / "unsafe-symlink-payload"
        unsafe_root.mkdir()
        try:
            (unsafe_root / "escape").symlink_to("../outside")
        except OSError:
            pass
        else:
            error = run_expected_failure(
                [
                    sys.executable,
                    str(ROOT / "scripts/manifest.py"),
                    "create",
                    str(unsafe_root),
                    "--output",
                    str(temp / "unsafe-manifest.json"),
                ],
                "unsafe symlink target",
            )
            if error:
                errors.append(error)

        source_root = temp / "dependency-source"
        source_file = source_root / "package/device.py"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("DEVICE = 'cuda'\n", encoding="utf-8")
        patch_bundle = temp / "patch-bundle"
        patch_file = patch_bundle / "patches/demo/0001-device.patch"
        patch_file.parent.mkdir(parents=True)
        patch_file.write_text("fixture patch\n", encoding="utf-8")
        patch_registry = {
            "schema_version": 1,
            "project": "self-check-fixture",
            "libraries": [
                {
                    "name": "demo",
                    "source": "https://example.invalid/demo.git",
                    "base_revision": "0" * 40,
                    "license_reference": "Apache-2.0",
                    "target_kind": "source-checkout",
                    "base_files": [
                        {
                            "path": "package/device.py",
                            "sha256": hashlib.sha256(source_file.read_bytes()).hexdigest(),
                        }
                    ],
                    "patches": [
                        {
                            "path": "patches/demo/0001-device.patch",
                            "size_bytes": patch_file.stat().st_size,
                            "sha256": hashlib.sha256(patch_file.read_bytes()).hexdigest(),
                        }
                    ],
                    "apply": "git apply --check PATCH && git apply PATCH",
                    "revert": "git apply --check -R PATCH && git apply -R PATCH",
                    "validation_commands": ["python3 -m pytest tests/test_device.py"],
                }
            ],
        }
        registry_path = patch_bundle / "dependency-patches.json"
        registry_path.write_text(json.dumps(patch_registry), encoding="utf-8")
        error = run(
            [
                sys.executable,
                str(ROOT / "scripts/validate_patch_registry.py"),
                str(registry_path),
                "--bundle-root",
                str(patch_bundle),
                "--source",
                f"demo={source_root}",
                "--require-base",
            ],
            "ASCEND_PATCH_REGISTRY_VALID",
        )
        if error:
            errors.append(error)
        patch_file.write_text("tampered fixture patch\n", encoding="utf-8")
        error = run_expected_failure(
            [
                sys.executable,
                str(ROOT / "scripts/validate_patch_registry.py"),
                str(registry_path),
                "--bundle-root",
                str(patch_bundle),
            ],
            "ASCEND_PATCH_REGISTRY_INVALID",
        )
        if error:
            errors.append(error)

        evidence_root = temp / "returned"
        log = evidence_root / "logs/smoke.log"
        log.parent.mkdir(parents=True)
        log.write_text("ASCEND_RUNTIME_PROBE_PASS\n", encoding="utf-8")
        evidence: dict[str, Any] = {
            "schema_version": 1,
            "project": "self-check-fixture",
            "source_revision": "0" * 40,
            "target_outcome": "runtime-ready",
            "status": "pass",
            "gate": "ASCEND_RUNTIME_PROBE_PASS",
            "runtime": {"fixture": True},
            "checks": [
                {
                    "name": "fixture",
                    "status": "pass",
                    "command": "fixture",
                    "exit_code": 0,
                    "gate": "ASCEND_RUNTIME_PROBE_PASS",
                    "artifacts": ["logs/smoke.log"],
                }
            ],
            "artifacts": [
                {
                    "path": "logs/smoke.log",
                    "size_bytes": log.stat().st_size,
                    "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                }
            ],
            "failures": [],
        }
        evidence_path = temp / "evidence.json"
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        error = run(
            [
                sys.executable,
                str(ROOT / "scripts/validate_evidence.py"),
                str(evidence_path),
                "--artifact-root",
                str(evidence_root),
            ],
            "ASCEND_HANDOFF_EVIDENCE_VALID",
        )
        if error:
            errors.append(error)
    return errors


def main() -> int:
    skipped_checks: list[str] = []
    errors = [f"missing required file: {relative}" for relative in REQUIRED if not (ROOT / relative).is_file()]
    errors.extend(local_link_errors())
    errors.extend(syntax_errors())
    if not errors:
        errors.extend(fixture_errors(skipped_checks))
    result = {
        "status": "pass" if not errors else "fail",
        "toolkit_root": str(ROOT),
        "python": sys.version.split()[0],
        "required_file_count": len(REQUIRED),
        "skipped_checks": skipped_checks,
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        print("ASCEND_SKILL_SELF_CHECK_FAIL", file=sys.stderr)
        return 1
    print("ASCEND_SKILL_SELF_CHECK_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
