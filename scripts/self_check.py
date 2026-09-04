#!/usr/bin/env python3
"""Offline, standard-library self-check for the portable Ascend porting kit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
REQUIRED = (
    "README.md",
    "SKILL.md",
    "PORTABLE_AGENT_GUIDE.md",
    "assets/training-job/torchrun_npu.sh",
    "references/compatibility-patterns.md",
    "references/dependency-patch-delivery.md",
    "references/glm-agent.md",
    "references/official-links.md",
    "references/offline-handoff.md",
    "references/porting-workflow.md",
    "references/serving-readiness.md",
    "references/training-performance.md",
    "references/training-readiness.md",
    "references/training-job-lifecycle.md",
    "scripts/manifest.py",
    "scripts/probe_ascend_runtime.py",
    "scripts/scan_npu_risks.py",
    "scripts/self_check.py",
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


def fixture_errors() -> list[str]:
    errors: list[str] = []
    for name in (
        "manifest.py",
        "probe_ascend_runtime.py",
        "scan_npu_risks.py",
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
    with tempfile.TemporaryDirectory(prefix="ascend-porting-self-check-") as raw_temp:
        temp = Path(raw_temp)
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
    errors = [f"missing required file: {relative}" for relative in REQUIRED if not (ROOT / relative).is_file()]
    errors.extend(local_link_errors())
    errors.extend(syntax_errors())
    if not errors:
        errors.extend(fixture_errors())
    result = {
        "status": "pass" if not errors else "fail",
        "toolkit_root": str(ROOT),
        "python": sys.version.split()[0],
        "required_file_count": len(REQUIRED),
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
