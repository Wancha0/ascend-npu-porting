#!/usr/bin/env python3
"""Offline, standard-library self-check for the portable Ascend porting kit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
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
    "references/compatibility-patterns.md",
    "references/glm-agent.md",
    "references/official-links.md",
    "references/offline-handoff.md",
    "references/porting-workflow.md",
    "references/serving-readiness.md",
    "references/training-performance.md",
    "references/training-readiness.md",
    "scripts/manifest.py",
    "scripts/probe_ascend_runtime.py",
    "scripts/scan_npu_risks.py",
    "scripts/validate_evidence.py",
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
    ):
        error = run([sys.executable, str(ROOT / "scripts" / name), "--help"])
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
