#!/usr/bin/env python3
"""Validate an offline Ascend run evidence envelope and its artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GATE_RE = re.compile(r"^[A-Z][A-Z0-9_]{3,127}$")
STATUSES = {"pass", "fail", "blocked"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def safe_relative(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        return None
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        return None
    return path.as_posix()


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output", type=Path, help="write validation result atomically")
    return parser.parse_args()


def validate(payload: Any, artifact_root: Path | None) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return {"status": "invalid", "errors": ["top-level evidence must be an object"]}

    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    for field in ("project", "source_revision", "target_outcome"):
        if not nonempty_string(payload.get(field)):
            errors.append(f"{field} must be a non-empty string")
    status = payload.get("status")
    if status not in STATUSES:
        errors.append(f"status must be one of {sorted(STATUSES)}")
    gate = payload.get("gate")
    if not nonempty_string(gate) or not GATE_RE.fullmatch(gate):
        errors.append("gate must be a non-empty uppercase machine-readable token")
    if not isinstance(payload.get("runtime"), dict) or not payload.get("runtime"):
        errors.append("runtime must be a non-empty object")

    checks = payload.get("checks")
    if not isinstance(checks, list) or not checks:
        errors.append("checks must be a non-empty list")
        checks = []
    check_artifacts: set[str] = set()
    pass_checks = 0
    for index, check in enumerate(checks):
        prefix = f"checks[{index}]"
        if not isinstance(check, dict):
            errors.append(f"{prefix} must be an object")
            continue
        for field in ("name", "command", "gate"):
            if not nonempty_string(check.get(field)):
                errors.append(f"{prefix}.{field} must be a non-empty string")
        if nonempty_string(check.get("gate")) and not GATE_RE.fullmatch(check["gate"]):
            errors.append(f"{prefix}.gate must be an uppercase machine-readable token")
        check_status = check.get("status")
        if check_status not in STATUSES:
            errors.append(f"{prefix}.status is invalid")
        if not isinstance(check.get("exit_code"), int):
            errors.append(f"{prefix}.exit_code must be an integer")
        if check_status == "pass" and check.get("exit_code") == 0:
            pass_checks += 1
        references = check.get("artifacts")
        if not isinstance(references, list):
            errors.append(f"{prefix}.artifacts must be a list")
            continue
        for raw in references:
            relative = safe_relative(raw)
            if relative is None:
                errors.append(f"{prefix}.artifacts contains an unsafe path: {raw!r}")
            else:
                check_artifacts.add(relative)

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("artifacts must be a non-empty list")
        artifacts = []
    declared_artifacts: set[str] = set()
    for index, artifact in enumerate(artifacts):
        prefix = f"artifacts[{index}]"
        if not isinstance(artifact, dict):
            errors.append(f"{prefix} must be an object")
            continue
        relative = safe_relative(artifact.get("path"))
        if relative is None:
            errors.append(f"{prefix}.path is unsafe")
            continue
        if relative in declared_artifacts:
            errors.append(f"duplicate artifact path: {relative}")
        declared_artifacts.add(relative)
        size = artifact.get("size_bytes")
        digest = artifact.get("sha256")
        if not isinstance(size, int) or size < 0:
            errors.append(f"{prefix}.size_bytes must be a non-negative integer")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            errors.append(f"{prefix}.sha256 must be 64 lowercase hexadecimal characters")
        if artifact_root is not None:
            path = artifact_root.joinpath(*PurePosixPath(relative).parts)
            try:
                path.parent.resolve(strict=False).relative_to(artifact_root.resolve())
            except (OSError, ValueError):
                errors.append(f"artifact path escapes root through a symlink: {relative}")
                continue
            try:
                info = path.lstat()
                if not path.is_file() or path.is_symlink():
                    errors.append(f"artifact is not a regular file: {relative}")
                else:
                    if isinstance(size, int) and info.st_size != size:
                        errors.append(
                            f"artifact size mismatch {relative}: expected {size}, got {info.st_size}"
                        )
                    if isinstance(digest, str) and SHA256_RE.fullmatch(digest):
                        if sha256_file(path) != digest:
                            errors.append(f"artifact sha256 mismatch: {relative}")
            except OSError as exc:
                errors.append(f"artifact missing/unreadable {relative}: {exc}")

    for relative in sorted(check_artifacts - declared_artifacts):
        errors.append(f"check references undeclared artifact: {relative}")

    failures = payload.get("failures")
    if not isinstance(failures, list):
        errors.append("failures must be a list")
        failures = []

    if status == "pass":
        if pass_checks != len(checks):
            errors.append("pass evidence requires every check to pass with exit_code 0")
        if failures:
            errors.append("pass evidence requires an empty failures list")
        if check_artifacts != declared_artifacts:
            errors.append("pass evidence requires every declared artifact to be referenced by a check")
    elif status in {"fail", "blocked"} and not failures:
        errors.append(f"{status} evidence requires at least one failure description")

    return {
        "status": "valid" if not errors else "invalid",
        "evidence_status": status,
        "project": payload.get("project"),
        "target_outcome": payload.get("target_outcome"),
        "check_count": len(checks),
        "artifact_count": len(artifacts),
        "artifacts_verified": artifact_root is not None,
        "errors": errors,
    }


def main() -> int:
    args = parse_args()
    evidence_path = args.evidence.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve() if args.artifact_root else None
    if artifact_root is not None and not artifact_root.is_dir():
        print(f"artifact root is not a directory: {artifact_root}", file=sys.stderr)
        return 2
    try:
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read evidence: {exc}", file=sys.stderr)
        return 2
    result = validate(payload, artifact_root)
    if args.output:
        atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] == "valid":
        print("ASCEND_HANDOFF_EVIDENCE_VALID")
        return 0
    print("ASCEND_HANDOFF_EVIDENCE_INVALID", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
