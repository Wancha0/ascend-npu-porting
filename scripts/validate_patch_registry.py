#!/usr/bin/env python3
"""Validate a multi-library source-patch registry and its hash guards."""

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
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
TARGET_KINDS = {"source-checkout", "installed-python", "overlay-only"}


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


def parse_source_mappings(raw_items: list[str]) -> tuple[dict[str, Path], list[str]]:
    mappings: dict[str, Path] = {}
    errors: list[str] = []
    for raw in raw_items:
        if "=" not in raw:
            errors.append(f"invalid --source mapping {raw!r}; expected NAME=/absolute/path")
            continue
        name, raw_path = raw.split("=", 1)
        if not NAME_RE.fullmatch(name):
            errors.append(f"invalid source name: {name!r}")
            continue
        if name in mappings:
            errors.append(f"duplicate source mapping: {name}")
            continue
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            errors.append(f"source root is not a directory for {name}: {path}")
            continue
        mappings[name] = path
    return mappings, errors


def regular_file(root: Path, relative: str) -> tuple[Path | None, str | None]:
    path = root.joinpath(*PurePosixPath(relative).parts)
    try:
        path.parent.resolve(strict=False).relative_to(root.resolve())
    except (OSError, ValueError):
        return None, f"path escapes root through a symlink: {relative}"
    try:
        if path.is_symlink() or not path.is_file():
            return None, f"not a regular file: {relative}"
    except OSError as exc:
        return None, f"unreadable file {relative}: {exc}"
    return path, None


def validate(
    payload: Any,
    bundle_root: Path,
    source_roots: dict[str, Path],
    require_base: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return {"status": "invalid", "errors": ["top-level registry must be an object"]}
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not nonempty_string(payload.get("project")):
        errors.append("project must be a non-empty string")

    libraries = payload.get("libraries")
    if not isinstance(libraries, list) or not libraries:
        errors.append("libraries must be a non-empty list")
        libraries = []

    names: set[str] = set()
    patch_paths: set[str] = set()
    checked_patches = 0
    verified_base_files = 0
    for index, library in enumerate(libraries):
        prefix = f"libraries[{index}]"
        if not isinstance(library, dict):
            errors.append(f"{prefix} must be an object")
            continue
        name = library.get("name")
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            errors.append(f"{prefix}.name must match {NAME_RE.pattern}")
            continue
        if name in names:
            errors.append(f"duplicate library name: {name}")
        names.add(name)
        for field in ("source", "base_revision", "license_reference", "apply", "revert"):
            if not nonempty_string(library.get(field)):
                errors.append(f"{prefix}.{field} must be a non-empty string")
        if library.get("target_kind") not in TARGET_KINDS:
            errors.append(f"{prefix}.target_kind must be one of {sorted(TARGET_KINDS)}")
        commands = library.get("validation_commands")
        if not isinstance(commands, list) or not commands or not all(
            nonempty_string(item) for item in commands
        ):
            errors.append(f"{prefix}.validation_commands must contain non-empty strings")

        base_files = library.get("base_files")
        if not isinstance(base_files, list) or not base_files:
            errors.append(f"{prefix}.base_files must be a non-empty list")
            base_files = []
        source_root = source_roots.get(name)
        if require_base and source_root is None:
            errors.append(f"missing required --source mapping for {name}")
        seen_base: set[str] = set()
        for item_index, item in enumerate(base_files):
            item_prefix = f"{prefix}.base_files[{item_index}]"
            if not isinstance(item, dict):
                errors.append(f"{item_prefix} must be an object")
                continue
            relative = safe_relative(item.get("path"))
            digest = item.get("sha256")
            if relative is None:
                errors.append(f"{item_prefix}.path is unsafe")
                continue
            if relative in seen_base:
                errors.append(f"duplicate base file for {name}: {relative}")
            seen_base.add(relative)
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                errors.append(f"{item_prefix}.sha256 must be lowercase SHA-256")
                continue
            if source_root is not None:
                path, error = regular_file(source_root, relative)
                if error:
                    errors.append(f"{name} base {error}")
                elif path is not None and sha256_file(path) != digest:
                    errors.append(f"base sha256 mismatch for {name}: {relative}")
                else:
                    verified_base_files += 1

        patches = library.get("patches")
        if not isinstance(patches, list) or not patches:
            errors.append(f"{prefix}.patches must be a non-empty list")
            patches = []
        for item_index, item in enumerate(patches):
            item_prefix = f"{prefix}.patches[{item_index}]"
            if not isinstance(item, dict):
                errors.append(f"{item_prefix} must be an object")
                continue
            relative = safe_relative(item.get("path"))
            digest = item.get("sha256")
            size = item.get("size_bytes")
            if relative is None:
                errors.append(f"{item_prefix}.path is unsafe")
                continue
            if relative in patch_paths:
                errors.append(f"duplicate patch path: {relative}")
            patch_paths.add(relative)
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
                errors.append(f"{item_prefix}.sha256 must be lowercase SHA-256")
            if not isinstance(size, int) or size < 0:
                errors.append(f"{item_prefix}.size_bytes must be a non-negative integer")
            path, error = regular_file(bundle_root, relative)
            if error:
                errors.append(f"patch {error}")
                continue
            if path is not None:
                checked_patches += 1
                actual_size = path.stat().st_size
                if isinstance(size, int) and actual_size != size:
                    errors.append(f"patch size mismatch: {relative}")
                if isinstance(digest, str) and SHA256_RE.fullmatch(digest):
                    if sha256_file(path) != digest:
                        errors.append(f"patch sha256 mismatch: {relative}")

    for name in sorted(source_roots.keys() - names):
        errors.append(f"--source mapping has no registry library: {name}")

    return {
        "status": "valid" if not errors else "invalid",
        "project": payload.get("project"),
        "library_count": len(libraries),
        "checked_patch_count": checked_patches,
        "verified_base_file_count": verified_base_files,
        "base_verification_required": require_base,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    parser.add_argument("--bundle-root", type=Path, help="defaults to the registry directory")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=/ABSOLUTE/PATH",
        help="verify base-file hashes for one library source root",
    )
    parser.add_argument(
        "--require-base",
        action="store_true",
        help="fail unless every library has a --source mapping",
    )
    parser.add_argument("--output", type=Path, help="write validation result atomically")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry_path = args.registry.expanduser().resolve()
    bundle_root = (args.bundle_root or registry_path.parent).expanduser().resolve()
    if not bundle_root.is_dir():
        print(f"bundle root is not a directory: {bundle_root}", file=sys.stderr)
        return 2
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read registry: {exc}", file=sys.stderr)
        return 2
    source_roots, mapping_errors = parse_source_mappings(args.source)
    result = validate(payload, bundle_root, source_roots, args.require_base)
    result["errors"] = [*mapping_errors, *result["errors"]]
    if result["errors"]:
        result["status"] = "invalid"
    if args.output:
        atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] == "valid":
        print("ASCEND_PATCH_REGISTRY_VALID")
        return 0
    print("ASCEND_PATCH_REGISTRY_INVALID", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
