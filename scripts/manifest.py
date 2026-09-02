#!/usr/bin/env python3
"""Create or verify a stable SHA-256 manifest for a directory tree."""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
from typing import Any, Iterable


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


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


def excluded(relative: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(relative, pattern) for pattern in patterns)


def inventory(
    root: Path, excludes: list[str], omitted_absolute: Path | None = None
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        relative_dir = current_path.relative_to(root).as_posix()
        kept_dirs: list[str] = []
        for directory in sorted(dirs):
            candidate = current_path / directory
            relative = candidate.relative_to(root).as_posix()
            if excluded(relative, excludes):
                continue
            if candidate.is_symlink():
                info = candidate.lstat()
                entries.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "target": os.readlink(candidate),
                        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
                    }
                )
            else:
                kept_dirs.append(directory)
        dirs[:] = kept_dirs
        for name in sorted(files):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if excluded(relative, excludes):
                continue
            try:
                if omitted_absolute is not None and path.resolve() == omitted_absolute:
                    continue
            except OSError:
                pass
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                entries.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "target": os.readlink(path),
                        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
                    }
                )
            elif stat.S_ISREG(info.st_mode):
                entries.append(
                    {
                        "path": relative,
                        "type": "file",
                        "size_bytes": info.st_size,
                        "sha256": sha256_file(path),
                        "mode": f"{stat.S_IMODE(info.st_mode):04o}",
                    }
                )
    entries.sort(key=lambda item: item["path"])
    return entries


def safe_relative(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        return None
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        return None
    return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    create = subparsers.add_parser("create", help="create a manifest")
    create.add_argument("root", type=Path)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--exclude", action="append", default=[], metavar="GLOB")

    verify = subparsers.add_parser("verify", help="verify a manifest")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--root", type=Path, help="defaults to the manifest directory")
    verify.add_argument("--allow-extra", action="store_true")
    return parser.parse_args()


def create_manifest(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    entries = inventory(root, args.exclude, output)
    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "root_name": root.name,
        "hash_algorithm": "sha256",
        "excludes": args.exclude,
        "entry_count": len(entries),
        "total_file_bytes": sum(item.get("size_bytes", 0) for item in entries),
        "entries": entries,
    }
    atomic_json(output, payload)
    result = {
        "status": "pass",
        "manifest": str(output),
        "entry_count": len(entries),
        "total_file_bytes": payload["total_file_bytes"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    print("ASCEND_MANIFEST_CREATE_PASS")
    return 0


def verify_manifest(args: argparse.Namespace) -> int:
    manifest_path = args.manifest.expanduser().resolve()
    root = (args.root or manifest_path.parent).expanduser().resolve()
    errors: list[str] = []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read manifest: {exc}", file=sys.stderr)
        return 2
    if payload.get("schema_version") != 1 or not isinstance(payload.get("entries"), list):
        print("unsupported or malformed manifest", file=sys.stderr)
        return 2
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    expected_paths: set[str] = set()
    for item in payload["entries"]:
        if not isinstance(item, dict):
            errors.append("manifest entry is not an object")
            continue
        relative = safe_relative(item.get("path"))
        if relative is None:
            errors.append(f"unsafe path: {item.get('path')!r}")
            continue
        if relative in expected_paths:
            errors.append(f"duplicate path: {relative}")
            continue
        expected_paths.add(relative)
        path = root.joinpath(*PurePosixPath(relative).parts)
        try:
            path.parent.resolve(strict=False).relative_to(root.resolve())
        except (OSError, ValueError):
            errors.append(f"path escapes root through a symlink: {relative}")
            continue
        try:
            info = path.lstat()
        except OSError as exc:
            errors.append(f"missing/unreadable {relative}: {exc}")
            continue
        expected_type = item.get("type")
        actual_mode = f"{stat.S_IMODE(info.st_mode):04o}"
        if item.get("mode") != actual_mode:
            errors.append(f"mode mismatch {relative}: expected {item.get('mode')}, got {actual_mode}")
        if expected_type == "file":
            if not stat.S_ISREG(info.st_mode):
                errors.append(f"type mismatch {relative}: expected file")
                continue
            if item.get("size_bytes") != info.st_size:
                errors.append(
                    f"size mismatch {relative}: expected {item.get('size_bytes')}, got {info.st_size}"
                )
                continue
            actual_hash = sha256_file(path)
            if item.get("sha256") != actual_hash:
                errors.append(f"sha256 mismatch {relative}")
        elif expected_type == "symlink":
            if not stat.S_ISLNK(info.st_mode):
                errors.append(f"type mismatch {relative}: expected symlink")
                continue
            target = os.readlink(path)
            if item.get("target") != target:
                errors.append(f"symlink target mismatch {relative}")
        else:
            errors.append(f"unsupported entry type for {relative}: {expected_type!r}")

    if not args.allow_extra:
        try:
            excludes = payload.get("excludes", [])
            if not isinstance(excludes, list) or not all(isinstance(item, str) for item in excludes):
                excludes = []
                errors.append("manifest excludes field is malformed")
            actual_entries = inventory(root, excludes, manifest_path)
            actual_paths = {item["path"] for item in actual_entries}
            for relative in sorted(actual_paths - expected_paths):
                errors.append(f"unexpected path: {relative}")
        except OSError as exc:
            errors.append(f"cannot enumerate extra paths: {exc}")

    result = {
        "status": "pass" if not errors else "fail",
        "manifest": str(manifest_path),
        "root": str(root),
        "checked_entries": len(expected_paths),
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if errors:
        print("ASCEND_MANIFEST_VERIFY_FAIL", file=sys.stderr)
        return 1
    print("ASCEND_MANIFEST_VERIFY_PASS")
    return 0


def main() -> int:
    args = parse_args()
    if args.action == "create":
        return create_manifest(args)
    return verify_manifest(args)


if __name__ == "__main__":
    raise SystemExit(main())
