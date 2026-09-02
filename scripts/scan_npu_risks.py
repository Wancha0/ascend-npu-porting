#!/usr/bin/env python3
"""Static inventory for common CUDA-to-Ascend porting risks.

This scanner is deliberately heuristic. Its output is a review queue, not a
claim that a line is wrong or that unreported code is safe.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterable


SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "site-packages",
}

TEXT_SUFFIXES = {
    ".cfg",
    ".cmake",
    ".ini",
    ".json",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

TEXT_NAMES = {
    "CMakeLists.txt",
    "Dockerfile",
    "Makefile",
    "environment.yml",
    "pyproject.toml",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
}

PATTERNS = [
    ("hardcoded-cuda", "high", re.compile(r"(?:\.cuda\s*\(|torch\.cuda\b|['\"]cuda(?::\d+)?['\"])")),
    ("cuda-identity", "high", re.compile(r"\.is_cuda\b")),
    ("cuda-build", "high", re.compile(r"(?:CUDA_HOME|\bnvcc\b|CUDAExtension|load_inline|cpp_extension)")),
    ("custom-kernel", "high", re.compile(r"(?:\btriton\b|xformers|flash[_-]?attn|bitsandbytes|\bapex\b)")),
    ("complex-or-fp64", "medium", re.compile(r"(?:complex128|complex64|torch\.complex|float64|torch\.double|\.double\s*\()")),
    ("autocast", "medium", re.compile(r"(?:autocast|GradScaler|amp\.)")),
    ("checkpoint-load", "medium", re.compile(r"(?:torch\.load\s*\(|mmap\s*=\s*True|weights_only\s*=)")),
    ("optimizer-memory", "medium", re.compile(r"(?:AdamW|foreach\s*=|fused\s*=)")),
    ("distributed", "medium", re.compile(r"(?:DistributedDataParallel|init_process_group|torchrun|Accelerator\s*\()")),
    ("compile-or-jit", "medium", re.compile(r"(?:torch\.compile|torch\.jit|torchscript)")),
    (
        "serving-backend",
        "medium",
        re.compile(r"(?:\bvllm\b|\bsglang\b|torchair|torch_aie|Triton-Ascend|triton_ascend)"),
    ),
    ("device-transfer", "low", re.compile(r"(?:\.to\s*\(|map_location\s*=|pin_memory\s*=|non_blocking\s*=)")),
    ("device-selection", "low", re.compile(r"(?:LOCAL_RANK|RANK|WORLD_SIZE|set_device|current_device)")),
]

ENTRYPOINT_NAMES = {
    "main.py",
    "run.py",
    "serve.py",
    "train.py",
    "trainer.py",
    "inference.py",
    "evaluate.py",
}

DEPENDENCY_NAMES = {
    "environment.yml",
    "pyproject.toml",
    "requirements.txt",
    "setup.cfg",
    "setup.py",
}

REFERENCE_LINKS = {
    "torch_npu": "https://github.com/Ascend/pytorch",
    "version_compatibility": "https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md",
    "api_reference": "https://ascend.github.io/docs/sources/pytorch/api_doc.html",
    "compatibility_shim": "https://github.com/Ascend/pytorch/blob/master/torch_npu/contrib/transfer_to_npu.py",
    "distributed": "https://docs.pytorch.org/docs/stable/distributed.html",
}

CATEGORY_GUIDANCE = {
    "hardcoded-cuda": "Route through one device resolver; preserve the original CUDA branch.",
    "cuda-identity": "Use tensor.device.type for CUDA-only kernels; compatibility shims can alter is_cuda.",
    "cuda-build": "Make CUDA build/JIT imports lazy and add a tested NPU-safe path; never fake CUDA_HOME.",
    "custom-kernel": "Prove the kernel/backend supports the installed Ascend stack or add a scoped fallback.",
    "complex-or-fp64": "Check dtype support; keep required high precision on CPU or use a parity-tested real formulation.",
    "autocast": "Use device-aware autocast and verify the installed stack supports the chosen dtype.",
    "checkpoint-load": "Validate strict keys/shapes/dtypes and test mmap/path behavior on the target tuple.",
    "optimizer-memory": "Measure optimizer-step peak; test foreach/fused behavior instead of assuming CUDA defaults.",
    "distributed": "Validate HCCL collectives before real DDP and require every rank to exit zero.",
    "compile-or-jit": "Disable or guard unproven compilation paths until eager NPU correctness passes.",
    "serving-backend": "Pin an Ascend-supported backend revision and test cold load plus a real request.",
    "device-transfer": "Review device, dtype, pinning, non_blocking, and map_location semantics.",
    "device-selection": "Record world/local rank to NPU mapping and set the selected device explicitly.",
    "read-error": "Inspect the unreadable source before declaring the static inventory complete.",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


def is_excluded(relative: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(relative, pattern) for pattern in patterns)


def candidate_files(
    root: Path, excludes: list[str], max_bytes: int, include_docs: bool
) -> Iterable[Path]:
    for current, dirs, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        relative_dir = current_path.relative_to(root).as_posix()
        dirs[:] = sorted(
            directory
            for directory in dirs
            if directory not in SKIP_DIRS
            and not is_excluded(
                (Path(relative_dir) / directory).as_posix() if relative_dir != "." else directory,
                excludes,
            )
        )
        for name in sorted(files):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if is_excluded(relative, excludes) or path.is_symlink():
                continue
            allowed_suffixes = TEXT_SUFFIXES | ({".md"} if include_docs else set())
            if name not in TEXT_NAMES and path.suffix.lower() not in allowed_suffixes:
                continue
            try:
                if path.stat().st_size > max_bytes:
                    continue
            except OSError:
                continue
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="repository root")
    parser.add_argument("--output", type=Path, help="write JSON atomically")
    parser.add_argument(
        "--exclude", action="append", default=[], metavar="GLOB", help="exclude a relative-path glob"
    )
    parser.add_argument("--max-file-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument(
        "--include-docs", action="store_true", help="also scan Markdown documentation"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir() or args.max_file_bytes <= 0:
        print("root must be a directory and --max-file-bytes must be positive", file=sys.stderr)
        return 2

    findings: list[dict[str, Any]] = []
    entrypoints: set[str] = set()
    dependency_files: set[str] = set()
    files_scanned = 0
    scanner_path = Path(__file__).resolve()

    for path in candidate_files(root, args.exclude, args.max_file_bytes, args.include_docs):
        if path.resolve() == scanner_path:
            continue
        relative = path.relative_to(root).as_posix()
        if path.name in ENTRYPOINT_NAMES:
            entrypoints.add(relative)
        if path.name in DEPENDENCY_NAMES or path.name.startswith("requirements"):
            dependency_files.add(relative)
        try:
            raw = path.read_bytes()
            if b"\x00" in raw:
                continue
            text = raw.decode("utf-8", errors="replace")
        except OSError as exc:
            findings.append(
                {
                    "category": "read-error",
                    "severity_hint": "medium",
                    "file": relative,
                    "line": 0,
                    "excerpt": str(exc)[:240],
                }
            )
            continue
        files_scanned += 1
        if "if __name__" in text and "__main__" in text:
            entrypoints.add(relative)
        for line_number, line in enumerate(text.splitlines(), start=1):
            for category, severity, regex in PATTERNS:
                if regex.search(line):
                    findings.append(
                        {
                            "category": category,
                            "severity_hint": severity,
                            "file": relative,
                            "line": line_number,
                            "excerpt": line.strip()[:240],
                        }
                    )

    counts: dict[str, int] = {}
    for finding in findings:
        category = finding["category"]
        counts[category] = counts.get(category, 0) + 1

    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "root_name": root.name,
        "files_scanned": files_scanned,
        "finding_count": len(findings),
        "counts_by_category": dict(sorted(counts.items())),
        "entrypoint_candidates": sorted(entrypoints),
        "dependency_files": sorted(dependency_files),
        "findings": findings,
        "category_guidance": CATEGORY_GUIDANCE,
        "reference_links": REFERENCE_LINKS,
        "limitations": [
            "Heuristic matches require human review.",
            "Runtime-generated code and files outside the scanned root are not covered.",
            "No finding is a compatibility guarantee.",
        ],
    }
    if args.output:
        atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    print("ASCEND_STATIC_INVENTORY_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
