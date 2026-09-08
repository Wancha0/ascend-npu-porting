#!/usr/bin/env python3
"""Probe an Ascend PyTorch runtime and run a tiny BF16 backward operation."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback
from typing import Any


SAFE_ENV_KEYS = (
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_AICPU_PATH",
    "HCCL_CONNECT_TIMEOUT",
    "PYTORCH_NPU_ALLOC_CONF",
    "LD_LIBRARY_PATH",
    "PATH",
)


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


def command_capture(argv: list[str], timeout: int = 15) -> dict[str, Any]:
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "available": True,
            "argv": argv,
            "exit_code": result.returncode,
            "stdout": result.stdout[-20000:],
            "stderr": result.stderr[-10000:],
        }
    except FileNotFoundError:
        return {"available": False, "argv": argv, "error": "command not found"}
    except subprocess.TimeoutExpired as exc:
        return {
            "available": True,
            "argv": argv,
            "error": f"timeout after {timeout}s",
            "stdout": (exc.stdout or "")[-20000:] if isinstance(exc.stdout, str) else "",
            "stderr": (exc.stderr or "")[-10000:] if isinstance(exc.stderr, str) else "",
        }


def cann_version_hints() -> list[dict[str, str]]:
    candidates: list[Path] = []
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if ascend_home:
        base = Path(ascend_home).expanduser()
        candidates.extend([base / "version.info", base.parent / "version.info"])
    candidates.extend(
        [
            Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg"),
            Path("/usr/local/Ascend/ascend-toolkit/latest/version.info"),
            Path("/usr/local/Ascend/driver/version.info"),
        ]
    )
    hints: list[dict[str, str]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen or not candidate.is_file():
            continue
        seen.add(key)
        try:
            hints.append({"path": key, "content": candidate.read_text(errors="replace")[:4000]})
        except OSError as exc:
            hints.append({"path": key, "error": str(exc)})
    return hints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write JSON atomically")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument(
        "--no-op",
        action="store_true",
        help="inventory only; do not allocate an NPU tensor",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "environment": {key: os.environ[key] for key in SAFE_ENV_KEYS if key in os.environ},
        "cann_version_hints": cann_version_hints(),
        "npu_smi": command_capture(["npu-smi", "info"]),
        "requested_device_index": args.device_index,
        "operation_skipped": args.no_op,
        "reference_links": {
            "torch_npu": "https://github.com/Ascend/pytorch",
            "version_compatibility": "https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md",
            "documentation": "https://www.hiascend.com/document/detail/zh/Pytorch/2610/index/index.html",
        },
        "status": "fail",
    }

    try:
        import torch

        payload["torch"] = {
            "version": getattr(torch, "__version__", "unknown"),
            "path": getattr(torch, "__file__", "unknown"),
        }
        import torch_npu

        payload["torch_npu"] = {
            "version": getattr(torch_npu, "__version__", "unknown"),
            "path": getattr(torch_npu, "__file__", "unknown"),
        }

        available = bool(torch.npu.is_available())
        device_count = int(torch.npu.device_count()) if available else 0
        payload["npu"] = {"available": available, "device_count": device_count}
        if not available:
            raise RuntimeError("torch_npu imported but torch.npu.is_available() is false")
        if args.device_index < 0 or args.device_index >= device_count:
            raise RuntimeError(
                f"device index {args.device_index} is outside available range 0..{device_count - 1}"
            )

        device = torch.device(f"npu:{args.device_index}")
        torch.npu.set_device(device)
        try:
            payload["npu"]["device_name"] = str(torch.npu.get_device_name(args.device_index))
        except Exception as exc:  # Runtime naming is optional across versions.
            payload["npu"]["device_name_error"] = str(exc)

        if args.no_op:
            payload["status"] = "pass"
            payload["gate"] = "ASCEND_RUNTIME_INVENTORY_PASS"
        else:
            torch.manual_seed(20260902)
            left = torch.randn((64, 64), device=device, dtype=torch.bfloat16, requires_grad=True)
            right = torch.randn((64, 64), device=device, dtype=torch.bfloat16, requires_grad=True)
            output = (left @ right).float().square().mean()
            output.backward()
            torch.npu.synchronize()
            finite = bool(torch.isfinite(output).item())
            grad_finite = bool(torch.isfinite(left.grad).all().item())
            if not finite or not grad_finite:
                raise RuntimeError("tiny BF16 matmul/backward produced a non-finite value")
            operation: dict[str, Any] = {
                "name": "bf16_matmul_backward",
                "loss": float(output.item()),
                "loss_finite": finite,
                "gradient_finite": grad_finite,
                "output_device": str(output.device),
            }
            for key, method_name in (
                ("memory_allocated_bytes", "memory_allocated"),
                ("memory_reserved_bytes", "memory_reserved"),
            ):
                method = getattr(torch.npu, method_name, None)
                if method is not None:
                    try:
                        operation[key] = int(method(args.device_index))
                    except Exception:
                        pass
            payload["operation"] = operation
            payload["status"] = "pass"
            payload["gate"] = "ASCEND_RUNTIME_PROBE_PASS"
    except Exception as exc:
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
        payload["traceback"] = traceback.format_exc()

    if args.output:
        atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if payload["status"] == "pass":
        print(payload["gate"])
        return 0
    print("ASCEND_RUNTIME_PROBE_FAIL", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
