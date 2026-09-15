#!/usr/bin/env python3
"""Validate a Bash script locally, then send it to one SSH host through stdin."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
HOST_NAME = re.compile(r"^[A-Za-z0-9_:.%][A-Za-z0-9_.:%-]*$")
USER_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


def parse_env(items: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise ValueError("environment entry must be NAME=VALUE")
        name, value = item.split("=", 1)
        if not ENV_NAME.fullmatch(name):
            raise ValueError("invalid environment name")
        if "\0" in value:
            raise ValueError("environment values cannot contain NUL")
        parsed.append((name, value))
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Bash stdin and execute it on one SSH target; no retries."
    )
    parser.add_argument("--host", required=True, help="SSH alias, hostname, or IP; no user prefix")
    parser.add_argument("--user")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--identity", type=Path, help="existing local private-key file")
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--connect-timeout", type=int, default=15)
    parser.add_argument("--server-alive-interval", type=int, default=20)
    parser.add_argument(
        "--host-key-policy", choices=("yes", "accept-new", "no"), default="accept-new"
    )
    parser.add_argument("--no-strict", action="store_true", help="omit set -euo pipefail")
    parser.add_argument("--dry-run", action="store_true", help="validate and print metadata without SSH")
    args = parser.parse_args()

    if not HOST_NAME.fullmatch(args.host):
        parser.error("--host must be an SSH alias, hostname, or IP without options or a user prefix")
    if args.user is not None and not USER_NAME.fullmatch(args.user):
        parser.error("--user contains unsupported characters")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.connect_timeout <= 0 or args.server_alive_interval <= 0:
        parser.error("timeout and keepalive values must be positive")
    if args.identity is not None:
        try:
            identity_valid = args.identity.is_file()
        except (OSError, ValueError):
            identity_valid = False
        if not identity_valid:
            parser.error("--identity must name an existing local file")

    try:
        env_items = parse_env(args.env)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        body = sys.stdin.read()
    except (OSError, UnicodeError, ValueError):
        parser.error("cannot read script stdin")
    if not body.strip():
        parser.error("script stdin is empty")
    if "\0" in body:
        parser.error("script stdin cannot contain NUL")

    exports = "".join(f"export {name}={shlex.quote(value)}\n" for name, value in env_items)
    strict = "" if args.no_strict else "set -euo pipefail\n"
    payload = strict + exports + body
    try:
        checked = subprocess.run(
            ["bash", "-n"], input=payload, text=True, encoding="utf-8", capture_output=True
        )
    except OSError:
        print("cannot run local Bash syntax check", file=sys.stderr)
        return 127
    if checked.returncode:
        # Bash diagnostics can quote source lines containing environment values.
        print(
            f"Bash syntax check failed (exit {checked.returncode}); inspect the script file locally",
            file=sys.stderr,
        )
        return checked.returncode

    target = f"{args.user}@{args.host}" if args.user else args.host
    command = [
        "ssh", "-T", "-p", str(args.port),
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={args.connect_timeout}",
        "-o", f"ServerAliveInterval={args.server_alive_interval}",
        "-o", f"StrictHostKeyChecking={args.host_key_policy}",
    ]
    if args.identity is not None:
        command.extend(["-i", os.fspath(args.identity)])
    command.extend([target, "bash -s"])

    if args.dry_run:
        print(json.dumps({
            "argv": command,
            "script_bytes": len(payload.encode("utf-8")),
            "script_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "environment_names": [name for name, _ in env_items],
            "syntax_checked": True,
        }, indent=2))
        return 0

    try:
        completed = subprocess.run(command, input=payload, text=True, encoding="utf-8")
    except OSError:
        print("cannot execute SSH client", file=sys.stderr)
        return 127
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
