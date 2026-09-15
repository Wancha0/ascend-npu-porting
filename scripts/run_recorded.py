#!/usr/bin/env python3
"""在目标机启动脱离控制连接的监督进程，持久记录命令的真实退出码。

只负责一个本地命令；不分配 NPU、不提交云作业、不自动重试、不判断模型通过。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                     prefix='.state-', delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        name = f.name
    os.replace(name, path)


def worker(run):
    os.umask(0o077)
    config = json.loads((run / 'command.json').read_text())
    state = {'status': 'starting', 'supervisor_pid': os.getpid(),
             'started_utc': now(), 'command_sha256': config['command_sha256'],
             'exit_code': None}
    save(run / 'runner.json', state)
    child = None
    requested_signal = None

    def forward(signum, frame):
        nonlocal requested_signal
        requested_signal = signum
        if child is not None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    try:
        with (run / 'output.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(config['argv'], cwd=config['cwd'],
                                     stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='running', child_pid=child.pid, process_group_id=child.pid)
            save(run / 'runner.json', state)
            if requested_signal is not None:
                forward(requested_signal, None)
            state['exit_code'] = child.wait()
            state['status'] = 'exited'
    except Exception as exc:
        # 不输出参数或完整环境；启动错误不伪装为子进程退出码。
        state.update(status='supervisor_error', error_type=type(exc).__name__)
    state.update(finished_utc=now(), requested_signal=requested_signal)
    save(run / 'result.json', state)
    save(run / 'runner.json', state)
    return 0 if state['status'] == 'exited' else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    start = sub.add_parser('start', help='新建唯一运行目录并启动监督器')
    start.add_argument('--run-dir', required=True, type=Path)
    start.add_argument('--cwd', required=True, type=Path)
    start.add_argument('command', nargs=argparse.REMAINDER)
    status = sub.add_parser('status', help='只读持久状态，不重启进程')
    status.add_argument('--run-dir', required=True, type=Path)
    internal = sub.add_parser('_worker', help=argparse.SUPPRESS)
    internal.add_argument('--run-dir', required=True, type=Path)
    args = p.parse_args()
    run = args.run_dir.expanduser().resolve()
    if args.action == '_worker':
        return worker(run)
    if args.action == 'status':
        path = run / 'result.json'
        if not path.exists():
            path = run / 'runner.json'
        if not path.exists():
            print(json.dumps({'status': 'unknown', 'exit_code': None,
                              'reason': '尚无状态记录，请核实启动进程；不要直接重试'}, ensure_ascii=False))
            return 2
        state = json.loads(path.read_text())
        if state.get('status') in ('starting', 'running'):
            try:
                os.kill(state['supervisor_pid'], 0)
                state['supervisor_pid_exists'] = True
            except ProcessLookupError:
                state.update(status='unknown', supervisor_pid_exists=False, exit_code=None)
            except PermissionError:
                state['supervisor_pid_exists'] = 'unverified'
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0
    argv = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not argv:
        p.error('需要 -- 后的实际命令')
    cwd = args.cwd.expanduser().resolve()
    if not cwd.is_dir():
        p.error('--cwd 必须是存在的目录')
    os.umask(0o077)
    # 已有目录绝不覆盖，避免重复启动；父目录由调用者明确选择。
    try:
        run.mkdir(mode=0o700)
    except FileExistsError:
        p.error('运行目录已存在：先读取状态，不能覆盖或自动重试')
    config = {'argv': argv, 'cwd': str(cwd)}
    config['command_sha256'] = hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    save(run / 'command.json', config)
    with (run / 'supervisor.log').open('ab', buffering=0) as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_worker',
                                 '--run-dir', str(run)], cwd=str(cwd),
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, close_fds=True)
    record = {'status': 'submitted', 'supervisor_pid': proc.pid,
              'submitted_utc': now(), 'run_dir': str(run), 'exit_code': None}
    save(run / 'submission.json', record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
