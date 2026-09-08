#!/usr/bin/env python3
"""离线检查随包工具；不联网、不导入 torch、不占用 NPU。"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent


def call(*argv, expected=0):
    r = subprocess.run(list(argv), capture_output=True, text=True, timeout=20)
    if r.returncode != expected:
        raise AssertionError(f'退出码 {r.returncode}，预期 {expected}: {r.stderr[-1000:]}')
    return r


def wait_result(run):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        path = run / 'result.json'
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(0.05)
    raise AssertionError('监督器未在限定时间保存结果')


def main():
    errors, passed, skipped = [], [], []
    required = ['SKILL.md', 'references/compatibility.md', 'references/validation.md',
                'references/glm-handoff.md', 'references/tools.md', 'references/job-lifecycle.md',
                'scripts/manifest.py', 'scripts/ssh_script.py', 'scripts/run_recorded.py',
                'scripts/probe_ascend_runtime.py', 'scripts/scan_npu_risks.py']
    for name in required:
        if not (ROOT / name).is_file():
            errors.append('缺少文件: ' + name)
    for p in ROOT.rglob('*.md'):
        for raw in re.findall(r'\[[^\]]*\]\(([^)]+)\)', p.read_text()):
            link = raw.strip('<>').split('#')[0]
            if not link or link.startswith(('https://', 'http://')):
                continue
            target = (p.parent / link).resolve()
            if ROOT not in target.parents or not target.exists():
                errors.append(f'无效本地链接: {p.name} -> {link}')
    for p in (ROOT / 'scripts').glob('*.py'):
        try:
            compile(p.read_text(), str(p), 'exec')
            call(sys.executable, str(p), '--help') if p.name != 'self_check.py' else None
        except Exception as e:
            errors.append(f'{p.name}: {e}')
    try:
        with tempfile.TemporaryDirectory(prefix='npu-skill-check-') as temp:
            tmp = Path(temp)
            fixture = tmp / 'fixture'
            fixture.mkdir()
            (fixture / 'literal.txt').write_text('中文与字面字符 $HOME `echo x`\n')
            (fixture / '._literal.txt').write_text('包装文件')
            manifest = tmp / 'manifest.json'
            tool = str(ROOT / 'scripts/manifest.py')
            call(sys.executable, tool, 'create', str(fixture), '--output', str(manifest),
                 '--exclude', '._*', '--exclude', '*/._*')
            call(sys.executable, tool, 'verify', str(manifest), '--root', str(fixture))
            (fixture / 'literal.txt').write_text('被修改')
            changed = subprocess.run([sys.executable, tool, 'verify', str(manifest),
                                      '--root', str(fixture)], capture_output=True, timeout=10)
            assert changed.returncode != 0, 'manifest 没有发现内容篡改'
            passed.append('manifest 校验及篡改检测')

            (fixture / 'train.py').write_text('x = torch.randn(1).cuda()\n')
            scan = call(sys.executable, str(ROOT / 'scripts/scan_npu_risks.py'), str(fixture))
            assert 'hardcoded-cuda' in scan.stdout, '静态扫描未报告 CUDA 线索'
            passed.append('静态扫描真实风险样例')

            runner = str(ROOT / 'scripts/run_recorded.py')
            for expected in (0, 37):
                run = tmp / f'job-{expected}'
                # start 进程退出后，独立监督器仍完成写文件与真实退出记录。
                child = 'import time,sys; time.sleep(0.2); print("fixture-done"); sys.exit(%d)' % expected
                call(sys.executable, runner, 'start', '--run-dir', str(run), '--cwd', str(tmp),
                     '--', sys.executable, '-c', child)
                record = wait_result(run)
                assert record['status'] == 'exited' and record['exit_code'] == expected
                assert 'fixture-done' in (run / 'output.log').read_text()
                status = json.loads(call(sys.executable, runner, 'status', '--run-dir', str(run)).stdout)
                assert status['exit_code'] == expected
                call(sys.executable, runner, 'start', '--run-dir', str(run), '--cwd', str(tmp),
                     '--', sys.executable, '-c', 'raise RuntimeError()', expected=2)
            passed.append('控制端退出后真实 0/37 退出码持久记录、拒绝重复启动')

            if shutil.which('bash') is None:
                skipped.append('缺少本地 Bash，SSH 行为未验证')
            else:
                helper = runpy.run_path(str(ROOT / 'scripts/ssh_script.py'))
                real_run = subprocess.run
                calls = []
                literal = "中文 '$HOME' $(echo unexpected) `echo unexpected`;\n第二行"

                def fake_run(argv, **kwargs):
                    if argv == ['bash', '-n']:
                        return real_run(argv, **kwargs)
                    assert argv[0] == 'ssh'
                    calls.append(argv)
                    result = real_run(['bash', '-s'], capture_output=True, timeout=5, **kwargs)
                    assert result.stdout == literal
                    return result

                with patch.object(sys, 'argv', ['ssh_script.py', '--host', 'fixture.invalid',
                                               '--env', 'NPU_FIXTURE=' + literal]), \
                     patch.object(sys, 'stdin', io.StringIO('printf \'%s\' "$NPU_FIXTURE"\nexit 37\n')), \
                     patch.object(subprocess, 'run', side_effect=fake_run):
                    assert helper['main']() == 37
                assert len(calls) == 1 and calls[0][-1] == 'bash -s'
                with patch.object(sys, 'argv', ['ssh_script.py', '--host', 'fixture.invalid']), \
                     patch.object(sys, 'stdin', io.StringIO('if then\n')), \
                     patch.object(subprocess, 'run', side_effect=fake_run), \
                     contextlib.redirect_stderr(io.StringIO()):
                    assert helper['main']() != 0
                assert len(calls) == 1, '错误脚本仍发起 SSH'
                passed.append('模拟 SSH 字面值、非零退出保留、语法错误阻止执行')
    except Exception as e:
        errors.append(f'行为检查失败: {type(e).__name__}: {e}')
    print(json.dumps({'status': 'pass' if not errors else 'fail', 'checks': passed,
                      'skipped': skipped, 'errors': errors,
                      'scope': '离线工具检查；不是 NPU 或模型验收'}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    raise SystemExit(main())
