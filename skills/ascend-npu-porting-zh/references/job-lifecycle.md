# 任务生命周期

## 保存位置与备份选择

正式长训练或大规模提取前，执行[产物保存与 OBS 选择](artifact-backup.md)，把用户选择写入本次契约。checkpoint 保存间隔与 OBS 同步间隔是两个字段；不能把本地落盘视为已备份。已有明确授权不重复询问，未选择不能默认不备份。

用户选择 OBS 后，监控独立上传器的最近已备份 step、待传资产和失败原因。训练退出 0 与备份完成分别报告；备份未完成时保留源文件，不因常规清理删除唯一副本或释放其节点。用户明确要求不备份释放时服从其指令。

## SSH 与命令

仅对已授权的节点和任务操作。先确认 hostname、代码/资产路径、卡占用与本次唯一目录。使用安全字段 `pid,ppid,pgid,user,stat,etime,comm` 查看进程；不转储完整参数、全量环境或服务凭证。

将多行内容写成实际脚本，经 `bash -n` 后用 [ssh_script.py](../scripts/ssh_script.py) 通过 stdin 发送。动态字符串用参数数组或工具提供的文件编辑能力，不套多层 shell 引号；`JSON.stringify` 不是 shell 转义。quoted heredoc 的终止符必须独占一行。本机 zsh、macOS 工具与目标 Linux Bash 不同，不能混用 `PIPESTATUS`、保留变量 `status` 或未经确认的 GNU `find -printf`。

保存生产者原始退出状态后再读日志。开启 pipefail 的管道接 `head`/`grep -q` 时可能触发 SIGPIPE，优先保存完整输出再过滤文件。

## 长训练不要依赖 SSH 父 shell

只让训练子进程 `setsid`，而让负责 `wait/$?` 的父 shell 留在 SSH 中，仍会丢失最终退出码。用目标节点上的持久监督器同时管理运行和退出记录。

随包 [run_recorded.py](../scripts/run_recorded.py) 支持 Linux/macOS 的本地进程；需要复制到**目标节点**，再用目标 Python 启动。它不分配资源、不提交平台作业、不自动重试。若平台会清理整个会话/cgroup，采用已授权的调度系统，由调度器保存最终状态；脱离 session 不能规避平台生命周期。

在目标端准备无凭证的 `command.sh`：写清解释器、源码、物理卡、batch、精度、输入和全新的 OUT；模型命令使用前台方式并以 `exec` 启动，不在脚本内再次加 `&`。只把非敏感参数写入命令；监督器会记录 argv。

```bash
bash -n "$COMMAND_FILE"
"$PYTHON_BIN" "$SKILL_DIR/scripts/run_recorded.py" start \
  --run-dir "$NEW_RUN_DIR" --cwd "$WORK_DIR" -- bash "$COMMAND_FILE"
"$PYTHON_BIN" "$SKILL_DIR/scripts/run_recorded.py" status --run-dir "$NEW_RUN_DIR"
```

上述变量由当前契约赋值，`NEW_RUN_DIR` 必须尚不存在，父目录已存在。`start` 返回 0 仅表示监督器被提交。读取 `result.json` 中实测 `exit_code`，并检查模型证据后才判定训练。状态为空或未知时检查 `submission.json`、监督器和子进程，不能自动重启。

`command.json` 固定 argv/cwd，`runner.json` 记录监督器/子进程 PID 与进程组，`output.log` 保存原始输出，`result.json` 原子写入实际退出码。正常结束为 `status=exited`，退出码仍可能非零；负数表示被信号终止。`status` 命令自身退出 0 只是状态读取成功。

## 运行与重试

- 启动前台账记录监督器、子进程、日志、源码身份、预期产物。NPU 可见列表与 local_rank 的映射在每个 rank 内输出，不能只记录 rank 0。
- 监控使用安全进程字段、`npu-smi info`、本次日志和更新产物。没有新日志不必然死锁；结合资源、耗时、进程状态判断，不凭短暂 AICore=0 立即杀任务。
- SSH 失败后重新只读获取已有任务状态，继续监控；训练完成但退出码丢失时标记 unknown。
- 失败先保存首个因果堆栈，检查本次进程与卡占用，再修复并用新目录重试；不重复下载未变资产、不删除失败日志、不启动重复训练。
- 停止必须先核实 PID/进程组仍属于本次运行。PID 存在不能排除被复用；结合开始时间、父子关系与 ledger。只向本次拥有的进程发送信号，不用 `pkill python`、整机进程清理或 NPU reset。
- 监督器收到 TERM/INT 会转发给它创建的子进程组并等待结果；若框架创建额外会话或节点，需按该 launcher 的所有权记录清理。该工具不声称已验证每个 rank 的退出或 NPU 释放，仍需外部核对。

## 两个实测操作陷阱

1. CANN 初始化若出现 `multiprocessing.Manager` 失败，先读最早异常。`AF_UNIX path too long` 应缩短本次 `TMPDIR`（如唯一的 `/tmp/npu-a01`），不是立即重装 CANN 或认定 HBM 不足。保持输出在持久目录；临时目录长度和权限在目标核实。
2. `result.json` 缺失不能据权重文件推断为退出 0。保留有效更新证据，把退出记录单独列为未验证，后续运行使用持久监督器补齐。
