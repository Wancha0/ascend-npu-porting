# 随包工具说明

除目标运行时探针按需导入 torch/torch_npu 外，工具只依赖 Python 3.9+ 标准库。命令帮助中的英文参数保持稳定；本页说明其用途和边界。全部命令在 skill 根目录执行，或替换为脚本绝对路径。

| 工具 | 作用 | 不能证明什么 |
|---|---|---|
| `self_check.py` | 本地链接、语法、manifest、SSH 参数及持久退出记录的离线检查 | 模型适配/真实 NPU 可用性 |
| `manifest.py` | 建立/核对文件大小、相对路径、SHA-256 | 模型数值正确或远端已经执行 |
| `scan_npu_risks.py` | 静态定位 CUDA 等风险线索 | 每条匹配都是 bug，或未匹配处安全 |
| `probe_ascend_runtime.py` | 环境清单；可选微型 BF16 前反向 | 完整模型训练或目标多卡通过 |
| `ssh_script.py` | 本地 Bash 语法检查，参数数组 SSH，stdin 脚本 | 长任务退出记录持久性 |
| `run_recorded.py` | 在执行节点保留子进程真实退出码和日志 | 分配 NPU、所有 rank 的数值正确性及清理 |

## 首次复制校验

```bash
python3 scripts/self_check.py
python3 scripts/manifest.py verify MANIFEST.json --root .
```

修改 skill 后重新冻结清单：

```bash
python3 scripts/manifest.py create . --output MANIFEST.json \
  --exclude '._*' --exclude '*/._*'
```

manifest 不依赖远端 `.git`。源码包也可用此工具建立独立清单，排除生成物需写在清单中；不要随意 `--allow-extra` 掩盖混入的代码。

## 风险和环境探测

```bash
python3 scripts/scan_npu_risks.py --help
python3 scripts/probe_ascend_runtime.py --no-op --output runtime.json
```

scan 对已追踪到的源码路径执行，先查看 `--help` 中当前参数。`--no-op` 不分配 NPU 张量；不加时会在选定 NPU 运行小型 BF16 前反向，必须属于当前已授权测试。已有相同运行时证据时不必再运行。输出只说明实际完成的探测层级。

## SSH

变量来自当前授权契约，不包含模型 API key 等秘密：

```bash
python3 scripts/ssh_script.py --host "$TARGET_HOST" --user "$TARGET_USER" \
  --port "$TARGET_PORT" --identity "$IDENTITY_FILE" < remote-script.sh
```

`--dry-run` 只做语法/参数校验，不连接目标。`--env NAME=VALUE` 按字面值传递非敏感配置；不要用它在可见命令行传凭证。工具不自动重试，返回 SSH 命令的原始退出码。长训练按[任务生命周期](job-lifecycle.md)在目标端使用持久监督器。

## 自检的边界

自检不联网、不调用模型、不申请 NPU。它在临时目录验证正常/非零退出、拒绝重复目录、静态扫描及 manifest 篡改检测，并用模拟 SSH 验证字面参数传递。通过仅说明这些工具行为在执行自检的系统/Python 上符合预期；在新宿主复制后可复验一次，不需要每轮适配都重跑。
