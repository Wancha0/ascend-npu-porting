# 正式训练的 OBS 与 MoXing 流程

OBS 资产传输在任务范围内时，配合[ModelArts 生产训练](modelarts-production.md)和[用户备份选择](artifact-backup.md)阅读。资料核对：2026-09-15。示例是应用层实现模式，尚不代表完整云端生产适配已实测。

## 每类资产明确一个传输负责人

| 机制 | 必须明确的契约 |
|---|---|
| 平台输入/输出映射 | 写清实际本地路径、准备/同步时机和最终交付状态。不要再用 MoXing 向同一目标重复复制。若平台在进程退出后上传，进程内无法提前证明上传完成，后续评估需等外部平台状态/回执核验。 |
| 显式 MoXing 下载/上传 | 作业内的准备和上传阶段负责不可变版本、频率、进度及完成标记。不能假定未映射目录会被平台自动收集。 |
| 共享持久文件系统 | 核对真实挂载、生命周期、所有读取节点可见性；如还需要 OBS 备份，独立确认同步状态。挂载成功不等于 OBS 已备份。 |

代码、权重、数据、特征可分别采用不同机制，但同一资产/目标只由一方交付。公网权重下载是独立准备任务；正式训练消费已固定版本的持久源，不依赖开发电脑中转。

## 环境、权限与小型连通测试

优先复用选定镜像里已工作的 MoXing，记录安装版本和解释器；import 通过不代表 OBS 可读写。若缺失，先用可信、固定版本的 MoXing 安装包及依赖构建训练兼容镜像，不猜测 PyPI 包名，不升级平台 torch 组合。

官方[Framework 介绍](https://support.huaweicloud.com/usermanual-standard-modelarts/modelarts_11_0001.html)区分普通 OBS 对象桶与并行文件系统，并说明版本相关加速能力。不要把旧 MA_MOXING_FWVER 值当通用配置，不假定所有自定义镜像都自带 MoXing；并行文件系统需另核兼容性。

使用作业支持的委托/临时凭证。Notebook 能访问不证明训练身份能访问。分别核对源读取/列举、目标写入、桶策略、区域/端点及作业内凭证续期，不写死 AK/SK、不输出凭证。官方[训练权限实践](https://support.huaweicloud.com/permission-modelarts/modelarts_24_00135.html)列出 ListBucket/GetObject/PutObject 委托，额外分片操作按实际客户端和策略确认。

在已有输出授权范围内，读取一个已知小对象，并在 run 前缀下写入/读回一个唯一的小回执即可，不用大权重测试连通。清理删除需要已有授权。403 优先修权限，缺失输入修清单；暂时性超时才按预算重试同一个已确认目标。

## 基本复制用法

官方[MoXing 文件说明](https://support.huaweicloud.com/intl/zh-cn/develop-modelarts/develop-moxing-0002.html)支持单文件与目录复制。以下变量由解析后的作业配置提供，不含凭证：

```python
import moxing as mox

# 单文件：OBS 到本地临时文件。
mox.file.copy(source_obs_file, local_staging_file)
# 目录：OBS 到隔离、尚未发布 READY 的本地临时目录。
mox.file.copy_parallel(source_obs_directory, local_staging_directory,
                       threads=4, is_processing=False)
# 上传冻结快照到本次 attempt 的全新 OBS 前缀。
mox.file.copy_parallel(frozen_snapshot_directory, new_obs_prefix,
                       threads=4, is_processing=False)
```

4 个线程只是示例起点。总并发考虑节点数 × 复制线程 × 内部分片并发，用小型代表性传输确定。[BrokenPipe 排障](https://support.huaweicloud.com/trouble-modelarts/modelarts_trouble_0042.html)说明了高并发及大文件参数，不要直接照搬其数值。目录复制不是原子事务，copy_parallel 也不能未经验证就宣传为字节级断点续传。

## 下载与完整快照发布示例

下面两个函数演示单文件下载和完整快照上传，采用大小校验。它们不实现分布式协调、可恢复队列、鉴权或超时监督，调用方必须补齐这些生命周期控制。相同大小的内容损坏无法检出；契约要求内容校验时使用可信摘要。[官方文件 API](https://github.com/huaweicloud/ModelArts-Lab/blob/master/docs/moxing_api_doc/MoXing_API_File.md)说明了 get_size/read/write。

```python
import json
import uuid
from pathlib import Path
import moxing as mox

def download_file(obs_uri, destination, expected_bytes):
    # Caller owns this destination; source identity is pinned in the job manifest.
    dst = Path(destination)
    if dst.exists():
        raise FileExistsError("Validate an existing cache before reusing it")
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp = dst.with_name(dst.name + ".part-" + uuid.uuid4().hex)
    mox.file.copy(obs_uri, str(temp))
    if temp.stat().st_size != expected_bytes:
        raise ValueError("Downloaded size mismatch; partial file is not published")
    # Same local filesystem. A single staging owner is required.
    temp.replace(dst)
    return {"source": obs_uri, "local": str(dst),
            "bytes": expected_bytes, "verification": "size_only"}

def publish_snapshot(snapshot_dir, unique_obs_prefix, identity):
    # Caller guarantees a frozen, complete snapshot and one publisher.
    # Prefix must be new and owned by this attempt, never a shared 'latest/'.
    base = Path(snapshot_dir)
    files = sorted(p for p in base.rglob("*") if p.is_file())
    if not files or any(p.is_symlink() for p in base.rglob("*")):
        raise ValueError("Need a nonempty regular-file snapshot")
    prefix = unique_obs_prefix.rstrip("/") + "/"
    if mox.file.exists(prefix):
        raise FileExistsError("Inspect a prior attempt before retrying")
    records = []
    for path in files:
        relative = path.relative_to(base).as_posix()
        if relative == "COMMITTED.json":
            raise ValueError("Reserved receipt name")
        target = prefix + relative
        size = path.stat().st_size
        mox.file.copy(str(path), target)
        if mox.file.get_size(target) != size:
            raise ValueError("Uploaded size mismatch")
        records.append({"path": relative, "bytes": size})
    receipt = {"identity": identity, "files": records,
               "verification": "size_only"}
    marker = prefix + "COMMITTED.json"
    mox.file.write(marker, json.dumps(receipt, sort_keys=True))
    if json.loads(mox.file.read(marker)) != receipt:
        raise ValueError("Receipt readback mismatch")
    return receipt
```

download_file 的输入来自固定清单，包含模型/revision、processor、特征生成身份。publish_snapshot 不能指向 trainer 正在写入的目录。FSDP/ZeRO 先收集或持久化全部必需分片和元数据，不能只保存 rank 0。identity 至少包含 run/attempt、step、普通/EMA/续训用途及源码/配置版本。

上传异常时不承诺存在有效完成标记。示例有意拒绝已有前缀；先核对，再用清单感知实现补传缺失文件，或另开 attempt 前缀，不覆盖已提交前缀。写标记过程中崩溃时，读取者必须解析完整标记并核对引用对象后才接受。

重复传输用台账复用已验证完整文件。除非验证过当前客户端的恢复行为，MoXing 未完成文件可能需要从头重传。保留部分状态供检查，不把“存在”或“目录总字节数相同”当作缓存有效。

## 多节点与训练衔接

节点本地缓存要**每节点准备一次**，通常在该节点启动训练 worker 前完成；只让 global rank 0 下载不能填满其他节点磁盘。共享存储才由一个持真实锁/租约的写入者准备，所有读者核验同一完成记录。OBS 对象的“先判断不存在再创建”不是分布式锁。

全部必需文件通过约定检查后才发布绑定资产身份的本地 READY，包含源清单身份、预期路径/数量、大小和校验强度。其他读者使用有界等待及失败信号，下载失败不能导致无限 barrier。平台逐 worker 启动时，使用节点内唯一下载者、本地锁和有界失败传播，不能 8 个 rank 同时复制。

checkpoint 流程：
1. Trainer 完成不可变快照；上传确认前保留文件。
2. 作业上传器写到独立 run/attempt/step 前缀，核验对象后最后发布 COMMITTED.json。
3. 记录本地 step、持久 step、待传字节、最近成功及首个错误。提交完成后才能更新 latest 指针；恢复时将指针解析为不可变清单。
4. 队列有上限，在待传快照占满磁盘前执行约定的背压策略。不能清理未确认快照；旧持久 checkpoint 删除遵守用户保留策略。
5. 训练完成后，在预设收尾预算内排空已选上传，分别记录训练、备份与评估状态。

除非明确测试过远端流式读取性能，不把每步 OBS 请求放进训练热路径；可复用特征优先放合适的本地/共享存储。特征清单保存 encoder 身份、相机覆盖、分辨率/token 形状、dtype、数据划分与预处理，不能只比文件名。

## 非交互作业必须预先决定退出策略

平台移除容器后，进程不能保证临时文件仍存在。因此“非零退出并保留 /cache”**不是持久化方案**。

选择 OBS 的生产作业，提交前提出并记录最终上传超时/重试预算，以及以下方案之一：已验证的持久 spool 保存待传文件；平台支持的有界容器保留时间用于补救；或用户明确接受可能损失最后未保存部分。不能静默选最后一种，不能无限等待人工，也不因此获得自动重复提交作业的授权。

最终交付失败时，保留训练退出状态及独立交付失败；若持久交付是完成条件，整体结果失败。能恢复的仅是实际存活的文件。平台保留/重试时序需按本模式核对。周期性已提交 checkpoint 限制损失窗口，但 OBS 长时间不可用时不承诺零损失。

应用自行上传时，整体成功要求训练成功、已选资产提交完成，以及属于本作业的已授权评估成功。平台退出后上传时，先记录应用完成，再由外部确认交付，之后才能声称整体完成或触发消费者。

## 有界验收

复用已有证据，仅检查变化的集成环节：小输入到实际作业路径；一份真实 checkpoint 提交到选定目标；新作业加载该已提交 checkpoint。核对普通/EMA 身份；声称续训时必须恢复训练状态。本地/模拟测试不证明云端权限、终止保留或生产恢复。
