# 昇腾 NPU 适配 skill

一个中文版本，支持 GPT 和 GLM 编码智能体。用于 PyTorch 到昇腾 NPU 的适配、训练/推理验证、多卡问题、ModelArts 正式训练，以及已授权的 OBS 下载与保存。

**从 [SKILL.md](SKILL.md) 开始，按当前任务读取参考。**

推荐流程：Notebook 开发 → 作业包准备 → 生产试运行 → 正式训练与周期备份 → 交付确认 → 评估。

## 使用

从当前发布分支获取：

```bash
git clone --branch codex/glm-zh-training-first --single-branch https://github.com/Wancha0/ascend-npu-porting.git
```

可将整个目录交给 GPT 或 GLM；宿主支持目录式 skill 时安装为 `ascend-npu-porting`，否则直接要求读取 SKILL.md 的绝对路径。智能体需要文件和命令执行能力；访问目标节点或云平台还需对应工具与授权。

```text
读取 [skill 绝对路径]/SKILL.md，为以下项目执行 NPU 适配：
源码与版本：[仓库/目录/revision]
目标环境：[节点访问方式、解释器、NPU/节点数]
资产：[权重、数据或特征位置]
目标与预算：[训练/推理/恢复/评估，参数与更新步数]
保存策略：[已有 OBS 选择或待确定的范围]
接续状态：[已有 ledger 和活跃任务]
复用有效验证，只补当前缺口，报告真实结果和未测项。
```

所有使用说明只维护中文。Python 标识符、CLI 参数和第三方 API 保持原名。生产流程文档及本地工具检查不等于云端生产实测通过。
