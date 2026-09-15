# 兼容问题与修复

以下是定位线索，不是要无条件套用的补丁。先确认触发点在本次真实执行路径。

## 设备和运行时

检查 `sys.executable`、模块 `__file__` 及实际版本，避免 launcher 调用了另一个环境。环境存在 Accelerate 包但没有 console script 时，可用选定解释器的 `-m accelerate.commands.launch`，仍须核实该版本参数及所有 rank 的设备分配。不要仅凭 `torch` 版本字符串含 `+cpu` 就判定 torch_npu 不可用。

在设备发现之前导入 torch_npu。把输入 device 转成 `torch.device`，比较其 `type`；`npu:0` 与字符串 `npu` 并不相等。兼容层可能让 `.is_cuda` 呈现 CUDA 风格行为，因此 CUDA 编译器或内核的分支判断使用真实设备类型。

## 内核、dtype 和数值

- 对 FlashAttention、Triton-CUDA、xFormers、bitsandbytes 和自定义 C++ 扩展，区分导入、构建和运行阶段。按设备延迟导入；NPU 优先选择已支持的实现或等价 PyTorch 路径，保留 CUDA 快路径。
- SDPA/eager fallback 也需要真实 attention mask、布局、序列长度、精度和 backward 验证。可执行不等于数值等价。
- 遇到复数或 float64 不支持，追踪是否仅为 RoPE/索引缓存构造。可在 CPU 构造后传递实数 cos/sin，或使用等价实数公式；不能为绕过报错全局降低参数精度。
- 对输出检查转换前浮点有限性。先 clamp/cast 成 uint8 再解码成功，不能证明模型输出没有 NaN/Inf。
- 测性能时使用目标设备同步，分别记录编译/首次加载和稳态。仅一轮 cold start 不用于宣称吞吐提升。

局部测试沿真实调用链覆盖输入/缓存构造、设备搬运、拼接或 reshape、核心算子及 backward。例如 RoPE 乘法通过不证明缓存的 `cat` 支持相同复数 dtype；某个算子自动降级 double 也不代表所有算子都这样处理。只围绕当前失败路径补测，不因此要求枚举模型全部算子。

数值容差依据实际 dtype、数值幅度、任务容忍度及参考实现确定，比较时统一初始化和输入转换。BF16 量化间隔随数值幅度变化；不能用 float32 的 `nextafter` 间隔冒充 BF16 ULP，也不能对近零参考只算相对误差。记录最大/平均绝对误差、差异比例及必要的梯度误差；不要把测试失败直接改名为通过。若断言公式确有错误，保留原失败，修正原因和阈值依据，然后重验必要部分。算子近似不能自动宣称 CUDA 数值等价。

## 数据、checkpoint 和模型语义

读取生产代码确定 metadata 是 list 还是 dict；路径是目录还是文件。缓存可能在 rank 子目录，不要猜测它位于根目录。先验证实际返回类型，再做索引或数组计算。

分片 checkpoint 按当前索引/加载器组织；某些模型检测器要求“一个嵌套 shard 列表”，不能把每个 shard 当独立模型。严格核对 keys/shapes；转换、裁剪或音频分支移除需有明确来源和规则。

缓存契约包含源码、预处理配置、分辨率/帧数、精度、文本/条件布局、音频模式。没有音频 token、静音 token 和缺失音频不是同一语义。历史 MiniMax 案例只在 `disable_audio` 的明确路径补充规范空张量；不能把这种模型专属形状复制给 FastWAM。

## 显存与 CPU offload

先列出参数/梯度/优化器状态/激活/临时张量的 dtype 与大小，再选择策略。

| 策略 | 需要说明及验证 |
|---|---|
| 数据并行复制 | 每卡仍有模型副本；增加卡数不会自动解决单副本放不下 |
| CPU offload | 必须有足够主存；模型参数、梯度和 optimizer state 在各阶段的设备及搬运时点需一致 |
| FSDP/其他分片 | 核对目标 torch_npu 版本支持、包装和 checkpoint 方案；旧版经验不能证明新版或相反 |
| 量化/裁剪 | 改变权重和数值契约；必须经任务范围允许并单独验收，不能静默替代完整模型 |

AdamW 的 foreach/fused 临时张量或 DDP bucket 可能提高峰值。`foreach=False`、bucket view 等仅作为有针对性的候选，不能预先替所有项目固定。

offload 特别检查：backward 前后的参数位置、梯度累积期间梯度是否仍在计算设备、更新前同步发生在何处、CPU optimizer 是否拿到正确梯度。累积 1 成功不证明累积 2；禁止在 microstep 中途把下一次 backward 需要的梯度移错设备。

## 分布式与计数

标准 DDP/FSDP 可用时优先使用已验证的框架方案。确需“CPU offload + 自定义 HCCL 梯度平均”时，不要再重复套一层梯度同步。

自定义同步需检查：

1. 所有 rank 的初始训练参数一致；固定种子、严格加载或显式广播须有证据。
2. 参数遍历顺序、shape、dtype 和梯度存在集合一致；按本地 `grad is None` 各自跳过可能死锁或错配。动态未使用参数需要一致的掩码与约定。
3. SUM 后除以实际 world size；不能重复平均或遗漏累积缩放。
4. 同步后的梯度数值、有限性和抽样校验；相同范数只是有限证据，不能单独证明全部张量一致。
5. optimizer 真正更新后再推进 scheduler/计数；AMP 跳步及不完整累积窗口需要明确处理。

辅助测试应先算正确的解析期望。例如每 rank loss 是 4 行输出之和再乘 `(rank+1)`，8 rank 平均梯度的系数是 `4×mean(1…8)=18`，不是 4.5。断言错误应修验证器，不能改正确算法去迎合断言。

## 官方资料与版本边界

使用时以目标已安装版本为准，必要时查询对应标签的文档和源码。更多按问题分类的官方入口及引用方法见[官方资料索引](official-links.md)。

- [昇腾 PyTorch 适配器及兼容矩阵入口](https://github.com/Ascend/pytorch)：核对 torch_npu、PyTorch、CANN 组合。
- [PyTorch 2.6 FSDP 文档](https://docs.pytorch.org/docs/2.6/fsdp.html)：本轮历史环境对应的包装、分片与 CPU offload 约束；不是所有版本的默认方案。

本文经验来自实际适配开发与原始日志复核。它提供诊断方法，不替代新模型、新版本或新拓扑的运行证据。
