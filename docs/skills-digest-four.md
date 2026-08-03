# 四份 skill 文档摘要(gaussian-map / color-consistency / retrieval-refit / lora-finetuning)

> 对 `.claude/skills/` 下除主实验档案外四份文档的通读摘要。跨文档时序:
> gaussian-map 与 color-consistency 主体 07-22 → LoRA saga 07-23~26(后被证伪)
> → retrieval-refit 07-27(Stage 1 NO-GO,Stage 3 前提随 LoRA 证伪瓦解)
> → color-consistency Plan 2 负结果 08-03(最新状态)。
> 当前权威训练结论在第五份文档 `splatt3r-finetuning-experiments`。

---

## 1. splatt3r-gaussian-map — 高斯地图重访重影(2026-07-22)

**根因**:MASt3R-SLAM 的点云地图天然抗漂移(每关键帧存相机系点云,绘制时按当前 `T_WC` 实时变换);原 Splatt3R 集成没学这招——追加时就把高斯按当时位姿一次性烘进世界系,写入只增不删的扁平环形缓冲。两个叠加 bug:(1) 每个跟踪帧都追加而非仅关键帧(非关键帧位姿不在位姿图里,无法校正);(2) `kf_id` 字段存的是推测性"下一个槽位"且无人读。净效果:重访区域叠加漂移过的第二副本,回环校正永远无法回溯修复。

**Plan A(已实现)**:完全镜像点云地图——按关键帧存相机系高斯、绘制时按当前 `T_WC` 实时烘焙。核心不变式:`SharedKeyframes.__setitem__` 只在 `value.gaussian_pred is not None` 时写 `gs_*` 字段(tracker 几乎每步重写最新关键帧的 `X_canon` 而那个 Frame 不带高斯,去掉守卫会静默清空全部高斯)。viz 端 `(T_WC快照, stride)` 缓存键,位姿变化自动重烘。

**关键数字/状态**:
- 显存预算:512 kf × 196608 像素 × 60B/点 ≈ **6.0 GB**;48GB A6000 舒适
- 出货默认:`spatial_stride` 1(用户指定,接受风险)、`max_gaussians` 16M、`min_opacity` 0.3 过滤恢复
- 崩溃链:`stride=1` 可复现 illegal memory access;四元数归一化+isfinite 是真 bug 但没治好;`stride=2` 只是缓解(room/floor 上 120–160 帧即崩);真凶后来在 LoRA 工作中查明:`scale_invariant=True` + 硬编码 `near=0.1` → 无条件 100× 协方差放大
- 已知限制:无跨 pass 几何去重(修的是"校正后错位"不是"冗余密度");`budget_stride` 全局均匀
- Plan B(仅设计):真实关键帧索引 + 每 kf 烘焙位姿快照 + 位姿校正时对 `kf_id` 匹配行施加刚性 delta(ΔT = T_new·T_old⁻¹)

**仍有效 vs 已取代**:
- 仍有效:根因诊断、Plan A 全部设计与不变式、显存预算、Plan B 规范、已知限制、`SPLATT3R_GS_DUMP_DIR` 调试模式
- 已取代:"单次调用高斯数量上限"的崩溃原理论——被 `scale_invariant` 的 100× 放大解释取代;"远景模糊需要训练"的指向——现由 head-only 路线接管(且 2026-08-03 光栅器 device guard 修复后,混合设备崩溃类已根除,见主档案 §15.11)

---

## 2. splatt3r-color-consistency — 重访表面颜色拼布(07-22,Plan 2 更新于 08-03)

**根因**:每个高斯的 DC 基色 = 网络残差 + `RGB2SH(该帧原始像素色)`。全管线没有任何环节调和同一物理表面的两次独立观测——自动曝光/白平衡漂移时,同一物理点的两份原始像素色不同并永久并排烘进地图。与重影 bug 是不同的失效模式:几何完全对齐也会发作。

**Plan 1(已实现,部分修复)**:因果式逐帧曝光归一化(`splatt3r_slam/image.py: normalize_exposure()`,各通道均值对齐到序列第一帧,增益钳 [0.4, 2.5])。实测:接缝从硬跳变软渐变但仍在——gray-world 只能修全局均匀漂移,修不了局部/方向性光照差异。

**Plan 2(08-03 实测 NEGATIVE)**:`scripts/color_harmonize.py`,因果序、共享 10mm 体素内对既有地图做逐通道最小二乘增益。结果:raw 12.4416/0.5027 → harmonized 11.8669/0.5158(**−0.57 dB**)。失败原因:拟合增益过强且单调(低至 0.64),在吸收头的逐视角色偏、视角方向 shading 与 10mm 配对误差,而非曝光漂移。**处置:不接进 bake 路径**;真正的调和器是 refiner 的光度优化(在线 +2.15 dB)。声明有界:desk 是轻漂移序列且 Plan 1 已在上游;强漂移无 Plan 1 时或许仍受益。

**Plan 3(未实现)**:重训让颜色头预测视角一致颜色,需新增跨视角一致性损失——现有 mse/lpips 全是单视角重建损失(该推理对 head-only 训练依然成立)。

---

## 3. splatt3r-retrieval-refit — 检索子系统去 MASt3R 化(2026-07-27)

**背景事实**:回路检索是唯一还吃 MASt3R 资产的子系统(`_trainingfree.pth` + `_codebook.pkl`)。倒出 .pth 验证:**整个检索头就是一个 PCA 白化,没有任何训练过的层**(prewhiten m/p,projector/postwhiten 均 Identity,top-300 token,64k 聚类,白化和码本各用 30k 张图)。特征流本身已是 Splatt3R 原生。潜在收益只有两条通道:域特化与漂移补偿(后者只有 encoder 被改动才存在)。

**Stage 0 基线(已实现,2026-07-27)**:回路开 vs 关(`--no-loop-closure`):
- fr1_room 0.0590 vs 0.0828(**+40%**);fr1_360 0.0421 vs 0.0770(**+83%**);fr1_desk(对照!) 0.0170 vs 0.0711(**+319%**)
- 结论:检索边是各种基线的重访约束,处处高价值;`--no-loop-closure` 只是消融工具,永远不是运行模式

**Recall@k 判定:NO-GO**。用全部 85,248 条 base 特征重拟合白化 + faiss 码本(2048/8192),重拟合白化在 3 条序列上 R@1 全部略差;码本对比样本不足无结论。**Stage 1 被拒,保留 MASt3R 资产**。附带观察:免量化 global-spoc 在 fr1_360 上打败所有 ASMK 配置(R@1 0.63 vs 0.50)——**ASMK 量化阶段本身可能才是检索瓶颈**,若检索质量成为优先级值得重查。

**Stage 3 状态**:前提(等 encoder-LoRA 适配器)已随 LoRA 证伪而瓦解;head-only 冻结 encoder、特征零漂移,按文档自己的逻辑"漂移补偿"通道不存在。后由 Route D(主档案 §15.12)再次结构性关闭:encoder 特征必须 bit-identical,decoder 适配不碰检索但也无益。**MASt3R 资产继续留任。**

**仍有效**:Stage 0 全部实现与基线数字、Recall 判定、三臂决策框架、阈值重扫与 asmk 风险登记、global-spoc 瓶颈观察、`eval_retrieval_recall.py` 可复用。

---

## 4. splatt3r-lora-finetuning — 已被证伪的 encoder-LoRA 路线(07-23~26,SUPERSEDED)

**证伪数字**(对照 base 的受控协议):psnr 9.50→7.41、lpips 0.3414→0.4545,约 **−49%**。根因:解冻了上游刻意冻结的 encoder,把 OOD 特征喂给 ~158M 冻结 matching/pointmap 头参数 → 高斯尺度爆炸(p99 为 base 的 **85×**)——也最可能是本项目长期 OOM/illegal memory access 史的源头。胜出的 route B:冻结 encoder 只训 Gaussian 头,+1.00 dB,峰值 6.3 GiB vs 42 GiB。

**工程遗产(仍有效,可复用)**:
- **数据管线**:TUM/7-Scenes 本地 RGB-D 足够,不需要 ScanNet++;EuRoC/ETH3D 无真深度,用 base 自预测伪深度(自视图前向取 `means` Z 通道、conf≥1.5——"自洽"而非"真值");四族适配器镜像 `ScanNetPPData` 接口;四元数序注意 TUM/ETH3D=xyzw、EuRoC=wxyz;train/val 连续切分防泄漏
- **覆盖矩阵**:dense `{i:{j:frac}}` 字典,desk 488 帧→15.3s,空间密集序列分钟级;磁盘缓存不按内容键控
- **bf16 兼容**:RoPE 内核 `AT_DISPATCH_...AND2(Half, BFloat16)` 补丁;`render_cuda()` 整体包 `autocast(enabled=False)`
- **崩溃/gotcha 目录**:DDP 子进程 abspath 一行修复;`scale_invariant=False` 根因(同时解释了 SLAM 侧 rasterizer 崩溃);resume 只带权重导致 val/loss 单调变差 → `RESUME_LR_FACTOR=0.1`;重试包装器孤儿打满双卡
- **最大隐藏 bug**:peft `save_pretrained()` 只序列化 `lora_` 前缀——**整场 saga 里 Gaussian 头的训练从未被保存/恢复**;修复 `modules_to_save=GAUSSIAN_HEAD_MODULES`(2.37M→49.4M)
- peft 实证:`get_peft_model()` 对非 HF encoder 可行;它会清掉 head 的 `requires_grad_(True)`,须手动重开

**已被证伪/取代**:LoRA 目标模块选择、全部超参、scale-penalty 补丁、热插拔适配器加载设计、`--lora` 用法。权威后继:`splatt3r-finetuning-experiments`。本文档只应作为工程考古与数据管线参考。
