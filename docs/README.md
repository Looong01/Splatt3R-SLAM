# Splatt3R-SLAM 研究工作总览(2026-07-20 → 2026-08-18)

**2026-10-07 更新：** 当前论文入口为 [IEEE RA-L 阅读稿](Thesis/ral.pdf) 和
[匿名 ieeeconf 稿](Thesis/ral-review.pdf)，两版均为 7 页。新增系统总览、
方法与性能 teaser、GT / Photo-SLAM / MonoGS / Ours 四列定性图。
新评测保留全部导出 SH；Replica 八场景均值为 Ours 23.02 dB、
Photo-SLAM 19.48 dB。构建与数据说明见 [稿件 README](Thesis/README.md)。
硕士论文 [master.pdf](Thesis/master.pdf) 已同步上述新内容与评测修正，
保留 A4 章节式毕业论文排版，独立于 RA-L 模板。
可编辑演示图单独打包为
[Splatt3R-SLAM-editable-figures.zip](Splatt3R-SLAM-editable-figures.zip)，
不参与两类论文的构建。
完整答辩演示位于 [Thesis/Splatt3R-SLAM-defense.pptx](Thesis/Splatt3R-SLAM-defense.pptx)，
PDF 预览为 [Thesis/Splatt3R-SLAM-defense.pdf](Thesis/Splatt3R-SLAM-defense.pdf)。
答辩交付包为 [Splatt3R-SLAM-defense-presentation.zip](Splatt3R-SLAM-defense-presentation.zip)。
以下研究时间线和分报告属于历史记录：其中“严格 held-out”“统一 8.7 dB
协议偏移”等旧解释已在新稿修正，不应直接作为当前结论引用。

本目录是全部研究工作的汇总索引。六份分报告:

| 文档 | 内容 |
|---|---|
| **[external-baselines.md](external-baselines.md)** | **外部基准对比(Photo-SLAM / MonoGS / MASt3R-SLAM / VGGT-SLAM):统一评测协议、Replica 全 8 场景与 TUM 全 9 序列结果、紧凑性曲线、系统开销、各 baseline 构建记录、可站得住的论文表述** |
| [chronicle-logs-git.md](chronicle-logs-git.md) | 按日编年史、checkpoints 产物对照、logs/ 十大产物 |
| [finetuning-experiments-part1.md](finetuning-experiments-part1.md) | 微调实验全史(Route A/B/C、协议、SLAM 验证、3DGS 细化、可部署性论证) |
| [online-refinement-campaign.md](online-refinement-campaign.md) | 在线精修战役((f)(g)(b′)(e) 判定、refiner 集成、双卡部署、科学闭环) |
| [skills-digest-four.md](skills-digest-four.md) | 四份专题 skill:重影修复、颜色一致性、检索子系统、LoRA 工程考古 |
| [online-eval-all-families.md](online-eval-all-families.md) | 全数据集实机评估(GUI 在线,四大家族 9 序列,ATE/地图质量/延迟/VRAM 全指标) |

> **对外报告数字前必读** [当前评测修正](Thesis/ral/REVISION_NOTES.md)：
> 历史位姿诊断说明评测对协议敏感，但不能推出统一的 8.7 dB 偏移。
> 新稿使用全 SH 评测与修正后的帧关联；旧记录中的绝对分数不能直接混用。

[`Thesis/`](Thesis/) 默认 `./build.sh` 生成 `ral.pdf`；
`./build.sh review` 生成 `ral-review.pdf`。两版 RA-L 共用 `ral/` 正文及图表。
硕士论文用 `./build.sh master`，正文独立位于 `chap/`；
旧会议稿及其专用模板和入口已移除。各版本共用 `main.bib` 和科学图源，
新渲染表由实验 JSON 生成到 `ral/generated/`，图表标题与排版各自独立。

写作状态、公式出处、结构决策、**不得润色的"难看"段落清单**与待作者填写的占位符,
记录在 skill `splatt3r-thesis-writing`——**改动 `docs/Thesis/` 下任何内容前先读它**。

原始权威记录:`.claude/skills/` 七份 skill(`splatt3r-finetuning-experiments`
已逾 14000 行)。其中论文素材集中在:§16(9 条 claim + 天花板 + 方法学陷阱)、
§17.93(外部 baseline 全过程)、§17.94(六张成稿数据表)、§17.95(图表清单与选帧规则)。

---

## 一、一句话总结

从"Splatt3R 权重直接驱动 SLAM"出发,两周内完成:微调路线去伪存真(encoder-LoRA 证伪、head-only 证实)→ 四家族生产头(+1.8~3.7 dB)→ 离线细化机制全部测穿(位姿闸门、共适应、双壳)→ **在线 refiner 落地并上线:双卡配置下地图质量 +2.15 dB、跟踪延迟 +1%、ATE 逐位一致**。

## 二、时间线五幕

1. **环境与原罪(07-21~22)**:CUDA 13.2 就位、SLAM 首跑、两个结构性 bug 修复(重影 → 关键帧局部存储 Plan A;色漂 → 逐帧曝光归一化 Plan 1)。
2. **微调路线证伪与确立(07-23~28)**:encoder-LoRA 全线崩溃(−49%,尺度爆炸 1821×)→ 根因是给 158M 冻结头喂 OOD 特征;head-only(route B)+1.00 dB 确立;测量协议修复(固定索引≠固定样本,0.6 dB 噪声带)。
3. **生产头与机制测量(07-28~31)**:四家族 40-epoch 头(TUM +1.78 / 7S +2.18 / EuRoC +3.69 / ETH3D +3.15);SLAM 级验证(ATE 位级不变、地图 +0.90 dB);跨家族仅域内有效;3DGS 细化纠偏链(对照退化、种子方差、几何 vs 光度);位姿闸门发现(1.8→4.5 dB)。
4. **可部署性与共适应(07-31~08-01)**:FramePoseLog 建成;视图 ~25 饱和;**GT 位姿建图恢复 +51.8% vs SLAM 位姿建图 −22.2%——共适应在烘焙瞬间产生**,P1 局部帧设计由此而来;CUDA 相机梯度建成(并揪出 sh[14] bug)。
5. **在线化落地(08-02~03)**:两个 go/no-go 判定(因果回放 GO、双卡无争用)→ refiner 接入三进程架构 → **+2.15 dB 在线交付**;光栅器 device guard 修复;stage 5 有据不建;route D 完成机理隔离。

## 三、最终交付物

**系统**(全部已推送 GitHub main,v0.2.0):

- `main.py --refiner`:在线地图精修进程(关键帧局部高斯、锚定随动监督、duty 限流、双卡计算、10mm 去重生命周期、精修快照发布)
- `main.py --head`:四家族 head-only 生产权重(`checkpoints/head_only_long/`)
- 持久化:`<seq>.txt`(轨迹)/ `_frames.txt`(逐帧位姿)/ `_gaussians.ply`(烘焙地图)/ `_refined.ply`(精修地图)/ `_kfgauss.pt`(关键帧高斯)
- 测量基建:`--frame-timing`、`--no-loop-closure`、`--dump-retrieval-features`、`--dump-keyframe-gaussians`

**数字**(desk,协议见各分报告):

```
地图质量(held-out NVS psnr)
10.66  烘焙地图(现状)
11.90  同卡 duty 0.25 在线精修
12.81  双卡不限速在线精修        ← 交付点
14.45  离线润色上限(估计位姿)
~19    GT 位姿离线上限(不可交付,缺口全是位姿)

控制指标:ATE 0.017158(逐位一致)、跟踪延迟 102ms vs 基线 101ms、7.6 fps
```

**判定一览**:

| 项 | 判定 |
|---|---|
| encoder-LoRA 微调 | 证伪(−49%,尺度爆炸) |
| head-only 微调 | **证实**(四家族 +1.78~+3.69 dB,SLAM 级 +0.90 dB,ATE 不变) |
| 跨家族泛化 | 仅域内;跨域主动有害 |
| decoder-only LoRA | 安全 2 轮后塌缩;head-only 是终点 |
| 因果约束下在线增益 | **存活**(严格实时 +1.21 dB) |
| 同卡 GPU 争用 | 延迟翻倍;**双卡是部署答案** |
| 回环接缝 | 无撕裂,修正保持(忠实监督语义下) |
| 在线位姿精化(stage 5) | **有据不建**(恢复率 ~11%,误差烘在簇内) |
| 每关键帧增益式颜色调和 | 证伪(−0.57 dB) |
| 检索资产重拟合 | 双重关闭(encoder 必须 bit-identical) |

## 四、方法学沉淀(写论文/复现前必读)

1. 固定索引 ≠ 固定样本——seeded draw 是相对比较的唯一有效形式(0.6 dB 采样噪声带)
2. 端到端运行间噪声地板 ~0.09 dB(atomicAdd)——废单次值、废峰值摘取
3. 两套 psnr 绝对尺度并存(Δ 一致,绝对值注明);全部 LPIPS 为 AlexNet 口径
4. 一个"什么都不做"的实验长得和强负结果一模一样——给你声称操纵的东西装仪表
5. 六层自我纠偏各自在不同代码状态测量——每个探针记录 git hash
6. 多个结论为 desk 单场景单种子;"plateau"判读三次全错,只报 best-so-far
7. Sim3 对齐残差是待测位姿效应量的 ~60%——est-vs-GT 比较全部继承该仪器误差

## 五、未竟事项(诚实的边界)

- 在线数字仅 TUM desk/room 两序列单种子;房间尺度复现是最近的下一步
- viz 精修地图显示:发布侧已验证,消费侧因 headless 环境从未绘制
- 长序列去重生命周期(仅 desk 尾部单次触发验证)
- 30 fps 不在当前地图规模的可及范围(工作点 ~7.6 fps @ 1.86M 高斯)
- 到 ~19 dB 的剩余缺口全是位姿,杠杆在融合阶段本身,未开工
