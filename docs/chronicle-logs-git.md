# Splatt3R-SLAM 实验活动编年史(2026-07-20 → 2026-08-03)

> 基于 logs/ 文件时间戳、各日志尾部结论、checkpoints 内容、脚本 docstring,并与 `.claude/skills/splatt3r-finetuning-experiments/SKILL.md`(3593 行实验档案,最后更新 08-03)交叉验证。

---

## 一、git 层面概览

时间窗内只有 3 个上游提交 + 2 个收尾提交:

- `7f8b535`(07-21)Update to CUDA 13.2;`389f706`(07-22)Update CMakeLists
- `ecd5182`(08-03)"Update" —— 一次性提交整场战役:5 个 skill 文档(5600+ 行)、~25 个新脚本、`main.py` +789 行(refiner 在线精化、`--frame-timing` 等)、`config/rt_calib.yaml` 等(已从提交中剔除 3 个光栅器崩溃转储大文件,共 ~790MB)
- `f82163c`(08-03)Docs: README/setup.py/pyproject.toml 适配在线精修(v0.2.0)

## 二、按日编年表

**07-21 — 环境就位**
- 下载 Splatt3R 预训练权重 `checkpoints/epoch=19-step=1200.ckpt`(3.2 GB);建好 4 个数据集的 download/eval 脚本;升级到 CUDA 13.2

**07-22 — 首次 SLAM 跑通**
- 下载 eth3d/euroc 数据集;`logs/tum/calib`、`logs/gaussian_renders/`、`logs/keyframes/`;产出第一批高斯地图 room/desk .ply

**07-23 — Route A(encoder LoRA)启动**
- `logs/lora_training/tum`(lightning 日志)、`checkpoints/lora_coverage_cache/tum_*`(coverage 矩阵缓存)

**07-25 → 07-26 — Route A 失败攻坚(encoder LoRA = −49%)**
- `full_run.log`(20 MB)+ `debug_run`~`debug_run6` 连续调试;`checkpoints/lora/` 下各代产物记录尺度爆炸(scale explosion)与单向 clamp ratchet 两个 bug 的排查

**07-27 — LoRA 四家族训练 + 检索重拟合预研**
- `train_lora_all_families.sh` / `precompute_pseudo_depth.py`(euroc/eth3d 无真深度,用 base 自预测);`eval_lora_scenes.py` → LoRA vs base 基本打平
- 检索线:`retrieval_features/`、`retrieval_ab_a/b`(loop closure 开关 ATE 对比)、`retrieval_recall/results.json` —— 重拟合白化+码本 Recall@1 判定 NO-GO

**07-28 — Route B(head-only)确立,全天高产**
- `headonly_training/` + `checkpoints/head_only/tum/head_best.pt`(6 epoch +16.5%);`exp_rescore_seeded.log` 修复"固定索引≠固定样本"的协议缺陷
- Route C:`exp_head_only_lpips.log`(LPIPS 权重 1.0,小增益);`render_compare/` 同样本渲染对比
- `SequenceExposureLock`(曝光锁)当日接入,**先于所有 40-epoch 生产头**
- 下午:三家族 coverage 预计算 + `exp_batch_scan` 吞吐/显存扫描
- 晚:40 epoch 长训启动,TUM **+23.2%**;`head_ate_base/` + `head_ate_head/` SLAM 级验证 → ATE 逐位一致;`diag_psnr.py` 解释读数刻度

**07-29 — 跨家族与几何验证**
- `exp_head_only_7scenes/euroc/eth3d` 三个 40-epoch 生产头(+2.18 / +3.69 / +3.15 dB);`eval_cross_family*` → **增益仅限 in-domain**;`factorial_desk.log`(2×2:致密化×初始化)→ 致密化救不了随机初始化;`geom_ab.log` 首次几何评估(head L1 0.133m vs base 0.181m)

**07-30 — 对照组退化诊断 + 模糊偏好诊断**
- `exp_head_only_eth3d_v2.log`(+40.5%);`multiscene*` 多场景复现(psnr 故事不泛化、lpips 故事泛化);`probe_map/probe_clone/fixed_control/isolate_*` 定位"随机初始化对照组在高斯数量级上退化";`diag_blur*` 证伪"LPIPS 导致模糊"假设

**07-31 — 种子方差、可部署性与相机梯度**
- `head_seeds/`(随机臂 ±0.66 dB,地图臂无);`sweep_v2.log`(修复对照组重扫)
- `kf_stride4/` + `missing_cell.log`(补齐 amortized 协议缺失格);`refined_geom.log` + `refined_maps/` → **精化的 +6.7 dB 纯是光度的,几何不改善**
- 下午重建光栅化器加入 CUDA 相机位姿梯度,`test_camera_gradient.py` / `test_sh_backward.py` 验证
- `frames_desk/room.log` 逐帧位姿持久化(FramePoseLog);`long_refine.log` GT 位姿 30k 迭代天花板 ~19.95 dB;位姿优化第一轮

**08-01 — "位姿精化器还是光度海绵"大辩论**
- `posediag.log` 建立判据;`inject/inject2/3/4.log` 受控注入位姿扰动测恢复率;各 LR/迭代臂;room/desk 复现;`map_gtposes_desk.ply`(GT 位姿烘焙地图)
- 结论三反转:SLAM 位姿图上恢复 −22.2%(越修越差),GT 位姿图上 +51.8% —— 问题是**地图与位姿共适应**,不是方法弱

**08-02 — 上线化战役(P1 stages + 两个 go/no-go)**
- 凌晨:`stage1/stage23.log`(trajectory-anchored 地图 stage 1-3)、`ceil30k.log`(GT 30k 天花板 20.61 dB)
- 下午:`stage3_seam.log`(回环 Sim3 校正接缝测试,通过);**go/no-go (g)** → 同卡延迟翻倍(101→206ms),跨卡免费
- 晚:**go/no-go (f)** `refine_causal/`(8 个臂)—— 因果重放下增益不蒸发,120 迭代严格实时也有 +1.21 dB;`refine_local/` 接缝差异化测试;room kfgauss 转储(51 关键帧)

**08-03 — 在线 refiner 落地 + 收尾判决**
- `refiner_eval/`(duty 0.25 同卡,+1.24 dB)→ `refiner_xgpu/`(第二块 GPU 不限速,**+2.15 dB**,tracker 延迟 +1%,ATE 逐位一致)→ `refiner_dedup/`(10mm 去重生命周期,地图 −27.6%)
- `color_harmonize_desk.log` —— Plan 2 色彩调和 **NEGATIVE**(−0.57 dB)
- `exp_dec_lora.log` —— Route D(decoder-only LoRA):第 1 epoch +0.37 dB,第 5 epoch 崩溃 −1.61 dB → "任何上游适配都会塌,head-only 是终点"
- skill 定稿 §16(paper-ready summary),全部提交并推送

## 三、checkpoints/ 产物对照

| 路径 | 内容 | 对应实验 |
|---|---|---|
| `epoch=19-step=1200.ckpt` | 官方 Splatt3R 预训练(3.2 GB) | 一切训练的 base |
| `MASt3R_...retrieval_trainingfree.pth` + `codebook.pkl` | MASt3R 官方检索资产 | 回环检索 baseline |
| `head_only/tum/head_best.pt` | 6-epoch head-only 头 | Route B 首轮 +1.00 dB |
| `head_only_lpips/tum/head_best.pt` | LPIPS 权重 1.0 的头 | Route C |
| `head_only_long/{tum,7-scenes,euroc,eth3d,eth3d_v2}/` | 5 个 40-epoch 生产头 | 主交付物(+1.78~+3.69 dB) |
| `head_seeds/{tum_s0,s1,s2,eth3d_s0}/` | 多种子复训头 | 种子方差测量 |
| `lora/tum*/` | Route A 各代(含 NEGATIVE_result) | encoder LoRA 失败案卷 |
| `lora_coverage_cache/*.pkl` | 4 家族 coverage 矩阵(~1 GB) | 训练数据加载加速 |
| `exp_dec_lora/dec_lora_best.pt` | decoder-LoRA 最佳(崩溃前) | Route D |

## 四、logs/ 下最重要的 10 个产物

1. **`logs/refiner_xgpu/`** —— 最终交付:第二 GPU 在线精化 +2.15 dB、tracker 延迟 +1%、ATE 不变
2. **`logs/refiner_eval/`** —— 单卡 duty 0.25 版本 +1.24 dB,证明无第二块卡也可部署
3. **`logs/refine_causal/`** —— go/no-go (f):因果重放 8 臂,增益在直播因果约束下不蒸发
4. **`logs/contention/*.csv`** —— go/no-go (g):逐帧延迟证明同卡翻倍、跨卡免费,决定部署形态
5. **`logs/head_ate_base/` + `head_ate_head/`** —— head-only 头的 SLAM 级验证:地图 +0.90 dB、高斯数 −16%,ATE 逐位一致
6. **`logs/exp_head_only_long.log`** —— Route B 40-epoch TUM 判决 +23.2%,head-only 路线基石
7. **`logs/exp_head_only_eth3d_v2.log`** —— eth3d 40-epoch +40.5%,跨家族最大增益(in-domain)
8. **`logs/refined_maps/` + `refined_geom.log`** —— 精化增益的诚实边界:光度 +6.7 dB 但几何不改善
9. **`logs/retrieval_recall/results.json`** —— 检索重拟合的唯一量化结论;后因 encoder 必须 bit-identical 而封存
10. **`logs/map_gtposes_desk.ply` + `posediag.log`** —— "位姿精化是海绵"的决定性证据(GT 图 +51.8% vs SLAM 图 −22.2%),直接导致 stage 5 被**刻意不建**

> 注:`.claude/skills/splatt3r-finetuning-experiments/SKILL.md` §16 是这场战役自带的 paper-ready 总结(C1–C9 九条 claim 均带产物指针),是解读这批日志最权威的索引。
