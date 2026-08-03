# 在线精修战役报告(2026-08-02/03,P1 全队列)

> 本轮工作的完整记录:从六个 go/no-go 实验到在线 refiner 落地、双卡部署、
> 以及三条科学闭环。全部数字为实测,协议见主档案 §15(逐节)与 §16(汇总)。
> 实验单元:TUM freiburg1_desk(14 关键帧 / 1,860,034 高斯 / 50 监督视角),
> 复制序列 freiburg1_room(51 关键帧 / 8.77M 高斯)。

---

## 1. 实验判定(先测后建)

### (f) 因果回放 — GO,增益在所有预算存活

事后测量的所有增益都是"优化器看到全序列监督"得出的;在线时关键帧 k 只有 ≤k 的监督。`scripts/refine_causal.py` 按时间戳注入关键帧、按到达解锁监督、迭代∝时间分配、注入即重建 Adam(与在线机制一致):

| 预算 | causal psnr | posthoc psnr | 结论 |
|---|---|---|---|
| 120 步(严格实时) | 13.6511 | 13.8002 | 比不优化 **+1.21 dB** |
| 500 步(~3min 润色) | 14.3942 | 14.3223 | 追平并略超事后 |
| 1000 步 | 14.4747 | 14.3516 | +2.03 dB vs init |
| 3000 步 | 14.4123 | 14.3747 | +1.97 dB vs init |

### (g) GPU 争用 — 答案是第二块卡

`main.py --frame-timing` 逐帧埋点,三臂对照(306 帧/臂,ATE 全部 0.017158±1e-6):

| 臂 | track p50 | iter mean | 可持续帧率 |
|---|---|---|---|
| 基线(GPU0 空) | 101 ms | 125 ms | 8.0 fps |
| 跨卡(负载在 GPU0) | 103 ms | 133 ms | 7.5 fps |
| 同卡争用 | 206 ms | 223 ms | 4.5 fps |

延迟翻倍全部来自同卡 SM 争抢;离线 ATE 不变(管线无丢帧机制,ATE 风险只在直播场景实体化)。吞吐标定:全图 1.86M 高斯 ~3-4 it/s。

### (b′) 忠实差分接缝测试 — 通过(desk + room 双向复制)

旧版扰动测试的监督相机是冻结的,真实系统里 `FramePoseLog` 会把非关键帧随锚点关键帧一起重算。新版(`refine_local.py --perturb-mode block`):轨迹后半段整块 Sim3(回环形态),监督随锚点随动,held-out 三分法:

- **两臂均无接缝撕裂**(6% 缩放块修正下,注入后单调恢复)
- **忠实臂保持修正**(残差 −0.88 dB 平台)——在线场景这正是所要:细化不撤销回环;跨块监督集中在 overlap 区(恢复 +0.81 vs +0.13,机理吻合)
- **冻结对照**把注入修正完全拉回(证明没有锚定随动时,细化 ~500 步就撤销回环修正)——`FramePoseLog` 的价值首次端到端量化
- room 上三分法正常区分(4/5/41),注入瞬间 low-only 类逐位不动(仪器自检通过)

### (e) 监督采样消融 — uniform 全胜

500 步预算,整体 / 早半区(遗忘) / 晚半区(同化):

| 采样 | 整体 | 早 | 晚 |
|---|---|---|---|
| uniform(全历史) | **14.40** | **14.09** | **14.72** |
| mixed 70/30 | 14.21 | 13.95 | 14.49 |
| recent-only | 14.06 | 13.79 | 14.35 |

recent 窗口的同化优势不存在(因果到达本身已带时近偏置),遗忘劣势随严格度放大 → `SupervisionFrames` 默认改储层主导(recent_frac 0.3)。

### 去簇消融 — 双壳假说证实,尺度 ~10mm

重叠区视角优化后反而比单覆盖区差(§13.14 遗留异常)。体素去重(共享体素只留最早关键帧):

| 臂 | 删除量 | 缺口(single−overlap) |
|---|---|---|
| 对照 | 0% | +0.62 dB |
| 5mm | 14.2% | +0.63(无效) |
| 10mm | 28.4% | **+0.18(消掉 2/3)** |

壳间距 ~10mm(与跨关键帧最近邻 p10≈11mm 互证);去重把质量从单覆盖区再分配到多覆盖区,总量近零成本。

---

## 2. 工程实现(全部由上面的测量驱动)

### (a) 进程集成(stage 4)— refiner 上线

`run_refiner` 接入 `main.py` 三进程架构(tracker / backend / refiner):

- **锚定相对监督**:CPU 共享 uint8 帧 + (anchor_idx, T_anchor_frame),采样时经锚点当前位姿合成——回环时监督跟地图走((b′) 结论);CPU 通道顺带解开跨卡限制
- **储层主导采样**((e) 结论)
- **duty 限流**((g) 结论):步间隔 EMA 睡眠,丢步不丢帧
- **终止保存**:后端排空后合成最终位姿,写 `<seq>_refined.ply`
- 预注册失败条件"ATE moves at all"**未触发**(0.016975 vs 基线 0.0170)

### 双卡部署 — 两个 vendor bug 的修复

1. **lietorch group ops 在 cuda:1 静默算错**(Sim3 组合返回零平移)——规避:位姿数学留在共享缓冲所在设备,只搬结果矩阵
2. **vendored 光栅器无 device guard**:多卡可见时缓冲分配在"当前设备"而输入指针在另一张卡 → illegal memory access。加 `c10::cuda::OptionalCUDAGuard` 于三个入口(forward/backward/markVisible)并重编译;cuda:1 探针轨迹与 cuda:0 一致。**历史上"高斯积累后 illegal memory access"的记录很可能同源**

### (c) 去重生命周期

`--refiner-dedup-voxel` / `--refiner-max-gaussians`:地图超阈值触发 10mm 去重(最早属主规则),Adam 动量随存活者子集迁移。desk 验证:1.85M→1.34M(−27.6%),ATE 不变;尾部一次性触发 −0.53 dB(缺恢复预算)——**定位是长序列规模控制,不是质量功能**。

### viz 发布侧

`RefinedMapSnapshot`:13 float/高斯、CPU 共享双缓冲 + 版本计数;消费侧(`Window._read_refined`)已接线,headless 环境无法绘制验证(如实标注)。

---

## 3. 科学闭环

### stage 5(在线位姿相位)— 测量决定:不建

三条证据:(1) 真实误差的事后恢复率仅 ~11%(iid 扰动 41-58% 是对照上限);(2) 重锚定到 GT 仅 +0.13 dB——误差已烘进簇内部(§13.14);(3) 位姿相位意味着 refiner 写位姿图——给 backend 引入第二写者,反馈环风险。重开条件:融合侧先恢复位姿信号。

### Route D(decoder-only LoRA)— 安全两轮后塌缩,§3.3 机理隔离

LoRA 只上 decoder(encoder 冻结,检索特征逐位不变),route B 同协议:

| epoch | psnr | scale_p99 |
|---|---|---|
| BASE | 15.0933 | 0.0728 |
| 1(最佳) | 15.4597(+0.37) | 0.0259 |
| 3 | 14.0916 | 0.0547 |
| 5 | **13.4834(−1.61)** | **0.3181(12× 爬升)** |

**机理隔离**:失败不在 encoder,而在"head 上游任何适配"——encoder 立即塌(−49%),decoder 两轮后塌,同一尺度爆炸签名。Route B(head-only)被论证为终点;检索资产问题双重关闭(encoder 必须 bit-identical;decoder 适配无益)。

### normalize_exposure — 早已闭环

训练侧 `SequenceExposureLock`(序列级锁定增益,首帧→0.5)于 07-28 上线,**先于全部 40-epoch 生产头**(mtime + 探针双重验证);部署侧逐帧对齐;§9 的 SLAM 级验证已实测该组合兼容(+0.90 dB,ATE 逐位一致)。§14 陈旧记录已修正。

### Plan 2 颜色调和 — 实测 NEGATIVE

−0.57 dB;拟合增益吸收的主要是 head 色偏/视角效应而非曝光漂移。不接烘焙路径(详见 color-consistency skill 与本目录 skills-digest-four.md)。

---

## 4. 最终效果(可交付形态)

```
10.66  烘焙地图(现状,eval_map_quality n=100)
11.90  同卡 duty 0.25 在线精修(56 步)
12.81  双卡不限速在线精修(225 步)  ← 交付:+2.15 dB,跟踪延迟 +1%,ATE 逐位一致
14.45  离线润色上限(估计位姿)
~19    GT 位姿离线上限(不可交付)
```

运行方式:

```bash
# 双卡(推荐)
CUDA_VISIBLE_DEVICES=0,1 python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
    --config config/eval_calib.yaml --refiner --refiner-gpu 1 --refiner-duty 1.0
# 单卡(限流)
python main.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
    --config config/eval_calib.yaml --refiner
```

边界声明:在线数字仅 TUM desk/room 两序列单种子;工作点 ~7.6fps@1.86M 高斯(非 30fps);viz 消费侧未绘制验证;长序列去重未验证。
