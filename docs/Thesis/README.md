# Splatt3R-SLAM 论文稿(CVPR 格式)

## 构建

```bash
./build.sh          # 生成 main.pdf
./build.sh clean    # 清理中间文件
```

TeX Live 2026 安装完成后即可直接运行。若 `pdflatex` 不在 PATH，脚本会自动尝试
`/usr/local/texlive/2026/bin/x86_64-linux`。

## 字体:为什么用 pdflatex 而不是 XeLaTeX

本文**只使用 TeX 发行版自带的 Type 1 字体**——`cvpr.sty` 要求的 `times`
(URW Nimbus Roman)、Nimbus Sans、以及 Computer Modern 数学字体。这些字体随
TeX Live 和 MiKTeX 一同分发,**Linux 与 Windows 上完全一致**,PDF 可复现。

**请勿改用 XeLaTeX + fontspec**:那会引用操作系统安装的字体(如 Arial、
SimSun),在另一台机器上会因缺字体而回退或报错,破坏跨平台一致性。

## 文件结构

| 文件 | 内容 |
|---|---|
| `main.tex` | 主文件(纸型、宏包、标题、章节 include) |
| `main.bib` | 参考文献库(24 条) |
| `cvpr.sty` / `ieeenat_fullname.bst` | CVPR 2026 官方 author-kit,未修改 |
| `sec/0_abstract.tex` | 摘要 |
| `sec/1_intro.tex` | 引言(三个核心结果 + 对比发现) |
| `sec/2_related.tex` | 相关工作 |
| `sec/3_method.tex` | **预备知识**(Splatt3R/MASt3R-SLAM 数学基础,式 1--8)+ **方法**(适配面、注入时削薄式 9、refiner 式 10、opacity 机制) |
| `sec/4_protocol.tex` | **评测协议与协议偏移**(本文方法论贡献) |
| `sec/5_experiments.tex` | 实验(全部数据表) |
| `sec/6_limitations.tex` | 局限(紧凑性曲线 + 未解决问题) |
| `sec/7_conclusion.tex` | 结论 |
| `fig/teaser.tex` | 跨栏 teaser 图(Replica room0 三方对比) |
| `fig/*.png` | 7 张插图 |

## 插图

| 文件 | 用途 | 位置 |
|---|---|---|
| `cmp_replica_room0_f0.png` | teaser:三方同协议对比 | 首页跨栏 |
| `cmp_replica_office0_f1.png` | 第二组 baseline 对比 | 实验节 |
| `eth3d_sofa1_f1.png` | head-only 消融:ETH3D | 图 3 第 1 行 |
| `euroc_v101_f0.png` | head-only 消融:EuRoC | 图 3 第 2 行 |
| `tum_rpy_f0.png` | head-only 消融:TUM | 图 3 第 3 行 |
| `7scenes_office_f0.png` | head-only 消融:**弱家族** | 图 3 第 4 行 |
| `replica_ps_office2_f1.png` | photospatial 头(机制节) | 方法节 |

对比图的代表帧按**逐帧 ΔPSNR 中位数**选取,而非人工挑选——随机取帧可能与图注
结论相反(详见 `docs/external-baselines.md` 第六节)。

## 投稿模式切换

`main.tex` 中三选一:

```latex
% \usepackage{cvpr}              % 盲审模式:加行号、匿名
\usepackage[final]{cvpr}         % 当前:camera-ready / preprint
% \usepackage[pagenumbers]{cvpr} % preprint 带页码
```

## 公式来源

| 式 | 内容 | 出处 |
|---|---|---|
| 1--4 | 高斯均值 $\mu=x+\Delta$、协方差、$\alpha$-合成、Splatt3R 掩码损失 | Splatt3R~\cite{} (`docs/third_party/2408.13912.pdf`) |
| 5--6 | $\mathrm{Sim}(3)$ 位姿与左加、置信度加权点图融合 | MASt3R-SLAM (`docs/third_party/2412.12392.pdf`) |
| 7 | 锚定关键帧到世界的复合 | 本工作 |
| 8 | **注入时削薄**(秩归一化置信度加权) | 本工作,对应 `--refiner-conf-fade` |
| 9 | refiner 目标函数(L1 + DSSIM) | 本工作,`splatt3r_slam/refiner.py` |
| 10 | Umeyama 对齐求逆 | 评测协议,`scripts/eval_map_quality.py` |

式 3(α-合成)是全文机制论证的支点:黑背景下降低 $\alpha$ 使区域变暗、升高使其变亮,
这正是 opacity 作为"亮度修剪盘"而非置信度门的依据。

## 数据来源

所有数字出自 `docs/external-baselines.md` 与
`.claude/skills/splatt3r-finetuning-experiments/SKILL.md`(§17.93–17.95),
均为本机实测,无一抄自论文。
