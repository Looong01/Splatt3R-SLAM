# Splatt3R-SLAM — RA-L 与硕士毕业论文

当前默认稿件为 **RA-L 版本**（2026-10-07 修订），正文与图表位于 `ral/`。

| 版本 | 构建命令 | PDF | 用途 |
|---|---|---|---|
| RA-L 阅读稿 | `./build.sh` | [ral.pdf](ral.pdf) | IEEEtran 期刊版式，含 Zelong Li 作者信息 |
| RA-L 匿名稿 | `./build.sh review` | [ral-review.pdf](ral-review.pdf) | ieeeconf 首次/修订投稿版式 |
| 硕士论文 | `./build.sh master` | [master.pdf](master.pdf) | 65 页，A4 单栏、12pt、1.5 倍行距，12 章与 4 个附录 |

两版 RA-L 共用 `ral/document.tex` 与 `ral/` 正文，均为 7 页，含参考文献。
硕士论文已同步新增方法图、四列定性图、最新评测与相关论述，
保留 `master.tex`、`chap/` 的长篇结构，不导入 RA-L 正文或导言区。
两类文档只复用科学图源、实验生成的纯表格数据和 `main.bib`。
RA-L 使用 IEEE 排版及 `IEEEtran.bst`；硕士论文使用 `report` 与
发行版自带 `unsrtnat.bst`，图题、表题和编号各自独立。
旧会议稿及其专用模板、图片和构建入口已移除。

## 模板与作者

[RA-L 当前官方说明](https://www.ieee-ras.org/publications/ra-l/information-for-authors-ra-l)
要求首次/修订稿使用 `ieeeconf`、双匿名审稿，最终期刊样式使用 `IEEEtran`。
提供两种版本便于阅读与后续投稿准备；没有虚构接收日期、DOI、编辑或共同作者。
匿名 PDF 的作者元数据为空。源代码包包含署名版本，因此它是复现包，不是匿名投稿包。

`ieeeconf.cls` 来自
<https://ras.papercept.net/conferences/support/files/ieeeconf.zip>；
`IEEEtran.cls` 和 `IEEEtran.bst` 使用 TeX Live/MiKTeX 自带版本。
均使用 **pdflatex**，不依赖操作系统字体。正文为发行版 Type 1 字体；
系统总览由可编辑 PowerPoint 图源导出；旧 TikZ 进程图仍作为实现参考。
其余图使用 Matplotlib 自带 DejaVu 字体。全部字体嵌入 PDF。

## 图与数值来源

| 图 | 可编辑文件 | 内容 |
|---|---|---|
| Teaser | [PDF](ral/fig/teaser.pdf) / [SVG](ral/fig/teaser.svg) | 方法特点、实测渲染和 Replica 八场景均值 |
| Editable figures | [PPTX](ral/fig/Splatt3R-SLAM-scientific-figures.pptx) | RA-L/MSc 方法导航、锚点细节、定性对比、细粒度消融；图形和文字均可编辑 |
| RA-L overview | [PDF](ral/fig/overview-ral.pdf) / [SVG](ral/fig/overview-ral.svg) | 真实输入与优化前后视图，标注 RA-L 章节及公式 |
| MSc overview | [PDF](ral/fig/overview-master.pdf) / [SVG](ral/fig/overview-master.svg) | 内容相同，引用硕士论文章节及公式 |
| Process diagram | [TikZ 源码](ral/fig/overview.tikz) / [PDF](ral/fig/overview.pdf) | 进程、双 GPU 与 CPU 共享内存实现图 |
| 定性对比 | [PDF](ral/fig/qualitative_tum.pdf) / [SVG](ral/fig/qualitative_tum.svg) | GT / Photo-SLAM / MonoGS / Ours 四列，两视角及统一区域放大 |
| 性能与规模 | [PDF](ral/fig/evidence.pdf) / [SVG](ral/fig/evidence.svg) | 八场景 PSNR 差、独立地图裁剪实验 |

所有图片来自实际地图渲染，没有生成图或局部修饰。每场景两张代表帧按
逐帧 ΔPSNR 的两个中间秩选择；TUM 对 MonoGS，Replica 对 Photo-SLAM。
裁剪坐标见 `ral/fig/crop_manifest.json`，所有方法使用相同坐标。

新实验数据在仓库根目录 `logs/paper_ral_20261007/<scene>/metrics.json`：
每场景 100 个相同非关键帧、GT Sim3 对齐、曝光归一化、**保留全部导出 SH**。
JSON 包含 PLY/轨迹 SHA256、对齐变换、逐帧结果与选帧记录。
主对比为发布版 head + 300 秒 polish，关闭 thinning。

Replica 新均值：Ours **23.019214 dB / 0.134862 LPIPS**，
Photo-SLAM **19.482997 / 0.162673**。TUM desk：
Ours **13.949010 / 0.411030**，MonoGS **13.913588 / 0.395311**，
Photo-SLAM **10.665049 / 0.562209**。
非关键帧可能参与在线监督，因此本文报告重建质量，不声称严格未见视角测试。
这些结果不足以支持无条件 SOTA 宣称；正文保留预算、地图规模和 LPIPS 的边界。

## 复现

从仓库根目录运行（生成图需要 numpy / Pillow / matplotlib；PPTX 需要
python-pptx；PDF 导出使用 LibreOffice；SVG/PNG 预览使用 Poppler）：

```bash
python scripts/paper/make_ral_figures.py
CUDA_VISIBLE_DEVICES=0 python scripts/paper/render_refinement_pair.py
PYTHONPATH=/path/to/python-pptx python scripts/paper/make_overview_pptx.py
cd docs/Thesis
./build.sh
./build.sh review
./build.sh master
```

论文使用 PPTX 导出的 `overview-ral.pdf` 和 `overview-master.pdf`。
`render_refinement_pair.py` 从保存的 TUM 地图在同一相机渲染真实优化前后图；
`make_overview_pptx.py` 生成六页可编辑图。旧实现图可用
`python scripts/paper/build_overview.py` 单独导出。

重新渲染需要现有 `splatt3r-slam` 环境、数据集和外部系统地图：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/paper/render_comparisons.py \
  --scenes office0 office1 office2 office3
CUDA_VISIBLE_DEVICES=1 python scripts/paper/render_comparisons.py \
  --scenes desk office4 room0 room1 room2
```

数值表 `ral/generated/*_table.tex` 由 JSON 自动生成。
`frame_distribution_table.tex` 是硕士论文的扩展逐帧分析；
主对比表通过 `tab/replica.tex` 与 `tab/tum.tex` 包装为硕士论文自己的表格。
轨迹、组件消融、在线代价与地图裁剪曲线沿用已记录实测，来源和修正详见
[REVISION_NOTES.md](ral/REVISION_NOTES.md)。本次是重新渲染已保存地图，
未重新训练或重建全部比较系统。

运行 `python scripts/paper/package_ral.py` 生成 `docs/Splatt3R-SLAM-RA-L-source.zip`，
内含 TeX、PDF、可编辑图、生成脚本、评测 JSON 和渲染 PNG。
无需 GPU 即可重建 PDF 和图表；重新运行 GPU 渲染还需要大体积数据与模型产物，
它们不包含在源代码包中。

运行 `python scripts/paper/package_master.py` 生成 `docs/Thesis.zip`，
内含可独立编译的硕士论文源码、PDF、共用科学图和复现证据，
不含 RA-L 正文、IEEE 模板或旧会议稿。解压后进入 `docs/Thesis`，
运行 `./build.sh master`。第二评阅人和个性化致谢仍待作者填写；
毕业论文排版沿用既有约定，不声称是已核实的学校官方模板。
