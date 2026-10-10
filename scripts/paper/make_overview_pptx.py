"""Create editable scientific figures for the RA-L and MSc manuscripts."""
from pathlib import Path
import json
import math

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[2]
PAPER_FIG = ROOT / "docs/Thesis/ral/fig"
OUT = ROOT / "docs/EditableFigures"
ASSET = PAPER_FIG / "overview_assets"
DATA = ROOT / "logs/paper_ral_20261007"

WHITE = "FFFFFF"
INK = "18323F"
MUTED = "607681"
LINE = "CAD6DA"
BLUE = "3478A5"
BLUE_BG = "EAF3F8"
GREEN = "168077"
GREEN_BG = "E9F6F3"
ORANGE = "C36B31"
ORANGE_BG = "FBEFE8"
MAGENTA = "A14D78"
MAGENTA_BG = "F8EDF3"
YELLOW = "D8A72F"


def rgb(hex_value):
    return RGBColor.from_string(hex_value)


def add_text(slide, x, y, w, h, value, size=12, color=INK, bold=False,
             align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.MIDDLE):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.margin_left = frame.margin_right = Inches(0.03)
    frame.margin_top = frame.margin_bottom = Inches(0.01)
    frame.vertical_anchor = valign
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.add_run()
    run.text = value
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    return box


def add_box(slide, x, y, w, h, title, subtitle="", fill=WHITE, stroke=LINE,
            title_color=INK):
    shape = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(stroke)
    shape.line.width = Pt(1)
    shape.adjustments[0] = 0.08
    add_text(slide, x + 0.12, y + 0.08, w - 0.24, 0.28, title, 12,
             title_color, True)
    if subtitle:
        add_text(slide, x + 0.12, y + 0.39, w - 0.24, h - 0.45, subtitle,
                 8.7, MUTED, False, valign=MSO_ANCHOR.TOP)
    return shape


def add_line(slide, x1, y1, x2, y2, color=MUTED, width=1.6):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(x1), Inches(y1), Inches(x2), Inches(y2),
    )
    line.line.color.rgb = rgb(color)
    line.line.width = Pt(width)
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    tri = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.ISOSCELES_TRIANGLE,
        Inches(x2 - 0.055), Inches(y2 - 0.055), Inches(0.11), Inches(0.11),
    )
    tri.rotation = angle + 90
    tri.fill.solid()
    tri.fill.fore_color.rgb = rgb(color)
    tri.line.fill.background()
    return line


def add_picture_cover(slide, path, x, y, w, h, stroke=WHITE):
    path = Path(path)
    with Image.open(path) as image:
        image_ratio = image.width / image.height
    box_ratio = w / h
    picture = slide.shapes.add_picture(
        str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h)
    )
    if image_ratio > box_ratio:
        visible = box_ratio / image_ratio
        picture.crop_left = picture.crop_right = (1 - visible) / 2
    else:
        visible = image_ratio / box_ratio
        picture.crop_top = picture.crop_bottom = (1 - visible) / 2
    picture.line.color.rgb = rgb(stroke)
    picture.line.width = Pt(0.8)
    return picture


def add_picture_roi(slide, path, roi, x, y, w, h, stroke=ORANGE):
    path = Path(path)
    with Image.open(path) as image:
        image_width, image_height = image.size
    roi_x, roi_y, roi_width, roi_height = roi
    picture = slide.shapes.add_picture(
        str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h)
    )
    picture.crop_left = roi_x / image_width
    picture.crop_right = (image_width - roi_x - roi_width) / image_width
    picture.crop_top = roi_y / image_height
    picture.crop_bottom = (image_height - roi_y - roi_height) / image_height
    picture.line.color.rgb = rgb(stroke)
    picture.line.width = Pt(1)
    return picture


def add_camera(slide, x, y, scale=1.0, color=BLUE):
    body = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.TRAPEZOID,
        Inches(x), Inches(y), Inches(0.34 * scale), Inches(0.24 * scale),
    )
    body.rotation = 90
    body.fill.solid()
    body.fill.fore_color.rgb = rgb(WHITE)
    body.line.color.rgb = rgb(color)
    body.line.width = Pt(1)
    add_line(slide, x + 0.17 * scale, y + 0.12 * scale,
             x + 0.43 * scale, y + 0.12 * scale, color, 1)


def add_gaussians(slide, x, y, scale=1.0):
    specs = [
        (0.00, 0.18, 0.34, 0.15, 15, BLUE),
        (0.28, 0.02, 0.30, 0.14, -20, GREEN),
        (0.38, 0.30, 0.36, 0.16, 10, ORANGE),
        (0.68, 0.12, 0.27, 0.13, -12, MAGENTA),
    ]
    for dx, dy, w, h, angle, color in specs:
        ellipse = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.OVAL,
            Inches(x + dx * scale), Inches(y + dy * scale),
            Inches(w * scale), Inches(h * scale),
        )
        ellipse.rotation = angle
        ellipse.fill.solid()
        ellipse.fill.fore_color.rgb = rgb(color)
        ellipse.fill.transparency = 22
        ellipse.line.color.rgb = rgb(color)
        ellipse.line.width = Pt(0.8)


def add_title(slide, title, subtitle):
    add_text(slide, 0.35, 0.12, 8.7, 0.36, title, 19, INK, True)
    add_text(slide, 9.2, 0.14, 3.75, 0.28, subtitle, 9, MUTED, False,
             PP_ALIGN.RIGHT)
    divider = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(0.35), Inches(0.55), Inches(12.98), Inches(0.55),
    )
    divider.line.color.rgb = rgb(LINE)
    divider.line.width = Pt(0.8)


def overview_slide(prs, master=False):
    from draw_paper_diagrams import method
    return method("master" if master else "ral").add_slide(
        prs, x=.30, y=.40, width=12.73)


def anchor_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Why the supervision camera shares the anchor",
              "Coordinate control and refinement loop")
    add_box(slide, 0.40, 0.90, 3.65, 2.28, "Before a loop closure",
            "Local map and camera use keyframe k.\nTheir relative transform is fixed.",
            BLUE_BG, BLUE)
    add_camera(slide, 0.86, 2.22, 1.4, BLUE)
    add_gaussians(slide, 2.07, 2.07, 1.35)
    add_line(slide, 1.45, 2.42, 2.10, 2.42, BLUE)
    add_text(slide, 0.72, 2.76, 2.98, 0.25, "T_WCf^-1 T_WCk = T_CkCf^-1", 13,
             INK, True, PP_ALIGN.CENTER)

    add_line(slide, 4.12, 2.02, 4.55, 2.02, ORANGE)
    add_box(slide, 4.58, 0.90, 3.62, 2.28, "Pose correction",
            "The pose graph changes T_WCk.\nThe map and camera move together.",
            ORANGE_BG, ORANGE)
    add_camera(slide, 5.12, 2.03, 1.4, ORANGE)
    add_gaussians(slide, 6.28, 1.76, 1.35)
    add_text(slide, 4.87, 2.76, 3.03, 0.25, "shared anchor: relation preserved",
             10, ORANGE, True, PP_ALIGN.CENTER)

    add_line(slide, 8.27, 2.02, 8.70, 2.02, MAGENTA)
    add_box(slide, 8.73, 0.90, 4.15, 2.28, "Controlled 6% scale test",
            "Fixed world camera: refinement reverses the correction.\n"
            "Shared anchor: correction remains after about 500 steps.",
            MAGENTA_BG, MAGENTA)
    add_text(slide, 8.98, 2.50, 3.65, 0.42, "Desk + room controls",
             13, MAGENTA, True, PP_ALIGN.CENTER)

    add_text(slide, 0.40, 3.58, 3.4, 0.30, "REFINEMENT LOOP", 11, GREEN, True)
    labels = [
        ("sample RGB", GREEN_BG, GREEN),
        ("compose pose", ORANGE_BG, ORANGE),
        ("render map", BLUE_BG, BLUE),
        ("L1 + SSIM", MAGENTA_BG, MAGENTA),
        ("Adam update", ORANGE_BG, ORANGE),
    ]
    x = 0.45
    for index, (label, fill, stroke) in enumerate(labels):
        add_box(slide, x, 4.08, 2.05, 1.08, label, "", fill, stroke)
        if index < len(labels) - 1:
            add_line(slide, x + 2.07, 4.62, x + 2.34, 4.62, stroke)
        x += 2.55
    add_text(slide, 0.55, 5.50, 12.0, 0.38,
             "Only Gaussian scale, rotation, opacity, and DC color receive gradients.",
             11, INK, True, PP_ALIGN.CENTER)
    add_picture_cover(slide, ASSET / "before.png", 2.12, 6.00, 2.52, 1.18, MUTED)
    add_picture_cover(slide, ASSET / "after.png", 8.70, 6.00, 2.52, 1.18, GREEN)
    add_line(slide, 4.73, 6.59, 8.60, 6.59, GREEN)


def qualitative_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Qualitative reconstruction at common cameras",
              "Measured renders; identical crops and scales")
    methods = [("gt", "Ground truth"), ("photo", "Photo-SLAM"),
               ("mono", "MonoGS"), ("ours", "Ours")]
    x0, width, gap = 0.40, 3.00, 0.18
    rois = [(155, 112, 220, 96), (58, 40, 220, 96)]
    for index, (method, label) in enumerate(methods):
        x = x0 + index * (width + gap)
        add_text(slide, x, 0.76, width, 0.30, label, 11,
                 GREEN if method == "ours" else INK, True, PP_ALIGN.CENTER)
        add_picture_cover(slide, DATA / "desk" / f"view0_{method}.png",
                          x, 1.10, width, 1.72, GREEN if method == "ours" else LINE)
        add_picture_roi(slide, DATA / "desk" / f"view0_{method}.png",
                        rois[0], x, 2.86, width, 0.72, ORANGE)
        add_picture_cover(slide, DATA / "desk" / f"view1_{method}.png",
                          x, 3.82, width, 1.72, GREEN if method == "ours" else LINE)
        add_picture_roi(slide, DATA / "desk" / f"view1_{method}.png",
                        rois[1], x, 5.58, width, 0.72, ORANGE)
    add_text(slide, 0.40, 6.48, 12.55, 0.28,
             "TUM desk: two middle-ranked Ours-minus-MonoGS PSNR views",
             9.5, MUTED, False, PP_ALIGN.CENTER)
    add_text(slide, 0.40, 6.86, 12.55, 0.28,
             "Orange panels are identical pixel crops across all methods.",
             9.5, ORANGE, True, PP_ALIGN.CENTER)


def replica_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Replica reconstruction across scenes",
              "No monocular MonoGS artifact was available")
    methods = [("gt", "Ground truth"), ("photo", "Photo-SLAM"), ("ours", "Ours")]
    scenes = [("office0", "office0"), ("office2", "office2"),
              ("room0", "room0"), ("room2", "room2")]
    x0, width, gap = 2.55, 3.30, 0.25
    for index, (_, label) in enumerate(methods):
        x = x0 + index * (width + gap)
        add_text(slide, x, 0.73, width, 0.30, label, 11,
                 GREEN if label == "Ours" else INK, True, PP_ALIGN.CENTER)
    for row, (scene, label) in enumerate(scenes):
        y = 1.08 + row * 1.48
        add_text(slide, 0.45, y + 0.47, 1.75, 0.28, label, 10, BLUE, True,
                 PP_ALIGN.RIGHT)
        for index, (method, _) in enumerate(methods):
            x = x0 + index * (width + gap)
            add_picture_cover(
                slide, DATA / scene / f"view0_{method}.png",
                x, y, width, 1.30, GREEN if method == "ours" else LINE,
            )
    add_text(slide, 0.45, 7.10, 12.4, 0.22,
             "All images use the same selected camera within each scene.",
             9, MUTED, False, PP_ALIGN.CENTER)


def ablation_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Component evidence",
              "All recorded attenuation cells and refinement controls")
    values = [-12.7, -12.6, -10.2, -8.9, -7.1, -1.6,
              -5.3, -1.8, -6.4, -0.6, -5.6, 0.16]
    labels = ["o3", "o0", "o2", "r0", "r1", "o1",
              "o0-k.9", "o0-k.6", "TUM-rel", "TUM-adapt", "EuRoC", "7S"]
    add_text(slide, 0.40, 0.76, 6.10, 0.30,
             "Opacity attenuation: relative LPIPS change (%)", 12, INK, True)
    zero_x = 6.05
    baseline = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Inches(zero_x), Inches(1.20), Inches(zero_x), Inches(6.35),
    )
    baseline.line.color.rgb = rgb(MUTED)
    baseline.line.width = Pt(1)
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 1.25 + i * 0.41
        add_text(slide, 0.48, y, 1.10, 0.24, label, 8.5, MUTED, False,
                 PP_ALIGN.RIGHT)
        scale = 0.31
        if value < 0:
            x = zero_x + value * scale
            w = -value * scale
            color = GREEN
        else:
            x = zero_x
            w = max(0.05, value * scale)
            color = ORANGE
        bar = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.RECTANGLE,
            Inches(x), Inches(y + 0.02), Inches(w), Inches(0.20),
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = rgb(color)
        bar.line.fill.background()
        add_text(slide, x - 0.62 if value < 0 else x + w + 0.05, y,
                 0.58, 0.24, f"{value:+.2g}", 8, color, True,
                 PP_ALIGN.RIGHT if value < 0 else PP_ALIGN.LEFT)

    add_text(slide, 6.82, 0.76, 5.98, 0.30,
             "Refiner on vs. off: PSNR gain (dB)", 12, INK, True)
    refine = [
        ("ETH3D sofa_1", 8.69),
        ("Replica office0", 4.62),
        ("TUM desk", 3.52),
        ("EuRoC V1_01", 1.00),
    ]
    for i, (label, value) in enumerate(refine):
        y = 1.35 + i * 0.92
        add_text(slide, 6.92, y, 1.62, 0.28, label, 9, MUTED)
        bar = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.RECTANGLE,
            Inches(8.58), Inches(y), Inches(value * 0.43), Inches(0.30),
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = rgb(BLUE)
        bar.line.fill.background()
        add_text(slide, 8.68 + value * 0.43, y, 0.65, 0.30,
                 f"+{value:.2f}", 9, BLUE, True)
    add_box(slide, 7.00, 5.25, 5.68, 1.05, "Interpretation",
            "Refinement gives the largest controlled gain.\n"
            "Opacity reduction is robust; confidence allocation is not.",
            GREEN_BG, GREEN)
    add_text(slide, 0.40, 6.78, 12.35, 0.28,
             "External baseline comparison uses the released head with attenuation off.",
             10, INK, True, PP_ALIGN.CENTER)


def teaser_slide(prs):
    from draw_paper_diagrams import teaser, SCENES
    records = {s: json.loads((DATA/s/"metrics.json").read_text()) for s in SCENES}
    return teaser(records).add_slide(prs, x=.30, y=1.72, width=12.73)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    required = [ASSET / "target.png", ASSET / "before.png", ASSET / "after.png"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Run render_refinement_pair.py first: " + ", ".join(missing))

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    overview_slide(prs, master=False)
    overview_slide(prs, master=True)
    anchor_slide(prs)
    qualitative_slide(prs)
    replica_slide(prs)
    ablation_slide(prs)
    teaser_slide(prs)
    path = OUT / "Splatt3R-SLAM-scientific-figures.pptx"
    prs.save(path)
    manifest = {
        "file": path.name,
        "slides": [
            "RA-L method navigation",
            "MSc method navigation",
            "Shared-anchor and refinement detail",
            "TUM four-column qualitative comparison with common crops",
            "Replica multi-scene qualitative comparison",
            "Fine-grained component ablations",
            "Editable paper teaser",
        ],
        "editable": "All text, connectors, boxes, cameras, Gaussians, and bars are native shapes.",
        "raster_assets": [str(path.relative_to(ROOT)) for path in required],
    }
    (OUT / "pptx_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
