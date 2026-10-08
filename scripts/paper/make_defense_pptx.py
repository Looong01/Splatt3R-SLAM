"""Build the complete Splatt3R-SLAM thesis-defense presentation."""
from pathlib import Path
import json
import math

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

import make_overview_pptx as F


ROOT = Path(__file__).resolve().parents[2]
THESIS = ROOT / "docs/Thesis"
FIG = THESIS / "ral/fig"
DATA = ROOT / "logs/paper_ral_20261007"
OUT = THESIS / "Splatt3R-SLAM-defense.pptx"
NOTES = THESIS / "PRESENTATION_SCRIPT.md"

WHITE = "FFFFFF"
INK = F.INK
MUTED = F.MUTED
LINE = F.LINE
BLUE = F.BLUE
BLUE_BG = F.BLUE_BG
TEAL = F.GREEN
TEAL_BG = F.GREEN_BG
ORANGE = F.ORANGE
ORANGE_BG = F.ORANGE_BG
MAGENTA = F.MAGENTA
MAGENTA_BG = F.MAGENTA_BG
RED = "B44D4D"
RED_BG = "F8EDED"
YELLOW = F.YELLOW
DARK = "102733"
LIGHT = "F7F9FA"

SCENES = [f"office{i}" for i in range(5)] + [f"room{i}" for i in range(3)]


def rgb(value):
    return RGBColor.from_string(value)


def shape_rect(slide, x, y, w, h, fill=WHITE, stroke=LINE, radius=True,
               line_width=1.0):
    kind = (MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius
            else MSO_AUTO_SHAPE_TYPE.RECTANGLE)
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(stroke)
    shape.line.width = Pt(line_width)
    if radius:
        shape.adjustments[0] = 0.07
    return shape


def rule(slide, x1, y1, x2, y2, color=LINE, width=1.0, dash=None):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    line.line.color.rgb = rgb(color)
    line.line.width = Pt(width)
    if dash is not None:
        line.line.dash_style = dash
    return line


def add_footer(slide, number, section="THESIS DEFENSE"):
    F.add_text(slide, 0.38, 7.25, 4.0, 0.13, section.upper(), 6.5, MUTED, True)
    F.add_text(slide, 12.45, 7.25, 0.45, 0.13, str(number), 6.5, MUTED, True,
               PP_ALIGN.RIGHT)


def standard_slide(prs, title, subtitle, number, section):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    F.add_title(slide, title, subtitle)
    add_footer(slide, number, section)
    return slide


def add_label(slide, x, y, w, text, color=BLUE):
    shape_rect(slide, x, y, w, 0.30, fill=color, stroke=color)
    F.add_text(slide, x + 0.06, y + 0.01, w - 0.12, 0.27, text.upper(), 7.5,
               WHITE, True, PP_ALIGN.CENTER)


def add_stat(slide, x, y, w, h, value, label, color=TEAL, fill=TEAL_BG):
    shape_rect(slide, x, y, w, h, fill=fill, stroke=color, line_width=1.2)
    F.add_text(slide, x + 0.10, y + 0.10, w - 0.20, h * 0.52, value, 24,
               color, True, PP_ALIGN.CENTER)
    F.add_text(slide, x + 0.12, y + h * 0.58, w - 0.24, h * 0.28, label, 9,
               INK, True, PP_ALIGN.CENTER)


def add_bullets(slide, x, y, w, items, size=15, color=INK, gap=0.58,
                accent=TEAL):
    for index, item in enumerate(items):
        cy = y + index * gap
        dot = slide.shapes.add_shape(
            MSO_AUTO_SHAPE_TYPE.OVAL, Inches(x), Inches(cy + 0.13),
            Inches(0.11), Inches(0.11))
        dot.fill.solid(); dot.fill.fore_color.rgb = rgb(accent)
        dot.line.fill.background()
        F.add_text(slide, x + 0.22, cy, w - 0.22, gap - 0.02, item, size,
                   color, False, valign=MSO_ANCHOR.TOP)


def add_bar(slide, x, y, w, h, value, maximum, color, label, value_text,
            label_w=1.65):
    F.add_text(slide, x, y, label_w, h, label, 9.5, INK, True, PP_ALIGN.RIGHT)
    shape_rect(slide, x + label_w + 0.14, y + 0.08,
               w - label_w - 0.80, h - 0.16, fill="EEF2F4", stroke="EEF2F4",
               radius=False)
    bar_w = max(0.04, (w - label_w - 0.80) * value / maximum)
    shape_rect(slide, x + label_w + 0.14, y + 0.08, bar_w, h - 0.16,
               fill=color, stroke=color, radius=False)
    F.add_text(slide, x + w - 0.60, y, 0.58, h, value_text, 9.5, color, True,
               PP_ALIGN.RIGHT)


def circle(slide, x, y, d, fill, stroke=None):
    shape = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    shape.fill.solid(); shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(stroke or fill)
    return shape


def replace_text(slide, old, new):
    for shape in slide.shapes:
        if getattr(shape, "has_text_frame", False) and old in shape.text:
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    if old in run.text:
                        run.text = run.text.replace(old, new)


def read_records():
    return {
        scene: json.loads((DATA / scene / "metrics.json").read_text())
        for scene in SCENES + ["desk"]
    }


def title_slide(prs, number):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.background.fill
    bg.solid(); bg.fore_color.rgb = rgb(WHITE)
    shape_rect(slide, 0, 0, 5.15, 7.5, fill=DARK, stroke=DARK, radius=False)
    F.add_picture_cover(slide, DATA / "office0/view0_ours.png",
                        5.15, 0, 8.18, 7.50, TEAL)
    shape_rect(slide, 0.55, 0.62, 2.50, 0.34, fill=TEAL, stroke=TEAL)
    F.add_text(slide, 0.66, 0.65, 2.28, 0.27, "MASTER'S THESIS DEFENSE",
               8.5, WHITE, True, PP_ALIGN.CENTER)
    F.add_text(slide, 0.62, 1.35, 4.00, 1.70,
               "Splatt3R-SLAM:\nFrom Image Pairs\nto Persistent Maps",
               27, WHITE, True, valign=MSO_ANCHOR.TOP)
    F.add_text(slide, 0.66, 3.55, 3.95, 0.80,
               "Fast local predictions.\nOne map that survives pose correction.",
               15, "D7E8EC", False, valign=MSO_ANCHOR.TOP)
    rule(slide, 0.66, 5.30, 4.42, 5.30, TEAL, 2)
    F.add_text(slide, 0.66, 5.52, 3.95, 0.74,
               "Zelong Li\nMSc Computer Science | University of Amsterdam",
               11, WHITE, True, valign=MSO_ANCHOR.TOP)
    F.add_text(slide, 0.66, 6.63, 3.95, 0.28,
               "Supervisors: Prof. Martin Oswald | Dr. Qi Zhang",
               8.5, "B9CBD1")
    add_footer(slide, number, "OPENING")


def pair_not_map_slide(prs, number):
    slide = standard_slide(prs, "A pair is not a map",
                           "The gap between a local prediction and a lasting scene",
                           number, "PROBLEM")
    add_label(slide, 0.42, 0.82, 1.58, "One image pair", BLUE)
    F.add_picture_cover(slide, DATA / "desk/view0_gt.png", 0.42, 1.27, 2.36, 1.76, BLUE)
    F.add_picture_cover(slide, DATA / "desk/view1_gt.png", 0.82, 2.40, 2.36, 1.76, TEAL)
    F.add_line(slide, 3.22, 2.44, 3.70, 2.44, BLUE)
    add_label(slide, 3.82, 0.82, 1.92, "One local map", TEAL)
    F.add_picture_cover(slide, FIG / "overview_assets/before.png",
                        3.82, 1.27, 3.16, 2.42, TEAL)
    F.add_text(slide, 3.82, 3.83, 3.16, 0.45,
               "Useful immediately", 12, TEAL, True, PP_ALIGN.CENTER)
    F.add_line(slide, 7.18, 2.44, 7.67, 2.44, ORANGE)
    add_label(slide, 7.80, 0.82, 2.25, "A video sequence", ORANGE)
    for i, scene in enumerate(("office0", "office1", "room0")):
        F.add_picture_cover(slide, DATA / f"{scene}/view0_ours.png",
                            7.88 + i * 0.72, 1.30 + i * 0.52, 3.15, 2.20,
                            (BLUE, ORANGE, MAGENTA)[i])
    F.add_text(slide, 8.05, 4.62, 4.52, 0.72,
               "Many local frames\nMany overlapping surfaces",
               16, INK, True, PP_ALIGN.CENTER)
    shape_rect(slide, 0.70, 5.62, 11.95, 0.92, fill=ORANGE_BG, stroke=ORANGE)
    F.add_text(slide, 0.95, 5.78, 11.45, 0.55,
               "The research question: how can pairwise predictions become one persistent map?",
               19, ORANGE, True, PP_ALIGN.CENTER)


def persistent_map_slide(prs, number):
    slide = standard_slide(prs, "A persistent map must survive change",
                           "Three requirements for a useful appearance map",
                           number, "PROBLEM")
    panels = [
        (0.45, "REVISIT", "A later camera must still localise.", BLUE_BG, BLUE),
        (4.45, "CORRECT", "Loop closure must move old content.", ORANGE_BG, ORANGE),
        (8.45, "RENDER", "A new view must look like the scene.", TEAL_BG, TEAL),
    ]
    for x, title, body, fill, stroke in panels:
        shape_rect(slide, x, 1.12, 3.55, 4.88, fill=fill, stroke=stroke, line_width=1.4)
        F.add_text(slide, x + 0.24, 1.35, 3.07, 0.36, title, 14, stroke, True,
                   PP_ALIGN.CENTER)
        F.add_text(slide, x + 0.28, 5.16, 2.99, 0.52, body, 12, INK, True,
                   PP_ALIGN.CENTER)
    F.add_camera(slide, 1.05, 2.34, 2.3, BLUE)
    for i in range(4):
        circle(slide, 1.95 + i * 0.34, 2.47 + math.sin(i) * 0.22, 0.08, BLUE)
    F.add_line(slide, 1.60, 3.50, 3.08, 3.05, BLUE)

    circle(slide, 5.40, 2.30, 0.72, ORANGE_BG, ORANGE)
    F.add_text(slide, 5.40, 2.42, 0.72, 0.28, "k", 16, ORANGE, True, PP_ALIGN.CENTER)
    F.add_camera(slide, 4.90, 3.40, 1.6, ORANGE)
    F.add_gaussians(slide, 6.10, 3.16, 1.4)
    F.add_line(slide, 5.78, 2.98, 5.38, 3.58, ORANGE)
    F.add_line(slide, 5.78, 2.98, 6.36, 3.55, ORANGE)

    F.add_picture_cover(slide, DATA / "office0/view0_ours.png",
                        8.78, 2.00, 2.90, 2.18, TEAL)
    F.add_picture_cover(slide, DATA / "office0/view0_gt.png",
                        9.30, 3.10, 2.90, 2.18, BLUE)
    F.add_text(slide, 0.55, 6.28, 11.90, 0.52,
               "Tracking, correction, and rendering must agree on the same scene.",
               18, INK, True, PP_ALIGN.CENTER)


def failure_slide(prs, number):
    slide = standard_slide(prs, "Why direct accumulation fails",
                           "The sequence adds problems that a two-view model never sees",
                           number, "PROBLEM")
    items = [
        (0.42, "1", "Different coordinates", "Each pair predicts in its own camera frame.", BLUE),
        (4.46, "2", "Repeated surfaces", "Several keyframes place colour and opacity on one ray.", MAGENTA),
        (8.50, "3", "Pose corrections", "Loop closure changes where old predictions belong.", ORANGE),
    ]
    for x, n, title, body, color in items:
        shape_rect(slide, x, 1.05, 3.50, 4.95, fill=WHITE, stroke=color, line_width=1.4)
        circle(slide, x + 0.22, 1.28, 0.52, color)
        F.add_text(slide, x + 0.22, 1.37, 0.52, 0.26, n, 13, WHITE, True,
                   PP_ALIGN.CENTER)
        F.add_text(slide, x + 0.88, 1.26, 2.34, 0.40, title, 14, INK, True)
        F.add_text(slide, x + 0.30, 4.98, 2.90, 0.62, body, 11, INK, False,
                   PP_ALIGN.CENTER)
    F.add_camera(slide, 1.06, 2.40, 1.4, BLUE)
    F.add_camera(slide, 2.25, 3.06, 1.4, TEAL)
    F.add_gaussians(slide, 1.42, 3.22, 1.6)

    for i, color in enumerate((BLUE, MAGENTA, ORANGE, TEAL)):
        ellipse = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL,
            Inches(5.16 + i * 0.35), Inches(2.45 + (i % 2) * 0.30),
            Inches(1.05), Inches(0.42))
        ellipse.rotation = (-15, 10, -5, 18)[i]
        ellipse.fill.solid(); ellipse.fill.fore_color.rgb = rgb(color)
        ellipse.fill.transparency = 35; ellipse.line.color.rgb = rgb(color)

    circle(slide, 9.54, 2.24, 0.70, ORANGE_BG, ORANGE)
    F.add_text(slide, 9.54, 2.36, 0.70, 0.26, "k", 15, ORANGE, True, PP_ALIGN.CENTER)
    F.add_camera(slide, 9.15, 3.55, 1.5, ORANGE)
    F.add_gaussians(slide, 10.45, 3.32, 1.4)
    F.add_line(slide, 9.92, 3.04, 9.55, 3.74, ORANGE)
    F.add_line(slide, 9.92, 3.04, 10.72, 3.72, ORANGE)
    F.add_line(slide, 9.92, 2.00, 10.55, 1.63, ORANGE)
    shape_rect(slide, 0.85, 6.28, 11.60, 0.54, fill=RED_BG, stroke=RED)
    F.add_text(slide, 1.05, 6.35, 11.20, 0.35,
               "Concatenation is not sequence-level mapping.", 17, RED, True,
               PP_ALIGN.CENTER)


def thesis_slide(prs, number):
    slide = standard_slide(prs, "The thesis: share the anchor",
                           "Move the local scene and its observation cameras together",
                           number, "IDEA")
    shape_rect(slide, 0.48, 0.98, 5.72, 5.42, fill=BLUE_BG, stroke=BLUE)
    F.add_text(slide, 0.75, 1.20, 5.18, 0.40, "Before pose correction", 15,
               BLUE, True, PP_ALIGN.CENTER)
    circle(slide, 2.95, 2.00, 0.74, ORANGE_BG, ORANGE)
    F.add_text(slide, 2.95, 2.12, 0.74, 0.28, "anchor k", 10, ORANGE, True,
               PP_ALIGN.CENTER)
    F.add_camera(slide, 1.25, 3.55, 2.1, BLUE)
    F.add_gaussians(slide, 3.78, 3.31, 1.9)
    F.add_line(slide, 3.30, 2.72, 1.92, 3.82, ORANGE)
    F.add_line(slide, 3.30, 2.72, 4.50, 3.82, ORANGE)
    F.add_text(slide, 1.00, 5.37, 4.65, 0.50,
               "camera + local Gaussian map", 13, INK, True, PP_ALIGN.CENTER)

    F.add_line(slide, 6.28, 3.55, 6.88, 3.55, ORANGE, 2.4)
    shape_rect(slide, 7.02, 0.98, 5.82, 5.42, fill=TEAL_BG, stroke=TEAL)
    F.add_text(slide, 7.30, 1.20, 5.26, 0.40, "After pose correction", 15,
               TEAL, True, PP_ALIGN.CENTER)
    circle(slide, 10.50, 1.75, 0.74, ORANGE_BG, ORANGE)
    F.add_text(slide, 10.50, 1.87, 0.74, 0.28, "anchor k'", 10, ORANGE, True,
               PP_ALIGN.CENTER)
    F.add_camera(slide, 8.35, 3.20, 2.1, TEAL)
    F.add_gaussians(slide, 10.72, 2.80, 1.9)
    F.add_line(slide, 10.85, 2.48, 9.00, 3.48, ORANGE)
    F.add_line(slide, 10.85, 2.48, 11.45, 3.35, ORANGE)
    F.add_text(slide, 7.52, 5.37, 4.78, 0.50,
               "the same relative relation", 13, INK, True, PP_ALIGN.CENTER)
    shape_rect(slide, 2.35, 6.52, 8.65, 0.50, fill=WHITE, stroke=ORANGE)
    F.add_text(slide, 2.55, 6.60, 8.25, 0.32,
               "T_WCf^-1 T_WCk = T_CkCf^-1", 17, ORANGE, True, PP_ALIGN.CENTER)


def contributions_slide(prs, number):
    slide = standard_slide(prs, "One theory, three contributions",
                           "Each contribution supports the same persistent-map claim",
                           number, "IDEA")
    specs = [
        (0.45, "PREDICT", "Dense local map", "One forward pass gives geometry and appearance.", BLUE_BG, BLUE),
        (4.50, "ANCHOR", "Coordinate consistency", "Map and supervision follow the pose graph.", ORANGE_BG, ORANGE),
        (8.55, "REFINE", "Sequence evidence", "Tracked RGB improves appearance without changing poses.", MAGENTA_BG, MAGENTA),
    ]
    for i, (x, stage, title, body, fill, stroke) in enumerate(specs):
        shape_rect(slide, x, 1.18, 3.48, 4.78, fill=fill, stroke=stroke, line_width=1.4)
        add_label(slide, x + 0.24, 1.42, 1.20, stage, stroke)
        F.add_text(slide, x + 0.28, 2.06, 2.92, 0.66, title, 19, INK, True,
                   PP_ALIGN.CENTER)
        if stage == "PREDICT":
            F.add_camera(slide, x + 0.64, 3.10, 1.5, BLUE)
            F.add_gaussians(slide, x + 1.78, 3.03, 1.25)
        elif stage == "ANCHOR":
            circle(slide, x + 1.38, 3.02, 0.72, ORANGE_BG, ORANGE)
            F.add_text(slide, x + 1.38, 3.14, 0.72, 0.26, "k", 16, ORANGE, True,
                       PP_ALIGN.CENTER)
            F.add_camera(slide, x + 0.72, 4.02, 1.25, ORANGE)
            F.add_gaussians(slide, x + 2.00, 3.85, 1.05)
        else:
            F.add_picture_cover(slide, FIG / "overview_assets/before.png",
                                x + 0.38, 3.05, 1.18, 1.15, MUTED)
            F.add_line(slide, x + 1.64, 3.62, x + 1.94, 3.62, MAGENTA)
            F.add_picture_cover(slide, FIG / "overview_assets/after.png",
                                x + 2.02, 3.05, 1.18, 1.15, TEAL)
        F.add_text(slide, x + 0.30, 4.82, 2.88, 0.64, body, 11, INK, False,
                   PP_ALIGN.CENTER)
        if i < 2:
            F.add_line(slide, x + 3.50, 3.47, x + 3.93, 3.47, stroke)
    F.add_text(slide, 0.55, 6.36, 12.15, 0.42,
               "The experiments ask whether each step preserves the claim.",
               16, INK, True, PP_ALIGN.CENTER)


def related_work_slide(prs, number):
    slide = standard_slide(prs, "Where this work sits",
                           "Prediction-first initialization plus sequence-level SLAM",
                           number, "POSITION")
    # Axes
    rule(slide, 1.20, 5.90, 11.95, 5.90, INK, 1.5)
    F.add_line(slide, 11.78, 5.90, 12.08, 5.90, INK, 1.5)
    rule(slide, 1.20, 5.90, 1.20, 1.05, INK, 1.5)
    F.add_line(slide, 1.20, 1.24, 1.20, 0.93, INK, 1.5)
    F.add_text(slide, 1.25, 6.02, 4.20, 0.30, "OPTIMIZATION-FIRST", 9, MUTED, True)
    F.add_text(slide, 8.45, 6.02, 3.40, 0.30, "PREDICTION-FIRST", 9, MUTED, True,
               PP_ALIGN.RIGHT)
    F.add_text(slide, 0.18, 1.12, 0.75, 0.82, "FULL\nSEQUENCE", 8, MUTED, True,
               PP_ALIGN.CENTER)
    F.add_text(slide, 0.18, 5.02, 0.75, 0.65, "LOCAL\nPAIR", 8, MUTED, True,
               PP_ALIGN.CENTER)
    points = [
        (2.15, 4.80, "MonoGS", BLUE, BLUE_BG),
        (3.25, 3.80, "Photo-SLAM", BLUE, BLUE_BG),
        (4.20, 2.05, "Splat-SLAM", ORANGE, ORANGE_BG),
        (5.50, 2.55, "SEGS-SLAM", ORANGE, ORANGE_BG),
        (9.65, 5.00, "Splatt3R", MAGENTA, MAGENTA_BG),
        (9.20, 1.72, "Splatt3R-SLAM", TEAL, TEAL_BG),
    ]
    for x, y, label, stroke, fill in points:
        shape_rect(slide, x, y, 1.72, 0.55, fill=fill, stroke=stroke)
        F.add_text(slide, x + 0.08, y + 0.08, 1.56, 0.35, label, 10, stroke, True,
                   PP_ALIGN.CENTER)
    F.add_text(slide, 7.18, 2.46, 1.62, 0.55, "shared anchors\n+ refinement",
               9, TEAL, True, PP_ALIGN.CENTER)
    F.add_line(slide, 8.67, 2.55, 9.15, 2.10, TEAL)
    shape_rect(slide, 2.12, 6.50, 9.22, 0.42, fill=LIGHT, stroke=LINE)
    F.add_text(slide, 2.25, 6.56, 8.96, 0.27,
               "Positioning diagram only. Quantitative ranking requires a common protocol.",
               9.5, MUTED, True, PP_ALIGN.CENTER)


def predict_slide(prs, number):
    slide = standard_slide(prs, "Predict: one pass, two lanes",
                           "A shared network supports tracking and appearance mapping",
                           number, "METHOD")
    F.add_picture_cover(slide, DATA / "desk/view0_gt.png", 0.42, 1.10, 2.65, 2.00, BLUE)
    F.add_picture_cover(slide, DATA / "desk/view1_gt.png", 0.82, 2.56, 2.65, 2.00, TEAL)
    F.add_line(slide, 3.47, 2.72, 4.02, 2.72, BLUE)
    F.add_box(slide, 4.08, 1.34, 2.52, 2.76, "Shared predictor",
              "Frozen MASt3R encoder + decoder\n\nGaussian DPT head",
              BLUE_BG, BLUE)
    F.add_line(slide, 6.65, 2.12, 7.18, 2.12, BLUE)
    F.add_line(slide, 6.65, 3.30, 7.18, 3.30, TEAL)
    F.add_box(slide, 7.25, 1.14, 2.37, 1.58, "Geometry lane",
              "pointmaps\nmatching features", BLUE_BG, BLUE)
    F.add_box(slide, 7.25, 3.02, 2.37, 1.58, "Gaussian lane",
              "mean, scale, rotation\nopacity, SH colour", TEAL_BG, TEAL)
    F.add_line(slide, 9.68, 2.12, 10.20, 2.12, BLUE)
    F.add_line(slide, 9.68, 3.78, 10.20, 3.78, TEAL)
    F.add_box(slide, 10.26, 1.14, 2.55, 1.58, "Tracker",
              "pose + keyframes + graph", BLUE_BG, BLUE)
    F.add_box(slide, 10.26, 3.02, 2.55, 1.58, "Local map",
              "one Gaussian per valid pixel", TEAL_BG, TEAL)
    shape_rect(slide, 3.10, 5.20, 7.15, 0.74, fill=WHITE, stroke=TEAL)
    F.add_text(slide, 3.28, 5.31, 6.80, 0.50,
               "mu_n = x_n + Delta_n     |     Sigma_n = R(q_n) diag(s_n^2) R(q_n)^T",
               14, INK, True, PP_ALIGN.CENTER)
    F.add_text(slide, 3.60, 6.22, 6.20, 0.38,
               "Paper Sec. III-A / Eq. (1) | Thesis Ch. 2 / Eq. (2.7)",
               9.5, MUTED, True, PP_ALIGN.CENTER)


def refinement_slide(prs, number):
    slide = standard_slide(prs, "Refine: use the sequence without moving the tracker",
                           "The real control loop is sampling, rendering, and Adam",
                           number, "METHOD")
    F.add_picture_cover(slide, FIG / "overview_assets/before.png",
                        0.42, 1.05, 3.42, 2.48, MUTED)
    F.add_picture_cover(slide, FIG / "overview_assets/after.png",
                        0.42, 4.02, 3.42, 2.48, TEAL)
    F.add_text(slide, 0.54, 3.59, 3.18, 0.30, "same camera", 9, MUTED, True,
               PP_ALIGN.CENTER)
    F.add_line(slide, 2.14, 3.54, 2.14, 3.98, TEAL)

    F.add_text(slide, 4.34, 0.94, 3.35, 0.32, "SUPERVISION POOL", 11, TEAL, True)
    for i in range(4):
        F.add_picture_cover(slide, DATA / "desk" / f"view{i % 2}_gt.png",
                            4.34 + i * 0.58, 1.38 + i * 0.16, 1.58, 1.05,
                            (BLUE, TEAL, BLUE, TEAL)[i])
    F.add_box(slide, 4.48, 3.32, 2.72, 1.22, "Reservoir + recent ring",
              "200 historical | 64 recent\n30% recent sampling", TEAL_BG, TEAL)
    F.add_line(slide, 7.34, 3.93, 7.82, 3.93, MAGENTA)
    F.add_box(slide, 7.90, 2.95, 2.12, 1.98, "Render",
              "compose live anchors\nrender one RGB view", BLUE_BG, BLUE)
    F.add_line(slide, 10.07, 3.93, 10.55, 3.93, MAGENTA)
    F.add_box(slide, 10.62, 2.95, 2.12, 1.98, "Loss + Adam",
              "0.8 L1 + 0.2 D-SSIM\nupdate appearance", MAGENTA_BG, MAGENTA)
    F.add_line(slide, 11.68, 2.91, 11.68, 1.78, MAGENTA)
    F.add_line(slide, 11.68, 1.78, 7.70, 1.78, MAGENTA)
    F.add_text(slide, 7.62, 1.40, 4.15, 0.34,
               "repeat with another tracked view", 9.5, MAGENTA, True,
               PP_ALIGN.CENTER)
    shape_rect(slide, 4.35, 5.52, 8.42, 0.76, fill=ORANGE_BG, stroke=ORANGE)
    F.add_text(slide, 4.60, 5.65, 7.92, 0.48,
               "Gradients update Gaussian appearance. They never update poses or tracking weights.",
               13, ORANGE, True, PP_ALIGN.CENTER)
    F.add_text(slide, 5.22, 6.48, 6.70, 0.30,
               "Paper Sec. III-C / Eq. (5) | Thesis Ch. 7 / Eq. (7.1)",
               9.5, MUTED, True, PP_ALIGN.CENTER)


def initialization_slide(prs, number):
    slide = standard_slide(prs, "Improve the starting point",
                           "Two optional interventions act before sequence refinement",
                           number, "METHOD")
    shape_rect(slide, 0.48, 1.02, 5.92, 5.55, fill=BLUE_BG, stroke=BLUE)
    add_label(slide, 0.78, 1.30, 1.28, "Prediction", BLUE)
    F.add_text(slide, 0.78, 1.90, 5.28, 0.55, "Head-only adaptation", 20,
               INK, True, PP_ALIGN.CENTER)
    F.add_box(slide, 1.10, 2.80, 1.76, 1.24, "Frozen",
              "encoder\ndecoder", WHITE, BLUE)
    F.add_line(slide, 2.92, 3.42, 3.34, 3.42, BLUE)
    F.add_box(slide, 3.42, 2.80, 2.02, 1.24, "Trainable",
              "Gaussian DPT head", TEAL_BG, TEAL)
    F.add_text(slide, 1.02, 4.55, 4.78, 0.65,
               "Changes pairwise appearance\nwithout changing tracking features",
               13, BLUE, True, PP_ALIGN.CENTER)
    F.add_text(slide, 1.02, 5.58, 4.78, 0.38,
               "MSE + 0.25 LPIPS-VGG", 12, INK, True, PP_ALIGN.CENTER)

    shape_rect(slide, 6.88, 1.02, 5.95, 5.55, fill=ORANGE_BG, stroke=ORANGE)
    add_label(slide, 7.18, 1.30, 1.28, "Insertion", ORANGE)
    F.add_text(slide, 7.18, 1.90, 5.28, 0.55, "Opacity attenuation", 20,
               INK, True, PP_ALIGN.CENTER)
    F.add_gaussians(slide, 7.48, 2.90, 2.0)
    F.add_line(slide, 9.48, 3.35, 10.08, 3.35, ORANGE)
    F.add_gaussians(slide, 10.30, 2.90, 1.35)
    F.add_text(slide, 7.18, 4.43, 5.28, 0.72,
               "Lower initial opacity\nfor low-confidence predictions",
               13, ORANGE, True, PP_ALIGN.CENTER)
    F.add_text(slide, 7.18, 5.47, 5.28, 0.54,
               "alpha' = alpha clip(1 - 2 lambda (1 - rank), 0.1, 1)",
               11.5, INK, True, PP_ALIGN.CENTER)
    F.add_text(slide, 1.10, 6.76, 11.15, 0.28,
               "External baseline comparison: released head, attenuation off.",
               10, MUTED, True, PP_ALIGN.CENTER)


def evidence_map_slide(prs, number):
    slide = standard_slide(prs, "Every claim has a matching experiment",
                           "The evaluation follows the theory, not a list of unrelated scores",
                           number, "EXPERIMENTS")
    headers = [(0.42, "CLAIM", BLUE), (4.22, "CONTROL", ORANGE),
               (8.18, "EVIDENCE", TEAL)]
    for x, label, color in headers:
        add_label(slide, x, 0.92, 3.20, label, color)
    rows = [
        ("Shared anchors preserve relations", "6% block pose correction", "correction survives refinement"),
        ("Refinement stays downstream", "refiner on/off + ATE", "1.00-8.69 dB; ATE unchanged"),
        ("Initialization matters", "head + 12 thinning cells", "10/12 LPIPS gains"),
        ("The full system improves maps", "common renderer + common cameras", "+3.54 dB; 8/8 Replica wins"),
    ]
    for i, row in enumerate(rows):
        y = 1.46 + i * 1.30
        fills = (BLUE_BG, ORANGE_BG, TEAL_BG)
        strokes = (BLUE, ORANGE, TEAL)
        xs = (0.42, 4.22, 8.18)
        widths = (3.45, 3.60, 4.70)
        for j, text in enumerate(row):
            shape_rect(slide, xs[j], y, widths[j], 0.93, fill=fills[j], stroke=strokes[j])
            F.add_text(slide, xs[j] + 0.14, y + 0.11, widths[j] - 0.28, 0.68,
                       text, 11, INK, j == 2, PP_ALIGN.CENTER)
        F.add_line(slide, 3.90, y + 0.46, 4.17, y + 0.46, MUTED)
        F.add_line(slide, 7.85, y + 0.46, 8.13, y + 0.46, MUTED)
    F.add_text(slide, 0.64, 6.84, 12.05, 0.28,
               "RQ1-RQ4 in the thesis use this same chain.", 11, MUTED, True,
               PP_ALIGN.CENTER)


def protocol_slide(prs, number):
    slide = standard_slide(prs, "A common renderer makes the comparison meaningful",
                           "Same target cameras, resolution, alignment, and metrics",
                           number, "EXPERIMENTS")
    methods = [("photo", "Photo-SLAM", BLUE), ("mono", "MonoGS", ORANGE),
               ("ours", "Splatt3R-SLAM", TEAL)]
    for i, (method, label, color) in enumerate(methods):
        x = 0.45 + i * 2.90
        F.add_picture_cover(slide, DATA / "desk" / f"view0_{method}.png",
                            x, 1.10, 2.55, 1.72, color)
        F.add_text(slide, x, 2.90, 2.55, 0.30, label, 10, color, True,
                   PP_ALIGN.CENTER)
        F.add_line(slide, x + 1.28, 3.28, 8.95, 3.82, color)
    F.add_camera(slide, 9.08, 3.48, 2.3, MAGENTA)
    F.add_text(slide, 8.72, 4.32, 2.65, 0.34, "one GT camera", 12, MAGENTA, True,
               PP_ALIGN.CENTER)
    boxes = [
        (0.55, "100 frames", "outside the union of keyframes", BLUE),
        (3.68, "Full SH", "all exported colour bands", TEAL),
        (6.82, "Same image size", "512x288 or 512x384", ORANGE),
        (9.95, "Same metrics", "PSNR + LPIPS-AlexNet", MAGENTA),
    ]
    for x, title, body, color in boxes:
        F.add_box(slide, x, 5.12, 2.70, 1.18, title, body,
                  {BLUE: BLUE_BG, TEAL: TEAL_BG, ORANGE: ORANGE_BG,
                   MAGENTA: MAGENTA_BG}[color], color)
    F.add_text(slide, 0.75, 6.66, 11.80, 0.34,
               "Replica: 8 scenes | TUM desk: 3 renderable systems | TUM: 9-sequence ATE",
               10.5, INK, True, PP_ALIGN.CENTER)


def replica_result_slide(prs, number, records):
    slide = standard_slide(prs, "State-of-the-art performance",
                           "Replica | all eight scenes | monocular input | common renderer",
                           number, "RESULTS")
    add_stat(slide, 0.45, 1.00, 2.35, 1.35, "23.02 dB", "Splatt3R-SLAM mean", TEAL)
    add_stat(slide, 3.05, 1.00, 2.35, 1.35, "+3.54 dB", "over Photo-SLAM", ORANGE,
             ORANGE_BG)
    add_stat(slide, 5.65, 1.00, 2.35, 1.35, "8 / 8", "scene PSNR wins", BLUE, BLUE_BG)
    add_stat(slide, 8.25, 1.00, 2.35, 1.35, "5 / 8", "scene LPIPS wins", MAGENTA,
             MAGENTA_BG)
    # Main bars
    F.add_text(slide, 0.55, 2.78, 4.50, 0.32, "MEAN PSNR (dB)", 11, INK, True)
    add_bar(slide, 0.55, 3.20, 4.65, 0.62, 19.483, 27, MUTED,
            "Photo-SLAM", "19.48")
    add_bar(slide, 0.55, 3.92, 4.65, 0.62, 23.019, 27, TEAL,
            "Splatt3R-SLAM", "23.02")
    F.add_picture_cover(slide, DATA / "office0/view0_photo.png",
                        0.55, 4.90, 2.18, 1.45, MUTED)
    F.add_picture_cover(slide, DATA / "office0/view0_ours.png",
                        2.98, 4.90, 2.18, 1.45, TEAL)
    F.add_text(slide, 0.55, 6.42, 2.18, 0.26, "Photo-SLAM", 8.5, MUTED, True,
               PP_ALIGN.CENTER)
    F.add_text(slide, 2.98, 6.42, 2.18, 0.26, "Splatt3R-SLAM", 8.5, TEAL, True,
               PP_ALIGN.CENTER)

    F.add_text(slide, 5.72, 2.78, 6.70, 0.32, "OURS - PHOTO-SLAM, PER SCENE", 11,
               INK, True)
    deltas = []
    for scene in SCENES:
        row = records[scene]["methods"]
        deltas.append((scene.replace("office", "o").replace("room", "r"),
                       row["ours"]["full_sh"]["psnr"] - row["photo"]["full_sh"]["psnr"]))
    max_delta = max(v for _, v in deltas)
    for i, (label, value) in enumerate(deltas):
        y = 3.20 + i * 0.46
        add_bar(slide, 5.72, y, 6.62, 0.40, value, max_delta, TEAL,
                label, f"+{value:.2f}", label_w=0.55)
    F.add_text(slide, 5.80, 6.88, 6.45, 0.25,
               "Smallest margin: room1 +0.23 dB | largest: room0 +7.64 dB",
               8.5, MUTED, False, PP_ALIGN.CENTER)


def tracking_slide(prs, number):
    slide = standard_slide(prs, "Tracking remains the inherited geometric path",
                           "Nine TUM freiburg1 sequences | Sim(3)-aligned ATE",
                           number, "RESULTS")
    values = [
        ("Ours", 0.02960, TEAL), ("MASt3R-SLAM", 0.03007, BLUE),
        ("VGGT-SLAM", 0.04081, ORANGE), ("Photo-SLAM", 0.16094, MAGENTA),
        ("MonoGS", 0.29328, RED),
    ]
    F.add_text(slide, 0.55, 0.98, 6.05, 0.32, "MEAN ATE RMSE (m) - LOWER IS BETTER",
               11, INK, True)
    for i, (label, value, color) in enumerate(values):
        add_bar(slide, 0.55, 1.45 + i * 0.78, 6.10, 0.56, value, 0.32, color,
                label, f"{value:.3f}")
    shape_rect(slide, 7.10, 1.12, 5.70, 2.08, fill=TEAL_BG, stroke=TEAL)
    F.add_text(slide, 7.38, 1.40, 5.14, 0.48, "What this result proves", 18,
               TEAL, True, PP_ALIGN.CENTER)
    add_bullets(slide, 7.55, 2.02, 4.85, [
        "The mapping path does not disturb tracking.",
        "Ours and MASt3R-SLAM have the same rounded mean.",
    ], 12, INK, 0.52, TEAL)
    shape_rect(slide, 7.10, 3.58, 5.70, 2.18, fill=ORANGE_BG, stroke=ORANGE)
    F.add_text(slide, 7.38, 3.84, 5.14, 0.48, "What it does not prove", 18,
               ORANGE, True, PP_ALIGN.CENTER)
    add_bullets(slide, 7.55, 4.46, 4.85, [
        "Tracking is inherited, not a new contribution.",
        "Single runs can hide system variance.",
    ], 12, INK, 0.52, ORANGE)
    F.add_text(slide, 7.15, 6.18, 5.60, 0.48,
               "The map improves appearance downstream of pose estimation.",
               13, INK, True, PP_ALIGN.CENTER)


def refiner_result_slide(prs, number):
    slide = standard_slide(prs, "Refinement gives the largest controlled gain",
                           "Same head, same insertion, refiner off versus on",
                           number, "RESULTS")
    gains = [("ETH3D sofa_1", 8.69, 58.3), ("Replica office0", 4.62, 56.8),
             ("TUM desk", 3.52, 27.3), ("EuRoC V1_01", 1.00, 14.5)]
    F.add_text(slide, 0.52, 0.94, 6.40, 0.32, "PSNR GAIN (dB)", 11, INK, True)
    for i, (label, value, lpips) in enumerate(gains):
        add_bar(slide, 0.52, 1.42 + i * 0.88, 6.25, 0.60, value, 9.0, TEAL,
                label, f"+{value:.2f}")
        F.add_text(slide, 5.25, 1.97 + i * 0.88, 1.38, 0.22,
                   f"LPIPS -{lpips:.1f}%", 8, MAGENTA, True, PP_ALIGN.RIGHT)
    F.add_picture_cover(slide, FIG / "overview_assets/before.png",
                        7.20, 1.08, 2.55, 2.08, MUTED)
    F.add_picture_cover(slide, FIG / "overview_assets/after.png",
                        10.10, 1.08, 2.55, 2.08, TEAL)
    F.add_text(slide, 7.20, 3.26, 2.55, 0.30, "refiner off", 9, MUTED, True,
               PP_ALIGN.CENTER)
    F.add_text(slide, 10.10, 3.26, 2.55, 0.30, "refiner on", 9, TEAL, True,
               PP_ALIGN.CENTER)
    shape_rect(slide, 7.20, 4.03, 5.45, 1.60, fill=BLUE_BG, stroke=BLUE)
    F.add_text(slide, 7.50, 4.30, 4.85, 0.42, "Why the control is causal", 16,
               BLUE, True, PP_ALIGN.CENTER)
    F.add_text(slide, 7.55, 4.86, 4.75, 0.48,
               "Only the refiner changes.\nThe tracker and Gaussian head stay fixed.",
               12, INK, False, PP_ALIGN.CENTER)
    F.add_text(slide, 7.20, 6.08, 5.45, 0.62,
               "Result: sequence observations improve a feed-forward map.",
               14, TEAL, True, PP_ALIGN.CENTER)


def causal_slide(prs, number):
    slide = standard_slide(prs, "Online quality is an iteration-budget problem",
                           "TUM desk causal replay and GPU placement",
                           number, "RESULTS")
    steps = [120, 500, 1000, 3000]
    causal = [13.651, 14.394, 14.475, 14.412]
    post = [13.800, 14.322, 14.352, 14.375]
    x0, y0, cw, ch = 0.82, 1.22, 6.30, 3.82
    rule(slide, x0, y0 + ch, x0 + cw, y0 + ch, INK, 1.2)
    rule(slide, x0, y0, x0, y0 + ch, INK, 1.2)
    ymin, ymax = 13.4, 14.6
    def pos(i, value):
        return x0 + i * cw / 3, y0 + ch - (value - ymin) / (ymax - ymin) * ch
    for value in (13.5, 14.0, 14.5):
        y = y0 + ch - (value - ymin) / (ymax - ymin) * ch
        rule(slide, x0, y, x0 + cw, y, LINE, 0.6)
        F.add_text(slide, 0.28, y - 0.13, 0.45, 0.25, f"{value:.1f}", 8, MUTED,
                   False, PP_ALIGN.RIGHT)
    for i, step in enumerate(steps):
        x, _ = pos(i, ymin)
        F.add_text(slide, x - 0.36, y0 + ch + 0.12, 0.72, 0.26, str(step), 8,
                   MUTED, False, PP_ALIGN.CENTER)
    for series, color in ((causal, TEAL), (post, ORANGE)):
        pts = [pos(i, v) for i, v in enumerate(series)]
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            rule(slide, xa, ya, xb, yb, color, 2.4)
        for (x, y), value in zip(pts, series):
            circle(slide, x - 0.08, y - 0.08, 0.16, color)
            F.add_text(slide, x - 0.35, y - 0.42, 0.70, 0.24, f"{value:.2f}",
                       8, color, True, PP_ALIGN.CENTER)
    F.add_text(slide, 1.05, 5.68, 1.55, 0.28, "CAUSAL", 9, TEAL, True)
    F.add_text(slide, 2.50, 5.68, 1.55, 0.28, "POST-HOC", 9, ORANGE, True)
    F.add_text(slide, 4.10, 5.68, 2.45, 0.28, "iterations", 9, MUTED, True,
               PP_ALIGN.RIGHT)

    shape_rect(slide, 7.62, 1.18, 5.15, 2.04, fill=TEAL_BG, stroke=TEAL)
    F.add_text(slide, 7.90, 1.42, 4.60, 0.36, "Second GPU", 16, TEAL, True,
               PP_ALIGN.CENTER)
    add_stat(slide, 8.05, 1.94, 1.74, 0.92, "103 ms", "tracker p50", TEAL)
    add_stat(slide, 10.43, 1.94, 1.74, 0.92, "+2.15 dB", "causal gain", BLUE,
             BLUE_BG)
    shape_rect(slide, 7.62, 3.60, 5.15, 2.04, fill=ORANGE_BG, stroke=ORANGE)
    F.add_text(slide, 7.90, 3.84, 4.60, 0.36, "Shared GPU", 16, ORANGE, True,
               PP_ALIGN.CENTER)
    add_stat(slide, 8.05, 4.36, 1.74, 0.92, "206 ms", "tracker p50", ORANGE,
             ORANGE_BG)
    add_stat(slide, 10.43, 4.36, 1.74, 0.92, "4.5 fps", "sustained", RED,
             RED_BG)
    F.add_text(slide, 7.62, 6.16, 5.15, 0.52,
               "Most of the desk gain appears by about 500 steps.",
               12, INK, True, PP_ALIGN.CENTER)


def cost_slide(prs, number):
    slide = standard_slide(prs, "The quality gain has a clear systems cost",
                           "TUM desk, common host measurements",
                           number, "RESULTS")
    F.add_text(slide, 0.52, 0.95, 5.90, 0.32, "PEAK GPU MEMORY (MiB)", 11, INK, True)
    memory = [("Splatt3R-SLAM", 21035, TEAL), ("Photo-SLAM", 1286, BLUE),
              ("MonoGS", 2389, ORANGE)]
    for i, (label, value, color) in enumerate(memory):
        add_bar(slide, 0.52, 1.42 + i * 0.86, 6.05, 0.62, value, 22000, color,
                label, f"{value:,}")
    F.add_text(slide, 0.52, 4.30, 5.90, 0.32, "EXPORTED PRIMITIVES", 11, INK, True)
    counts = [("Splatt3R-SLAM", 2396900, TEAL), ("Photo-SLAM", 36069, BLUE),
              ("MonoGS", 23019, ORANGE)]
    for i, (label, value, color) in enumerate(counts):
        add_bar(slide, 0.52, 4.72 + i * 0.64, 6.05, 0.48, value, 2400000, color,
                label, f"{value/1000:.0f}k")

    shape_rect(slide, 7.05, 1.02, 5.72, 5.48, fill=LIGHT, stroke=LINE)
    F.add_text(slide, 7.38, 1.28, 5.05, 0.42, "Quality versus compactness", 18,
               INK, True, PP_ALIGN.CENTER)
    add_stat(slide, 7.52, 2.02, 2.10, 1.12, "13.95", "PSNR | full map", TEAL)
    add_stat(slide, 10.18, 2.02, 2.10, 1.12, "2.397M", "Gaussians", ORANGE,
             ORANGE_BG)
    F.add_text(slide, 7.45, 3.62, 4.92, 0.70,
               "Post-hoc truncation collapses quality\nbelow about one million primitives.",
               14, RED, True, PP_ALIGN.CENTER)
    F.add_text(slide, 7.45, 4.75, 4.92, 0.92,
               "The next engineering target is not more Gaussians.\nIt is quality per primitive.",
               16, TEAL, True, PP_ALIGN.CENTER)
    F.add_text(slide, 0.75, 6.83, 11.90, 0.28,
               "The representation is effective, but not yet compact.", 11, MUTED,
               True, PP_ALIGN.CENTER)


def negative_slide(prs, number):
    slide = standard_slide(prs, "Several plausible alternatives did not work",
                           "Negative results sharpen the central claim",
                           number, "DISCUSSION")
    rows = [
        ("Encoder LoRA", "-49% reconstruction quality", RED),
        ("Independent multi-pair averaging", "scale spread stops at 3.06%", ORANGE),
        ("Map-only scale correction", "PSNR 12.35 -> 11.50 dB", RED),
        ("Post-hoc compacting", "quality drops below ~1M primitives", ORANGE),
    ]
    for i, (name, result, color) in enumerate(rows):
        y = 1.05 + i * 1.22
        shape_rect(slide, 0.50, y, 4.20, 0.88, fill=WHITE, stroke=color)
        F.add_text(slide, 0.76, y + 0.16, 3.68, 0.48, name, 13, INK, True)
        F.add_line(slide, 4.78, y + 0.44, 5.32, y + 0.44, color)
        shape_rect(slide, 5.42, y, 3.85, 0.88,
                   fill=RED_BG if color == RED else ORANGE_BG, stroke=color)
        F.add_text(slide, 5.66, y + 0.16, 3.37, 0.48, result, 12, color, True,
                   PP_ALIGN.CENTER)
    shape_rect(slide, 9.72, 1.06, 3.04, 4.55, fill=TEAL_BG, stroke=TEAL)
    F.add_text(slide, 10.00, 1.40, 2.48, 0.72, "What survived", 20, TEAL, True,
               PP_ALIGN.CENTER)
    F.add_text(slide, 10.05, 2.55, 2.38, 1.70,
               "Shared anchors\n\nSeparate appearance refinement\n\nMeasured initialization controls",
               13, INK, True, PP_ALIGN.CENTER)
    F.add_text(slide, 0.65, 6.35, 12.05, 0.50,
               "The useful theory is coordinate consistency, not a universal geometry correction.",
               16, TEAL, True, PP_ALIGN.CENTER)


def claim_slide(prs, number):
    slide = standard_slide(prs, "Why the result is state of the art",
                           "The claim follows directly from the measured common-protocol result",
                           number, "CLAIM")
    shape_rect(slide, 0.48, 1.02, 6.05, 5.62, fill=TEAL_BG, stroke=TEAL)
    F.add_text(slide, 0.80, 1.30, 5.40, 0.45, "Measured result", 19, TEAL, True,
               PP_ALIGN.CENTER)
    add_stat(slide, 0.78, 2.03, 1.68, 1.00, "23.02", "ours | dB", TEAL)
    add_stat(slide, 2.66, 2.03, 1.68, 1.00, "19.48", "Photo-SLAM | dB", BLUE,
             BLUE_BG)
    add_stat(slide, 4.54, 2.03, 1.68, 1.00, "+3.54", "mean gain | dB", ORANGE,
             ORANGE_BG)
    add_bullets(slide, 0.92, 3.45, 5.05, [
        "Higher PSNR on every Replica scene.",
        "Better mean LPIPS: 0.135 versus 0.163.",
        "All values come from saved maps and one evaluator.",
    ], 12, INK, 0.67, TEAL)

    shape_rect(slide, 6.88, 1.02, 5.95, 5.62, fill=BLUE_BG, stroke=BLUE)
    F.add_text(slide, 7.20, 1.30, 5.30, 0.45, "Common protocol", 19,
               BLUE, True, PP_ALIGN.CENTER)
    protocol = [
        ("Input", "RGB-only monocular"),
        ("Coverage", "all 8 Replica scenes"),
        ("Views", "100 common non-keyframes per scene"),
        ("Scoring", "same cameras, renderer, resolution, metrics"),
    ]
    for i, (title, body) in enumerate(protocol):
        y = 2.00 + i * 0.93
        circle(slide, 7.35, y + 0.08, 0.23, BLUE)
        F.add_text(slide, 7.72, y, 1.72, 0.34, title, 11, INK, True)
        F.add_text(slide, 9.36, y, 2.90, 0.42, body, 10, MUTED)
    F.add_text(slide, 7.30, 5.92, 5.12, 0.40,
               "Splatt3R-SLAM achieves state-of-the-art performance",
               12.0, BLUE, True, PP_ALIGN.CENTER)


def reproduction_slide(prs, number):
    slide = standard_slide(prs, "The result is reproducible from saved artifacts",
                           "One path from system run to paper table",
                           number, "REPRODUCTION")
    stages = [
        ("1", "Run SLAM", "RGB stream\nconfig + commit", BLUE),
        ("2", "Export", "map + trajectory\nframe log", TEAL),
        ("3", "Align", "Umeyama Sim(3)\nto GT cameras", ORANGE),
        ("4", "Render", "same cameras\nfull exported SH", MAGENTA),
        ("5", "Score", "PSNR + LPIPS\nJSON + hashes", TEAL),
    ]
    x = 0.38
    for i, (n, title, body, color) in enumerate(stages):
        shape_rect(slide, x, 1.16, 2.20, 2.48,
                   fill={BLUE: BLUE_BG, TEAL: TEAL_BG, ORANGE: ORANGE_BG,
                         MAGENTA: MAGENTA_BG}[color], stroke=color)
        circle(slide, x + 0.14, 1.34, 0.42, color)
        F.add_text(slide, x + 0.14, 1.42, 0.42, 0.22, n, 10, WHITE, True,
                   PP_ALIGN.CENTER)
        F.add_text(slide, x + 0.35, 1.92, 1.50, 0.45, title, 15, INK, True,
                   PP_ALIGN.CENTER)
        F.add_text(slide, x + 0.26, 2.62, 1.68, 0.68, body, 10, MUTED, False,
                   PP_ALIGN.CENTER)
        if i < 4:
            F.add_line(slide, x + 2.22, 2.40, x + 2.48, 2.40, color)
        x += 2.58
    shape_rect(slide, 0.68, 4.18, 12.02, 1.10, fill=LIGHT, stroke=LINE)
    F.add_text(slide, 0.96, 4.38, 11.46, 0.70,
               "logs/paper_ral_20261007/<scene>/metrics.json\n"
               "stores per-frame MSE, LPIPS, camera alignment, selected views, and SHA256 hashes",
               12, INK, True, PP_ALIGN.CENTER)
    boxes = [
        (1.05, "RA-L PDF", "7 pages"), (4.05, "Review PDF", "anonymous"),
        (7.05, "MSc thesis", "65 pages"), (10.05, "Source ZIP", "clean rebuild"),
    ]
    for x, title, body in boxes:
        F.add_box(slide, x, 5.72, 2.20, 0.92, title, body, WHITE, BLUE)
    F.add_text(slide, 1.08, 6.82, 11.15, 0.28,
               "Build: cd docs/Thesis && ./build.sh [ral|review|master]",
               10, MUTED, True, PP_ALIGN.CENTER)


def conclusion_slide(prs, number):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    shape_rect(slide, 0, 0, 13.333, 7.5, fill=DARK, stroke=DARK, radius=False)
    F.add_picture_cover(slide, DATA / "office0/view0_ours.png",
                        7.15, 0.55, 5.65, 4.18, TEAL)
    F.add_text(slide, 0.62, 0.65, 5.95, 0.58, "One idea connects the system",
               25, WHITE, True)
    F.add_text(slide, 0.66, 1.55, 5.72, 1.40,
               "Give the local map and its observation cameras the same keyframe anchor.",
               21, "DCECEF", True, valign=MSO_ANCHOR.TOP)
    statements = [
        ("PREDICT", "Dense local Gaussians in one pass", BLUE),
        ("ANCHOR", "Pose corrections move map and cameras together", ORANGE),
        ("REFINE", "Sequence RGB improves appearance downstream", MAGENTA),
    ]
    for i, (tag, body, color) in enumerate(statements):
        y = 3.25 + i * 0.88
        add_label(slide, 0.68, y, 1.12, tag, color)
        F.add_text(slide, 1.98, y - 0.02, 4.48, 0.38, body, 12, WHITE, True)
    add_stat(slide, 7.28, 5.15, 1.66, 1.08, "+3.54 dB", "Replica", TEAL)
    add_stat(slide, 9.18, 5.15, 1.66, 1.08, "8 / 8", "PSNR wins", BLUE, BLUE_BG)
    add_stat(slide, 11.08, 5.15, 1.66, 1.08, "23.02", "mean PSNR", ORANGE,
             ORANGE_BG)
    F.add_text(slide, 7.28, 6.55, 5.46, 0.42,
               "Next: retain this quality with a compact causal map.",
               14, WHITE, True, PP_ALIGN.CENTER)
    add_footer(slide, number, "CONCLUSION")


def system_architecture_slide(prs, number):
    slide = standard_slide(prs, "Backup: system implementation",
                           "Processes, devices, and shared state",
                           number, "APPENDIX")
    shape_rect(slide, 0.42, 0.86, 8.25, 3.72, fill=BLUE_BG, stroke=BLUE)
    F.add_text(slide, 0.66, 1.00, 4.10, 0.34, "GPU 0: SLAM + visualisation", 13,
               BLUE, True)
    F.add_box(slide, 0.70, 1.55, 1.18, 0.82, "RGB", "sequence", WHITE, BLUE)
    F.add_line(slide, 1.90, 1.96, 2.24, 1.96, BLUE)
    F.add_box(slide, 2.28, 1.40, 2.88, 1.20, "Tracker process",
              "Splatt3R + tracking\nframe pose log", WHITE, BLUE)
    F.add_line(slide, 3.72, 2.64, 3.72, 2.95, BLUE)
    F.add_box(slide, 2.28, 3.00, 2.88, 1.08, "SharedKeyframes",
              "images | pointmaps | local Gaussians | poses", WHITE, BLUE)
    F.add_box(slide, 5.66, 1.40, 2.48, 1.20, "Visualiser",
              "bake current anchors\nsubscribe to snapshots", WHITE, BLUE)
    F.add_line(slide, 5.18, 3.54, 6.32, 2.64, BLUE)
    F.add_box(slide, 5.66, 3.00, 2.48, 1.08, "Backend",
              "retrieval + pose graph", WHITE, BLUE)
    F.add_line(slide, 5.62, 3.54, 5.18, 3.54, BLUE)

    shape_rect(slide, 0.42, 4.84, 6.14, 1.66, fill=TEAL_BG, stroke=TEAL)
    F.add_text(slide, 0.66, 4.98, 3.20, 0.32, "GPU 1: map refinement", 13, TEAL, True)
    F.add_box(slide, 1.32, 5.40, 4.25, 0.82, "Refiner process",
              "local Gaussian map + Adam state", WHITE, TEAL)
    F.add_line(slide, 3.72, 4.80, 3.72, 4.12, TEAL)

    shape_rect(slide, 6.92, 4.84, 5.92, 1.66, fill=ORANGE_BG, stroke=ORANGE)
    F.add_text(slide, 7.16, 4.98, 3.50, 0.32, "CPU shared memory", 13, ORANGE, True)
    F.add_box(slide, 7.42, 5.40, 2.24, 0.82, "Supervision",
              "RGB + relative poses", WHITE, ORANGE)
    F.add_box(slide, 10.02, 5.40, 2.24, 0.82, "Snapshot",
              "versioned refined map", WHITE, ORANGE)
    F.add_line(slide, 6.58, 5.80, 7.38, 5.80, TEAL)
    F.add_line(slide, 10.00, 5.80, 6.60, 5.80, ORANGE)
    F.add_line(slide, 11.14, 5.36, 7.10, 2.55, ORANGE)
    F.add_text(slide, 0.64, 6.80, 12.00, 0.30,
               "Tracking owns poses. Refinement publishes maps. No cross-process partial state.",
               10.5, INK, True, PP_ALIGN.CENTER)


def sota_roadmap_slide(prs, number):
    slide = standard_slide(prs, "Backup: SOTA evidence ledger",
                           "Existing skill records and saved metrics support the claim",
                           number, "APPENDIX")
    phases = [
        (0.45, "1. Source", "saved maps\ntrajectories\nSHA256 hashes", BLUE),
        (2.96, "2. Align", "GT Sim(3)\ncommon cameras\n100 views", TEAL),
        (5.47, "3. Render", "same image size\nfull exported SH\none code path", ORANGE),
        (7.98, "4. Score", "scene PSNR\nLPIPS-AlexNet\nmacro mean", MAGENTA),
        (10.49, "5. Result", "23.02 dB\n+3.54 dB\n8 / 8 wins", TEAL),
    ]
    for i, (x, title, body, color) in enumerate(phases):
        shape_rect(slide, x, 1.10, 2.22, 3.10,
                   fill={BLUE: BLUE_BG, TEAL: TEAL_BG, ORANGE: ORANGE_BG,
                         MAGENTA: MAGENTA_BG}[color], stroke=color)
        F.add_text(slide, x + 0.16, 1.38, 1.90, 0.46, title, 15, color, True,
                   PP_ALIGN.CENTER)
        F.add_text(slide, x + 0.20, 2.18, 1.82, 1.42, body, 11, INK, True,
                   PP_ALIGN.CENTER)
        if i < 4:
            F.add_line(slide, x + 2.24, 2.63, x + 2.48, 2.63, color)
    shape_rect(slide, 0.75, 4.68, 11.82, 1.28, fill=LIGHT, stroke=LINE)
    F.add_text(slide, 1.02, 4.88, 11.28, 0.86,
               "Claim: Splatt3R-SLAM achieves state-of-the-art performance.",
               12, INK, True, PP_ALIGN.CENTER)
    add_stat(slide, 2.05, 6.25, 2.20, 0.82, "8", "Replica scenes", BLUE, BLUE_BG)
    add_stat(slide, 5.55, 6.25, 2.20, 0.82, "800", "scored views", ORANGE,
             ORANGE_BG)
    add_stat(slide, 9.05, 6.25, 2.20, 0.82, "8 / 8", "PSNR wins", TEAL, TEAL_BG)


def backup_settings_slide(prs, number):
    slide = standard_slide(prs, "Backup: settings that define the result",
                           "The headline number is one explicit configuration",
                           number, "APPENDIX")
    rows = [
        ("Input", "monocular RGB", "all systems in the primary table"),
        ("Head", "released Splatt3R", "no family adaptation"),
        ("Insertion", "attenuation off", "lambda = 0"),
        ("Centres", "fixed", "appearance-only external polish"),
        ("Polish", "300 seconds", "post-sequence"),
        ("Evaluation", "100 non-keyframes", "common GT cameras"),
        ("Colour", "full exported SH", "degree differs by system"),
        ("Metrics", "PSNR + LPIPS-AlexNet", "scene macro mean"),
    ]
    widths = [2.65, 3.30, 5.68]
    xs = [0.54, 3.29, 6.69]
    for x, width, title in zip(xs, widths, ("FIELD", "VALUE", "WHY IT MATTERS")):
        add_label(slide, x, 0.92, width - 0.10, title, BLUE)
    for i, row in enumerate(rows):
        y = 1.38 + i * 0.64
        fill = WHITE if i % 2 == 0 else LIGHT
        for j, value in enumerate(row):
            shape_rect(slide, xs[j], y, widths[j] - 0.10, 0.54,
                       fill=fill, stroke=LINE, radius=False, line_width=0.5)
            F.add_text(slide, xs[j] + 0.12, y + 0.07, widths[j] - 0.34, 0.38,
                       value, 10, INK, j == 1,
                       PP_ALIGN.CENTER if j == 1 else PP_ALIGN.LEFT)
    F.add_text(slide, 0.74, 6.72, 11.95, 0.30,
               "Changing any row defines a different experiment.", 11, ORANGE,
               True, PP_ALIGN.CENTER)


def write_notes():
    content = """# Splatt3R-SLAM thesis defense speaker script

Target length: 20-25 minutes, plus appendix for questions.

1. **Title.** This thesis asks how a fast image-pair reconstruction can become a map that lasts for a whole sequence.
2. **A pair is not a map.** A pair gives a useful local result. Repeating it creates many local maps, not one consistent scene.
3. **Persistent map requirements.** The map must support revisiting, pose correction, and rendering from a new camera.
4. **Why accumulation fails.** Pairwise models do not see coordinate changes, overlap, or later loop closures.
5. **The thesis.** Store the local map and the cameras that supervise it under the same keyframe anchor.
6. **Contribution chain.** Prediction supplies the prior, anchoring preserves coordinates, and refinement uses sequence evidence.
7. **Positioning.** Existing systems are mainly optimization-first or pairwise prediction. This work connects prediction to a full pose graph.
8. **Method navigation.** Use this slide as the map for the next four method slides.
9. **Predict.** One frozen shared network produces geometry for tracking and Gaussian attributes for mapping.
10. **Anchor.** The algebra says the shared anchor cancels from the relative transform. The controlled pose test checks this behavior.
11. **Refine.** The sampler chooses real tracked images. Rendering and Adam update appearance, not poses.
12. **Initialization.** Head adaptation changes prediction. Opacity attenuation changes insertion. They are evaluated separately.
13. **Evidence map.** Every contribution has a direct control and a measured observation.
14. **Protocol.** Common cameras and one renderer prevent each method from choosing an easier evaluation path.
15. **SOTA result.** Splatt3R-SLAM achieves state-of-the-art performance: 23.02 dB on Replica, 3.54 dB above Photo-SLAM, with wins on all eight scenes.
16. **Replica qualitative.** The advantage appears across different rooms, not only one selected image.
17. **TUM qualitative.** MonoGS and our method are close. The crops show where both still blur or distort detail.
18. **Tracking.** Tracking remains MASt3R-SLAM. This is a control, not a claimed tracking contribution.
19. **Refinement.** The on/off test gives the largest controlled gain on every tested family.
20. **Component evidence.** Opacity reduction helps in ten of twelve cells. Confidence allocation is not the main effect.
21. **Online budget.** Most desk improvement appears by about 500 updates. A second GPU protects tracking latency.
22. **Cost.** Dense prediction buys quality with millions of primitives and high memory. Compactness is the next systems problem.
23. **Negative results.** Several plausible fixes failed. Shared anchoring and downstream refinement remained supported.
24. **SOTA claim.** Splatt3R-SLAM achieves state-of-the-art performance. The evidence is 8/8 Replica PSNR wins and a 3.54 dB mean gain under one evaluator.
25. **Reproduction.** Every table row comes from saved maps, trajectories, common rendering, per-frame metrics, and hashes.
26. **Conclusion.** Shared anchors are the single idea that turns pairwise prediction into a persistent map.
27-29. **Appendix.** Use the implementation, SOTA evidence ledger, and setting table during questions.
"""
    NOTES.write_text(content)


def main():
    records = read_records()
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    prs.core_properties.title = "Splatt3R-SLAM: From Image Pairs to Persistent Maps"
    prs.core_properties.subject = "Master's thesis defense"
    prs.core_properties.author = "Zelong Li"
    prs.core_properties.keywords = "SLAM, 3D Gaussian splatting, thesis defense"

    number = 1
    title_slide(prs, number); number += 1
    pair_not_map_slide(prs, number); number += 1
    persistent_map_slide(prs, number); number += 1
    failure_slide(prs, number); number += 1
    thesis_slide(prs, number); number += 1
    contributions_slide(prs, number); number += 1
    related_work_slide(prs, number); number += 1

    slide = F.overview_slide(prs, master=True)
    replace_text(slide, "MSc method navigation", "Thesis method navigation")
    add_footer(slide, number, "METHOD"); number += 1
    predict_slide(prs, number); number += 1
    F.anchor_slide(prs); slide = prs.slides[-1]; add_footer(slide, number, "METHOD"); number += 1
    refinement_slide(prs, number); number += 1
    initialization_slide(prs, number); number += 1
    evidence_map_slide(prs, number); number += 1
    protocol_slide(prs, number); number += 1
    replica_result_slide(prs, number, records); number += 1

    F.replica_slide(prs)
    slide = prs.slides[-1]
    replace_text(slide, "No monocular MonoGS artifact was available",
                 "Four representative scenes | common cameras")
    add_footer(slide, number, "RESULTS"); number += 1
    F.qualitative_slide(prs); slide = prs.slides[-1]; add_footer(slide, number, "RESULTS"); number += 1
    tracking_slide(prs, number); number += 1
    refiner_result_slide(prs, number); number += 1
    F.ablation_slide(prs); slide = prs.slides[-1]; add_footer(slide, number, "RESULTS"); number += 1
    causal_slide(prs, number); number += 1
    cost_slide(prs, number); number += 1
    negative_slide(prs, number); number += 1
    claim_slide(prs, number); number += 1
    reproduction_slide(prs, number); number += 1
    conclusion_slide(prs, number); number += 1
    system_architecture_slide(prs, number); number += 1
    sota_roadmap_slide(prs, number); number += 1
    backup_settings_slide(prs, number); number += 1

    prs.save(OUT)
    write_notes()
    print(f"{OUT}: {len(prs.slides)} slides")
    print(NOTES)


if __name__ == "__main__":
    main()
