"""Shared visual rules for the non-isolated HTML reports."""

from __future__ import annotations

BAR_EDGE_COLOR = "#111827"
GRID_COLOR = "#D8DEE8"
TEXT_COLOR = "#172033"
MUTED_COLOR = "#667085"

MODALITY_COLORS = {
    "M": "#264653",
    "MB": "#2A9D8F",
    "MP": "#E9C46A",
    "MR": "#E76F51",
    "MUV": "#6D597A",
}

COMBINATION_COLORS = {
    "M+MB": MODALITY_COLORS["MB"],
    "M+MR": MODALITY_COLORS["M"],
    "M+MB+MR": MODALITY_COLORS["MP"],
    "MR+MB": MODALITY_COLORS["MR"],
    "M+MB+MP+MR+MUV": MODALITY_COLORS["MUV"],
}

REPORT_CSS = """
:root { --ink:#172033; --muted:#667085; --line:#D8DEE8; --paper:#F7F9FC; --card:#FFFFFF; }
* { box-sizing:border-box; }
body { margin:0 auto; max-width:1600px; padding:38px 28px 70px; background:var(--paper); color:var(--ink); font-family:Inter,Arial,"Microsoft YaHei",sans-serif; line-height:1.6; }
h1 { margin:0 0 24px; color:var(--ink); font-size:clamp(28px,4vw,46px); letter-spacing:-.03em; border-bottom:3px solid #264653; padding-bottom:12px; }
h2 { margin:30px 0 14px; color:#264653; font-size:22px; border-left:5px solid #2A9D8F; padding-left:12px; }
h3 { color:var(--ink); margin:22px 0 10px; }
section, figure { margin:24px 0; padding:20px; background:var(--card); border:1px solid var(--line); border-radius:14px; box-shadow:0 10px 26px rgba(23,32,51,.05); }
img { display:block; width:100%; max-width:1250px; height:auto; margin:14px auto 26px; border:1px solid var(--line); border-radius:10px; background:#fff; }
table { width:100%; border-collapse:collapse; margin:14px 0 26px; font-size:12px; background:#fff; }
th, td { border:1px solid var(--line); padding:7px 8px; text-align:right; vertical-align:top; }
th { background:#EEF3F5; color:#264653; font-weight:700; }
td:first-child, th:first-child { text-align:left; }
pre { margin:14px 0; padding:14px; overflow:auto; white-space:pre-wrap; background:#F4F7F9; border:1px solid var(--line); border-radius:8px; font-size:12px; }
p.note { padding:13px 16px; background:#FFF8E7; border-left:4px solid #E9C46A; }
figcaption { color:var(--muted); font-size:12px; text-align:center; }
.gallery { display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:18px; }
.confusion-card { padding:14px; background:#FFFFFF; border:1px solid var(--line); border-radius:12px; }
@media (max-width:900px) { body { padding:24px 16px 50px; } section, figure { padding:14px; } table { display:block; overflow-x:auto; } }
"""


def modality_color(modality: str) -> str:
    return MODALITY_COLORS.get(modality, "#8A94A6")


def combination_color(combination: str) -> str:
    return COMBINATION_COLORS.get(combination, "#8A94A6")


def style_axis(axis) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.85, alpha=0.7)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(MUTED_COLOR)
    axis.spines["bottom"].set_color(MUTED_COLOR)
    axis.tick_params(axis="x", length=0, pad=8)
    axis.tick_params(axis="y", colors="#4B5563")


def style_bars(bars) -> None:
    for bar in bars:
        bar.set_edgecolor(BAR_EDGE_COLOR)
        bar.set_linewidth(1.35)


def combination_axis_label(combination: str) -> str:
    if combination == "M+MB+MP+MR+MUV":
        return "All\nchannels"
    if combination == "M (baseline)":
        return "M\nbaseline"
    if combination == "M+MB+MR":
        return "M+MB+\nMR"
    return combination
