#!/usr/bin/env python
"""Measured schematic-diagram toolkit shared by the hand-composed figures.

Layout is measured, not hand-tuned. Text is wrapped against the real font metrics of the
box it lives in, so card heights follow from their own content and a figure can assert its
own fit instead of trusting eyeballed coordinates. Data units equal inches, which makes
every measurement portable between figures.

Inline markup inside content strings:
    **bold**   `mono`   *italic*
Runs that touch without whitespace stay glued, so `**7.38%**, over` keeps its comma tight.

Consumers: fig_preprocessing_defect.py, fig_benchmark_curation.py.
"""
from __future__ import annotations

import re

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MPath

# ---------------------------------------------------------------------------
# Palette: neutral body text, colour reserved for routing and status.
# ---------------------------------------------------------------------------
PAPER = "#ffffff"
WASH = "#f8fafc"
HAIR = "#e2e8f0"
INK = "#0f172a"
BODY = "#334155"
MUTED = "#64748b"
DOT = "#94a3b8"

SLATE = dict(line="#475569", tint="#f1f5f9", edge="#e2e8f0", head="#334155", body="#475569")
EMERALD = dict(line="#2f8f5b", tint="#eaf6ef", edge="#c8e6d3", head="#155c37", body="#1f7a4c")
ROSE = dict(line="#d34a5c", tint="#fbeef0", edge="#f2c9ce", head="#8a1f2c", body="#b23347")
INDIGO = dict(line="#4f5fc4", tint="#eef0fb", edge="#c9cfef", head="#28307d", body="#3a45a3")
VIOLET = dict(line="#7c5cc7", tint="#f3eefb", edge="#ddd0f0", head="#432d7a", body="#5c3fa3")
AMBER = dict(line="#c07f1a", tint="#fbf3e4", edge="#f0dca8", head="#7a4e0c", body="#9c661a")

# ---------------------------------------------------------------------------
# Type scale (points) and vertical rhythm (inches).
# ---------------------------------------------------------------------------
FS_TITLE = 11.6
FS_SUB = 8.1
FS_BODY = 7.8
FS_PANEL = 7.6
FS_NOTE = 7.1
FS_TAG = 6.3
FS_GLYPH = 6.4

LEAD_BODY = 0.150
LEAD_PANEL = 0.146
LEAD_NOTE = 0.138
BULLET_GAP = 0.074
BULLET_INDENT = 0.155
TAG_H = 0.175

# Journal artwork specs ask for sans-serif figure lettering (Nature: Helvetica or Arial,
# 5-7 pt at final size). Liberation Sans is metric-compatible with Arial and freely
# licensed, so every checkout renders identical figures.
TEXT_FAMILY = "Liberation Sans"
MONO_FAMILY = "Liberation Mono"

FAMILY = {"n": TEXT_FAMILY, "b": TEXT_FAMILY, "i": TEXT_FAMILY,
          "m": MONO_FAMILY, "s": TEXT_FAMILY, "sb": TEXT_FAMILY}
WEIGHT = {"b": "bold", "sb": "bold"}
SLANT = {"i": "italic"}


def require_fonts(*families: str) -> None:
    """Fails loudly instead of letting matplotlib substitute DejaVu behind our backs."""
    for family in families or (TEXT_FAMILY, MONO_FAMILY):
        try:
            font_manager.findfont(FontProperties(family=family),
                                  fallback_to_default=False)
        except ValueError as exc:
            raise RuntimeError(
                f"font {family!r} is not installed; figures would silently fall back to "
                f"DejaVu and change every measured layout ({exc})") from exc


require_fonts()

_MARKUP = re.compile(r"\*\*(.+?)\*\*|`(.+?)`|\*(.+?)\*")
_SPLIT = re.compile(r"(\s+)")


def parse(text: str) -> list[tuple[str, str]]:
    """Splits inline markup into (text, style) runs."""
    runs: list[tuple[str, str]] = []
    pos = 0
    for m in _MARKUP.finditer(text):
        if m.start() > pos:
            runs.append((text[pos:m.start()], "n"))
        bold, mono, italic = m.groups()
        if bold is not None:
            runs.append((bold, "b"))
        elif mono is not None:
            runs.append((mono, "m"))
        else:
            runs.append((italic, "i"))
        pos = m.end()
    if pos < len(text):
        runs.append((text[pos:], "n"))
    return runs


def tokenize(runs: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Words as fragment lists. Style changes without whitespace do not break a word."""
    words: list[list[tuple[str, str]]] = []
    glued = False
    for text, style in runs:
        for part in _SPLIT.split(text):
            if not part:
                continue
            if part.isspace():
                glued = False
                continue
            if glued and words:
                words[-1].append((part, style))
            else:
                words.append([(part, style)])
            glued = True
    return words


class Ctx:
    """Drawing context in inches: data units equal inches, so measurement is portable."""

    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.renderer = fig.canvas.get_renderer()
        self._cache: dict[tuple, float] = {}

    def kw(self, style: str) -> dict:
        return dict(family=FAMILY[style], weight=WEIGHT.get(style, "normal"),
                    style=SLANT.get(style, "normal"))

    def measure(self, text: str, size: float, style: str = "n") -> float:
        """Text advance width in inches."""
        key = (text, size, style)
        hit = self._cache.get(key)
        if hit is None:
            probe = self.fig.text(0, 0, text, fontsize=size, **self.kw(style))
            hit = probe.get_window_extent(self.renderer).width / self.fig.dpi
            probe.remove()
            self._cache[key] = hit
        return hit

    def text(self, x: float, y: float, s: str, size: float, style: str, color: str,
             ha: str = "left", va: str = "top") -> None:
        self.ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, zorder=6,
                     **self.kw(style))


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def card(ax, x: float, y_top: float, w: float, h: float, accent: str, *,
         face: str = PAPER, edge: str = HAIR, radius: float = 0.10,
         stripe: float = 0.0, lw: float = 0.8, zorder: float = 2) -> None:
    """Rounded card with a CSS-style left accent rule, clipped to the rounded corners."""
    box = FancyBboxPatch((x, y_top - h), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         facecolor=face, edgecolor=edge, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    if stripe > 0:
        rule = Rectangle((x, y_top - h), stripe, h, facecolor=accent, edgecolor="none",
                         zorder=zorder + 0.5)
        ax.add_patch(rule)
        rule.set_clip_path(box)


def pill(ctx: Ctx, x: float, y_mid: float, label: str, *, fg: str, bg: str, edge: str,
         size: float = FS_TAG, style: str = "sb", anchor: str = "left",
         pad_x: float = 0.085, h: float = TAG_H, radius: float = 0.09,
         track: str = "\u200a") -> float:
    """Tag pill sized to its text. Returns its width."""
    text = track.join(label) if track else label
    w = ctx.measure(text, size, style) + 2 * pad_x
    x0 = {"left": x, "right": x - w, "center": x - w / 2}[anchor]
    ctx.ax.add_patch(FancyBboxPatch((x0, y_mid - h / 2), w, h,
                                    boxstyle=f"round,pad=0,rounding_size={radius}",
                                    facecolor=bg, edgecolor=edge, linewidth=0.7, zorder=5))
    ctx.ax.text(x0 + w / 2, y_mid, text, fontsize=size, color=fg, ha="center", va="center",
                zorder=6, **ctx.kw(style))
    return w


def paragraph(ctx: Ctx, runs: list[tuple[str, str]], x: float, y_top: float, w: float, *,
              size: float, leading: float, color: str, draw: bool) -> float:
    """Greedy word wrap against measured widths. Returns consumed height."""
    space = ctx.measure(" ", size, "n")
    lines: list[list[list[tuple[str, str]]]] = []
    line: list[list[tuple[str, str]]] = []
    used = 0.0

    for word in tokenize(runs):
        ww = sum(ctx.measure(frag, size, style) for frag, style in word)
        nxt = ww if not line else used + space + ww
        if line and nxt > w:
            lines.append(line)
            line, used = [word], ww
        else:
            line.append(word)
            used = nxt
    if line:
        lines.append(line)

    if draw:
        y = y_top
        for row in lines:
            x_cursor = x
            for word in row:
                for frag, style in word:
                    ctx.text(x_cursor, y, frag, size, style, color)
                    x_cursor += ctx.measure(frag, size, style)
                x_cursor += space
            y -= leading
    return len(lines) * leading


def bullets(ctx: Ctx, items: list[str], x: float, y_top: float, w: float, *,
            size: float = FS_BODY, leading: float = LEAD_BODY, gap: float = BULLET_GAP,
            color: str = BODY, draw: bool) -> float:
    """Hanging-indent bullet list. Returns consumed height."""
    total = 0.0
    for i, item in enumerate(items):
        y = y_top - total
        if draw:
            ctx.text(x, y, "\u2022", size + 0.6, "n", DOT)
        total += paragraph(ctx, parse(item), x + BULLET_INDENT, y, w - BULLET_INDENT,
                           size=size, leading=leading, color=color, draw=draw)
        if i < len(items) - 1:
            total += gap
    return total


def sub_panel(ctx: Ctx, x: float, y_top: float, w: float, head: str, body: str,
              tone: dict, draw: bool) -> float:
    """Tinted record panel. Measures its own height, then paints the fill behind the text."""
    pad_x, pad_y = 0.145, 0.125
    if draw:
        h = sub_panel(ctx, x, y_top, w, head, body, tone, draw=False)
        card(ctx.ax, x, y_top, w, h, tone["line"], face=tone["tint"], edge=tone["edge"],
             radius=0.07, stripe=0.0, zorder=3)
    y = y_top - pad_y
    if draw:
        ctx.text(x + pad_x, y, head, FS_PANEL + 0.4, "b", tone["head"])
    y -= 0.185
    for para in body.split("\n"):
        y -= paragraph(ctx, parse(para), x + pad_x, y, w - 2 * pad_x, size=FS_PANEL,
                       leading=LEAD_PANEL, color=tone["body"], draw=draw)
    return y_top - y + pad_y


def flow(ax, p0: tuple[float, float], p1: tuple[float, float], color: str, *,
         dashed: bool = False, lw: float = 1.9) -> tuple[float, float]:
    """Horizontal S-curve connector with an anchored origin. Returns the curve midpoint."""
    (x0, y0), (x1, y1) = p0, p1
    dx = (x1 - x0) * 0.52
    c0, c1 = (x0 + dx, y0), (x1 - dx, y1)
    path = MPath([p0, c0, c1, p1],
                 [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=12,
                                 color=color, linewidth=lw, shrinkA=0, shrinkB=0,
                                 linestyle=(0, (3.0, 2.2)) if dashed else "solid",
                                 capstyle="round", joinstyle="round", zorder=1))
    ax.add_patch(Circle(p0, 0.032, facecolor=color, edgecolor=PAPER, linewidth=0.8,
                        zorder=5))
    return ((x0 + 3 * c0[0] + 3 * c1[0] + x1) / 8.0, (y0 + 3 * c0[1] + 3 * c1[1] + y1) / 8.0)


def descend(ax, x: float, y0: float, y1: float, color: str, *, lw: float = 1.9,
            dashed: bool = False) -> None:
    """Vertical spine connector between stacked cards."""
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=12,
                                 color=color, linewidth=lw, shrinkA=0, shrinkB=0,
                                 linestyle=(0, (3.0, 2.2)) if dashed else "solid",
                                 capstyle="round", zorder=1))
