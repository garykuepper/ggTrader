#!/usr/bin/env python3
"""Render the four root-cause charts from numbers traced to cited source lines."""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = "research/audits/root-cause-2026-11-04/figures"

# Source files (repo-relative, date prefix kept). Every plotted number cites one of these lines.
_BASE = "research/audits/root-cause-2026-11-04"
CEILING = f"{_BASE}/lenses/2026-11-04-lens4-feasibility-and-ceiling.md"
SYNTHESIS = f"{_BASE}/findings/2026-11-04-root-cause-investigation.md"
PARITY = f"{_BASE}/lenses/2026-11-04-lens3-live-parity.md"
REVIEW = f"{_BASE}/reviews/2026-11-05-direction-reviews-consolidated.md"

# Output size: 8 x 4.5 inches at 200 dpi is 1600 x 900 px.
FIG_SIZE = (8.0, 4.5)
DPI = 200

# Light card surface and ink, plus Okabe-Ito colours (colour-blind safe). Colour is never the
# only signal: every bar carries a direct label.
SURFACE = "#FCFCFB"
INK = "#1F2328"
MUTED = "#4B5563"
GRID = "#D9DCE0"
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREY = "#7A8088"
NEED_FILL = "#E5E7EB"
NEED_EDGE = "#4B5563"

# Benchmark each gap row is measured against, drawn under its window label.
BENCHMARK_TAGS = {
    "2021-2026": "vs SPY\ntotal return",
    "2006-2026": "vs SPY\ntotal return",
    "1994-2026": "vs French\nmarket index",
    "1927-2026": "vs French\nmarket index",
}

FONT = 10.5  # points; 10 pt or more keeps text near 15 px at 900 px display width
FONT_SMALL = 10.0
FONT_TITLE = 14.0


@dataclass(frozen=True)
class Point:
    """One plotted number with the repo-relative `path:line` where the source states it."""

    label: str
    value: float
    source: str


@dataclass(frozen=True)
class FigureSpec:
    """A figure: its message title, alt text, caption and the traced numbers it plots."""

    name: str
    title: str
    alt: str
    caption: str
    points: tuple[Point, ...]


@dataclass(frozen=True)
class OptionBlock:
    """One option column; each text cites the review line it was shortened from."""

    option: int
    title: str
    title_source: str
    question: str
    question_source: str
    step: str
    step_source: str
    stop: str
    stop_source: str


GAP_POINTS: tuple[Point, ...] = (
    Point("gap 2x needs, 2021-2026", 9.57, f"{SYNTHESIS}:67"),
    Point("gap 2x needs, 2006-2026", 3.54, f"{SYNTHESIS}:67"),
    Point("gap 2x needs, 1994-2026", 2.34, f"{CEILING}:38"),
    Point("gap 2x needs, 1927-2026", 0.77, f"{SYNTHESIS}:68"),
    Point("gap 2x needs, 2022-2026 (momentum window)", 9.54, f"{SYNTHESIS}:432"),
    Point("2x leverage arm, 2006-2026", 4.15, f"{SYNTHESIS}:69"),
    Point("2x leverage worst drawdown", -84.8, f"{SYNTHESIS}:69"),
    Point("SPY worst drawdown over the same window", -55.3, f"{CEILING}:70"),
    Point("average funding rate in the leverage window, percent", 1.64, f"{CEILING}:68"),
    Point("momentum replay, OOS 2022-2026", 4.37, f"{CEILING}:50"),
    Point("momentum interval low", -7.07, f"{CEILING}:51"),
    Point("momentum interval high", 16.05, f"{CEILING}:51"),
    Point("stock-picking results after 1994, low", -0.62, f"{SYNTHESIS}:72"),
    Point("stock-picking results after 1994, high", 0.83, f"{SYNTHESIS}:72"),
)

VERDICT_POINTS: tuple[Point, ...] = (
    Point("reliable", 0, f"{SYNTHESIS}:59"),
    Point("unreliable", 11, f"{SYNTHESIS}:59"),
    Point("inconclusive", 58, f"{SYNTHESIS}:59"),
    Point("stopped on their own terms", 3, f"{SYNTHESIS}:60"),
    Point("measured tests in total", 72, f"{SYNTHESIS}:59"),
)

SHORTFALL_POINTS: tuple[Point, ...] = (
    Point("behind SPY at run-time broker value", -3.76, f"{PARITY}:33"),
    Point("behind SPY at Alpaca end-of-day equity", -4.57, f"{PARITY}:195"),
    Point("idle cash before the SPY sweep (estimate)", -2.46, f"{PARITY}:40"),
    Point("MNST split never applied by the broker (estimate)", -0.85, f"{PARITY}:45"),
)

OPTIONS: tuple[OptionBlock, ...] = (
    OptionBlock(
        1,
        "Paper-trader measurement and parity",
        f"{REVIEW}:68",
        "Do live decisions differ from replica? Partial bar, halt base, broker artifacts?",
        f"{REVIEW}:70",
        "Write the parity proposal: archive, halt base, bar choice",
        f"{REVIEW}:71",
        "Stop: parity is measured and archived",
        f"{REVIEW}:74",
    ),
    OptionBlock(
        2,
        "Funding-honest leverage figures",
        f"{REVIEW}:76",
        "Does 2.0x still reach the 2006-2026 gap at today-like funding?",
        f"{REVIEW}:80",
        "None beyond restating existing figures with their funding caveat",
        f"{REVIEW}:81",
        "2.0x below the 2006-2026 gap at today-like funding: no level reaches 2x",
        f"{REVIEW}:84",
    ),
    OptionBlock(
        3,
        "One pre-registered long-history stock screen (accruals)",
        f"{REVIEW}:86",
        "One registered screen: about +1 pp/yr? Can the design resolve its bar?",
        f"{REVIEW}:90",
        "Write the brief and state the power limit before any run",
        f"{REVIEW}:91",
        "Stop: a long-history-test fail closes it",
        f"{REVIEW}:94",
    ),
)

FIGURES: dict[str, FigureSpec] = {
    "gap-vs-measured": FigureSpec(
        name="gap-vs-measured",
        title="Only one plotted result reached its gap: 2x leverage",
        alt=(
            "Horizontal bars in percentage points per year, beyond SPY total return for "
            "2021-2026 and 2006-2026 and beyond the French market index for 1994-2026 and "
            "1927-2026. For each window a grey bar shows the gap that doubling the benchmark "
            "needs: 9.57 for 2021-2026, 3.54 for 2006-2026, 2.34 for 1994-2026 and 0.77 for "
            "1927-2026. Beside it a blue mark shows the best measured result: momentum at +4.37 "
            "(tested 2022-2026, interval -7.07 to +16.05), 2x leverage at +4.15 which reaches "
            "its gap, and a range of stock-picking results from -0.62 to +0.83."
        ),
        caption=(
            "Source: the Ceiling check report and the root-cause synthesis. Percentage points per "
            "year beyond SPY total return for 2021-2026 and 2006-2026, and beyond the French "
            "market index for 1994-2026 and 1927-2026. Momentum was tested on 2022-2026, where "
            "the gap is 9.54. The momentum interval is the 95% interval on the arithmetic mean "
            "excess. The 2x leverage arm is the best of five arms, picked after seeing results. "
            "Leverage is more market exposure, not stock-picking skill: worst drawdown -84.8% "
            "against SPY's -55.3%, and the figure holds only at the window's 1.64% average "
            "funding. Pre-1994 French momentum also reached its own window's gap and is not "
            "plotted. The stock-picking range is the registered long-history screens after 1994."
        ),
        points=GAP_POINTS,
    ),
    "verdict-trust": FigureSpec(
        name="verdict-trust",
        title="None of the 72 measured tests is reliable",
        alt=(
            "One stacked bar of 72 measured past tests: 0 reliable, 11 unreliable, 58 "
            "inconclusive and 3 stopped on their own terms. The inconclusive segment is the "
            "largest by far. Inconclusive means the test could not decide, which is not the same "
            "as finding nothing."
        ),
        caption=(
            "Source: the Verdict Trust check, as summarised in the root-cause synthesis. A test "
            "is reliable only if its data were clean, its code checked, its number recomputed "
            "and its power adequate. Inconclusive means the test could not decide, which is not "
            "the same as finding nothing. 42 more rows are status labels, not tests, and are "
            "not plotted."
        ),
        points=VERDICT_POINTS,
    ),
    "paper-shortfall": FigureSpec(
        name="paper-shortfall",
        title="Paper account: 3.76 to 4.57 points behind SPY in total since 06-24",
        alt=(
            "Horizontal bars of percentage points behind SPY, totals since 2026-06-24 and not "
            "yearly rates. At run-time broker value the account trailed by 3.76, and at Alpaca "
            "end-of-day equity by 4.57. Below, two stated parts of the 3.76, both estimates: "
            "idle cash about 2.46, and the MNST split that the broker never applied, 0.85."
        ),
        caption=(
            "Source: the Paper-versus-Backtest check. Totals since 2026-06-24, not yearly rates. "
            "The split by part is an estimate. The two parts shown are the ones the report "
            "states. They are not a full breakdown and no remainder is computed."
        ),
        points=SHORTFALL_POINTS,
    ),
    "options-at-a-glance": FigureSpec(
        name="options-at-a-glance",
        title="Each option: one question, one cheap step, one stop gate",
        alt=(
            "Three columns. Option 1, paper-trader measurement and parity. Option 2, "
            "funding-honest leverage figures. Option 3, one pre-registered long-history stock "
            "screen. Each column gives the question, the cheapest step and the stop gate."
        ),
        caption=(
            "Source: the consolidated direction review, section 4, shortened. Options are in "
            "Opus's order. The wording is shortened from the review's option blocks. Option 2's "
            "cheapest step is derived by the review page, not by Opus. The owner decides which, "
            "if any, to run."
        ),
        points=(),
    ),
}


def _new_figure(title: str, subtitle: str | None = None) -> plt.Figure:
    """Create the light card figure with a left-aligned message title."""
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI, facecolor=SURFACE)
    fig.text(0.035, 0.94, title, fontsize=FONT_TITLE, fontweight="bold", color=INK, va="center")
    if subtitle:
        fig.text(0.035, 0.905, _wrap(subtitle, 88), fontsize=FONT, color=MUTED, va="top",
                 linespacing=1.3)  # fmt: skip
    return fig


def _style_axes(ax: plt.Axes) -> None:
    """Recessive axes: no frame except the bottom grid line."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=MUTED, labelsize=FONT_SMALL, length=0)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _value(name: str, label: str) -> float:
    """Look up a plotted value by label so the drawing code never repeats a literal."""
    for point in FIGURES[name].points:
        if point.label == label:
            return point.value
    raise KeyError(label)


def _wrap(text: str, width: int) -> str:
    """Word wrap to `width` characters per line without splitting hyphenated words."""
    return textwrap.fill(text, width, break_on_hyphens=False)


def _draw_gap(out: Path) -> None:
    """Draw the gap-versus-measured chart: grey gap bars beside blue measured marks."""
    fig = _new_figure(
        FIGURES["gap-vs-measured"].title,
        "Growth beyond the benchmark per year, in percentage points: "
        "what 2x needs against what was measured",
    )
    ax = fig.add_axes((0.13, 0.12, 0.84, 0.62))
    _style_axes(ax)
    v = partial(_value, "gap-vs-measured")
    windows = ("2021-2026", "2006-2026", "1994-2026", "1927-2026")
    needs = [v(f"gap 2x needs, {window}") for window in windows]
    pitch, height = 1.5, 0.3
    need_dy, got_dy = 0.2, -0.14

    def base(row: int) -> float:
        return (len(windows) - 1 - row) * pitch

    for row, (window, need) in enumerate(zip(windows, needs, strict=True)):
        y = base(row)
        ax.barh(
            y + need_dy, need, height, color=NEED_FILL, edgecolor=NEED_EDGE, linewidth=1.0, zorder=2
        )
        ax.text(need + 0.35, y + need_dy, f"{need:.2f}", va="center", fontsize=FONT, color=INK)
        ax.text(
            -8.7, y + 0.12, window, va="center", ha="right", fontsize=FONT, color=INK,
            fontweight="bold",
        )  # fmt: skip
        ax.text(
            -8.7, y - 0.06, BENCHMARK_TAGS[window], va="top", ha="right", fontsize=FONT_SMALL,
            color=MUTED, linespacing=1.2,
        )  # fmt: skip

    # 2021-2026 row: momentum bar with its interval whisker and a one-line interval label
    mom = v("momentum replay, OOS 2022-2026")
    mom_gap = v("gap 2x needs, 2022-2026 (momentum window)")
    lo, hi = v("momentum interval low"), v("momentum interval high")
    y0 = base(0) + got_dy
    ax.barh(y0, mom, height, color=BLUE, zorder=3)
    ax.plot([lo, hi], [y0 - 0.55] * 2, color=INK, linewidth=1.3, zorder=3)
    for end in (lo, hi):
        ax.plot([end, end], [y0 - 0.61, y0 - 0.49], color=INK, linewidth=1.3, zorder=3)
    ax.text(
        mom + 0.35, y0 + 0.12,
        f"+{mom:.2f} momentum replay (tested 2022-2026),\nshort of its {mom_gap:.2f} gap",
        va="top", fontsize=FONT, color=INK, linespacing=1.2,
    )  # fmt: skip
    ax.text(
        0.35, y0 - 0.62, f"95% interval on the arithmetic mean: {lo:.2f} to +{hi:.2f}",
        va="top", ha="left", fontsize=FONT_SMALL, color=MUTED,
    )  # fmt: skip

    # 2006-2026 row: 2x leverage reaches its gap, with the drawdown caveat beside it
    lev = v("2x leverage arm, 2006-2026")
    y1 = base(1) + got_dy
    ax.barh(y1, lev, height, color=BLUE, zorder=3)
    ax.text(
        lev + 0.35, y1, f"+{lev:.2f} 2x leverage reaches its gap", va="center",
        fontsize=FONT, color=INK, fontweight="bold",
    )  # fmt: skip
    ax.text(
        lev + 0.35, y1 - 0.3,
        f"best of five arms, picked after seeing results\n"
        f"worst drawdown {v('2x leverage worst drawdown'):.1f}% "
        f"against SPY's {v('SPY worst drawdown over the same window'):.1f}%",
        va="top", fontsize=FONT_SMALL, color=MUTED, linespacing=1.3,
    )  # fmt: skip

    # 1994-2026 row: stock-picking results drawn as a range, not a bar from zero
    low = v("stock-picking results after 1994, low")
    high = v("stock-picking results after 1994, high")
    y2 = base(2) + got_dy
    ax.barh(y2, high - low, height, left=low, color="white", edgecolor=BLUE, linewidth=2, zorder=3)
    ax.text(
        low - 0.5, y2, "range of registered\nscreens after 1994",
        va="center", ha="right", fontsize=FONT, color=INK, linespacing=1.3,
    )  # fmt: skip
    ax.text(
        high + 0.5, y2, f"{low:.2f} to +{high:.2f}, short of the gap",
        va="center", fontsize=FONT, color=INK,
    )  # fmt: skip

    ax.text(
        0.35, base(3) + got_dy - 0.12, "no stock-picking result plotted for this window",
        va="center", fontsize=FONT_SMALL, color=MUTED, style="italic",
    )  # fmt: skip

    ax.axvline(0, color=INK, linewidth=1.0, zorder=4)
    ax.set_xlim(-9, 19)
    ax.set_ylim(-0.85, 5.2)
    ax.set_yticks([])
    ax.set_xticks([-5, 0, 5, 10, 15])
    ax.set_xlabel("percentage points per year", fontsize=FONT_SMALL, color=MUTED)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=NEED_FILL, edgecolor=NEED_EDGE, label="gap 2x needs"),
        Rectangle((0, 0), 1, 1, facecolor=BLUE, label="best measured result"),
        Rectangle(
            (0, 0), 1, 1, facecolor="white", edgecolor=BLUE, linewidth=2, label="range of results"
        ),
    ]
    ax.legend(
        handles=handles, loc="lower right", frameon=False, fontsize=FONT_SMALL, labelcolor=INK,
        bbox_to_anchor=(1.0, 1.0), ncol=3, columnspacing=1.2, handlelength=1.6,
    )  # fmt: skip
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def _draw_verdict(out: Path) -> None:
    """Draw the verdict-trust chart: one stacked bar of the 72 measured tests."""
    fig = _new_figure(
        FIGURES["verdict-trust"].title,
        "72 measured tests, sorted by whether data, code, recomputed number and power hold up",
    )
    ax = fig.add_axes((0.035, 0.06, 0.93, 0.76))
    ax.set_facecolor(SURFACE)
    ax.axis("off")
    v = partial(_value, "verdict-trust")
    total = v("measured tests in total")
    un, inc, stop = v("unreliable"), v("inconclusive"), v("stopped on their own terms")
    ax.set_xlim(0, total)
    ax.set_ylim(0, 4)
    bar_y, bar_h = 1.9, 1.0
    left = 0.0
    for count, colour in ((un, VERMILLION), (inc, BLUE), (stop, GREY)):
        ax.add_patch(
            Rectangle((left, bar_y), count, bar_h, facecolor=colour, edgecolor=SURFACE, linewidth=2)
        )
        left += count
    for count, word, centre in (
        (un, "unreliable", un / 2),
        (inc, "inconclusive", un + inc / 2),
    ):
        ax.text(centre, bar_y + 0.62, f"{count:.0f}", ha="center", va="center", fontsize=24,
                fontweight="bold", color="white")  # fmt: skip
        ax.text(centre, bar_y + 0.24, word, ha="center", va="center", fontsize=FONT,
                fontweight="bold", color="white")  # fmt: skip
    # reliable is empty: a zero-width slot at the start of the bar, named by a leader line
    ax.plot([0.05, 0.05], [bar_y + bar_h, 3.35], color=INK, linewidth=1.2)
    ax.text(0.6, 3.55, "0 reliable", fontsize=FONT + 3, fontweight="bold", color=INK, va="center")
    # leader lines from the bar down to its descriptions
    ax.plot([un / 2] * 2, [bar_y, 1.5], color=INK, linewidth=1.0)
    ax.plot([un + inc / 2] * 2, [bar_y, 1.5], color=INK, linewidth=1.0)
    ax.text(0.0, 1.42, _wrap("Unreliable: judged on a voided baseline or built with leaks.", 22),
            fontsize=FONT_SMALL, color=MUTED, va="top", ha="left", linespacing=1.4)  # fmt: skip
    ax.text(un + 8.0, 1.42,
            _wrap("Inconclusive: the test could not decide, which is not the same as finding "
                  "nothing. Data, code, recompute or power fails on at least one point.", 50),
            fontsize=FONT_SMALL, color=MUTED, va="top", ha="left", linespacing=1.4)  # fmt: skip
    ax.plot([total - stop / 2] * 2, [bar_y, 0.75], color=INK, linewidth=1.0)
    ax.text(total, 0.45, f"{stop:.0f} stopped on their own terms", fontsize=FONT, color=INK,
            va="center", ha="right", fontweight="bold")  # fmt: skip
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def _draw_shortfall(out: Path) -> None:
    """Draw the paper-shortfall chart: two totals, then two stated (estimated) parts."""
    fig = _new_figure(
        FIGURES["paper-shortfall"].title,
        "Percentage points behind SPY, totals over the whole period and not yearly rates",
    )
    ax = fig.add_axes((0.035, 0.14, 0.93, 0.66))
    _style_axes(ax)
    v = partial(_value, "paper-shortfall")
    rows = [
        ("At run-time broker value", v("behind SPY at run-time broker value"), BLUE, None, 5.0),
        (
            "At Alpaca end-of-day equity (to 11-03)",
            v("behind SPY at Alpaca end-of-day equity"),
            BLUE,
            None,
            4.0,
        ),
        (
            "Part: idle cash before the SPY sweep (estimate)",
            v("idle cash before the SPY sweep (estimate)"),
            VERMILLION,
            "////",
            2.0,
        ),
        (
            "Part: MNST split never applied by the broker (estimate)",
            v("MNST split never applied by the broker (estimate)"),
            VERMILLION,
            "////",
            1.0,
        ),
    ]
    height = 0.42
    for name, value, colour, hatch, y in rows:
        ax.barh(
            y, value, height, color=colour, hatch=hatch, edgecolor=SURFACE if hatch else colour,
            linewidth=0.0, zorder=3,
        )  # fmt: skip
        ax.text(-5.4, y + 0.43, name, fontsize=FONT, color=INK, va="center", fontweight="bold")
        ax.text(value - 0.1, y, f"{value:.2f}", fontsize=FONT + 1, color=INK, va="center",
                ha="right", fontweight="bold")  # fmt: skip
    ax.text(
        -5.4, 3.0, "Stated parts of the run-time figure, not a full breakdown:",
        fontsize=FONT_SMALL, color=MUTED, va="center", style="italic",
    )  # fmt: skip
    ax.axhline(3.45, color=GRID, linewidth=1.0)
    ax.axvline(0, color=INK, linewidth=1.0, zorder=4)
    ax.set_xlim(-5.45, 0.3)
    ax.set_ylim(0.3, 5.7)
    ax.set_yticks([])
    ax.set_xticks([-5, -4, -3, -2, -1, 0])
    ax.set_xlabel(
        "percentage points versus SPY (negative = behind)", fontsize=FONT_SMALL, color=MUTED
    )
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def _draw_options(out: Path) -> None:
    """Draw the options chart: three columns of question, cheapest step and stop gate."""
    fig = _new_figure(
        FIGURES["options-at-a-glance"].title,
        "In Opus's order. The owner decides which, if any, to run.",
    )
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.patch.set_alpha(0)
    headings = (("THE QUESTION", 0.625), ("CHEAPEST STEP", 0.42), ("STOP GATE", 0.215))
    col_w, gap, x0 = 0.2975, 0.0185, 0.035
    top, bottom, head_h = 0.84, 0.03, 0.18
    for i, block in enumerate(OPTIONS):
        x = x0 + i * (col_w + gap)
        ax.add_patch(
            Rectangle((x, bottom), col_w, top - bottom, facecolor="white", edgecolor=GRID,
                      linewidth=1.2)
        )  # fmt: skip
        # one neutral header colour: blue and vermilion carry data meaning in the other charts
        ax.add_patch(Rectangle((x, top - head_h), col_w, head_h, facecolor=MUTED, edgecolor="none"))
        ax.text(x + 0.012, top - 0.028, f"Option {block.option}", fontsize=FONT_SMALL,
                color="white", va="center", fontweight="bold")  # fmt: skip
        ax.text(x + 0.012, top - 0.055, _wrap(block.title, 29), fontsize=FONT_SMALL,
                color="white", va="top", linespacing=1.3)  # fmt: skip
        for (heading, y), body in zip(headings, (block.question, block.step, block.stop)):
            ax.text(x + 0.012, y, heading, fontsize=FONT_SMALL - 0.5, color=MUTED,
                    va="center", fontweight="bold")  # fmt: skip
            width = 26 if heading == "STOP GATE" else 28  # keeps a margin from the column edge
            ax.text(x + 0.012, y - 0.03, _wrap(body, width), fontsize=FONT, color=INK, va="top",
                    linespacing=1.35)  # fmt: skip
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


_DRAWERS = {
    "gap-vs-measured": _draw_gap,
    "verdict-trust": _draw_verdict,
    "paper-shortfall": _draw_shortfall,
    "options-at-a-glance": _draw_options,
}


def render(name: str, out_dir: Path) -> Path:
    """Render one figure into `out_dir` and return the PNG path."""
    if name not in FIGURES:
        raise KeyError(f"unknown figure {name!r}; choose from {sorted(FIGURES)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.png"
    _DRAWERS[name](out)
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: render all figures, or one with --only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=None,
        help=f"output directory (default: {DEFAULT_OUT_DIR} under the repo root)",
    )
    parser.add_argument("--only", choices=sorted(FIGURES), help="render only this figure")
    args = parser.parse_args(argv)
    out_dir = Path(args.out) if args.out else REPO_ROOT / DEFAULT_OUT_DIR
    names = [args.only] if args.only else list(FIGURES)
    for name in names:
        print(render(name, out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
