from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, VPacker, TextArea
from matplotlib.patches import Circle
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd

from .config import FIGURES, MODELS, METHODS, METHOD_COLORS


def _pick_installed_font(candidates: tuple[str, ...], fallback: str) -> str:
    """Return the first installed font name."""
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    return next((name for name in candidates if name in installed), fallback)


# Use the exact Windows font files whenever they exist.  This removes all
# font-manager ambiguity: Chinese is drawn from simsun.ttc and Latin/unit text
# is drawn from times.ttf.  Name-based fallbacks remain only for non-Windows
# environments.
WINDOWS_FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
TIMES_REGULAR_FILE = WINDOWS_FONT_DIR / "times.ttf"
TIMES_BOLD_FILE = WINDOWS_FONT_DIR / "timesbd.ttf"
SIMSUN_FILE = WINDOWS_FONT_DIR / "simsun.ttc"

LATIN_FONT = _pick_installed_font(
    ("Times New Roman", "Times New Roman PS MT", "Times", "Nimbus Roman", "Liberation Serif"),
    "DejaVu Serif",
)
CN_FONT = _pick_installed_font(
    ("SimSun", "Songti SC", "STSong", "Noto Serif CJK SC", "Source Han Serif SC"),
    "DejaVu Sans",
)


def _font_from_file_or_family(path: Path, family: str, *, weight: str = "normal") -> FontProperties:
    if path.exists():
        return FontProperties(fname=str(path), weight=weight)
    try:
        resolved = font_manager.findfont(FontProperties(family=family), fallback_to_default=False)
        return FontProperties(fname=resolved, weight=weight)
    except Exception:
        return FontProperties(family=family, weight=weight)


LATIN_FONT_PROP = _font_from_file_or_family(TIMES_REGULAR_FILE, LATIN_FONT)
LATIN_BOLD_PROP = _font_from_file_or_family(TIMES_BOLD_FILE, LATIN_FONT, weight="bold")
CN_FONT_PROP = _font_from_file_or_family(SIMSUN_FILE, CN_FONT)


def _print_font_diagnostics() -> None:
    """Print and validate the exact font files used by the exporter."""
    cn_file = CN_FONT_PROP.get_file() or CN_FONT
    latin_file = LATIN_FONT_PROP.get_file() or LATIN_FONT
    print("[visualization] mixed-font renderer: exact-v4")
    print(f"[visualization] Chinese font: {cn_file}")
    print(f"[visualization] Latin font:   {latin_file}")
    if os.name == "nt":
        if not SIMSUN_FILE.exists():
            raise FileNotFoundError(f"Required SimSun file not found: {SIMSUN_FILE}")
        if not TIMES_REGULAR_FILE.exists():
            raise FileNotFoundError(f"Required Times New Roman file not found: {TIMES_REGULAR_FILE}")
        if Path(str(cn_file)).resolve() != SIMSUN_FILE.resolve():
            raise RuntimeError(f"Chinese font is not SimSun: {cn_file}")
        if Path(str(latin_file)).resolve() != TIMES_REGULAR_FILE.resolve():
            raise RuntimeError(f"Latin font is not Times New Roman: {latin_file}")


def _font_collection_requires_type3(font_name: str) -> bool:
    """Older Matplotlib/fontTools releases handle TTC/OTC CJK fonts most reliably as Type 3."""
    try:
        font_path = font_manager.findfont(
            FontProperties(family=font_name), fallback_to_default=False
        )
    except Exception:
        return False
    return Path(font_path).suffix.lower() in {".ttc", ".otc"}


PDF_FONT_TYPE = 3 if (CN_FONT_PROP.get_file() and Path(CN_FONT_PROP.get_file()).suffix.lower() in {".ttc", ".otc"}) else 42

DOUBLE_COLUMN_SIZE = (7.20, 5.15)
BASE_FONT_SIZE = 11.2
TICK_FONT_SIZE = 10.0
LEGEND_FONT_SIZE = 9.6
ANNOTATION_FONT_SIZE = 9.2
RED_CMAP = LinearSegmentedColormap.from_list("white_red", ["#FFFFFF", "#FF4F4F"])
DIV_CMAP = LinearSegmentedColormap.from_list("blue_white_red", ["#007FFF", "#FFFFFF", "#FF4F4F"])
CONDITION_ORDER = ("标称", "重尾扰动", "混合偏移", "突发失配")
LAYOUT_ORDER = ("交叉口", "环岛", "仓储通道")
METHOD_TICKS = ("Tube", "CC", "CVaR", "ECBF", "Uniform-DR", "DREAM")


def _style() -> None:
    mpl.rcParams.update({
        # Latin-only strings use Times New Roman. Chinese strings are assigned
        # an explicit CJK font immediately before saving (see _apply_text_fonts).
        "font.family": LATIN_FONT,
        "font.serif": [LATIN_FONT],
        "mathtext.fontset": "custom",
        "mathtext.rm": LATIN_FONT,
        "mathtext.it": f"{LATIN_FONT}:italic",
        "mathtext.bf": f"{LATIN_FONT}:bold",
        "mathtext.default": "regular",
        # Use Type 3 only for TTC/OTC CJK collections on older Matplotlib;
        # otherwise embed TrueType outlines as Type 42. Explicit font assignment
        # below is what prevents blank Chinese glyphs in mixed labels.
        "axes.unicode_minus": False, "pdf.fonttype": PDF_FONT_TYPE, "ps.fonttype": PDF_FONT_TYPE,
        "pdf.use14corefonts": False,
        "font.size": BASE_FONT_SIZE, "axes.labelsize": BASE_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE, "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE, "axes.titlesize": BASE_FONT_SIZE,
        "axes.linewidth": 0.9, "lines.linewidth": 1.65, "lines.markersize": 4.8,
        "figure.constrained_layout.use": False,
    })


def _contains_cjk(text: str) -> bool:
    return any(
        "\u3400" <= char <= "\u9fff"
        or "\u3000" <= char <= "\u303f"
        or "\uff00" <= char <= "\uffef"
        for char in text
    )


def _contains_latin_or_math(text: str) -> bool:
    """Whether text contains Latin letters, digits, or a MathText segment."""
    return "$" in text or bool(re.search(r"[A-Za-z0-9²³⁻·]", text))


def _font_prop(base: FontProperties, size: float, weight: str | int = "normal") -> FontProperties:
    """Copy a concrete font property while preserving its resolved font file."""
    if base.get_file():
        return FontProperties(fname=base.get_file(), size=size, weight=weight)
    return FontProperties(family=base.get_family(), size=size, weight=weight)


def _split_font_runs(text: str) -> list[tuple[str, FontProperties]]:
    """Split one mixed label into Chinese and Latin/MathText runs.

    Chinese runs use the exact simsun.ttc file. Latin letters, numbers,
    units, variables, slash separators, and optional MathText expressions use
    the exact times.ttf file. Each run is a separate Text artist, so no glyph
    fallback is involved.
    """
    runs: list[tuple[str, FontProperties]] = []
    # Keep every $...$ expression intact, then split the remaining text by CJK.
    chunks = re.split(r"(\$[^$]*\$)", text)
    for chunk in chunks:
        if not chunk:
            continue
        if chunk.startswith("$") and chunk.endswith("$"):
            runs.append((chunk, LATIN_FONT_PROP))
            continue
        # Treat Chinese punctuation and full-width punctuation as Chinese too,
        # so characters such as “；”“，”“（”“）” are drawn by SimSun.
        pattern = r"[\u3000-\u303f\u3400-\u9fff\uff00-\uffef]+|[^\u3000-\u303f\u3400-\u9fff\uff00-\uffef]+"
        for part in re.findall(pattern, chunk):
            if not part:
                continue
            prop = CN_FONT_PROP if _contains_cjk(part) else LATIN_FONT_PROP
            runs.append((part, prop))
    return runs


def _mixed_text_box(text: str, fontsize: float, color: str = "black",
                    rotation: float = 0.0, weight: str | int = "normal"):
    """Build a packed text box whose runs can use different font files."""
    children = []
    for run, base_prop in _split_font_runs(text):
        props = {
            "fontproperties": _font_prop(base_prop, fontsize, weight),
            "color": color,
            "rotation": rotation,
        }
        area = TextArea(run, textprops=props)
        area._text._mixed_font_locked = True
        children.append(area)
    if not children:
        children = [TextArea("", textprops={"fontsize": fontsize})]

    normalized_rotation = float(rotation) % 360.0
    if normalized_rotation in (90.0, 270.0):
        # Standard Matplotlib y labels read in the direction of rotation. For
        # 90 degrees, reverse the packed run order so the complete phrase keeps
        # the same reading order as one ordinary rotated Text object.
        ordered = list(reversed(children)) if normalized_rotation == 90.0 else children
        return VPacker(children=ordered, align="center", pad=0, sep=0)
    return HPacker(children=children, align="baseline", pad=0, sep=0)


def _replace_one_axis_label(fig: plt.Figure, label: mpl.text.Text) -> None:
    """Replace a mixed CJK/Latin axis label with separately fonted text runs."""
    raw = label.get_text() or ""
    if not (_contains_cjk(raw) and _contains_latin_or_math(raw)):
        return

    # Reuse Matplotlib's already-computed label centre, including labelpad,
    # rotated tick labels, left/right y-axis placement, and colorbar placement.
    display_xy = label.get_transform().transform(label.get_position())
    figure_xy = fig.transFigure.inverted().transform(display_xy)
    rotation = float(label.get_rotation())
    fontsize = float(label.get_fontsize())
    color = label.get_color()
    weight = label.get_fontweight()

    box = _mixed_text_box(raw, fontsize, color=color, rotation=rotation, weight=weight)
    artist = AnnotationBbox(
        box,
        figure_xy,
        xycoords=fig.transFigure,
        box_alignment=(0.5, 0.5),
        frameon=False,
        pad=0,
    )
    artist.set_in_layout(True)
    fig.add_artist(artist)
    label.set_visible(False)


def _replace_mixed_axis_labels(fig: plt.Figure) -> None:
    """Apply exact SimSun/Times fonts within every mixed axis or colorbar label."""
    # Give mixed placeholders a temporary Chinese-capable font before the first
    # layout draw. They are hidden immediately afterwards, but this avoids any
    # transient missing-glyph warnings while Matplotlib measures their bounds.
    candidates: list[mpl.text.Text] = []
    for ax in fig.axes:
        for label in (ax.xaxis.label, ax.yaxis.label):
            raw = label.get_text() or ""
            if _contains_cjk(raw) and _contains_latin_or_math(raw):
                label.set_fontproperties(CN_FONT_PROP)
                candidates.append(label)

    # A first draw resolves the final locations chosen by Matplotlib's axis
    # label machinery. The replacement boxes are then placed at those centres.
    fig.canvas.draw()
    for label in candidates:
        _replace_one_axis_label(fig, label)


def _add_mixed_line_legend(ax: plt.Axes, text: str, *, xy=(0.03, 0.96)) -> AnnotationBbox:
    """Draw a compact dashed-line legend with mixed Chinese/Times text."""
    line_area = DrawingArea(28, 10, 0, 0)
    sample = Line2D([1, 27], [5, 5], color="#333333", linestyle="--", linewidth=1.0)
    line_area.add_artist(sample)
    text_box = _mixed_text_box(text, LEGEND_FONT_SIZE)
    packed = HPacker(children=[line_area, text_box], align="center", pad=0, sep=4)
    artist = AnnotationBbox(
        packed,
        xy,
        xycoords=ax.transAxes,
        box_alignment=(0.0, 1.0),
        frameon=False,
        pad=0,
    )
    artist.set_in_layout(True)
    ax.add_artist(artist)
    return artist


def _add_mixed_note(ax: plt.Axes, text: str, *, xy: tuple[float, float],
                    edgecolor: str, fontsize: float = 9.5) -> AnnotationBbox:
    """Add a boxed annotation with Chinese in SimSun and units in Times."""
    packed = _mixed_text_box(text, fontsize)
    artist = AnnotationBbox(
        packed,
        xy,
        xycoords=ax.transAxes,
        box_alignment=(0.0, 0.0),
        frameon=True,
        pad=0.22,
        bboxprops={
            "boxstyle": "round,pad=.22",
            "facecolor": "white",
            "edgecolor": edgecolor,
            "alpha": .88,
            "linewidth": .8,
        },
    )
    artist.set_in_layout(True)
    ax.add_artist(artist)
    return artist


def _apply_text_fonts(fig: plt.Figure) -> None:
    """Apply CJK/Latin fonts to every Matplotlib Text object before export.

    Matplotlib does not reliably fall back per glyph inside a mixed Chinese/
    English string. Chinese text is therefore assigned an explicit CJK font
    file, which avoids blank glyphs in exported PDF figures.
    """
    for text in fig.findobj(match=mpl.text.Text):
        if getattr(text, "_mixed_font_locked", False):
            continue
        raw = text.get_text() or ""
        text.set_fontproperties(CN_FONT_PROP if _contains_cjk(raw) else LATIN_FONT_PROP)
        if "\n" in raw:
            text.set_multialignment("center")
            text.set_linespacing(0.95)


def _unit(unit_text: str) -> str:
    """Return a plain unit run; the mixed-label renderer assigns Times New Roman."""
    return unit_text


def _decorate(ax: plt.Axes, letter: str) -> None:
    # Panel identifiers remain outside the plotting rectangle after Word scaling.
    ax.text(-0.16, 1.075, f"({letter})", transform=ax.transAxes, va="bottom", ha="left",
            fontsize=BASE_FONT_SIZE, fontweight="bold", fontproperties=LATIN_BOLD_PROP, clip_on=False)
    ax.grid(True, color="#E7E7E7", linewidth=0.55, alpha=0.75, zorder=0)
    ax.tick_params(direction="in", length=3.0, width=0.8)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontproperties(CN_FONT_PROP if _contains_cjk(label.get_text()) else LATIN_FONT_PROP)
        if "\n" in (label.get_text() or ""):
            label.set_multialignment("center")
            label.set_linespacing(0.95)


def _shift_xtick_labels(ax: plt.Axes, points: float = 2.2) -> None:
    """Move x tick labels horizontally without changing tick/data positions."""
    offset = ScaledTranslation(points / 72.0, 0.0, ax.figure.dpi_scale_trans)
    for label in ax.get_xticklabels():
        label.set_transform(label.get_transform() + offset)


def _wrap_labels(labels: tuple[str, ...] | list[str], width: int = 2) -> list[str]:
    """Wrap compact Chinese category names at a fixed character width."""
    return ["\n".join(text[i:i + width] for i in range(0, len(text), width))
            if len(text) > width else text for text in labels]


def _common_legend(fig: plt.Figure, y: float = 0.955) -> None:
    handles = [Line2D([0], [0], color=METHOD_COLORS[m], marker="o", lw=2, label=m) for m in METHODS]
    # Centre the legend over the complete three-column subplot grid.
    legend_x = (0.105 + 0.985) / 2.0
    fig.legend(handles=handles, labels=METHODS, loc="upper center", bbox_to_anchor=(legend_x, y),
               ncol=6, frameon=False, columnspacing=.75, handlelength=1.45,
               handletextpad=.28, borderaxespad=0,
               prop=FontProperties(fname=LATIN_FONT_PROP.get_file(), size=LEGEND_FONT_SIZE)
               if LATIN_FONT_PROP.get_file() else FontProperties(family=LATIN_FONT, size=LEGEND_FONT_SIZE))


def _save(fig: plt.Figure, name: str, legend: bool = True, legend_y: float = 0.955,
          wspace: float = 0.43, top: float | None = None) -> None:
    if legend:
        _common_legend(fig, legend_y)
    resolved_top = top if top is not None else (0.865 if legend else 0.945)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.105,
                        top=resolved_top, wspace=wspace, hspace=0.50)
    path = FIGURES / name
    _replace_mixed_axis_labels(fig)
    _apply_text_fonts(fig)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def _bootstrap_mean(values: np.ndarray, seed: int = 19) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(1600, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def _heatmap(ax: plt.Axes, values: np.ndarray, xlabels: list[str], ylabels: list[str],
             fmt: str = ".1f", diverging: bool = False, cbar_label: str = "",
             cbar_position: str = "right", x_rotation: float = 25.0,
             x_ha: str = "right") -> None:
    values = np.asarray(values, dtype=float)
    if diverging:
        lim = max(abs(np.nanmin(values)), abs(np.nanmax(values)), 1e-9)
        im = ax.imshow(values, cmap=DIV_CMAP, norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), aspect="auto")
    else:
        vmax = max(float(np.nanmax(values)), 1e-9)
        im = ax.imshow(values, cmap=RED_CMAP, vmin=0, vmax=vmax, aspect="auto")
    if values.size <= 80:
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=8.8,
                        fontproperties=LATIN_FONT_PROP,
                        color="#222222" if not diverging or abs(val) < .65 * max(abs(np.nanmin(values)), abs(np.nanmax(values)), 1e-9) else "white")
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=x_rotation, ha=x_ha)
    ax.set_yticks(range(len(ylabels)), ylabels)
    for label in ax.get_xticklabels():
        label.set_fontproperties(CN_FONT_PROP if _contains_cjk(label.get_text()) else LATIN_FONT_PROP)
    for label in ax.get_yticklabels():
        label.set_fontproperties(CN_FONT_PROP if _contains_cjk(label.get_text()) else LATIN_FONT_PROP)
        if "\n" in (label.get_text() or ""):
            label.set_multialignment("center")
            label.set_linespacing(0.95)
    if cbar_position == "top":
        # An inset colorbar does not shrink panel (b), so all six panels retain
        # exactly the same grid dimensions as Figures 3 and 4.
        cax = ax.inset_axes([0.06, 1.065, 0.88, 0.045], transform=ax.transAxes)
        cb = ax.figure.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.tick_params(labelsize=8.4, pad=1.0, direction="in")
        if cbar_label:
            cb.ax.set_title(cbar_label, fontsize=9.2, pad=2.0)
        for label in cb.ax.get_xticklabels():
            label.set_fontproperties(LATIN_FONT_PROP)
    else:
        cb = ax.figure.colorbar(im, ax=ax, fraction=.046, pad=.025)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=BASE_FONT_SIZE)
        cb.ax.tick_params(labelsize=TICK_FONT_SIZE)
        for label in cb.ax.get_yticklabels():
            label.set_fontproperties(LATIN_FONT_PROP)


def main_benchmark(records: pd.DataFrame) -> None:
    data = records[records.phase == "main"].copy()
    fig, axes = plt.subplots(2, 3, figsize=DOUBLE_COLUMN_SIZE)
    ax = axes[0, 0]
    means, lows, highs = zip(*[_bootstrap_mean(data[data.method == m].success.to_numpy(), 31 + i) for i, m in enumerate(METHODS)])
    x = np.arange(len(METHODS)); err = np.vstack((np.asarray(means)-np.asarray(lows), np.asarray(highs)-np.asarray(means))) * 100
    ax.bar(x, np.asarray(means)*100, color=[METHOD_COLORS[m] for m in METHODS], width=.72, zorder=2)
    ax.errorbar(x, np.asarray(means)*100, yerr=err, fmt="none", ecolor="#303030", capsize=3, lw=1.0, zorder=3)
    ax.set_ylabel("安全完成率 / " + "%"); ax.set_xticks(x, METHOD_TICKS, rotation=32, ha="right")
    ax.set_ylim(0, 108); _decorate(ax, "a"); _shift_xtick_labels(ax)

    ax = axes[0, 1]
    arrays = [data[data.method == m].clearance_q05.to_numpy() for m in METHODS]
    bp = ax.boxplot(arrays, positions=x, patch_artist=True, widths=.62, showfliers=False,
                    medianprops={"color":"#222222","lw":1.2})
    for patch, m in zip(bp["boxes"], METHODS): patch.set_facecolor(METHOD_COLORS[m]); patch.set_alpha(.72)
    ax.axhline(0, color="#444444", ls="--", lw=.9); ax.set_ylabel("5%" + "分位净间距 / " + _unit("m"))
    ax.set_xticks(x, METHOD_TICKS, rotation=32, ha="right"); _decorate(ax, "b"); _shift_xtick_labels(ax)

    ax = axes[0, 2]
    grouped = data.groupby("method")[["completion_time_s", "energy", "success"]].mean().reindex(METHODS)
    for m, row in grouped.iterrows():
        ax.scatter(row.completion_time_s, row.energy, s=50+110*row.success, color=METHOD_COLORS[m], edgecolor="white", lw=.8, zorder=3)
    ax.set_xlabel("完成时间 / " + _unit("s")); ax.set_ylabel("控制能耗 / " + "m²·s⁻³"); _decorate(ax, "c")

    ax = axes[1, 0]
    for m in METHODS:
        v = np.sort(data[data.method == m].goal_error_m.to_numpy()); y = np.arange(1, len(v)+1)/len(v)
        ax.step(v, y, where="post", color=METHOD_COLORS[m])
    ax.set_xscale("log"); ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5], ["0.02", "0.05", "0.10", "0.20", "0.50"])
    ax.set_xlabel("终端位置误差 / " + _unit("m")); ax.set_ylabel("经验累积分布"); ax.set_ylim(0,1.02); _decorate(ax, "d")

    ax = axes[1, 1]
    for i, m in enumerate(METHODS):
        v = data[data.method == m].solve_ms_p95.to_numpy(); mean, lo, hi = _bootstrap_mean(v, 60+i)
        jitter = np.linspace(-.08, .08, min(32, len(v)))
        ax.scatter(np.full(len(jitter), i)+jitter, np.sort(v)[:len(jitter)], s=9, color=METHOD_COLORS[m], alpha=.28)
        ax.errorbar(i, mean, yerr=[[mean-lo],[hi-mean]], fmt="o", color=METHOD_COLORS[m], capsize=4, ms=5, zorder=4)
    ax.axhline(180, color="#333333", ls="--", lw=1.0)
    ax.set_ylabel("单步" + "95%" + "分位求解时间 / " + _unit("ms")); ax.set_xticks(x, METHOD_TICKS, rotation=32, ha="right")
    _add_mixed_line_legend(ax, "控制周期 " + "180 ms", xy=(0.04, 0.96))
    _decorate(ax, "e"); _shift_xtick_labels(ax)

    ax = axes[1, 2]
    matrix = data.pivot_table(index="condition", columns="method", values="success", aggfunc="mean").reindex(index=CONDITION_ORDER, columns=METHODS).to_numpy()*100
    _heatmap(ax, matrix, list(METHOD_TICKS), _wrap_labels(CONDITION_ORDER), ".1f", False, "安全完成率 / " + "%")
    ax.set_xlabel("控制方法"); ax.set_ylabel("不确定性条件"); _decorate(ax, "f"); _shift_xtick_labels(ax)
    _save(fig, "图3_综合性能基准.pdf", legend_y=0.952)


def robustness(records: pd.DataFrame) -> None:
    data = records[records.phase == "main"].copy()
    fig, axes = plt.subplots(2, 3, figsize=DOUBLE_COLUMN_SIZE); x = np.arange(len(METHODS))
    ax = axes[0, 0]
    for m in METHODS:
        s = data[data.method == m].groupby("condition").success.mean().reindex(CONDITION_ORDER)*100
        ax.plot(range(4), s, marker="o", color=METHOD_COLORS[m])
    ax.set_xticks(range(4), CONDITION_ORDER, rotation=18); ax.set_ylabel("安全完成率 / " + "%"); ax.set_ylim(-3,105); _decorate(ax,"a")

    ax = axes[0, 1]
    dream = data[data.method == "DREAM-MPC"].copy()
    dream["combined_clearance"] = dream[["min_pair_clearance", "min_obstacle_clearance"]].min(axis=1)
    mat = dream.pivot_table(index="layout", columns="condition", values="combined_clearance", aggfunc="mean").reindex(index=LAYOUT_ORDER, columns=CONDITION_ORDER).fillna(0).to_numpy()
    _heatmap(ax, mat, list(CONDITION_ORDER), _wrap_labels(LAYOUT_ORDER), ".2f", False, "最小净距 / " + _unit("m"))
    ax.set_xlabel("不确定性条件"); ax.set_ylabel("场景布局"); _decorate(ax,"b"); _shift_xtick_labels(ax)

    ax = axes[0, 2]
    q = np.linspace(.02, .98, 49)
    for m in METHODS:
        vals = data[data.method == m].clearance_q05.to_numpy(); ax.plot(q*100, np.quantile(vals,q), color=METHOD_COLORS[m])
    ax.axhline(0,color="#333",ls="--",lw=.9); ax.set_xlabel("样本分位数 / " + "%"); ax.set_ylabel("净间距 / " + _unit("m"))
    ax.yaxis.set_label_position("right"); ax.yaxis.tick_right(); _decorate(ax,"c")

    ax = axes[1, 0]
    collision = data.groupby("method")[["collision_steps"]].mean().reindex(METHODS).collision_steps
    ax.barh(np.arange(6), collision, color=[METHOD_COLORS[m] for m in METHODS])
    ax.set_yticks(np.arange(6), METHOD_TICKS); ax.set_xlabel("平均碰撞离散步数 / 步"); ax.invert_yaxis(); _decorate(ax,"d")

    ax = axes[1, 1]
    base = data[data.condition=="标称"].groupby("method").success.mean().reindex(METHODS)
    adverse = data[data.condition!="标称"].groupby("method").success.mean().reindex(METHODS)
    delta=(adverse-base)*100
    ax.vlines(x, 0, delta, color=[METHOD_COLORS[m] for m in METHODS], lw=3); ax.scatter(x,delta,color=[METHOD_COLORS[m] for m in METHODS],zorder=3)
    ax.axhline(0,color="#333",lw=.8); ax.set_ylabel("相对标称性能变化 / 百分点"); ax.set_xticks(x,METHOD_TICKS,rotation=32,ha="right"); _decorate(ax,"e"); _shift_xtick_labels(ax)

    ax = axes[1, 2]
    for i,m in enumerate(METHODS):
        sub=data[data.method==m]; xx=sub.path_length_m.to_numpy(); yy=sub.energy.to_numpy()
        ax.scatter(xx,yy,s=10,color=METHOD_COLORS[m],alpha=.25)
        ax.scatter(xx.mean(),yy.mean(),s=65,color=METHOD_COLORS[m],edgecolor="white",lw=.8,zorder=4)
    ax.set_xlabel("累计路径长度 / " + _unit("m")); ax.set_ylabel("控制能耗 / " + "m²·s⁻³"); _decorate(ax,"f")
    _save(fig,"图4_不确定性鲁棒性.pdf", legend_y=0.952)


def mechanism(traces: dict[str, object]) -> None:
    trace = traces["DREAM-MPC-warehouse"]
    pos=np.asarray(trace["positions"]); vel=np.asarray(trace["velocities"]); cmd=np.asarray(trace["commands"]); solver=trace["solver"]
    fig,axes=plt.subplots(2,3,figsize=DOUBLE_COLUMN_SIZE); ax=axes[0,0]
    for obstacle in trace["obstacles"]: ax.add_patch(Circle(obstacle["center"],obstacle["radius"],facecolor="#D9D9D9",edgecolor="#666",lw=1))
    for i in range(pos.shape[1]):
        ax.plot(pos[:,i,0],pos[:,i,1],lw=1.5,color=plt.cm.tab10(i)); ax.scatter(pos[0,i,0],pos[0,i,1],marker="o",s=28,color=plt.cm.tab10(i)); ax.scatter(trace["goals"][i][0],trace["goals"][i][1],marker="*",s=55,color=plt.cm.tab10(i))
    ax.set_aspect("equal"); ax.set_xlabel("横向位置 / " + _unit("m")); ax.set_ylabel("纵向位置 / " + _unit("m")); _decorate(ax,"a")

    ax=axes[0,1]; top=np.zeros((len(solver),8))
    for t,row in enumerate(solver):
        vals=[z[1] for z in row["top_risk"]][:8]; top[t,:len(vals)]=vals
    _heatmap(ax,top.T,np.linspace(0,len(solver)*.18,6).round(1).astype(str).tolist(),
             [f"约束{i+1}" for i in range(8)],".3f",False,"风险份额",
             cbar_position="top", x_rotation=0, x_ha="center")
    ax.set_xticks(np.linspace(0,len(solver)-1,6),np.linspace(0,len(solver)*.18,6).round(1)); ax.set_xlabel("时间 / " + _unit("s")); _decorate(ax,"b")

    ax=axes[0,2]; t=np.arange(len(solver))*.18; unc=np.asarray([r["uncertainty_scale"] for r in solver]); active=np.asarray([r["active_constraints"] for r in solver])
    ax.plot(t,unc,color="#FF6666",label="尺度"); ax.set_xlabel("时间 / " + _unit("s")); ax.set_ylabel("尺度 / -", labelpad=1)
    twin=ax.twinx(); twin.plot(t,active,color="#3399FF",alpha=.75,label="约束数"); twin.set_ylabel("约束数 / 个", labelpad=2)
    for label in twin.get_yticklabels(): label.set_fontproperties(LATIN_FONT_PROP)
    h1,l1=ax.get_legend_handles_labels();h2,l2=twin.get_legend_handles_labels();ax.legend(h1+h2,l1+l2,frameon=False,loc="lower left",ncol=1); _decorate(ax,"c")

    ax=axes[1,0]; slack=np.sort(np.asarray([r["max_slack"] for r in solver])); qq=np.arange(1,len(slack)+1)/len(slack)
    ax.plot(slack,qq,color="#FF6666"); ax.set_xlim(left=0); ax.set_xlabel("最大安全松弛 / " + _unit("m")); ax.set_ylabel("经验累积分布"); _decorate(ax,"d")

    ax=axes[1,1]; interventions=np.asarray([r["interventions"] for r in solver]); ax.step(t,interventions,where="post",color="#FF6666")
    ax.fill_between(t,0,interventions,step="post",color="#FF6666",alpha=.18); ax.set_xlabel("时间 / " + _unit("s")); ax.set_ylabel("累计安全干预 / 次"); _decorate(ax,"e")

    ax=axes[1,2]; speed=np.linalg.norm(vel,axis=2).ravel(); effort=np.linalg.norm(cmd,axis=2).ravel(); time_color=np.repeat(np.arange(len(cmd))*.18,cmd.shape[1])
    sc=ax.scatter(speed,effort,c=time_color,cmap=RED_CMAP,s=9,alpha=.7); cb=fig.colorbar(sc,ax=ax,fraction=.046,pad=.025);cb.set_label("时间 / " + _unit("s"))
    ax.set_xlabel("机器人速度 / " + "m·s⁻¹");ax.set_ylabel("控制加速度 / " + "m·s⁻²");_decorate(ax,"f")
    _save(fig,"图5_闭环机制解析.pdf",legend=False,wspace=0.43,top=0.865)


def scalability_and_sensitivity(records: pd.DataFrame) -> None:
    scale=records[records.phase=="scalability"]; sens=records[records.phase=="sensitivity"]
    fig,axes=plt.subplots(2,3,figsize=DOUBLE_COLUMN_SIZE)
    specs=[("success", "安全完成率 / " + "%", 100),("min_pair_clearance", "最小机器人净距 / " + _unit("m"), 1),("solve_ms_p95", "95%" + "分位求解时间 / " + _unit("ms"), 1)]
    for k,(metric,label,factor) in enumerate(specs):
        ax=axes[0,k]
        for m in METHODS:
            sub=scale[scale.method==m]; groups=sub.groupby("n_agents")[metric]
            xx=np.array(sorted(sub.n_agents.unique())); means=[];lo=[];hi=[]
            for n in xx:
                a,b,c=_bootstrap_mean(groups.get_group(n).to_numpy(),n+17);means.append(a*factor);lo.append(b*factor);hi.append(c*factor)
            ax.plot(xx,means,marker="o",color=METHOD_COLORS[m]); ax.fill_between(xx,lo,hi,color=METHOD_COLORS[m],alpha=.10)
        ax.set_xlabel("机器人数量 / 个"); ax.set_ylabel(label);_decorate(ax,chr(ord('a')+k))

    ax=axes[1,0]
    budgets=sorted(sens.risk_budget.unique()); means=[];lo=[];hi=[]
    for b in budgets:
        a,c,d=_bootstrap_mean(sens[sens.risk_budget==b].min_pair_clearance.to_numpy(),int(b*1000));means.append(a);lo.append(c);hi.append(d)
    ax.errorbar(budgets,means,yerr=[np.asarray(means)-lo,np.asarray(hi)-means],color="#FF6666",marker="o",capsize=4)
    ax.set_xlabel("联合风险预算 / -");ax.set_ylabel("最小机器人净距 / " + _unit("m"));_decorate(ax,"d")

    ax=axes[1,1]
    s=sens.groupby("risk_budget")[["completion_time_s","energy","success"]].mean()
    ax.scatter(s.completion_time_s,s.energy,c=s.index,cmap=RED_CMAP,s=70,edgecolor="#555",lw=.6)
    label_offsets = {0.10: (3, 3), 0.14: (-2, -16), 0.18: (-18, -18),
                     0.22: (-18, 9), 0.26: (3, 3)}
    for b,row in s.iterrows():
        dx, dy = label_offsets[round(float(b), 2)]
        ax.annotate(f"{b:.2f}",(row.completion_time_s,row.energy),xytext=(dx,dy),
                    textcoords="offset points",fontsize=ANNOTATION_FONT_SIZE,
                    fontproperties=LATIN_FONT_PROP,ha="right" if dx < 0 else "left")
    ax.set_ylim(41.25, 48.0)
    ax.set_xlabel("完成时间 / " + _unit("s"));ax.set_ylabel("控制能耗 / " + "m²·s⁻³");_decorate(ax,"e")

    ax=axes[1,2]
    entropy=sens.groupby("risk_budget").risk_entropy.agg(["mean","std"])
    ax.plot(entropy.index,entropy["mean"],color="#FF6666",marker="o");ax.fill_between(entropy.index,entropy["mean"]-entropy["std"],entropy["mean"]+entropy["std"],color="#FF6666",alpha=.18)
    ax.set_xlabel("联合风险预算 / -");ax.set_ylabel("归一化风险分配熵 / -");_decorate(ax,"f")
    _save(fig,"图6_规模与参数敏感性.pdf",wspace=0.60)


def mujoco_validation(records: pd.DataFrame, traces: dict[str, object]) -> None:
    fig,axes=plt.subplots(2,3,figsize=DOUBLE_COLUMN_SIZE); x=np.arange(len(METHODS))
    ax=axes[0,0]
    preferred_conditions = ["刚体标称", "低附着", "载荷失配", "复合失配"]
    conditions = [name for name in preferred_conditions if name in set(records.condition)]
    conditions += [name for name in records.condition.unique() if name not in conditions]
    for m in METHODS:
        vals=records[records.method==m].groupby("condition").success.mean().reindex(conditions)*100
        ax.plot(range(len(conditions)),vals,marker="o",color=METHOD_COLORS[m])
    ax.set_xticks(range(len(conditions)),conditions,rotation=24,ha="right");ax.set_ylabel("刚体安全完成率 / " + "%");_decorate(ax,"a")

    ax=axes[0,1]
    mat=(records.pivot_table(index="condition", columns="method", values="contact_episodes", aggfunc="mean")
         .reindex(index=conditions, columns=METHODS).to_numpy())
    display_conditions = {
        "刚体标称": "刚体\n标称",
        "低附着": "低附\n着",
        "载荷失配": "载荷\n失配",
        "复合失配": "复合\n失配",
    }
    ylabels = [display_conditions.get(name, "\n".join(name[i:i+2] for i in range(0, len(name), 2))) for name in conditions]
    _heatmap(ax, mat, list(METHOD_TICKS), ylabels, ".1f", False, "",
             x_rotation=38, x_ha="right")
    for label in ax.get_yticklabels():
        label.set_multialignment("center")
        label.set_linespacing(0.95)
        label.set_fontproperties(CN_FONT_PROP)
    ax.set_xlabel("控制方法");ax.set_ylabel("物理失配条件", labelpad=2);_decorate(ax,"b");_shift_xtick_labels(ax)

    ax=axes[0,2]
    for i,m in enumerate(METHODS):
        v=records[records.method==m].peak_contact_force_N.to_numpy();
        ax.scatter(np.full(len(v),i)+np.linspace(-.07,.07,len(v)),v+1,s=10,color=METHOD_COLORS[m],alpha=.25)
        ax.scatter(i,np.mean(v)+1,s=55,color=METHOD_COLORS[m],edgecolor="white",zorder=4)
    ax.set_yscale("log");ax.set_ylabel("峰值接触力" + "+1" + " / " + _unit("N"));ax.set_xticks(x,METHOD_TICKS,rotation=32,ha="right")
    ax.yaxis.set_label_position("right"); ax.yaxis.tick_right(); _decorate(ax,"c"); _shift_xtick_labels(ax)

    ax=axes[1,0]
    for m in METHODS:
        sub=records[records.method==m];ax.scatter(sub.tracking_rmse,sub.lateral_slip_mps,s=11,color=METHOD_COLORS[m],alpha=.24)
        ax.scatter(sub.tracking_rmse.mean(),sub.lateral_slip_mps.mean(),s=60,color=METHOD_COLORS[m],edgecolor="white",zorder=4)
    ax.set_xlabel("加速度跟踪均方根误差 / " + "m·s⁻²");ax.set_ylabel("横向滑移速度 / " + "m·s⁻¹");_decorate(ax,"d")

    ax=axes[1,1]; trace=traces["DREAM-MPC-mujoco"]; pos=np.asarray(trace["positions"])
    for obstacle in trace["obstacles"]:ax.add_patch(Circle(obstacle["center"],obstacle["radius"],facecolor="#D9D9D9",edgecolor="#666"))
    for i in range(pos.shape[1]):ax.plot(pos[:,i,0],pos[:,i,1],lw=1.5,color=plt.cm.tab10(i));ax.scatter(pos[0,i,0],pos[0,i,1],s=25,color=plt.cm.tab10(i));ax.scatter(trace["goals"][i][0],trace["goals"][i][1],s=55,marker="*",color=plt.cm.tab10(i))
    ax.set_aspect("equal");ax.set_xlabel("横向位置 / " + _unit("m"));ax.set_ylabel("纵向位置 / " + _unit("m"));_decorate(ax,"e")

    ax=axes[1,2]
    g=records.groupby("method")[["completion_time_s","energy_J","success"]].mean().reindex(METHODS)
    for m,row in g.iterrows():ax.scatter(row.completion_time_s,row.energy_J,s=45+100*row.success,color=METHOD_COLORS[m],edgecolor="white",lw=.7)
    ax.set_xlabel("完成时间 / " + _unit("s"));ax.set_ylabel("机械能耗 / " + _unit("J"));_decorate(ax,"f")
    _save(fig,"图7_MuJoCo刚体验证.pdf", legend_y=0.952,wspace=0.43)


def mujoco_snapshot_montage() -> None:
    snapshot_path = MODELS / "mujoco_snapshots.npz"
    metadata_path = MODELS / "mujoco_snapshots.json"
    if not snapshot_path.exists() or not metadata_path.exists():
        return
    frames = np.load(snapshot_path)
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    fig, axes = plt.subplots(2, 3, figsize=(7.20, 4.45))
    for index, (ax, method) in enumerate(zip(axes.ravel(), METHODS)):
        ax.imshow(frames[method])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(method, pad=4, color=METHOD_COLORS[method],
                     fontproperties=LATIN_BOLD_PROP)
        row = metadata["methods"][method]
        if row.get("contact_episodes", 0) > 0:
            note = (f"接触事件 {int(row['contact_episodes'])} 次；持续 "
                    f"{row['contact_duration_s']:.2f} s")
        else:
            clearance = min(row["min_pair_clearance"], row["min_obstacle_clearance"])
            note = f"零接触；最小净距 {clearance:.3f} m"
        _add_mixed_note(ax, note, xy=(.02, .035),
                        edgecolor=METHOD_COLORS[method], fontsize=9.5)
        ax.text(-.035, 1.035, f"({chr(ord('a') + index)})", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=BASE_FONT_SIZE, fontweight="bold",
                fontproperties=LATIN_BOLD_PROP, clip_on=False)
        for spine in ax.spines.values():
            spine.set_color(METHOD_COLORS[method]); spine.set_linewidth(1.1)
    fig.subplots_adjust(left=.025, right=.992, bottom=.025, top=.935, wspace=.035, hspace=.17)
    _apply_text_fonts(fig)
    fig.savefig(FIGURES / "图8_MuJoCo场景快照.pdf", format="pdf",
                bbox_inches="tight", pad_inches=.025)
    plt.close(fig)


def generate_all() -> None:
    _style(); _print_font_diagnostics(); FIGURES.mkdir(parents=True,exist_ok=True)
    records=pd.read_json(MODELS/"episode_records.jsonl",lines=True)
    with (MODELS/"representative_traces.json").open(encoding="utf-8") as f: traces=json.load(f)
    main_benchmark(records); robustness(records); mechanism(traces); scalability_and_sensitivity(records)
    mujoco_path=MODELS/"mujoco_records.jsonl"
    if mujoco_path.exists():
        mj=pd.read_json(mujoco_path,lines=True)
        with (MODELS/"mujoco_traces.json").open(encoding="utf-8") as f:mj_traces=json.load(f)
        mujoco_validation(mj,mj_traces)
        mujoco_snapshot_montage()


if __name__ == "__main__":
    generate_all()
