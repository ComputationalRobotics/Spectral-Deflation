"""Shared combined-figure layout for the per-tier ablation curve figures,
matching the main-benchmark figure paradigm: four columns side by side
(validation loss vs iterations | vs time, then training loss vs iterations |
vs time), x-axis starting partway into the run, a per-tier auto-tightened
y-window (top = the worst displayed value at the crop point, so the panel has
no dead space), an optional compressed top strip for an outlier arm,
and a compact white-backed legend.

split_figure() is the standalone-figure companion: ONE loss type (validation
or training) with the arms split by polar method -- Newton Schulz (plain +
deflated) in the left column pair, Polar Express in the right, four panels
total -- so a tier's deflation effect reads once per method instead of four
curves in one axes.  family_figures() draws the same panels as SEPARATE
figures, one per polar method (the layout the paper uses).  Same convention as the main-benchmark standalone
figures (plot_main_bench.fig_val / fig_train); keep the two in sync.

Series format: one dict per arm,
    {label, color, ls, val_steps, val_losses, val_times,
     train_losses, train_times}
plus, for split_figure, {family} -- the polar method the arm shares with one
other arm.  *_times are the per-eval cumulative times / per-step cumulative
times.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def smooth(y, w):
    if w <= 1 or len(y) < w:
        return np.asarray(y, float)
    return np.convolve(np.asarray(y, float), np.ones(w) / w, mode="valid")


def _value_at(x, y, x0):
    i = int(np.searchsorted(np.asarray(x, float), x0))
    return float(np.asarray(y, float)[min(i, len(y) - 1)])


def combined_figure(series, *, time_label="training time (s)", strip=None,
                    xmin_iter=None, figwidth=16, aspect=0.55,
                    val_interval=100, train_smooth=100, tag=None, pad=0.015):
    """-> matplotlib Figure.  strip=(lo, hi): compressed top band holding any
    arm whose whole displayed curve sits above the main band."""
    it_max = max(len(s["train_losses"]) for s in series)
    t_max = max(float(np.asarray(s["train_times"], float)[-1]) for s in series)
    x0 = xmin_iter or 0
    in_band = [s for s in series
               if strip is None or min(s["val_losses"]) < strip[0]]
    lo = min(min(np.min(s["val_losses"]),
                 np.min(smooth(s["train_losses"], train_smooth)))
             for s in in_band) - 0.02
    band_top = max(max(_value_at(s["val_steps"], s["val_losses"], x0),
                       _value_at(np.arange(1, len(s["train_losses"]) + 1),
                                 smooth(s["train_losses"], train_smooth), x0))
                   for s in in_band) + pad

    panel_w = figwidth / 4
    bot_h = panel_w * aspect
    if strip:
        fig = plt.figure(figsize=(figwidth, bot_h * 5.5 / 4.5 + 0.85))
        gs = fig.add_gridspec(2, 4, height_ratios=[1, 4.5],
                              hspace=0.07, wspace=0.10)
    else:
        fig = plt.figure(figsize=(figwidth, bot_h + 0.85))
        gs = fig.add_gridspec(1, 4, wspace=0.10)
    cols = []
    for c, xlab in enumerate(["iterations", time_label,
                              "iterations", time_label]):
        if strip:
            top = fig.add_subplot(gs[0, c])
            bot = fig.add_subplot(gs[1, c], sharex=top)
            top.set_ylim(strip[0], strip[1])
            bot.set_ylim(lo, band_top)
            top.tick_params(labelbottom=False, bottom=False)
            top.spines["bottom"].set_visible(False)
            top.yaxis.set_major_locator(MaxNLocator(3))
            d = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=7,
                     linestyle="none", color="k", mec="k", mew=0.7,
                     clip_on=False)
            top.plot([0], [0], transform=top.transAxes, **d)
            bot.plot([0], [1], transform=bot.transAxes, **d)
            axes = [bot, top]
        else:
            bot = fig.add_subplot(gs[0, c])
            bot.set_ylim(lo, band_top)
            axes = [bot]
        bot.set_xlabel(xlab)
        bot.xaxis.set_major_locator(MaxNLocator(4))
        for a in axes:
            a.grid(True)
            if c > 0:                      # shared y-range: label col 0 only
                a.tick_params(labelleft=False)
        xmax = it_max if c in (0, 2) else t_max
        bot.set_xlim((x0 / it_max) * xmax, 1.02 * xmax)
        cols.append(axes)
    for x, txt in ((0.30, "validation loss"), (0.72, "training loss")):
        fig.text(x, 0.985, txt, ha="center", va="top", fontsize=10)
    if tag:
        fig.text(0.008, 0.985, tag, ha="left", va="top", fontsize=9)
    cols[0][0].set_ylabel("loss")

    off = train_smooth - 1
    for s in series:
        vs = np.asarray(s["val_steps"], float)
        vv = np.asarray(s["val_losses"], float)
        vt = np.asarray(s["val_times"], float)
        m = (vs % val_interval) == 0
        for a in cols[0]:
            a.plot(vs[m], vv[m], color=s["color"], linestyle=s["ls"],
                   label=s["label"])
        for a in cols[1]:
            a.plot(vt[m], vv[m], color=s["color"], linestyle=s["ls"])
        y = np.asarray(s["train_losses"], float)
        it = np.arange(1, len(y) + 1)
        tt = np.asarray(s["train_times"], float)
        ys = smooth(y, train_smooth)
        for axes, x in ((cols[2], it), (cols[3], tt)):
            for a in axes:
                a.plot(x, y, color=s["color"], linewidth=0.4, alpha=0.10)
                a.plot(x[off:], ys, color=s["color"], linestyle=s["ls"])
    cols[0][0].legend(loc="lower left", frameon=True, framealpha=0.85,
                      edgecolor="none", facecolor="white", fontsize=7,
                      handlelength=1.7, labelspacing=0.25, borderpad=0.3,
                      borderaxespad=0.25)
    # keep an overflowing legend above the neighbouring panel's background
    cols[0][0].set_zorder(cols[1][0].get_zorder() + 1)
    return fig


def short_label(label, family):
    """Legend label with the family suffix dropped -- in a split figure the
    column-pair header already names it ("Deflated Muon (Polar Express)" ->
    "Deflated Muon")."""
    return label.replace(f" ({family})", "")


# file-name slug per polar-method family, for the per-family figures
FAMILY_SLUG = {"Newton Schulz": "ns", "Polar Express": "pe"}


def family_figures(series, kind, *, family_order=None, figwidth=16,
                   lead_panel=True, **kw):
    """-> [(slug, Figure), ...]: ONE standalone figure per polar-method
    family -- Newton Schulz (plain + deflated) and Polar Express in separate
    files rather than side by side.  Each is split_figure() restricted to
    one family, at figwidth/4 per panel so the panels keep the size they
    have in the four-panel layouts.  With lead_panel (default) the
    validation figure is three panels -- training loss vs iterations, then
    validation loss vs iterations | vs time; the training figure is the
    two-panel pair.  Every panel carries its own y tick labels and y-axis
    label."""
    arms = [s for s in series if s.get("family")]
    order = family_order or list(dict.fromkeys(s["family"] for s in arms))
    lead_panel = bool(lead_panel) and kind == "val"
    npanel = 3 if lead_panel else 2
    out = []
    for fam in order:
        fig = split_figure(series, kind, family_order=[fam],
                           figwidth=figwidth * npanel / 4,
                           lead_panel=lead_panel, **kw)
        if fig is not None:
            out.append((FAMILY_SLUG.get(fam, fam.lower().replace(" ", "_")),
                        fig))
    return out


def split_figure(series, kind, *, family_order=None,
                 time_label="training time (s)", xmin_iter=None,
                 figwidth=16, aspect=0.55, val_interval=100, train_smooth=100,
                 tag=None, pad=0.015, lead_panel=False, ytop=None):
    """-> matplotlib Figure: ONE loss type, arms split by polar method.

    kind is "val" or "train".  One (iterations | time) column pair per
    family, in family_order (default: order of first appearance in series),
    with a narrow empty grid column between the pairs.  Arms carrying no
    family are dropped -- a split by polar method has no column for an arm
    that is not a polar method, which is also why the y-axis here is
    always continuous (no compressed strip).  The y-window follows the same
    auto-tightening as combined_figure, but over this loss type only, so the
    panel has no dead space.

    ytop: fixed top of every panel's y-window instead of the auto-tightened
    one (the bottom stays auto); curves then enter each panel from the top.

    lead_panel: prepend a panel of the OTHER loss type vs iterations to
    each family's pair, so a family reads left to right as training loss |
    validation loss | validation loss vs time (kind "val"), or validation
    loss | training loss | training loss vs time (kind "train").  The lead
    panel has its own auto-tightened y-window and y-axis; the pair shares
    theirs.
    """
    arms = [s for s in series if s.get("family")]
    if not arms:
        return None
    order = family_order or list(dict.fromkeys(s["family"] for s in arms))
    families = [(f, [s for s in arms if s["family"] == f]) for f in order]
    families = [(f, ss) for f, ss in families if ss]
    if not families:
        return None
    other = "train" if kind == "val" else "val"

    it_max = max(len(s["train_losses"]) for s in arms)
    t_max = max(float(np.asarray(s["train_times"], float)[-1]) for s in arms)
    x0 = xmin_iter or 0
    off = train_smooth - 1
    sm = {id(s): smooth(s["train_losses"], train_smooth) for s in arms}
    # y-windows: bottom = the best value anywhere, top = the worst value still
    # displayed at the crop point (same rule as combined_figure), one window
    # per loss type that appears in the figure
    ylim = {}
    if kind == "val" or lead_panel:
        ylim["val"] = (min(float(np.min(s["val_losses"])) for s in arms) - 0.02,
                       max(_value_at(s["val_steps"], s["val_losses"], x0)
                           for s in arms) + pad)
    if kind == "train" or lead_panel:
        ylim["train"] = (min(float(np.min(sm[id(s)])) for s in arms) - 0.02,
                         max(_value_at(np.arange(1, len(s["train_losses"]) + 1),
                                       sm[id(s)], x0) for s in arms) + pad)
    if ytop is not None:
        ylim = {k: (lo, ytop) for k, (lo, _) in ylim.items()}
    ylabel = {"val": "validation loss", "train": "training loss"}

    # panels per family: (loss type, x kind); the family's own y-axis is
    # drawn on the first panel of each loss type
    panels = [(kind, "it"), (kind, "time")]
    if lead_panel:
        panels = [(other, "it")] + panels
    # grid columns: the panels of a family, then a narrow empty spacer column
    # before the next family, widening the between-family gap past the
    # within-family wspace
    widths, slots = [], []
    for k in range(len(families)):
        if k:
            widths.append(0.30)
        cols = []
        for _ in panels:
            cols.append(len(widths))
            widths.append(1.0)
        slots.append(cols)
    npanel = len(panels) * len(families)
    single = len(families) == 1
    # a single-family figure (the per-family paper figures) uses constrained
    # layout, so the room for each panel's own y tick labels and y label
    # follows the actual font size at any figure size -- a fixed wspace only
    # fits one size.  Multi-family figures keep the fixed grid because their
    # headers are placed from the axes boxes below.
    fig = plt.figure(figsize=(figwidth, figwidth / npanel * aspect + 0.85),
                     layout="constrained" if single else None)
    if single:
        fig.get_layout_engine().set(wspace=0.06, w_pad=0.04, h_pad=0.04)
        gs = fig.add_gridspec(1, len(widths), width_ratios=widths)
    else:
        gs = fig.add_gridspec(1, len(widths), width_ratios=widths,
                              wspace=0.28)
    # fewer x ticks on narrow panels so the labels cannot collide
    xbins = 4 if figwidth / npanel >= 2.5 else 3
    axes = []            # [family][panel]
    for k, (fam, _) in enumerate(families):
        row = []
        for (ltype, xkind), c in zip(panels, slots[k]):
            ax = fig.add_subplot(gs[0, c])
            ax.set_ylim(*ylim[ltype])
            ax.set_xlabel("iterations" if xkind == "it" else time_label)
            ax.xaxis.set_major_locator(MaxNLocator(xbins))
            ax.grid(True)
            # every panel carries its own y tick labels and y-axis label
            # (the ranges are shared per loss type, the labels are not)
            ax.set_ylabel(ylabel[ltype])
            xmax = it_max if xkind == "it" else t_max
            ax.set_xlim((x0 / it_max) * xmax, 1.02 * xmax)
            row.append((ltype, xkind, ax))
        axes.append(row)
    # family header: a suptitle when there is one family (constrained layout
    # places it); otherwise centred over each family's panels, from the axes
    # boxes themselves (a fixed figure-x guess drifts once the ylabel strip
    # widens the left margin)
    if single:
        fig.suptitle(families[0][0], fontsize=10)
    else:
        for k, (fam, _) in enumerate(families):
            b0 = axes[k][0][2].get_position()
            b1 = axes[k][-1][2].get_position()
            fig.text((b0.x0 + b1.x1) / 2, b0.y1 + 0.045, fam,
                     ha="center", va="bottom", fontsize=10)
    if tag:
        fig.text(0.008, 0.985, tag, ha="left", va="top", fontsize=9)

    for k, (fam, ss) in enumerate(families):
        for s in ss:
            lab = short_label(s["label"], fam)
            vs = np.asarray(s["val_steps"], float)
            vv = np.asarray(s["val_losses"], float)
            vt = np.asarray(s["val_times"], float)
            m = (vs % val_interval) == 0
            y = np.asarray(s["train_losses"], float)
            it = np.arange(1, len(y) + 1)
            tt = np.asarray(s["train_times"], float)
            for ltype, xkind, ax in axes[k]:
                if ltype == "val":
                    ax.plot(vs[m] if xkind == "it" else vt[m], vv[m],
                            color=s["color"], linestyle=s["ls"], label=lab)
                else:
                    x = it if xkind == "it" else tt
                    # raw series faintly underneath, smoothed on top
                    ax.plot(x, y, color=s["color"], linewidth=0.4, alpha=0.10)
                    ax.plot(x[off:], sm[id(s)], color=s["color"],
                            linestyle=s["ls"], label=lab)
        # one legend per family, on its first panel
        ax_0 = axes[k][0][2]
        ax_0.legend(loc="lower left", frameon=True, framealpha=0.85,
                    edgecolor="none", facecolor="white", fontsize=7,
                    handlelength=1.7, labelspacing=0.25, borderpad=0.3,
                    borderaxespad=0.25)
        # the legend can run past the panel's right edge; draw this axes
        # after its neighbour so the neighbour's background does not paint
        # over the legend text
        ax_0.set_zorder(axes[k][1][2].get_zorder() + 1)
    return fig
