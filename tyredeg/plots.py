"""Figures for the README. F1 compound colours, plus marker shapes and
direct labels so compounds are readable without relying on colour alone."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .model import DegModel, detrend_for_plot

STYLE = {
    "SOFT":   {"color": "#D62828", "marker": "o"},
    "MEDIUM": {"color": "#C99700", "marker": "s"},
    "HARD":   {"color": "#6E6E6E", "marker": "^"},
}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=MUTED); ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)


def plot_degradation(df, model: DegModel, title: str, path: str):
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    y = detrend_for_plot(df, model)
    for c in model.compounds:
        m = df["Compound"] == c
        st = STYLE[c]
        ax.scatter(df.loc[m, "TyreLife"], y[m], s=10, alpha=0.25, color=st["color"],
                   marker=st["marker"], linewidths=0)
        ages = np.arange(1, df.loc[m, "TyreLife"].max() + 1)
        line = model.lap_time(c, ages)
        ax.plot(ages, line, color=st["color"], linewidth=2.2)
        ax.annotate(f"{c.title()}  {model.deg[c]:+.3f} s/lap", (ages[-1], line[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    lo, hi = np.percentile(y, [1, 99])
    ax.set_ylim(lo - 0.5, hi + 0.5)
    ax.set_xlim(0, df["TyreLife"].max() * 1.28)
    _style(ax, title, "Tyre age (laps)", "Fuel-corrected lap time (s), driver pace removed")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_kalman(laps, pace, deg, deg_std, batch_deg, label: str, compound: str, path: str):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.5), dpi=150, sharex=True)
    c = STYLE.get(compound, STYLE["MEDIUM"])["color"]
    x = laps["TyreLife"].to_numpy()
    a1.scatter(x, laps["FuelCorrected_s"], s=18, color=c, label="Lap time (fuel-corrected)")
    a1.plot(x, pace, color=INK, linewidth=2, label="Kalman pace estimate")
    a1.legend(frameon=False, fontsize=9, loc="upper left")
    _style(a1, f"Live degradation estimate - {label}", "", "Lap time (s)")

    a2.fill_between(x, deg - 1.96 * deg_std, deg + 1.96 * deg_std, color=c, alpha=0.2,
                    linewidth=0, label="95% interval")
    a2.plot(x, deg, color=c, linewidth=2, label="Kalman deg estimate (laps so far only)")
    a2.axhline(batch_deg, color=MUTED, linestyle="--", linewidth=1.5,
               label=f"Whole-race fit for {compound.title()}: {batch_deg:+.3f} s/lap")
    a2.legend(frameon=False, fontsize=9, loc="upper right")
    _style(a2, "", "Tyre age (laps)", "Degradation (s/lap)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_strategy(strats, best, total_laps: int, title: str, path: str):
    """Race time vs pit lap for each 1-stop compound pair."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    one = strats[strats["stops"] == 1]
    t0 = strats["race_time_s"].min()
    pairs = best[best["stops"] == 1]["compounds"].head(3)
    dashes = ["-", "--", ":"]
    for k, seq in enumerate(pairs):
        s = one[one["compounds"] == seq].sort_values("pit_laps")
        pit = s["pit_laps"].map(lambda p: p[0])
        parts = seq.split("-")
        softest = min(parts, key=["SOFT", "MEDIUM", "HARD"].index)
        col = STYLE[softest]["color"]
        ax.plot(pit, s["race_time_s"] - t0, linewidth=2, color=col, linestyle=dashes[k])
        i = s["race_time_s"].idxmin()
        ax.scatter([s.at[i, "pit_laps"][0]], [s.at[i, "race_time_s"] - t0], s=40,
                   color=col, marker=STYLE[softest]["marker"], zorder=3)
        ax.annotate(f"{' → '.join(p.title() for p in parts)}  (best: pit lap {s.at[i, 'pit_laps'][0]}, "
                    f"+{s.at[i, 'race_time_s'] - t0:.1f} s)",
                    (pit.iloc[-1], (s["race_time_s"] - t0).iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    two = best[best["stops"] == 2].head(1)
    if len(two):
        g = two["race_time_s"].iloc[0] - t0
        ax.axhline(g, color=MUTED, linestyle=":", linewidth=1.5)
        ax.annotate(f"Best 2-stop: {two['plan'].iloc[0]}", (1, g), xytext=(0, 4),
                    textcoords="offset points", fontsize=9, color=MUTED)
    ax.set_ylim(-1, min(40, ax.get_ylim()[1]))
    ax.set_xlim(0, total_laps * 1.45)
    _style(ax, title, "Pit lap (1-stop)", "Race time vs fastest strategy (s)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_strategy_ranking(best, max_age: dict, title: str, path: str, top: int = 6):
    """Horizontal bars: time lost vs the fastest plan, one bar per tyre combination."""
    b = best.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(b) + 1.8), dpi=150)
    gaps = b["gap_to_best_s"].to_numpy()
    y = np.arange(len(b))
    colors = [STYLE[min(c.split("-"), key=["SOFT", "MEDIUM", "HARD"].index)]["color"]
              for c in b["compounds"]]
    ax.barh(y, np.maximum(gaps, 0.15), height=0.55, color=colors)
    for yi, g, plan, st in zip(y, gaps, b["plan"], b["stops"]):
        ax.annotate(f"{plan}   ({st}-stop)  " + ("fastest" if g < 0.05 else f"+{g:.1f} s"),
                    (max(g, 0.15), yi), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_yticks([])
    ax.set_xlim(0, max(gaps.max(), 1) * 2.2)
    note = "Stint lengths capped at the oldest tyre seen: " + ", ".join(
        f"{c.title()} {n} laps" for c, n in max_age.items())
    _style(ax, title, "Time lost vs fastest plan (s)   ·   S = Soft, M = Medium, H = Hard", "")
    ax.grid(axis="y", visible=False)
    fig.text(0.01, 0.015, note, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(path); plt.close(fig)
