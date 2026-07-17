"""2 rows (Anthropic / Qwen3.5) x 2 columns (P(win) overall / P(build) overall)
summary figure for the Lomekwi dense sweep.

Same family/point filtering and ordinal-x convention as plot_lomekwi_decomposition
(Anthropic + Qwen3.5 only, Qwen3.5 2B dropped, x = within-family capability
rank) so the two figures read the same way.

Usage: PYTHONPATH=. python -m scripts.lomekwi.plotters.plot_lomekwi_summary [dir ...]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.lomekwi.plotters.plot_lomekwi_decomposition import (
    DENSE_DIR, FIGS_DIR, FAMILY_STYLE, ROW_FAMILIES, PANELS, discover,
    filtered_families, wilson)

COLUMN_TITLES = ["P(win) overall", "P(build) overall"]


def _draw_panel(ax, fam, members, title, denom, outcome, row_i, col_i):
    style = FAMILY_STYLE.get(fam, dict(color="gray", marker="^"))
    xs, ys, los, his, labels = [], [], [], [], []
    for rank, (size, label, facts) in enumerate(members, start=1):
        sel = [f for f in facts if denom(f)]
        k = int(sum(outcome(f) for f in sel)); n = len(sel)
        p = k / n if n else float("nan")
        lo, hi = wilson(k, n)
        xs.append(rank); ys.append(p)
        los.append(p - lo); his.append(hi - p); labels.append(label)
        print(f"  {fam:<10} {label:<11} {title[:12]}: {k:>3}/{n:<3} = {p:.3f}")
    # small-capped whiskers: readable but lighter than the line so CIs don't shout
    ax.errorbar(xs, ys, yerr=[los, his], capsize=3, capthick=1.0, elinewidth=1.2,
                ecolor=style["color"], alpha=0.6, zorder=1, lw=0, marker="none")
    ax.plot(xs, ys, lw=2, markersize=9, markeredgecolor="black",
            markeredgewidth=0.6, zorder=2, **style)
    for x, y, lab in zip(xs, ys, labels):
        if not math.isnan(y):
            if title == "P(build) overall" and lab == "Sonnet 4.6":
                # local minimum -- both neighboring lines converge from above,
                # so drop the label below instead.
                offset, ha, va = (0, -11), "center", "top"
            else:
                offset, ha, va = (0, 9), "center", "bottom"
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=offset,
                        ha=ha, va=va, fontsize=8, color=style["color"])
    if row_i == 0:
        ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(0.5, len(members) + 0.5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(1, len(members) + 1))
    ax.set_xticklabels([])
    if row_i == len(ROW_FAMILIES) - 1:
        ax.set_xlabel("Model size (ordinal)", fontweight="bold")
    if col_i == 0:
        ax.set_ylabel("Probability", fontweight="bold")
        ax.text(-0.32, 0.5, fam, transform=ax.transAxes, rotation=90,
                va="center", ha="center", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, which="major")


def main():
    args = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
    dirs = args if args else sorted(p for p in DENSE_DIR.iterdir() if p.is_dir())
    fams = filtered_families(discover(dirs))
    if not fams:
        raise SystemExit(f"no classifiable Anthropic/Qwen3.5 run dirs found under {DENSE_DIR}")

    # PANELS[3] = P(won) overall. Build an unconditional-built panel: denom=all, outcome=built.
    _, won_denom, won_outcome = PANELS[3]
    built_denom = lambda f: True
    built_outcome = lambda f: 1.0 if f["built"] else 0.0

    n_rows = len(ROW_FAMILIES)
    fig, axes = plt.subplots(n_rows, 2, figsize=(6.5, 2.8 * n_rows), squeeze=False,
                              constrained_layout=True)

    for row_i, fam in enumerate(ROW_FAMILIES):
        members = fams.get(fam)
        if not members:
            continue
        _draw_panel(axes[row_i][0], fam, members, COLUMN_TITLES[0], won_denom, won_outcome, row_i, 0)
        _draw_panel(axes[row_i][1], fam, members, COLUMN_TITLES[1], built_denom, built_outcome, row_i, 1)

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGS_DIR / "fig_grid_dense_summary_lines.png"
    fig.savefig(out, dpi=130); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out}")
    print(f"  wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
