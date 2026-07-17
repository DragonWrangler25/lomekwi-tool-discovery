"""Capability-axis metric panels for the Deep Lomekwi (two-layer recipe), 3 Claude models.

C-R-E decomposition + solve, one panel per metric, models as a connected series ordered by
capability (Haiku < Sonnet < Opus). Recognition is split into its two recipe subparts:

  Curiosity           C  = P(held the base ingredients)              [held_base]
  Recognition step 1  R1 = P(built part | held base)                [step-1 discovery]
  Recognition step 2  R2 = P(built axe  | built part)               [step-2 follow-through]
  Efficiency          E  = P(solved | built axe)                    [exploit the built tool]
  Solve rate             = P(solved)                                [overall]

Layout: Curiosity / Recognition step 1 / Recognition step 2 on the top row, Efficiency /
Solve rate on the bottom row. x = within-family capability rank (ordinal, not a size
estimate); each point is labeled with its model name; error bars are 95% Wilson CIs.

Computed on the full grid T2-6 x N10-30 (all three models fully cover it).

Run from repo root:
  PYTHONPATH=. python -m scripts.lomekwi.plotters.plot_deep_lomekwi
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - half) / d, (c + half) / d)


MODELS = [("Haiku", "deep_lomekwi/haiku_deep_lomekwi_T2-6_N10-30_*"),
          ("Sonnet", "deep_lomekwi/sonnet_deep_lomekwi_T2-6_N10-30_*"),
          ("Opus", "deep_lomekwi/opus_deep_lomekwi_T2-6_N10-30_*")]

T_LO, T_HI = 2, 6
N_LO, N_HI = 10, 30          # full grid (all three models now fully cover it)

# panel name -> (row, col) in the 2x3 grid; (1, 2) is left empty.
GRID_POS = {
    "Curiosity":          (0, 0),
    "Recognition step 1": (0, 1),
    "Recognition step 2": (0, 2),
    "Efficiency":         (1, 0),
    "Solve rate":         (1, 1),
}
# panels whose column has nothing below them -- these get the x-axis title.
BOTTOM_PANELS = {"Recognition step 2", "Efficiency", "Solve rate"}


def load(glob):
    d = sorted(Path("runs").glob(glob))[-1]
    rows = [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines()
            if l.strip()]
    return [r for r in rows if not r.get("error")
            and T_LO <= r["n_types"] <= T_HI and N_LO <= r["n"] <= N_HI]


def metrics(rows):
    """Return {panel name: (value, numerator, denominator)} for one model."""
    n = len(rows)
    base = [r for r in rows if r.get("held_base")]
    part = [r for r in rows if r.get("built_part")]
    axe = [r for r in rows if r.get("built_axe")]
    def frac(num, den):
        return (num / den if den else float("nan"), num, den)
    return {
        "Curiosity":          frac(len(base), n),
        "Recognition step 1": frac(sum(r.get("built_part") for r in base), len(base)),
        "Recognition step 2": frac(sum(r.get("built_axe") for r in part), len(part)),
        "Efficiency":         frac(sum(r.get("solved") for r in axe), len(axe)),
        "Solve rate":         frac(sum(r.get("solved") for r in rows), n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="figs/deep-lomekwi")
    args = ap.parse_args()

    names = [nm for nm, _ in MODELS]
    xs = list(range(1, len(MODELS) + 1))
    data = {nm: metrics(load(glob)) for nm, glob in MODELS}

    fig, axes = plt.subplots(2, 3, figsize=(9, 6.2), constrained_layout=True)
    axes[1, 2].set_visible(False)
    color = "#1f77b4"

    for panel, (row_i, col_i) in GRID_POS.items():
        ax = axes[row_i, col_i]
        ys, los, his = [], [], []
        for nm in names:
            v, num, den = data[nm][panel]
            lo, hi = wilson(num, den)
            ys.append(v); los.append(v - lo); his.append(hi - v)
        ax.errorbar(xs, ys, yerr=[los, his], capsize=3, capthick=1.0, elinewidth=1.2,
                    ecolor=color, alpha=0.6, zorder=1, lw=0, marker="none")
        ax.plot(xs, ys, "-o", color=color, lw=2, ms=8, zorder=2,
                 markeredgecolor="black", markeredgewidth=0.6)
        for x, y, nm in zip(xs, ys, names):
            # Per-point label placement overrides where the default (centered,
            # above the point) collides with the figure edge or the line itself.
            if panel in ("Curiosity", "Efficiency") and nm == "Sonnet":
                # sits at the top of the panel -- drop it below instead.
                offset, ha, va = (0, -11), "center", "top"
            elif panel == "Recognition step 1" and nm == "Sonnet":
                # the steep Haiku->Sonnet drop and the shallow Sonnet->Opus
                # line both crowd the point -- tuck the label up and to the
                # left, into the open space above the drop.
                offset, ha, va = (-6, 13), "right", "bottom"
            elif panel in ("Efficiency", "Solve rate") and nm == "Haiku":
                # steep rise to Sonnet passes through an above-point label --
                # tuck it up and to the left instead.
                offset, ha, va = (-6, 8), "right", "bottom"
            else:
                offset, ha, va = (0, 9), "center", "bottom"
            ax.annotate(nm, (x, y), textcoords="offset points", xytext=offset,
                        ha=ha, va=va, fontsize=8, color=color)

        ax.set_title(panel, fontsize=11, fontweight="bold")
        ax.set_xlim(0.5, len(MODELS) + 0.5)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xticks(xs)
        ax.set_xticklabels([])
        if panel in BOTTOM_PANELS:
            ax.set_xlabel("Model size (ordinal)", fontsize=10, fontweight="bold")
        if col_i == 0:
            ax.set_ylabel("Probability", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "fig_deep_lomekwi_two_layer_metric_panels.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out} (+ .pdf)")

    # also print the table
    print(f"\n{'metric':22}" + "".join(f"{n:>14}" for n in names))
    for panel in GRID_POS:
        cells = "".join(f"{data[nm][panel][0]:>8.2f}{('('+str(data[nm][panel][1])+'/'+str(data[nm][panel][2])+')'):>6}" for nm in names)
        print(f"{panel:22}{cells}")


if __name__ == "__main__":
    main()
