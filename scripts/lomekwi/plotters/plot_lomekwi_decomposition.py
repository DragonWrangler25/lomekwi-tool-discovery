"""Cross-model build-chain LINE graphs for the Lomekwi dense sweep, split by family.

Three statistics (curiosity/recognition/efficiency), drawn as lines -- one ROW
per model family (Anthropic; Qwen3.5), one column per statistic. The x-axis is
the within-family capability rank (1 = weakest), so a family's own scaling
trend reads cleanly without conflating Anthropic's (supposed) sizes with
Qwen's actual ones.

  Curiosity   = P(picked up both shards)
  Recognition = P(built | both shards)
  Efficiency  = P(won | built)

Only Anthropic and Qwen3.5 are plotted (Qwen2.5 and OpenAI are dropped); Qwen3.5
2B is dropped too, so both rows end up with the same number of points (3).

Reads all run dirs under runs/grid/dense/ (override with explicit dir args).
Usage: PYTHONPATH=. python -m scripts.lomekwi.plotters.plot_lomekwi_decomposition [dir ...]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.lomekwi.replay_lomekwi import replay

DENSE_DIR = Path("runs") / "grid" / "dense"
FIGS_DIR = Path("figs") / "grid"

# (title, predicate to select the denominator, outcome) -- each over episode facts.
# denom=None means "all episodes"; outcome maps a fact to 0/1.
PANELS = [
    ("P(picked up both shards)",
     lambda f: f["has_both"] is not None,
     lambda f: 1.0 if f["has_both"] else 0.0),
    ("P(built | both shards)",
     lambda f: f["has_both"] is True,
     lambda f: 1.0 if f["built"] else 0.0),
    ("P(won | built)",
     lambda f: f["built"],
     lambda f: 1.0 if f["solved"] else 0.0),
    ("P(won) overall (end-to-end)",
     lambda f: True,
     lambda f: 1.0 if f["solved"] else 0.0),
]


def load(ep: Path) -> list[dict]:
    return [json.loads(l) for l in ep.read_text().splitlines() if l.strip()]


def episode_facts(row: dict) -> dict | None:
    """Replay one episode -> {n, t, has_both, built, solved}. None if errored."""
    if row.get("error"):
        return None
    s, _ = replay(row)
    recipe = (row.get("labels") or {}).get("recipe", []) or []
    built = bool(getattr(s, "has_machine", False))
    solved = bool(s.solved())
    has_both = None
    if len(recipe) >= 2:
        has_both = (s.byproducts.get(recipe[0], 0) >= 1
                    and s.byproducts.get(recipe[1], 0) >= 1)
    return {"n": row["n"], "t": row["n_types"], "has_both": has_both,
            "built": built, "solved": solved}


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - half) / d, (c + half) / d)

# (family, size in B params, short label) keyed by substring of the dir name.
# Qwen sizes are exact; Anthropic sizes are SUPPOSED (unpublished) estimates --
# both are used only to ORDER models within a family; the plotted x-axis is
# ordinal rank, not size.
CLASSIFY = [
    ("haiku",        ("Anthropic", 20.0, "Haiku 4.5")),
    ("sonnet",       ("Anthropic", 150.0, "Sonnet 4.6")),
    ("opus",         ("Anthropic", 600.0, "Opus 4.8")),
    ("Qwen2-5-7B",   ("Qwen2.5", 7.0, "7B")),
    ("Qwen2-5-14B",  ("Qwen2.5", 14.0, "14B")),
    ("Qwen2-5-72B",  ("Qwen2.5", 72.0, "72B")),
    ("Qwen3-5-2B",   ("Qwen3.5", 2.0, "2B")),
    ("Qwen3-5-4B",   ("Qwen3.5", 4.0, "4B")),
    ("Qwen3-5-9B",   ("Qwen3.5", 9.0, "9B")),
    ("Qwen3-5-27B",  ("Qwen3.5", 27.0, "27B")),
    # OpenAI sizes are SUPPOSED (unpublished); order matters here (match the
    # specific -mini/-nano BEFORE bare gpt-5, since it's a substring of both).
    ("gpt-5-nano",   ("OpenAI", 4.0, "gpt-5-nano")),
    ("gpt-5-mini",   ("OpenAI", 8.0, "gpt-5-mini")),
    ("gpt-5",        ("OpenAI", 200.0, "gpt-5")),
]
FAMILY_STYLE = {  # colour + marker per family
    "Anthropic": dict(color="#C44E52", marker="o"),
    "Qwen2.5":   dict(color="#4C72B0", marker="s"),
    "Qwen3.5":   dict(color="#9467BD", marker="^"),
    "OpenAI":    dict(color="#2CA02C", marker="D"),
}

# Families/points actually plotted -- see module docstring.
ROW_FAMILIES = ["Anthropic", "Qwen3.5"]
EXCLUDE = {"Qwen3.5": {"2B"}}

PANEL_TITLES = ["Curiosity", "Recognition", "Efficiency"]


def classify(name: str):
    for key, val in CLASSIFY:
        if key in name:
            return val
    return None


def discover(dirs: list[Path]) -> dict:
    """family -> list of (size_b, label, facts) sorted by size. Unfiltered --
    includes every family CLASSIFY knows about; see filtered_families() for the
    Anthropic/Qwen3.5-only, 2B-dropped view this module's figures use."""
    fams: dict[str, list] = {}
    for d in dirs:
        ep = d if d.suffix == ".jsonl" else d / "episodes.jsonl"
        if not ep.exists():
            continue
        info = classify(d.name)
        if not info:
            print(f"  (skipping unclassified {d.name})")
            continue
        fam, size, label = info
        facts = [f for f in (episode_facts(r) for r in load(ep)) if f]
        fams.setdefault(fam, []).append((size, label, facts))
    for fam in fams:
        fams[fam].sort(key=lambda x: x[0])
    return fams


def filtered_families(fams_all: dict) -> dict:
    """Restrict to ROW_FAMILIES, drop EXCLUDE'd points, keep sorted by size."""
    fams = {}
    for fam in ROW_FAMILIES:
        members = [m for m in fams_all.get(fam, []) if m[1] not in EXCLUDE.get(fam, set())]
        if members:
            fams[fam] = sorted(members, key=lambda x: x[0])
    return fams


def main():
    args = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
    dirs = args if args else sorted(p for p in DENSE_DIR.iterdir() if p.is_dir())
    fams = filtered_families(discover(dirs))
    if not fams:
        raise SystemExit(f"no classifiable Anthropic/Qwen3.5 run dirs found under {DENSE_DIR}")
    for fam, members in fams.items():
        print(f"{fam}: " + ", ".join(f"{lab}(n={len(f)})" for _, lab, f in members))

    n_rows = len(ROW_FAMILIES)
    fig, axes = plt.subplots(n_rows, 3, figsize=(7.5, 2.6 * n_rows), squeeze=False,
                              constrained_layout=True)

    for row_i, fam in enumerate(ROW_FAMILIES):
        members = fams.get(fam)
        if not members:
            continue
        style = FAMILY_STYLE.get(fam, dict(color="gray", marker="^"))
        for col_i, (panel_title, (_, denom, outcome)) in enumerate(zip(PANEL_TITLES, PANELS[:3])):
            ax = axes[row_i][col_i]
            xs, ys, los, his, labels = [], [], [], [], []
            for rank, (size, label, facts) in enumerate(members, start=1):
                sel = [f for f in facts if denom(f)]
                k = int(sum(outcome(f) for f in sel)); n = len(sel)
                p = k / n if n else float("nan")
                lo, hi = wilson(k, n)
                xs.append(rank); ys.append(p)
                los.append(p - lo); his.append(hi - p); labels.append(label)
            # small-capped whiskers: readable but lighter than the line so CIs don't shout
            ax.errorbar(xs, ys, yerr=[los, his], capsize=3, capthick=1.0,
                        elinewidth=1.2, ecolor=style["color"], alpha=0.6, zorder=1,
                        lw=0, marker="none")
            ax.plot(xs, ys, lw=2, markersize=9, markeredgecolor="black",
                    markeredgewidth=0.6, zorder=2, **style)
            for x, y, lab in zip(xs, ys, labels):
                if not math.isnan(y):
                    # Per-point label placement overrides where the default
                    # (centered, above the point) collides with the axes edge
                    # or the connecting line itself.
                    if panel_title == "Curiosity" and lab in ("Sonnet 4.6", "Opus 4.8"):
                        # sit at the top of the panel -- drop below instead.
                        offset, ha, va = (0, -11), "center", "top"
                    elif panel_title == "Efficiency" and lab == "Haiku 4.5":
                        # steep rise to Sonnet passes through an above-point
                        # label -- tuck it up and to the left instead.
                        offset, ha, va = (-6, 8), "right", "bottom"
                    elif panel_title == "Recognition" and lab == "Sonnet 4.6":
                        # local minimum -- both neighboring lines converge from
                        # above, so drop the label below instead.
                        offset, ha, va = (0, -11), "center", "top"
                    else:
                        offset, ha, va = (0, 9), "center", "bottom"
                    ax.annotate(lab, (x, y), textcoords="offset points",
                                xytext=offset, ha=ha, va=va, fontsize=8,
                                color=style["color"])
            if row_i == 0:
                ax.set_title(panel_title, fontsize=12, fontweight="bold")
            ax.set_xlim(0.5, len(members) + 0.5)
            ax.set_ylim(0, 1.08)
            ax.set_xticks(range(1, len(members) + 1))
            ax.set_xticklabels([])
            if row_i == n_rows - 1:
                ax.set_xlabel("Model size (ordinal)", fontweight="bold")
            if col_i == 0:
                ax.set_ylabel("Probability", fontweight="bold")
                ax.text(-0.32, 0.5, fam, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=11, fontweight="bold")
            ax.grid(alpha=0.3, which="major")

    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGS_DIR / "fig_grid_dense_chain_lines.png"
    fig.savefig(out, dpi=130); fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"  wrote {out}")
    print(f"  wrote {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
