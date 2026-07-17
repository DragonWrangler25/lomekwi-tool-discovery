"""Shared configuration for the Lomekwi sweeps (run_grid_sweep.py,
run_deep_lomekwi_sweep.py) and the action-budget appendix figure
(plotters/plot_fig_build_vs_grind_region_2n.py).

budget_for(n) is DERIVED from n via the coupon-collector (grind) expectation.
Callers must keep their turn cap strictly above budget_for(n), or the turn
cap -- not the budget -- becomes the binding constraint and the "build vs.
grind" decision stops being economic.
"""

from __future__ import annotations

import scripts.shared.obfuscation as _obfuscation

# --- obfuscation scheme ------------------------------------------------
# "tokens" = pronounceable nonsense tokens (default). "letter" = a single
# random letter per element. "alnum" = a random alphanumeric string (length
# 4-8) per element. Flipping this here propagates to every sweep that imports
# sweep_config (assign() reads the module default).
OBFUSCATION_SCHEME = "letter"
_obfuscation.DEFAULT_SCHEME = OBFUSCATION_SCHEME

N = 8              # default door count for budget_for's default arg
HINT = True        # subtle "feels active alongside a different kind" hint
BUDGET_MULT = 1.2  # budget as a multiple of E[grind] (coupon-collector + n)


def _Hn(n: int) -> float:
    return sum(1.0 / k for k in range(1, n + 1))


def _grind_cost(n: int) -> float:
    """E[actions] to brute-force open all n doors: coupon-collector over keys
    (n*H_n) plus the n opens themselves."""
    return n * _Hn(n) + n


def budget_for(n: int = N) -> int:
    """Strict action budget = BUDGET_MULT x E[grind] (grind-calibrated). Brute
    force is always a viable escape hatch, so building stays OPTIONAL -- feasible
    only where it is genuinely cheaper than grinding (E[build] < E[grind]), which
    is exactly what makes built_tool a clean disposition signal rather than a
    feasibility outcome."""
    return round(BUDGET_MULT * _grind_cost(n))


# Per-provider in-flight episode cap (bounds rate-limit / token bursts). A
# single shared dict is fine -- each sweep only indexes the providers it runs.
CONCURRENCY = {"anthropic": 6, "openai": 4, "google": 4, "ollama": 2}
