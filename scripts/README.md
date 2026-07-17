# scripts/

Two independent environments (`lomekwi/`, `oracle/`), each with its
runners/environment code and a `plotters/` subpackage for the matplotlib
figures, plus `shared/` for code both environments import. Run everything
from the repo root, e.g.:

```bash
PYTHONPATH=. python -m scripts.lomekwi.run_grid_sweep
PYTHONPATH=. python -m scripts.oracle.plotters.plot_qwen_oracle
```

`scripts/lomekwi/` and `scripts/oracle/` have no dependency on each other;
both import `scripts/shared/`.

Plotters write to `figs/...` (created on demand, not checked in); the
curated figures actually cited in the paper live in `../figures/`.

## `scripts/lomekwi/`

**Environments.** `lomekwi.py` is Lomekwi (Section 4, explicit `pickup`,
matching the appendix's action table) — the environment every runner in this
repo actually executes. `lomekwi_world.py` is a library, not a second
runnable variant: it holds `make_world`/`recipe_indices`/`world_from_labels`/
`State`, which `lomekwi.py` calls directly so worlds stay byte-identical
across the same seeds, and which `replay_lomekwi.py` reuses to reconstruct
episodes. It has no `run()`/CLI of its own. `run_lomekwi_llm.py` is the v1
world/parser library (`ACT`/`SYS`/`extract`/`parse`/`_clean`), reused the
same way — also no runner of its own. `deep_lomekwi.py` is Deep Lomekwi
(Appendix, "Notable Variations") — a coupon-collector goal plus a two-layer
recipe. `sweep_config.py` is the shared budget config (`budget_for`) and
model-roster/concurrency config used by the runners and
`plotters/plot_fig_build_vs_grind_region_2n.py`. `replay_lomekwi.py`
deterministically replays a logged episode (used by
`plotters/plot_lomekwi_decomposition.py` to recover per-episode build-chain
facts).

**Runners.** `run_grid_sweep.py` (Lomekwi — both the random-scatter sweep
and, with `--dense`, the sweep behind the dense-grid figures);
`run_deep_lomekwi_sweep.py` (Deep Lomekwi, sweeps `T in [2,6]`, `N in [10,30]`).

**`plotters/`.** Pruned to just the scripts that produce a figure cited in
the paper, named `plot_<figure>.py` after the file it writes into
`../figures/`; each is self-contained (only imports from outside
`plotters/`), except `plot_lomekwi_summary.py`, which imports its family
discovery and `PANELS`/`wilson` from `plot_lomekwi_decomposition.py`:
- `plot_lomekwi_decomposition.py` (`lomekwi_decomposition.pdf`,
  Curiosity/Recognition/Efficiency; owns the episode loader/replay/Wilson-CI
  helpers and the `PANELS` outcome definitions).
- `plot_lomekwi_summary.py` (`lomekwi_summary.pdf`, P(win)/P(build) overall).
- `plot_deep_lomekwi.py` (`deep_lomekwi.pdf`, C/R1/R2/E/solve vs.
  capability; own Wilson-CI helper).
- `plot_fig_build_vs_grind_region_2n.py` (`fig_build_vs_grind_region_2n.pdf`,
  action-budget appendix). Depends only on `sweep_config.py`.

Obfuscation-scheme ablation (Appendix D) is not a separate script: it's
`shared/obfuscation.py`'s three schemes (`tokens` / `alnum` / `letter`) run
through the same Lomekwi environment with `scheme=` overridden per sweep.

## `scripts/shared/`

Imported by both environments. `obfuscation.py` is the per-episode
relabeling protocol — three label pools (`tokens` / `alnum` / `letter`;
Appendix D). `raw_chat.py` is a thin multi-provider chat client (Claude,
GPT, Gemini, vLLM, Ollama) used by every runner.

## `scripts/oracle/`

**Environment driver.** `run_oracle_llm.py` (single episode) and
`run_oracle_sweep.py` (the production sweep) drive the recognition MCP
environment (`config.py`, `problems.py`, `server.py`, `mcp_host.py`).

**Config / calibration.** `oracle_sweep_config.py` (Qwen3.5 + Claude
roster), `oracle_qwen25_config.py`, `oracle_qwen3_dense_config.py` —
together the three Qwen families reported in Section 5.2. `oracle_pilot.py`
is the tools-off calibration run behind the budget formula quoted in the
appendix.

**Scoring / analysis.** `oracle_counterfactual.py` scores one all-decoy
rollout into `n` scored oracle-assignment episodes; both kept plotters import
only this (plus stdlib) to regenerate their figures from existing
`runs/oracle/` data.

**`plotters/`.** `plot_qwen_oracle_with_hint.py` (`qwen_oracle_with_hint.pdf`)
and `plot_qwen_oracle.py` (`qwen_oracle.pdf`) — the 3x3 (family x outcome)
panel figure, old per-size-directory sweep vs. the newer combined-episodes
vLLM sweep.
