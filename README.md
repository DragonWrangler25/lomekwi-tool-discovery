# Lomekwi: Resource-Bounded Tool Discovery in LLM Agents

Code, environments, and run logs for the paper *Lomekwi: Resource-Bounded Tool
Discovery in LLM Agents* (Klein-Seetharaman, Wang, Xu — Sea12 Technologies /
Yale University).

## Repository layout

- **`lomekwi.tex`, `lomekwi.bib`** — the paper.
- **`scripts/`** — every runner, sweep, and plotting script that produces a
  number or figure in the paper, split by environment, plus a `shared/`
  library (the per-episode relabeling/obfuscation protocol — three schemes,
  see Appendix D — and a multi-provider chat client) used by all of them.
  See [`scripts/README.md`](scripts/README.md).
- **`runs/`** — raw episode logs (`episodes.jsonl`) for the sweeps reported
  in the paper, one directory per sweep, grouped by environment
  (`lomekwi/`, `deep_lomekwi/`, `oracle/`).
- **`figures/`** — the exact figure files cited in the paper.
- **`requirements.txt`** — Python dependencies.

## Setup

```bash
pip install -r requirements.txt
```

Set whichever provider keys you need as environment variables (or in a
`.env` file, loaded automatically): `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GEMINI_API_KEY` (or `GOOGLE_API_KEY`); for local/open-weight models point
`VLLM_BASE_URL` or `OLLAMA_BASE_URL` at a running server.

## Reproducing results

All scripts run from the repo root with `PYTHONPATH=.`, e.g.:

```bash
PYTHONPATH=. python -m scripts.lomekwi.run_grid_sweep
PYTHONPATH=. python -m scripts.lomekwi.plotters.plot_lomekwi_summary
```

See [`scripts/README.md`](scripts/README.md) for what each script
reproduces. Sweeps write fresh logs under `runs/`; plotting scripts read
those logs and write figures alongside them (or to `figures/`).

## Citation

```
Klein-Seetharaman, R., Wang, D., and Xu, A.
Lomekwi: Resource-Bounded Tool Discovery in LLM Agents.
```
