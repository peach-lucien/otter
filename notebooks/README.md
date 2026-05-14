# Notebooks

Four interactive walkthroughs. None re-fit the model from scratch — they load pre-computed outputs from `outputs/`. Run `pipeline/04_solve_production.py` + `experiments/anchor_packs/compose_all.py` first if you need to regenerate.

## Reading order

| # | Notebook | What it does | Time |
|---|---|---|---|
| 01 | [`01_quickstart.ipynb`](01_quickstart.ipynb) | Interactive: pick a mouse region, see top-K human partners. Compare strict-π vs packed-π. | 5 min |
| 02 | [`02_trust_map.ipynb`](02_trust_map.ipynb) | Per-parcel multi-source evidence tier. Interactive filter by tier. Which parts of the brain to trust. | 5 min |
| 03 | [`03_anchor_packs.ipynb`](03_anchor_packs.ipynb) | Side-by-side Beauchamp comparison: production vs production+all-packs. Per-region lift table + bar plot. | 5 min |
| 04 | [`04_methodology.ipynb`](04_methodology.ipynb) | Step-by-step FGW: raw FC → costs → M → solver → eval. For people who want to see what's under the hood. | 15 min |

## Required widgets

Notebooks 01 and 02 use `ipywidgets` for interactivity. If you see a static-looking output, install:

```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

## Archive

`archive/` keeps two older notebooks from the iteration period:
- `02_explore_results.ipynb` — the historical "comprehensive comparison" notebook (superseded by `03_anchor_packs.ipynb` + `02_trust_map.ipynb`).
- `03_compare_model_levels.ipynb` — fits all four model levels side-by-side (UnsupervisedGW / SupervisedFGW / MultimodalFGW / HierarchicalFGW). Re-fits from scratch (~5 min), so it's slow but instructive.
