# OTTER pipeline

The pipeline builds processed inputs, generated artefacts and the mapping explorer. Run commands
from the repository root in the `otter` environment.

## Use the released coupling

Most users do not need to refit the model:

```bash
python scripts/fetch_data.py
```

This downloads `outputs/coupling/pi_canonical.npy` and the processed inputs used by the notebooks.
`load_pi()` returns this coupling by default.

## Rebuild processed inputs

Download the raw tier and then run the preparation steps:

```bash
python scripts/fetch_data.py --tier raw
PYTHONPATH=src python pipeline/00_external/00_inspect_masks.py
PYTHONPATH=src python pipeline/00_external/00b_verify_alignment.py
PYTHONPATH=src python pipeline/00_external/01_mouse_sc.py
PYTHONPATH=src python pipeline/00_external/02_mouse_genes.py
PYTHONPATH=src python pipeline/00_external/03_human_sc.py
PYTHONPATH=src python pipeline/00_external/04_human_genes.py
PYTHONPATH=src python pipeline/00_external/05_orthologs.py
PYTHONPATH=src python pipeline/02_build_anndata.py
PYTHONPATH=src python pipeline/03_build_costs.py
```

See [`00_external/README.md`](00_external/README.md) for source-specific requirements.

## Refit the canonical model

The public fitting recipe is defined in `otter.repro`:

```python
import numpy as np
from otter.repro import CANONICAL, anchor_warped_xyz, fit_coupling, load_inputs

mouse, human, costs, regional_entries = load_inputs()
spatial_cost = anchor_warped_xyz(mouse, human)
pi = fit_coupling(
    mouse, human, costs, regional_entries, spatial_cost, **CANONICAL
)
np.save("outputs/coupling/pi_canonical_refit.npy", pi)
```

The canonical fit combines 21 Garin homology classes, 26 curated regional entries and the
anchor-warped spatial cost. Save a refit under a new name and compare its provenance with the
released coupling before using existing result logs.

## Analysis and interface artefacts

Analysis scripts are grouped under [`experiments/`](../experiments/). General pipeline components
can also be run directly:

```bash
PYTHONPATH=src python pipeline/08_build_gui.py --publish
```

Generated files are written beneath `outputs/`. The released display categories used by the
explorer are heuristic interface metadata rather than confidence or validation tiers.

All scripts use paths relative to the repository root. Existing cached inputs are reused where
supported; consult a script's `--help` output before forcing recomputation.
