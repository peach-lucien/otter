# Extending the pipeline

How to add a new modality, a new species, or a new model class without
touching the rest of the codebase.

## New modalities

A within-species cost matrix from a new modality (e.g. dMRI tractography)
has three integration points.

### 1. A within-species cost function

Add to `otter.costs.relational`:

```python
def tractography_correlation_distance(tract: np.ndarray) -> np.ndarray:
    """Within-species relational distance from a tractography matrix."""
    # ... return symmetric, zero-diagonal, finite (n, n) distance
```

It must return a `(n, n)` symmetric, zero-diagonal, non-negative, finite
matrix. The tests in `tests/test_costs.py::_is_valid_cost_matrix`
encode the contract.

### 2. A pipeline step to precompute and cache it

Add a `pipeline/03d_build_tractography_costs.py` script that:
- loads the raw tractography from `data_external/`
- calls `tractography_correlation_distance`
- stashes the result in `outputs/anndata/full_costs.npz` as `Cm_tract` and
  `Ch_tract`

Then add the script name to `pipeline/03_build_costs.py`'s `STEPS` list so
the orchestrator runs it.

### 3. A model parameter

Add to `MultimodalFGW.__init__`:

```python
def __init__(self, ..., use_tract: bool = False, tract_weight: float = 0.0,
             ...):
    ...
    self.config.update(dict(use_tract=use_tract, tract_weight=tract_weight))
```

And in `_solve()`, mix it into the relational cost:

```python
if self.config["use_tract"]:
    if Cm_tract is None or Ch_tract is None:
        raise ValueError("use_tract=True but Cm_tract/Ch_tract not supplied to fit()")
    weights["tract"] = self.config["tract_weight"]
    Cm = Cm + weights["tract"] * Cm_tract.astype(np.float64)
    Ch = Ch + weights["tract"] * Ch_tract.astype(np.float64)
```

### 4. A new test config

In `pipeline/05a_anchor_cv.py`, add a `CONFIGS` entry:

```python
"fc_plus_tract": dict(
    relational={"FC": 0.7, "tract": 0.3},
    M={"xyz": 0.5},
),
```

Then re-run `pipeline/05a_anchor_cv.py --configs fc_plus_tract`. The result
will appear in the comparison table the next time `pipeline/07_build_artefacts.py`
runs.

## New species

Adding macaque alongside mouse and human is a data-layer change. The rest of
the codebase is species-agnostic.

### 1. Update the I/O constants

`otter.data.io._MAT_TOPKEY` currently maps `"mouse" → "m"` and `"human" → "h"`.
Extend it for the new species and the corresponding `corrs_<species>.mat`
file under `data_external/`.

### 2. Anchor definition

The 42 Garin anchors are mouse-human-specific. A new species pair needs new
putative homologue pairs. Two options:

- Hand-curate them and update `otter.data.networks.PAIRID_TO_NETWORK` with
  the new species' pair labels.
- Use a published atlas (e.g. for macaque-human, the Markov 2014 hierarchy
  has ~30 well-defined cortical homologues).

### 3. Model use

Once the data layer accepts the new species, the model classes work
unchanged:

```python
from otter.data import load_cached
from otter.models import MultimodalFGW
M, _ = load_cached("macaque", cache_dir=...)
H, _ = load_cached("human", cache_dir=...)
model = MultimodalFGW(use_sc=True).fit(M, H)
```

## New region anchors (supplementary supervision)

A new mouse↔human region pair on top of the 15 atlas-derived region anchors is
added either through a YAML config or programmatically.

YAML form (see `config/supplementary_anchors_motor.yaml` for an example):

```yaml
- pair_id: 30        # must be > 21 to avoid clashing with Garin point anchors
  label: "Some new region"
  mouse:
    node_ids: ["L_708", "R_808"]   # explicit
    # OR
    centroid_mm: [-1.5, 2.6, 1.8]
    radius_mm: 1.5
  human:
    node_ids: ["L_935"]
    # OR
    centroid_mm: [-35, -20, 55]
    radius_mm: 15
```

Then in code:

```python
from otter.data.region_anchors import parse_region_anchors_config, apply_region_supervision

entries = parse_region_anchors_config("config/my_region.yaml", M.var, H.var)
# Soft constraint (default 0.15), gives room for FC/SC structure to push back
M_cost = apply_region_supervision(M_cost, entries)
# Hard constraint (legacy 0/1 wall), use when you want strict enforcement
M_cost = apply_region_supervision(M_cost, entries, lam_outside=1.0)
```

Region anchors apply when the mouse or human side is a multi-parcel set rather
than a single point, and when the set of permitted partners is known but a
specific 1-to-1 pairing is not.

## New model classes

The base class contract is in `otter.models.base.FGWModel`. To add a new
solver:

```python
from otter.models.base import FGWModel, FitInfo


class MyNewSolverFGW(FGWModel):
    """One-line description."""
    _name = "MyNewSolverFGW"

    def __init__(self, *, my_param: float = 1.0, **kwargs):
        super().__init__(my_param=my_param, **kwargs)

    def _solve(self, *, mouse_ad, human_ad, **kw):
        # ... build Cm, Ch, M, etc.
        # ... call your solver
        # return (pi, FitInfo(loss=..., n_iter=..., converged=...))
        return pi, FitInfo(loss=loss, n_iter=n_iter, converged=True)
```

Then export it from `otter.models.__init__`:

```python
from otter.models.my_new_solver import MyNewSolverFGW
__all__.append("MyNewSolverFGW")
```

Test it the same way the other models are tested in `tests/test_models.py`
(parametrise over the new class).

## Custom evaluation metrics

Add a new function to `otter.eval` and re-export from `otter.eval.__init__`.
Then add it to `pipeline/07_build_artefacts.py` for it to appear in the
headline comparison table.

The library API assumes `model.evaluate()` returns a dict; downstream code (notebooks, comparison generators) iterates
over the dict keys generically.
