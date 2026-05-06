# Cross-Species Brain Mapping with Fused Gromov–Wasserstein

**Goal.** Produce a probabilistic (fuzzy, many-to-many) coupling π between mouse brain
nodes (1864) and human brain nodes (2094) from resting-state functional connectivity,
with the 42 Garin-atlas anchor pairs used as supervision and evaluation.

**Non-goal (for now).** A hard 1-to-1 permutation. We can discretise π post-hoc if a
downstream consumer needs it.

**Design ethos.** Build the simplest thing that could work first; treat each layer as an
independent, replaceable module; gate every step on a numeric sanity check that fails
loudly if the data / math is wrong.

---

## 0. Why this approach and not the colleague's

Two structural problems with the spectral + Procrustes + FAQ pipeline:

1. **The 42 Garin anchors are unused.** Spectral embedding ignores known homology;
   Procrustes is a global alignment with no per-node supervision; FAQ refines without
   anchor constraints. We have ground truth and we should *use* it.
2. **A hard permutation is the wrong output type.** Mouse has 1864 nodes, human has
   2094. Even if every mouse node had a unique human counterpart (it doesn't), a
   permutation can't represent partial homology, evolutionary expansion, or
   one-to-many relationships. A soft transport plan can.

Fused Unbalanced Gromov–Wasserstein addresses both directly, with a single global
objective and a soft, probabilistic output.

---

## 1. Repository layout

```
moscot/
├── PLAN.md                      ← this document
├── README.md                    ← quickstart once code exists
├── pyproject.toml               ← deps, lint config
├── env.yml                      ← reproducible conda environment
├── src/
│   └── retune/
│       ├── __init__.py
│       ├── data.py              ← .mat / NIfTI loaders, AnnData builders
│       ├── features.py          ← per-node embeddings (PCA, FC fingerprint)
│       ├── costs.py             ← intra-species distance matrices
│       ├── fgw.py               ← Fused (Unbalanced) GW wrapper around POT
│       ├── anchors.py           ← Garin anchor handling, partial-coupling priors
│       ├── eval.py              ← anchor recovery, hemispheric symmetry, bootstrap
│       ├── viz.py               ← plotting (FC heatmaps, π heatmaps, 3D nodes)
│       └── flow.py              ← (Phase 8) OT-CFM refinement, optional
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_garin_only_sanity.ipynb
│   ├── 04_features.ipynb
│   ├── 05_cost_matrices.ipynb
│   ├── 06_fgw_full.ipynb
│   ├── 07_evaluation.ipynb
│   └── 08_flow_matching.ipynb     ← optional
├── scripts/
│   ├── build_anndata.py            ← one-shot data prep, caches .h5ad
│   └── run_fgw.py                  ← CLI for full sweep
├── tests/
│   ├── test_data.py
│   ├── test_costs.py
│   ├── test_fgw.py
│   └── test_anchors.py
└── outputs/
    ├── anndata/                    ← cached human.h5ad, mouse.h5ad
    ├── coupling/                   ← π matrices per hyperparam
    ├── figures/
    └── logs/
```

`src/retune/` rather than `src/moscot/` so we don't shadow the real `moscot` package
if it ever gets installed.

---

## 2. Environment

Pinned-but-light. We'll start with a fresh venv to avoid contamination.

```yaml
# env.yml
name: retune
channels: [conda-forge, defaults]
dependencies:
  - python=3.11
  - numpy>=1.26
  - scipy>=1.12
  - h5py
  - mat73          # for v7.3 .mat (HDF5-backed) files
  - pandas
  - anndata
  - scanpy
  - nibabel        # NIfTI masks
  - matplotlib
  - seaborn
  - plotly         # 3D node viz
  - scikit-learn
  - pytorch        # for OT-CFM / flow matching later
  - pip
  - pip:
    - pot          # Python Optimal Transport — main workhorse
    - jupyterlab
    - ruff
    - pytest
```

Why no JAX / OTT / moscot proper? We don't need their scaling tricks at 2k nodes, and
the JAX setup is fragile across platforms. We *can* add `ott-jax` later as a
cross-check on FGW results.

Sanity: `python -c "import ot; print(ot.__version__)"` and `python -c "import torch;
print(torch.cuda.is_available())"` (CUDA is nice-to-have, not required).

---

## 3. Data layer (`src/retune/data.py`)

### 3.1 What we're loading

| File | Content | Backend |
|---|---|---|
| `corrs_human.mat` | struct with `rr [2094×2094×113]`, `t [2094×7 cell]`, `ht`, `dirs`, `species` | v7.3 HDF5 — use `mat73` or `h5py` |
| `corrs_mouse.mat` | struct with `rr [1864×1864×105]`, `t [1864×7 cell]`, … | same |
| `_human_mask/rsmask_human.nii` | brain mask in MNI | `nibabel` |
| `_human_mask/single_subj_T1.nii` | T1 template | `nibabel` |
| `_mouse_mask/rsmask.nii` | mouse mask in AIBS / Allen space | `nibabel` |
| `_mouse_mask/RS_AVGT.nii.gz` | reference template | `nibabel` |

### 3.2 AnnData layout

We follow the moscot convention so we can later port to its API if useful.

For each species we build an `AnnData` with shape `(n_subjects, n_nodes)`:

- `adata.X` — *unused* (or set to mean FC of each subject as a placeholder); keep small.
- `adata.obs` — subject-level metadata (subject ID, source path, dataset).
- `adata.var` — node-level metadata: `numid, pairid, type, region, subregion, x, y, z,
  hemisphere ∈ {L, R}, garin_anchor ∈ {True, False}, anchor_pair_id (NaN for
  non-anchors)`.
- `adata.varm["voxel_indices"]` — ragged voxel index lists (store as object array).
- `adata.layers["fc"]` — full FC tensor reshaped to `(n_subjects, n_nodes*n_nodes)` —
  too big? alternative: store as a separate `.npy` and reference by path.
- `adata.uns["fc_mean"]` — precomputed `(n_nodes × n_nodes)` average FC.
- `adata.uns["fc_z"]` — Fisher-z transformed mean FC.

Store: `outputs/anndata/human.h5ad`, `outputs/anndata/mouse.h5ad`, plus
`outputs/anndata/fc_human.npy`, `fc_mouse.npy` for the heavy tensor.

### 3.3 Loader skeleton

```python
# src/retune/data.py
from __future__ import annotations
import numpy as np, pandas as pd, h5py, mat73, nibabel as nib, anndata as ad
from pathlib import Path

DATA = Path("data_crossspecies")

def _load_struct(mat_path: Path, species: str) -> dict:
    """Load v7.3 .mat into a python dict. Try mat73 first, fall back to h5py."""
    try:
        d = mat73.loadmat(mat_path)[species]   # mat73 unwraps nicely
    except Exception:
        d = _h5_to_dict(h5py.File(mat_path, "r")[species])
    return d

def _parse_t_table(t, ht) -> pd.DataFrame:
    """t is a cell array [n_nodes × 7]; columns named by ht."""
    cols = [str(h) for h in ht]
    df = pd.DataFrame({c: t[i] for i, c in enumerate(cols)})
    # center is a 1×3 array per row, indices is a list — flatten as needed
    centers = np.stack(df["center"].values)
    df[["x", "y", "z"]] = centers
    df["hemisphere"] = df["region"].str.startswith("L_").map({True: "L", False: "R"})
    df["garin_anchor"] = df["type"].astype(int) == 1
    df["anchor_pair_id"] = df["pairid"].where(df["garin_anchor"]).astype("Int64")
    return df

def build_anndata(species: str, *, cache: Path | None = None) -> ad.AnnData:
    mat = DATA / f"corrs_{species}.mat"
    raw = _load_struct(mat, species)
    rr  = np.ascontiguousarray(raw["rr"])      # (n_nodes, n_nodes, n_subj)
    var = _parse_t_table(raw["t"], raw["ht"])
    n_subj = rr.shape[2]
    obs = pd.DataFrame({
        "subject_id": [f"{species}_{i:04d}" for i in range(n_subj)],
        "source_path": raw["dirs"],
    }).set_index("subject_id")
    obs.index = obs.index.astype(str)
    var.index = var["numid"].astype(int).astype(str)
    A = ad.AnnData(
        X=np.zeros((n_subj, rr.shape[0]), dtype=np.float32),  # placeholder
        obs=obs, var=var, uns={"species": species},
    )
    A.uns["fc_mean"] = rr.mean(axis=2).astype(np.float32)
    A.uns["fc_z"]    = np.arctanh(np.clip(A.uns["fc_mean"], -0.999, 0.999))
    if cache:
        np.save(cache.with_suffix(".fc.npy"), rr)
        A.write_h5ad(cache.with_suffix(".h5ad"))
    return A
```

### 3.4 Validation block (must pass before moving on)

```python
# tests/test_data.py
def test_shapes():
    H = build_anndata("human")
    M = build_anndata("mouse")
    assert H.uns["fc_mean"].shape == (2094, 2094)
    assert M.uns["fc_mean"].shape == (1864, 1864)
    assert (H.var["garin_anchor"]).sum() == 42
    assert (M.var["garin_anchor"]).sum() == 42

def test_anchor_pairing_consistent():
    H = build_anndata("human"); M = build_anndata("mouse")
    h_anchor_pairs = set(H.var.loc[H.var.garin_anchor, "anchor_pair_id"])
    m_anchor_pairs = set(M.var.loc[M.var.garin_anchor, "anchor_pair_id"])
    # Same 21 pair IDs (1..21), each appearing L+R, in both species
    assert h_anchor_pairs == m_anchor_pairs == set(range(1, 22))

def test_fc_symmetry():
    H = build_anndata("human")
    fc = H.uns["fc_mean"]
    assert np.allclose(fc, fc.T, atol=1e-5)
    assert np.allclose(np.diag(fc), 1.0, atol=1e-3)  # self-correlation = 1
```

These three tests are the minimum bar. **Do not start §4 until they all pass.**

---

## 4. Exploratory analysis (`notebooks/02_eda.ipynb`)

Designed to find data problems before they sink the modeling work.

1. **FC value distribution.** Histogram of off-diagonal FC values per species. Expect
   roughly Gaussian-ish, mean weakly positive, no spikes at ±1.
2. **Subject-to-subject FC similarity.** Vec each subject's upper triangle, compute
   the 113×113 (or 105×105) subject correlation. Look for outlier subjects to
   potentially exclude. Threshold candidate: mean inter-subject correlation < 0.3.
3. **Within-species L/R symmetry.** For each `pairid`, correlate left and right node's
   FC fingerprints (rows of mean FC). If left/right homologues have correlation < 0.5
   for many pairs, the FC may be too noisy — flag this.
4. **Garin anchor connectivity sanity.** Pull the 42 anchor nodes per species.
   Compute their inter-anchor FC matrix (42×42). Compute Spearman correlation between
   the human and mouse anchor FC matrices. *This is the cross-species signal we'll
   ride.* Expect ρ between ~0.3 and ~0.6 — anything below 0.2 is a red flag.
5. **3D node positions.** Plotly scatter, coloured by `garin_anchor`, for each species.
   Confirm anchors are sensibly placed.

### Validation block

The notebook must print and save the following numbers; we'll record them in
`outputs/logs/eda_summary.json`:

```json
{
  "human": {"n_subj": 113, "fc_mean_off_diag": "...", "lr_symmetry_median": "..."},
  "mouse": {"n_subj": 105, "fc_mean_off_diag": "...", "lr_symmetry_median": "..."},
  "anchor_fc_spearman": "..."
}
```

If `anchor_fc_spearman < 0.2`, **stop and reconsider.** The whole approach assumes
inter-anchor connectivity patterns are conserved across species; if they aren't, no
amount of OT machinery will save us.

---

## 5. Sanity check #1 — Garin-only FGW (`notebooks/03_garin_only_sanity.ipynb`)

This is the most important early experiment. **If this fails we stop and rethink.**

**Setup.** Take only the 42 anchor nodes per species (strip out type-2 grid nodes).
Compute cost matrices `C_h, C_m` from the 42×42 mean FC submatrices. Do **not** tell
the algorithm which nodes are paired. Run Fused-GW with no anchor supervision.

**Question.** Does the recovered coupling assign each mouse anchor to the correct
human anchor (or to the L/R partner), purely from connectivity structure?

```python
import ot, numpy as np
from retune.data import build_anndata
from retune.costs import correlation_distance

H = build_anndata("human"); M = build_anndata("mouse")
h_idx = H.var.query("garin_anchor").index.astype(int).values - 1   # 0-indexed
m_idx = M.var.query("garin_anchor").index.astype(int).values - 1

C_h = correlation_distance(H.uns["fc_mean"][np.ix_(h_idx, h_idx)])
C_m = correlation_distance(M.uns["fc_mean"][np.ix_(m_idx, m_idx)])

a = np.ones(42) / 42; b = np.ones(42) / 42
pi, log = ot.gromov.entropic_gromov_wasserstein(
    C_m, C_h, a, b, "square_loss", epsilon=5e-3, log=True,
)
```

**Evaluation.** True pairing maps `pairid_m == pairid_h` (and same hemisphere).
Compute:
- **Top-1 anchor accuracy**: fraction of mouse anchors whose argmax assignment is the
  correct human anchor.
- **Hemisphere accuracy**: fraction whose argmax is at least the correct hemisphere.
- **Pair accuracy** (ignoring L/R): fraction matched to correct `pairid` ignoring side.

**Pass criteria for moving on.** All three above 50% is fine; pair accuracy should
ideally exceed 70%. Top-1 of 100% is unlikely without anchor supervision and not
required at this stage.

**If it fails.** Try (a) Fisher-z FC, (b) thresholded-FC geodesic distance as cost,
(c) augmenting the cost with `xyz`-spatial term in a fused setup but *with both
species rescaled to a common bounding box first*. Record everything in
`outputs/logs/garin_only_sweep.json`.

---

## 6. Per-species node embeddings (`src/retune/features.py`)

We'll start dumb and add complexity only if needed.

### 6.1 Tier 1 — FC fingerprint

Each node's embedding = its row in the mean FC matrix, optionally Fisher-z'd, then
PCA'd to ~50 dims.

```python
def fc_fingerprint(adata, *, n_components=50, fisher_z=True) -> np.ndarray:
    fc = adata.uns["fc_z"] if fisher_z else adata.uns["fc_mean"]
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components, random_state=0).fit_transform(fc)
```

This is **per-species**, so the 50-dim spaces are not aligned. We'll align them in §7.

### 6.2 Tier 2 — per-subject FC stack

Stack each subject's FC row: `(n_nodes, n_subj × n_nodes)`. Compress with PCA. This
preserves cross-subject variability that the mean discards. Costs more memory.

### 6.3 Tier 3 — graph autoencoder (only if Tiers 1–2 underperform)

GraphSAGE / GAT on the thresholded FC graph, trained per-species with a
reconstruction objective. Yields embeddings that respect graph neighbourhoods.

### Validation

- For each tier, compute within-species L/R symmetry: do paired homologues sit close
  in embedding space? Median cosine similarity of L/R partners > 0.7 is the bar.
- Cluster anchors by region: do the 42 anchors form sensible clusters (limbic vs
  motor vs sensory)? Visualise with UMAP coloured by region.

---

## 7. Cost matrices and shared feature space (`src/retune/costs.py`)

### 7.1 Intra-species relational cost

```python
def correlation_distance(fc: np.ndarray) -> np.ndarray:
    """1 - r, clipped, zero-diagonal. Symmetric."""
    d = 1.0 - fc
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d.astype(np.float64)

def geodesic_fc_distance(fc: np.ndarray, *, threshold: float = 0.2) -> np.ndarray:
    """Threshold FC to a graph, take 1/|r| as edge weight, shortest-path distance."""
    import scipy.sparse as sp; from scipy.sparse.csgraph import shortest_path
    W = np.where(np.abs(fc) >= threshold, 1.0 / np.maximum(np.abs(fc), 1e-3), 0.0)
    np.fill_diagonal(W, 0.0)
    return shortest_path(sp.csr_matrix(W), directed=False)
```

We'll evaluate both. The geodesic option is more "structural" but adds a threshold
hyperparameter.

### 7.2 Cross-species feature cost (the *fused* term)

Two embedding spaces of dim 50 each, not natively aligned. Use the 42 Garin anchors
to learn a Procrustes / orthogonal map that brings mouse embeddings into the human
space:

```python
def align_embeddings(F_m, F_h, anchor_pairs):
    """anchor_pairs: list of (mouse_idx, human_idx)."""
    Xm = F_m[[p[0] for p in anchor_pairs]]
    Xh = F_h[[p[1] for p in anchor_pairs]]
    U, _, Vt = np.linalg.svd(Xm.T @ Xh, full_matrices=False)
    R = U @ Vt                         # orthogonal map mouse → human
    return F_m @ R, F_h, R
```

Then the cross-species feature cost is just:

```python
def feature_cost(F_m_aligned, F_h):
    from sklearn.metrics import pairwise_distances
    return pairwise_distances(F_m_aligned, F_h, metric="cosine")
```

### Validation

After alignment, the 42 anchors should sit on the diagonal of the cosine-similarity
matrix between the two embedding spaces. Mean cosine similarity of held-out anchors
(use 5-fold CV) should be > 0.5.

---

## 8. Fused (Unbalanced) GW (`src/retune/fgw.py`)

### 8.1 Balanced FGW first

```python
import ot, numpy as np

def fused_gw(C_m, C_h, M, *, alpha=0.5, epsilon=5e-3, max_iter=2000):
    """
    C_m: (n_m, n_m) intra-mouse cost
    C_h: (n_h, n_h) intra-human cost
    M:   (n_m, n_h) cross-species feature cost
    alpha in [0,1]; alpha=1 → pure GW, alpha=0 → pure W on features
    """
    a = np.ones(C_m.shape[0]) / C_m.shape[0]
    b = np.ones(C_h.shape[0]) / C_h.shape[0]
    pi, log = ot.gromov.entropic_fused_gromov_wasserstein(
        M=M, C1=C_m, C2=C_h, p=a, q=b,
        loss_fun="square_loss", alpha=alpha, epsilon=epsilon,
        max_iter=max_iter, log=True,
    )
    return pi, log
```

### 8.2 Anchor-constrained FGW

Two ways to inject the 42 known pairings:

**(a) Hard mask.** Set `M[i, j] = +∞` whenever node `i` is a mouse anchor and `j` is
a human anchor *not* in the corresponding pair. POT respects this through the cost.
Equivalent in the limit to a Lagrangian.

**(b) Soft prior on π.** Add a KL term to a prior coupling π₀:
`π₀[i,j] = 1` for known anchor pairs, `1/(n_m·n_h)` elsewhere; renormalised. Use
`ot.gromov.entropic_fused_gromov_wasserstein` with the `init` kwarg pointing at π₀,
or solve a "partial" variant.

Implement both, treat (a) as the default.

```python
def anchor_constrained_fgw(C_m, C_h, M, anchor_pairs, **kwargs):
    M = M.copy().astype(np.float64)
    m_anchors = {p[0] for p in anchor_pairs}
    h_anchors = {p[1] for p in anchor_pairs}
    pair_set  = set(anchor_pairs)
    BIG = 1e6
    for i in m_anchors:
        for j in h_anchors:
            if (i, j) not in pair_set:
                M[i, j] = BIG
    return fused_gw(C_m, C_h, M, **kwargs)
```

### 8.3 Unbalanced FGW

`ot.unbalanced.entropic_fused_unbalanced_gromov_wasserstein` (POT ≥ 0.9.4) — relaxes
the marginal constraints with KL penalties of strength `tau_a, tau_b`. Run this for
the full 1864 × 2094 problem.

### 8.4 Validation gates

- Solver returned without warning, π is non-negative, row sums ≈ a, col sums ≈ b
  (within tolerance for unbalanced).
- Anchor recovery on **held-out** anchors (5-fold CV over the 42 pairs):
  - top-1 accuracy on the 8–9 held-out mouse anchors per fold;
  - mean rank of correct human anchor in `argsort(π[i, :])`.
- Hemispheric symmetry of the recovered π: if you flip `pairid_h` for L/R, does π
  symmetrise? (A reasonable sanity check that the model isn't picking up a global
  L/R flip.)
- Hyperparameter sweep: `alpha ∈ {0.1, 0.3, 0.5, 0.7, 0.9}`, `epsilon ∈ {1e-3, 5e-3,
  1e-2}`, unbalanced `tau ∈ {1, 5, 50, ∞}`. Grid → JSON.

---

## 9. Evaluation & uncertainty (`src/retune/eval.py`)

### 9.1 Anchor metrics

```python
def anchor_topk(pi, anchor_pairs, k=1):
    correct = 0
    for i, j_true in anchor_pairs:
        if j_true in np.argsort(pi[i])[-k:]:
            correct += 1
    return correct / len(anchor_pairs)
```

### 9.2 Cross-validated anchor recovery

5-fold over the 21 pair IDs (so L/R stay together in the same fold). For each fold:
strip those anchors from supervision, run FGW, compute top-1, top-5 accuracy.

### 9.3 Hemispheric consistency

For each mouse node `i` with hemisphere `H(i)`, compute the marginal probability
mass of π that lands on human nodes of the same hemisphere:
`mass_same_hemi(i) = π[i, H_h == H(i)].sum()`. Median should exceed 0.7. (Note: the
*Garin anchor* L/R partners share `pairid`, so the model should learn this.)

### 9.4 Subject-bootstrap uncertainty

Resample 80% of subjects with replacement, recompute mean FC and π, repeat 100×.
Report per-pair stability:
`stability(i, j) = 1/B Σ_b 1[argmax π_b[i, :] == j]` for top-1, plus the standard
deviation of π entries. Replace the colleague's "add noise to FC" with this — it's
honest sampling variability.

### 9.5 Out-of-anchor sanity

Pick 5–10 mouse nodes whose nearest human assignments are well-known from
neuroimaging literature (e.g., mouse hippocampus → human hippocampus, mouse V1 → human
V1) but are *type-2 grid* nodes, not Garin anchors. Inspect manually that π does the
right thing.

---

## 10. (Optional) Phase 8 — Flow matching (`src/retune/flow.py`)

Only after §1–9 are working.

**Goal.** Train a continuous-time flow `v_θ(x, t)` that transports mouse-node
embeddings to human-node embeddings, using π from §8 as supervision. Yields:

- A deterministic, invertible map per mouse node into "human-space".
- Sampling-based uncertainty: perturb input, run forward, see distribution of outputs.
- Cycle consistency: `flow_h2m(flow_m2h(x))` ≈ x without an adversarial loss.

**Recipe (OT-CFM, Tong et al. 2023).**

1. Sample a coupling `(x_0, x_1) ∼ π` (mouse, human) for many node pairs.
2. Sample `t ∼ U(0, 1)`, `x_t = (1-t)·x_0 + t·x_1`.
3. Target velocity `u_t = x_1 - x_0`.
4. Train `v_θ(x_t, t)` with MSE loss against `u_t`.
5. At inference, integrate `v_θ` with an ODE solver from `x_0` to `t=1`.

```python
# pseudo-code, torch
class VField(nn.Module):
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),  nn.SiLU(),
            nn.Linear(hidden, dim),
        )
    def forward(self, x, t):
        return self.net(torch.cat([x, t.expand(*x.shape[:-1], 1)], -1))

def train_otcfm(emb_m, emb_h, pi, *, epochs=1000):
    ...
```

**Validation for the flow.** Anchor recovery via flow's nearest-neighbour assignment
should at least match π's. Cycle consistency error should converge below some
threshold (e.g., < 0.05 in cosine distance).

---

## 11. Risks & open questions

1. **Cost matrix definition.** `1 - r` vs geodesic vs partial-correlation can change
   the FGW solution materially. We'll evaluate all three on the Garin-only sanity.
2. **Subject averaging vs subject-aware cost.** Mean FC throws away variability.
   Alternative: distributional GW with per-subject cost tensors. Defer until baseline
   is working.
3. **Mass imbalance.** 1864 vs 2094 nodes. Default to unbalanced; diagnose where mass
   is dropped on the human side (PFC expansion?) as qualitative validation.
4. **Local minima in GW.** The objective is non-convex. Use multiple restarts and
   anchor-based init. Compare against a random-init baseline.
5. **Coordinate space mismatch.** Mouse mm and human mm aren't comparable. We'll only
   use spatial features *after* aligning each species' bounding box to a unit cube,
   and treat spatial cost as a soft prior (`alpha` controls weight).
6. **Subject quality.** Some ABIDE / Grandjean subjects are noisy. EDA in §4 will
   flag outliers; we may exclude before computing mean FC.
7. **Partial homology.** Some nodes genuinely have no counterpart. Unbalanced GW
   handles this on the marginal level, but we may need a more explicit mechanism
   (e.g., a "null" target node) if biology demands.

---

## 12. Working order

| # | Block | Output | Pass criterion |
|---|---|---|---|
| 1 | Env + skeleton repo | venv green, tests collect | `pytest --collect-only` lists planned tests |
| 2 | `data.py` + tests | `human.h5ad`, `mouse.h5ad` cached | §3.4 tests pass |
| 3 | EDA notebook | `eda_summary.json` | anchor FC Spearman > 0.2 |
| 4 | **Garin-only FGW** | `garin_only_sweep.json` | pair accuracy > 70% |
| 5 | Embeddings | `features_*.npy` | L/R median cosine > 0.7 |
| 6 | Cost matrices + alignment | `costs.npz` | held-out anchor cosine > 0.5 |
| 7 | Full FGW (balanced + anchor) | `pi_balanced.npy` | held-out anchor top-1 > 50% |
| 8 | Unbalanced FGW + sweep | `pi_unbalanced.npy`, `sweep.json` | matches/beats balanced |
| 9 | Evaluation + uncertainty | `eval_report.html` | bootstrap stable, hemi-consistent |
| 10 | (Optional) OT-CFM flow | `flow.pt` | cycle err < 0.05 |

We do not advance to step N+1 until step N's pass criterion is met.

---

## 13. Snippets we will lift / adapt from moscot

The moscot codebase patterns worth borrowing (re-implemented in pure POT/numpy):

- **AnnData-as-input convention** for the data layer (already adopted in §3).
- **Problem / Solver split**: a `CrossSpeciesProblem` class exposing `prepare()`,
  `solve()`, `pull()/push()` (inherit nothing, just match the API). Useful if we
  later add other formulations (e.g., per-subject coupling).
- **Anchor-as-prior pattern** from `moscot.problems.cross_modality.TranslationProblem`
  — they use a "joint attribute" mechanism that's basically partial constraints on π.
- **Cost-matrix factory** abstraction: a pluggable `cost_fn(adata, **kwargs)
  -> (n×n)` so we can swap correlation / geodesic / partial-correlation cleanly.

We don't borrow the JAX backend, the low-rank GW, or the gigascale sinkhorn — none of
those are needed at our problem size.

---

## 14. What we'll write up if it works

A short methods note: data, FGW formulation, anchor handling, validation. Position
explicitly against the colleague's pipeline as a soft, supervised, principled
alternative, and report the same metrics the colleague's pipeline would report
(anchor recovery, bootstrap stability) so comparison is direct.

If the colleague's pipeline outperforms us on Garin recovery — which would surprise
me — that's still a useful negative result.
