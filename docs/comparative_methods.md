# Comparative methods — what didn't work, and what we now believe

After the colleague-review fixes (May 2026) revealed that the production
`fc_plus_SC` model is statistically tied with `fc_only` and that full-space
anchor recovery is only 2.4% (vs the 81% restricted-set ranking we report
prominently), we tested two alternatives suggested in critique:

1. **FUGW** — Fused Unbalanced Gromov-Wasserstein (Thual et al. 2022 NeurIPS,
   `fugw` PyPI package). Tests the hypothesis: "the per-voxel mapping is weak
   because semirelaxed FGW lets the human marginal float, so the model parks
   mass on convenient grid nodes near anchors rather than on the anchors
   themselves. FUGW's KL marginal control should fix this."

2. **Knox 2019 leaf-level cortical SC** (instead of Allen summary-structure
   SC). Tests the hypothesis: "SC isn't helping because all 47 visual cortex
   parcels share one SC vector at summary-structure resolution. A finer SC
   should distinguish parcels and help anchor recovery."

Both implementations are kept in the repository as **comparative additions**,
not replacements. They can be re-run any time.

---

## 1. FUGW (`homer.models.FUGWModel`)

`pip install fugw torch` — available as an optional model class. Same
`FGWModel` API as the existing classes; full integration with all evaluation
pipelines.

### Key parameters

- `rho_s` (source/mouse marginal relaxation strength)
- `rho_t` (target/human marginal relaxation strength)

`rho → ∞` recovers balanced FGW (forces marginal); `rho → 0` recovers fully
unconstrained mass.

### Headline results (visual network held out, fc_plus_SC settings)

| Model | restricted top-1 | full top-1 | mean rank /2094 | row max conc. | uncovered humans |
|---|---|---|---|---|---|
| `MultimodalFGW` (semirelaxed) | 50% | 0% | 682 | 0.977 | 762 (36%) |
| `FUGWModel` (rho_s=rho_t=1)   | 50% | 0% | 511 | 0.077 | **0** |
| `FUGWModel` (rho_s=100, rho_t=1) | 50% | 0% | 512 | 0.080 | 0 |
| `FUGWModel` (rho_s=1, rho_t=100) | 50% | 0% | 511 | 0.080 | 0 |

### What we learned

- FUGW with default `rho_s=rho_t=1.0` produces a fundamentally different π:
  - **Soft (mean row max concentration 0.077 vs 0.977)** — the coupling is
    genuinely probabilistic instead of essentially one-hot.
  - **Zero uncovered human nodes** (vs 762 = 36% in MultimodalFGW) — every
    human parcel receives some mass.
  - **Mean rank improved 682 → 511** — the correct anchor is somewhat closer
    to the top of π's row, but not enough to flip the binary top-1.

- Sweeping `rho_s` and `rho_t` (1→100) does NOT measurably affect any
  recovery metric. The marginal-control parameters change π's calibration
  but not its ability to identify the correct held-out anchor.

- **Restricted-anchor ranking and full-space top-1 are unchanged**.
  The semirelaxed-vs-FUGW distinction matters for *probabilistic mass
  distribution* and *coverage*, but not for *anchor identification*.

### Honest interpretation

FUGW addresses one real critique (the coverage problem: 36% of human nodes
were uncovered in the production solve) but not the other (the per-voxel
mapping problem: full-space top-1 ≈ 0% on hard networks). The two issues
have different root causes; FUGW only attacks the first one.

If a downstream task wants a probabilistic π with full human coverage,
FUGW is the right model. For anchor identification it's no better than the
existing semirelaxed formulation.

### How to reproduce

```python
from homer.models import FUGWModel
from homer.data import load_cached
import numpy as np

M, _ = load_cached('mouse', cache_dir='outputs/anndata')
H, _ = load_cached('human', cache_dir='outputs/anndata')
costs = np.load('outputs/anndata/full_costs.npz')

model = FUGWModel(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                  epsilon=5e-3, xyz_weight=0.5,
                  rho_s=1.0, rho_t=1.0,
                  nits_bcd=10, nits_uot=500)
model.fit(M, H, Cm_SC=costs['Cm_SC'], Ch_SC=costs['Ch_SC'])
print(model.evaluate(eval_kind='anchor'))
```

---

## 2. Knox 2019 leaf-level cortical SC

The Allen summary-structure SC we use has only **192 unique row-fingerprints
across our 1864 mouse parcels** — i.e., on average 9.7 parcels share an
identical SC profile. For example, `L_Visual striate cortex` and
`R_Visual pre and extra striate cortex` literally have the same SC vector.
The hypothesis: a finer-grained SC dataset should distinguish these parcels.

We downloaded the [Knox 2019 voxel-resolved model](https://download.alleninstitute.org/publications/A_high_resolution_data-driven_model_of_the_mouse_connectome/)
and used its **leaf-level cortical SC** (43 cortical regions × 43 ipsi targets) to
augment our SC. Mapped 22 of 42 anchors (11 of 21 pair_ids) to Knox cortical
leaves via hand-curated region-name match (`pipeline/00_external/06_knox_sc.py`).

### Resolution improvement

The 192-vs-469 figure we initially reported compared apples to oranges: 192
was the unique row count of the **raw** Allen streamline-density matrix
(`data_external/mouse_sc.npy`), 469 was the unique row count of the
post-processed **cost** matrix `Cm_SC_knox`. The fair comparison is
cost-vs-cost:

| SC source | Unique row-fingerprints (out of 1864) at the *cost-matrix* level |
|---|---|
| Allen summary-structure → `Cm_SC` | **454** |
| Knox-augmented → `Cm_SC_knox` | **469** |

So at the level the FGW solver actually sees, Knox adds **15 new fingerprints
(1.03×)**, not 2.4×. The log1p + correlation-distance + max-normalisation
pipeline applied to both sources collapses much of the apparent advantage of
having raw Knox vectors. (The raw-SC count, 192 vs 1864 parcels, looked dire
because many parcels share an SC vector identically — but the cost transform
already smooths most of that out via correlation across the full row.)

### Recovery results — DOES it help anchor identification?

This table is the **fair** version: both `Cm_SC` and `Cm_SC_knox` are
normalised to `[0, 1]` so the only thing changing between configs is the SC
content, not its scale. (An earlier version of this section used a
non-normalised `Cm_SC_knox` with range `[0, 1.32]`, which silently
over-weighted SC by ~30% and produced spuriously different mean-rank numbers.
Re-run with `pipeline/05e_knox_vs_standard_sc.py --recompute`.)

All 11 networks, leave-one-network-out, MultimodalFGW with `fc_weight=0.7`,
`sc_weight=0.3`, `xyz_weight=0.5`, `epsilon=5e-3`:

| Network | n | Allen SC mean rank | Knox SC mean rank | Δ rank |
|---|---|---|---|---|
| auditory       |  2 |  324 |  324 | +0.0 |
| brainstem      |  4 |  420 |  419 | -1.0 |
| frontal_dmn    |  2 |  254 |  253 | -0.5 |
| frontoparietal |  2 |  226 |  226 | +0.0 |
| limbic         |  6 |  164 |  164 | +0.0 |
| olfactory      |  2 |   66 |   66 | +0.0 |
| salience       |  4 |   30 |   30 | -0.2 |
| sensorimotor   |  4 |   69 |   69 | +0.0 |
| subcortical    | 10 |   18 |   18 | +0.0 |
| temporal_dmn   |  2 |  470 |  470 | +0.0 |
| visual         |  4 |  682 |  680 | -2.5 |
| **Weighted (n=42)** | **42** | **205.9** | **205.5** | **-0.4** |

| Aggregate metric (n=42) | Allen | Knox |
|---|---|---|
| Full-space top-1                  | 2.4%  | 2.4% |
| Full-space top-5                  | 11.9% | 11.9% |
| `frac_argmax_is_anchor`           | 4.8%  | 4.8% |

### What we learned

**Knox-leaf-level SC produces statistically indistinguishable recovery
numbers when scaled correctly.** Top-1, top-5, and argmax-is-anchor are
identical at the aggregate level; mean rank improves by 0.4 out of 2094
(noise). The 1.03× cost-matrix resolution gain — itself much smaller than
the raw-SC gain we initially reported — does not translate into anchor
identification.

### Honest interpretation

Combined with FUGW's null result, this strongly suggests **the bottleneck
is not in the modality data** but in either:

1. **Information redundancy**: SC at parcel level is highly correlated with
   FC + xyz, so adding richer SC adds no orthogonal information.
2. **The OT objective itself**: anchor supervision via the M cost matrix
   dominates everything. Held-out anchors can't be reliably placed because
   the model has no information about *where they should go* once the
   supervision is removed; the structural cost provides only weak guidance.
3. **Per-voxel correctness is genuinely impossible** at this dataset/model
   scale: with 1864 mouse parcels and 2094 human parcels, there's no
   1-to-1 correspondence to find. The model lands on grid neighbours, and
   that's a fundamentally honest answer.

### Path forward

The next experiment that would actually test interpretation #3 would be to
**get more anchors** (Garin's 42 is the bottleneck on supervision).
Roadmap items V1 (compare to colleague's spectral pipeline) and E5 (validate
against published cross-species correspondences from Beauchamp 2022 etc.)
are the way to externalize the question rather than keep iterating on
modalities.

### How to reproduce

```bash
# Build the Knox-augmented SC (depends on data_external/knox_sc/* — see README)
PYTHONPATH=src python pipeline/00_external/06_knox_sc.py

# All-network LONO comparison vs the Allen baseline:
PYTHONPATH=src python pipeline/05e_knox_vs_standard_sc.py --recompute

# Then use it in any model:
import numpy as np
from homer.data import load_cached
from homer.models import MultimodalFGW
M, _ = load_cached('mouse', cache_dir='outputs/anndata')
H, _ = load_cached('human', cache_dir='outputs/anndata')
costs = np.load('outputs/anndata/full_costs.npz')

model = MultimodalFGW(use_sc=True, sc_weight=0.3, fc_weight=0.7,
                      epsilon=5e-3, xyz_weight=0.5)
model.fit(M, H, Cm_SC=costs['Cm_SC_knox'], Ch_SC=costs['Ch_SC'])
```

---

## Summary recommendation

**Both FUGW and Knox-leaf SC are kept in the codebase as comparative methods
but neither becomes production.** The colleague's specific critiques —
SC isn't helping; the model identifies anchor candidates but not per-voxel
homologies — were correct, AND the natural fixes (better SC resolution,
better marginal control) don't help either.

This is an **important convergent negative**: it bounds where the problem
is. The bottleneck is not in modality resolution and not in marginal
control; it's in the underlying ambiguity of cross-species correspondence
at the parcel level given only 42 anchor constraints.

Future work should focus on:
- Adding more anchor pairs (manual curation or external published correspondences)
- External validation of the existing π against published cross-species maps
  (Beauchamp 2022 etc. — roadmap E5)
- Comparison vs alternative methods (spectral / manifold alignment, seeded
  graph matching — roadmap V1) to determine whether OT is even the right
  framework
