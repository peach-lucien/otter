"""Cell-type ABUNDANCE / composition test through π (real cell types, not markers).

The marker test correlates single-gene smooth maps (weak, mean r=0.089). The
contrast reframe (03) already recovered a real positive — the excitatory↔inhibitory
axis survives a spin null (r=+0.262, p=0.001). This script is the higher-resolution
upgrade flagged as future work: instead of ~23 gene-marker proxies, use the ACTUAL
per-region cell-type ABUNDANCE tables and ask whether π maps mouse regions to human
regions of matching cell-type *composition*.

  Mouse  : Yao 2023 Allen Brain Cell (ABC) atlas — per-CCFv3-region class/subclass
           abundance (fraction of cells per type).  Access via `abc_atlas_access`.
  Human  : Siletti 2023 human brain cell atlas — per-dissection-region class/subclass
           abundance.  Access via `cellxgene-census` (CELLxGENE) or the published
           supplementary cell-metadata table.

Pipeline (once the two abundance tables are present):
  1. Build composition vectors C_mouse[region, type] and C_human[region, type] over a
     shared class/subclass label set.
  2. Map each abundance table's regions to HOMER's parcels (CCFv3 centroids for mouse;
     MNI/dissection centroids for human) → per-parcel composition.
  3. For each cell type t: route the mouse abundance map of t through π and correlate
     with the human abundance map of t — but TEST THE CONTRAST/COMPOSITION, not the
     single smooth map: (a) per-type, against the fair translation-spin null; and
     (b) compositional — does π map mouse parcels to human parcels whose dominant
     cell class matches (argmax-class agreement, scored vs a spin null)?

ENVIRONMENT NOTE: this needs the two cell atlases (multi-GB) + abc_atlas_access /
cellxgene-census, which are NOT available in the Cowork sandbox (disk-limited, and
the packages aren't on the mirror). Run locally:

    pip install abc_atlas_access cellxgene-census
    # mouse: download ABC atlas cell metadata, group by CCF region + subclass
    # human: open CELLxGENE census 'homo_sapiens', filter Siletti 2023, group by
    #        dissection region + subclass
    python experiments/biccn_2023_cell_types/04_abundance_composition.py \
        --mouse-abundance <yao2023_region_by_subclass.csv> \
        --human-abundance <siletti2023_region_by_subclass.csv>

The script below implements steps 1–3 given those two CSVs (rows=regions with
centroid columns x,y,z; remaining columns = per-type fractions). It exits with this
recipe if they are absent, so the analysis is reproducible the moment the data lands.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached                              # noqa: E402
from homer.eval.nulls import translation_spin_null, _route_normalized  # noqa: E402
from scipy.spatial import cKDTree                               # noqa: E402
from scipy.stats import pearsonr                                # noqa: E402

RECIPE = __doc__


def _need(path, what):
    if path is None or not Path(path).exists():
        print(f"\n[04_abundance_composition] MISSING {what}.\n{RECIPE}")
        raise SystemExit(0)


def parcel_composition(abund_df, parcel_xyz):
    """Assign each HOMER parcel the composition of its nearest abundance-table region."""
    cents = abund_df[["x", "y", "z"]].to_numpy(float)
    type_cols = [c for c in abund_df.columns if c not in ("region", "x", "y", "z")]
    frac = abund_df[type_cols].to_numpy(float)
    _, nn = cKDTree(cents).query(parcel_xyz)
    return frac[nn], type_cols                                   # (n_parcels, n_types)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mouse-abundance", default=None)
    ap.add_argument("--human-abundance", default=None)
    args = ap.parse_args()
    _need(args.mouse_abundance, "mouse Yao-2023 region×subclass abundance CSV")
    _need(args.human_abundance, "human Siletti-2023 region×subclass abundance CSV")

    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs/anndata"))
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    pi = np.load(ROOT / "outputs/coupling/pi_fc_plus_SC_with_all_packs.npy")
    m_xyz = M.var[["x", "y", "z"]].to_numpy(float)
    h_xyz = H.var[["x", "y", "z"]].to_numpy(float)

    m_comp, m_types = parcel_composition(pd.read_csv(args.mouse_abundance), m_xyz)
    h_comp, h_types = parcel_composition(pd.read_csv(args.human_abundance), h_xyz)
    shared = [t for t in m_types if t in h_types]
    mi = [m_types.index(t) for t in shared]; hi = [h_types.index(t) for t in shared]
    m_comp, h_comp = m_comp[:, mi], h_comp[:, hi]
    print(f"shared cell types: {len(shared)}")

    # per-type translation-spin test
    results = {}
    for j, t in enumerate(shared):
        spin = translation_spin_null(m_comp[:, j], h_comp[:, j], pi, m_xyz,
                                     n_trials=500, seed=j)
        results[t] = {"pearson_r": float(pearsonr(_route_normalized(m_comp[:, j], pi),
                                                   h_comp[:, j])[0]),
                      "spin_p": spin["p_translation_spin"]}
        print(f"  {t:<28} r={results[t]['pearson_r']:+.3f}  spin p={results[t]['spin_p']:.3f}")

    # compositional argmax-class agreement
    pred_dom = np.array([_route_normalized(m_comp[:, j], pi) for j in range(len(shared))]).T.argmax(1)
    obs_dom = h_comp.argmax(1)
    ok = np.isfinite(pred_dom)
    agree = float((pred_dom == obs_dom).mean())
    print(f"\ndominant-class argmax agreement: {agree:.3f}")
    out = {"shared_types": shared, "per_type": results, "argmax_class_agreement": agree}
    (ROOT / "outputs/logs/biccn_abundance_composition.json").write_text(json.dumps(out, indent=2))
    print("Wrote outputs/logs/biccn_abundance_composition.json")


if __name__ == "__main__":
    main()
