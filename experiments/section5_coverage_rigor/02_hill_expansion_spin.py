"""External validation: OTTER coverage vs Hill 2010 evolutionary cortical expansion,
tested with the repo's spin null. Asks whether the human cortex OTTER leaves
uncovered by the mouse is the cortex that expanded most in evolution.

Data: neuromaps `source='hill2010', desc='evoexp'` (fsLR 164k, RIGHT hemisphere only).
The map is resampled to Schaefer-400 and compared to OTTER's per-Schaefer coverage,
with `otter.eval.nulls.spin_null` (continuous) + a tertile-gap spin (most- vs least-
expanded third). Centroids = mean OTTER parcel MNI coord per Schaefer region, so the
spin runs in the repo's own coordinate frame.

CAVEATS: (i) Hill is macaque->human expansion, OTTER is mouse->
human, so primary areas (e.g. V1) may deviate; (ii) right hemisphere only; (iii) an
in-sandbox check gave continuous rho=-0.21 (spin p~0.10) and tertile gap 3.66 log units
(spin p~0.15) -> directionally consistent but NOT spin-significant at Schaefer-400. Run
this to confirm at full rigor and, if desired, at parcel resolution.

Requires: pip install neuromaps netneurotools ; Connectome Workbench (wb_command) on PATH.
Run: cd otter && PYTHONPATH=src python experiments/section5_coverage_rigor/02_hill_expansion_spin.py
Writes outputs/logs/section5_hill_expansion.json
"""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import numpy as np, nibabel as nib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance
from otter.eval.nulls import spin_null, _haar_rotation
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

N_SPIN = 1000


def tertile_gap_spin(cov, axis, coords, n_trials=N_SPIN, seed=0):
    o = np.argsort(axis); t = len(o) // 3; lo, hi = o[:t], o[-t:]
    obs = cov[lo].mean() - cov[hi].mean()                 # least-expanded minus most-expanded coverage
    c = coords - np.nanmean(coords, 0); sph = c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-12, None)
    rng = np.random.default_rng(seed); null = np.empty(n_trials)
    for i in range(n_trials):
        _, perm = cKDTree(sph @ _haar_rotation(rng).T).query(sph); cp = cov[perm]
        null[i] = cp[lo].mean() - cp[hi].mean()
    an = np.abs(null)
    return {"gap_observed": float(obs), "p_spin": float((np.sum(an >= abs(obs)) + 1) / (n_trials + 1)),
            "null_abs_p95": float(np.percentile(an, 95))}


def hill_per_schaefer():
    """Return {schaefer_id: expansion} for the right hemisphere via nearest-vertex
    resampling of the fsLR-164k Hill map onto the Schaefer-400 fsLR-32k parcellation.
    (Area-accurate alternative: neuromaps.parcellate.Parcellater with wb_command.)"""
    from neuromaps.datasets import fetch_annotation
    from netneurotools.datasets import fetch_schaefer2018
    NM = Path(fetch_annotation(source="hill2010", desc="evoexp")).parent  # triggers download
    atl = NM.parents[3] / "atlases" / "fsLR"
    hillf = glob.glob(str(NM / "*hemi-R*.gii"))[0]
    hill = np.asarray(nib.load(hillf).agg_data(), float)
    sc = lambda p: np.asarray(nib.load(p).agg_data()[0])
    sph164 = sc(atl / "tpl-fsLR_den-164k_hemi-R_sphere.surf.gii")
    sph32 = sc(atl / "tpl-fsLR_den-32k_hemi-R_sphere.surf.gii")
    import os
    kw = {"data_dir": os.environ["NNT_DATA_DIR"]} if os.environ.get("NNT_DATA_DIR") else {}
    lab = np.asarray(nib.load(fetch_schaefer2018("fslr32k", **kw)["400Parcels17Networks"]).get_fdata()).ravel().astype(int)
    labR = lab[sph32.shape[0]:2 * sph32.shape[0]]
    _, idx = cKDTree(sph32).query(sph164)
    lab164 = labR[idx]
    ids = [k for k in range(201, 401) if (labR == k).any()]
    return {k: hill[(lab164 == k) & np.isfinite(hill) & (hill != 0)].mean()
            for k in ids if ((lab164 == k) & (hill != 0)).any()}


def main():
    pi = load_pi()
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    node_region = np.asarray(json.loads((ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    col_mass = pi.sum(0)

    hill = hill_per_schaefer()
    ids = sorted(hill)
    exp = np.array([hill[k] for k in ids])
    # MASS-NORMALISED MEAN, not sum: summing makes coverage scale with the number of
    # OTTER parcels in a Schaefer region (rho = 0.35 with parcel count), which is a
    # size confound, not a biological signal.
    cov = np.array([np.log10(col_mass[node_region == k].mean()) if (node_region == k).any() else np.nan for k in ids])
    cen = np.array([xyz[node_region == k].mean(0) if (node_region == k).any() else [np.nan] * 3 for k in ids])
    m = np.isfinite(exp) & np.isfinite(cov) & np.isfinite(cen).all(1)
    exp, cov, cen = exp[m], cov[m], cen[m]

    cont = spin_null(cov, exp, cen, n_trials=N_SPIN)
    tert = tertile_gap_spin(cov, exp, cen)
    rho = spearmanr(cov, exp).statistic
    print(f"n R-hemi Schaefer regions = {len(cov)}")
    print(f"[continuous] coverage vs Hill expansion: Pearson r = {cont['r_observed']:+.3f} (Spearman {rho:+.3f})  spin p = {cont['p_spin']:.4f}")
    print(f"[tertile]    least- minus most-expanded coverage gap = {tert['gap_observed']:.2f} log units  spin p = {tert['p_spin']:.4f}")

    out = {"map": "hill2010 evoexp (fsLR164k, R hemi)", "n_regions": int(len(cov)),
           "spearman": float(rho), "coverage_vs_expansion_continuous": cont,
           "coverage_vs_expansion_tertile": tert,
           "caveats": ["macaque->human vs mouse->human", "right hemisphere only",
                       "Schaefer-400 resolution; nearest-vertex resample"]}
    out.update(pi_provenance())   # which coupling produced these numbers
    (ROOT / "outputs/logs/section5_hill_expansion.json").write_text(json.dumps(out, indent=2))
    print("wrote outputs/logs/section5_hill_expansion.json")


if __name__ == "__main__":
    main()
