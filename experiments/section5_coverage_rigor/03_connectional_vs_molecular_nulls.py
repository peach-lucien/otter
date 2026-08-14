"""The homology boundary is CONNECTIONAL, not MOLECULAR — with proper spin nulls.

`make_boundary_connectional.py` shows the descriptive picture: OTTER coverage (a
connectivity + anchor quantity) has a large association-cortex deficit, while an
independent transcriptomic similarity (51-gene homolog panel, best mouse match per
human parcel) has essentially none. This script attaches significance via the repo's
spatial-autocorrelation (spin) null, and tests the DISSOCIATION directly.

Three tertile statistics along the sensorimotor->association (myelin) axis, all
z-scored across cortical parcels so they are on the same scale:
  (1) coverage gap (connectivity)            -> expected LARGE, spin-significant
  (2) transcriptomic-similarity gap (molecular) -> expected ~0, spin-NS
  (3) DIFFERENCE map (coverage_z - transcriptomic_z) gap -> the dissociation test:
      is the connectional deficit specifically greater than the molecular one over
      association cortex, beyond spatial autocorrelation?

Run: cd otter && PYTHONPATH=src python experiments/section5_coverage_rigor/03_connectional_vs_molecular_nulls.py
Writes outputs/logs/section5_connectional_vs_molecular.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached
from otter.eval.nulls import _haar_rotation
from scipy.spatial import cKDTree

DATA = ROOT / "data_external"
N_SPIN = 1000


def tertile_gap_spin(sig, axis, coords, n_trials=N_SPIN, seed=0):
    """sensorimotor(high-axis) minus association(low-axis) gap of `sig`, spin p."""
    o = np.argsort(axis); t = len(o) // 3; lo, hi = o[:t], o[-t:]
    obs = sig[hi].mean() - sig[lo].mean()
    c = coords - np.nanmean(coords, 0)
    sph = c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-12, None)
    rng = np.random.default_rng(seed); null = np.empty(n_trials)
    for i in range(n_trials):
        _, perm = cKDTree(sph @ _haar_rotation(rng).T).query(sph)
        sp = sig[perm]; null[i] = sp[hi].mean() - sp[lo].mean()
    an = np.abs(null)
    return {"gap_sd": float(obs), "p_spin": float((np.sum(an >= abs(obs)) + 1) / (n_trials + 1)),
            "null_abs_p95": float(np.percentile(an, 95))}


def main():
    # independent transcriptomic best-match per human parcel (51-gene homolog panel)
    mg = np.load(DATA / "mouse_genes_aligned.npy"); hg = np.load(DATA / "human_genes_aligned.npy")
    def zc(A):
        A = A.astype(float); mu = np.nanmean(A, 0); sd = np.nanstd(A, 0); sd[sd < 1e-9] = 1
        return (A - mu) / sd
    Mz, Hz = zc(mg), zc(hg); mok = np.isfinite(Mz).all(1); hok = np.isfinite(Hz).all(1)
    Mc = Mz[mok] - Mz[mok].mean(1, keepdims=True); Mn = Mc / np.linalg.norm(Mc, axis=1, keepdims=True)
    Hc = Hz - np.nanmean(Hz, 1, keepdims=True)
    Hn = np.full_like(Hc, np.nan); Hn[hok] = Hc[hok] / np.linalg.norm(Hc[hok], axis=1, keepdims=True)
    best = np.nanmax(Hn @ Mn.T, axis=1)                       # molecular similarity per human parcel

    bk = json.load(open(ROOT / "outputs/logs/buckner_krienen_2013_tethering.json"))
    cov = np.array(bk["coverage_per_parcel"]); mye = np.array(bk["myelin_per_parcel"], dtype=float)
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    xyz = H.var[["x", "y", "z"]].to_numpy(float)

    m = np.isfinite(cov) & np.isfinite(best) & np.isfinite(mye)
    covz = (cov[m] - cov[m].mean()) / cov[m].std()
    bestz = (best[m] - best[m].mean()) / best[m].std()
    diff = covz - bestz                                        # connectional minus molecular
    axis, coords = mye[m], xyz[m]

    g_cov = tertile_gap_spin(covz, axis, coords)
    g_mol = tertile_gap_spin(bestz, axis, coords)
    g_dif = tertile_gap_spin(diff, axis, coords)

    print(f"n cortical parcels = {m.sum()}")
    print(f"connectivity coverage gap  = {g_cov['gap_sd']:+.2f} SD   spin p = {g_cov['p_spin']:.4f}")
    print(f"transcriptomic similarity gap = {g_mol['gap_sd']:+.2f} SD   spin p = {g_mol['p_spin']:.4f}")
    print(f"DISSOCIATION (cov - molecular) gap = {g_dif['gap_sd']:+.2f} SD   spin p = {g_dif['p_spin']:.4f}")

    out = {"n_cortical_parcels": int(m.sum()), "n_spin": N_SPIN,
           "connectivity_coverage_gap": g_cov, "transcriptomic_similarity_gap": g_mol,
           "dissociation_gap": g_dif,
           "interpretation": ("Coverage (connectional) collapses over association cortex; transcriptomic "
                              "similarity (molecular) does not. The dissociation-gap spin p tests whether "
                              "the connectional deficit specifically exceeds the molecular one.")}
    (ROOT / "outputs/logs/section5_connectional_vs_molecular.json").write_text(json.dumps(out, indent=2))
    print("wrote outputs/logs/section5_connectional_vs_molecular.json")


if __name__ == "__main__":
    main()
