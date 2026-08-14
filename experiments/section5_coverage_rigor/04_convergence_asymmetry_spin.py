"""Directional asymmetry / convergence of the coupling, with a spin null.

Two faces of human cortical reorganisation:
  - ABSENCE  : human territory that receives almost no mouse mass (coverage).
  - CONVERGENCE : human territory that pools MANY mouse sources (this script).

Per human parcel, the effective number of mouse sources = exp(entropy of the
column-normalised coupling). Mouse->human is near one-to-one (forward effective
targets ~1-2); human->mouse is heavier-tailed (reverse effective sources ~3), i.e.
some human cortex integrates several mouse regions. This script tests whether the
convergence is organised along the sensorimotor->association (myelin) axis with the
repo's spin null, and whether convergence tracks (low) coverage.

Run: cd otter && PYTHONPATH=src python experiments/section5_coverage_rigor/04_convergence_asymmetry_spin.py
Writes outputs/logs/section5_convergence_asymmetry.json
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance
from otter.eval.nulls import spin_null, _haar_rotation
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

DATA = ROOT / "data_external"
N_SPIN = 1000


def eff_number(P, axis):
    """Effective number of contributors along `axis` = exp(Shannon entropy)."""
    P = P / np.clip(P.sum(axis=axis, keepdims=True), 1e-300, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.nansum(np.where(P > 0, P * np.log(P), 0.0), axis=axis)
    return np.exp(H)


def tertile_gap_spin(sig, ax, coords, n=N_SPIN, seed=0):
    o = np.argsort(ax); t = len(o) // 3; lo, hi = o[:t], o[-t:]
    obs = sig[hi].mean() - sig[lo].mean()
    c = coords - np.nanmean(coords, 0); sph = c / np.clip(np.linalg.norm(c, axis=1, keepdims=True), 1e-12, None)
    rng = np.random.default_rng(seed); null = np.empty(n)
    for i in range(n):
        _, perm = cKDTree(sph @ _haar_rotation(rng).T).query(sph); s = sig[perm]
        null[i] = s[hi].mean() - s[lo].mean()
    an = np.abs(null)
    return {"gap": float(obs), "p_spin": float((np.sum(an >= abs(obs)) + 1) / (n + 1))}


def main():
    pi = load_pi()
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    node_region = np.asarray(json.loads((DATA / "human_sc_meta.json").read_text())["node_region"], int)

    eff_targets = eff_number(pi, axis=1)                        # per mouse parcel (forward)
    eff_sources = eff_number(pi, axis=0)                        # per human parcel (reverse = convergence)
    coverage = np.log10(np.maximum(pi.sum(0), 1e-300))
    print(f"forward eff targets: mean {eff_targets.mean():.2f} median {np.median(eff_targets):.2f}")
    print(f"reverse eff sources: mean {eff_sources.mean():.2f} median {np.median(eff_sources):.2f}")

    myelin_reg = {}
    with open(DATA / "fulcher_2019_gradients/human_myelinmap_schaefer400_HOMERorder.csv") as f:
        for row in csv.DictReader(f):
            myelin_reg[int(row["otter_region_id"])] = float(row["t1t2_myelin"])
    myelin = np.array([myelin_reg.get(r, np.nan) for r in node_region])

    m = np.isfinite(myelin)
    conv, mye, cov, coords = eff_sources[m], myelin[m], coverage[m], xyz[m]

    conv_axis = tertile_gap_spin(conv, mye, coords)             # convergence along sensorimotor->association
    conv_cont = spin_null(conv, mye, coords, n_trials=N_SPIN)
    rho_cc = spearmanr(conv, cov).statistic                    # convergence vs coverage

    print(f"convergence vs myelin axis: continuous r = {conv_cont['r_observed']:+.3f} spin p = {conv_cont['p_spin']:.4f}")
    print(f"convergence tertile gap (sensorimotor - association) = {conv_axis['gap']:+.2f}  spin p = {conv_axis['p_spin']:.4f}")
    print(f"convergence vs coverage: Spearman rho = {rho_cc:+.3f}  (are pooled-source parcels also low-mass?)")

    out = {"forward_eff_targets_mean": float(eff_targets.mean()),
           "reverse_eff_sources_mean": float(eff_sources.mean()),
           "convergence_vs_myelin_continuous": conv_cont,
           "convergence_vs_myelin_tertile": conv_axis,
           "convergence_vs_coverage_spearman": float(rho_cc),
           "note": "Convergence is the reverse-direction complement of coverage; test if it is spatially organised."}
    out.update(pi_provenance())   # which coupling produced these numbers
    (ROOT / "outputs/logs/section5_convergence_asymmetry.json").write_text(json.dumps(out, indent=2))
    print("wrote outputs/logs/section5_convergence_asymmetry.json")


if __name__ == "__main__":
    main()
