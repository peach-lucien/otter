#!/usr/bin/env python3
"""Coverage redefined as CONNECTIVITY RECONSTRUCTION FIDELITY (not transported mass).

"A human parcel is covered if its connectivity can be rebuilt from the mouse." For human parcel j:
  - column-normalise pi so each human parcel is a mouse-representative distribution:
        pihat[:,j] = pi[:,j] / pi[:,j].sum()   (well-defined even at tiny total mass -> avoids the
        mass-normalisation instability of the raw GW-distortion prototype)
  - push mouse connectivity into human space:  pred = pihat.T @ Mconn @ pihat   (n_h x n_h)
  - coverage_recon(j) = Pearson r between pred[j, :] and the TRUE human connectivity Hconn[j, :],
    over all other parcels l != j.

High = j's wiring has a mouse basis (covered); low = j's connectivity cannot be reproduced from the
mouse (genuinely no connectional counterpart, e.g. reorganised/expanded cortex). Computed for FC, SC,
and their fc0.7/sc0.3 average. Validates: position-decoupling, L/R reliability, expansion, dlPFC.

Writes: outputs/logs/section5_reconstruction_coverage.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance  # noqa: E402
np.seterr(divide="ignore", invalid="ignore")


def row_corr(pred, true, exclude_self=True):
    """Per-row Pearson r between pred and true (n x n), excluding the diagonal."""
    n = pred.shape[0]
    out = np.full(n, np.nan)
    for j in range(n):
        a = pred[j].copy(); b = true[j].copy()
        if exclude_self:
            a[j] = np.nan; b[j] = np.nan
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 10 and a[ok].std() > 1e-9 and b[ok].std() > 1e-9:
            out[j] = np.corrcoef(a[ok], b[ok])[0, 1]
    return out


def recon_coverage(pi, Mconn, Hconn):
    col = pi.sum(0)
    pihat = pi / np.maximum(col, 1e-300)          # column-normalised (n_m x n_h)
    pred = pihat.T @ Mconn @ pihat                # (n_h x n_h) mouse connectivity in human space
    return row_corr(pred, Hconn)


def main():
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    Mfc = np.asarray(M.uns["fc_mean"], float); Hfc = np.asarray(H.uns["fc_mean"], float)
    Msc = np.load(ROOT / "data_external/mouse_sc.npy").astype(float)
    Hsc = np.load(ROOT / "data_external/human_sc.npy").astype(float)
    # SC: log1p to tame heavy tails (matches Cm_SC construction)
    Msc = np.log1p(np.maximum(Msc, 0)); Hsc = np.log1p(np.maximum(Hsc, 0))

    pi = load_pi()
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    iso = np.load(ROOT / "outputs/anndata/full_costs.npz")["M_xyz"].min(0)
    mye = np.asarray(json.loads((ROOT / "outputs/logs/buckner_krienen_2013_tethering.json").read_text())["myelin_per_parcel"], float)
    nr = np.asarray(json.loads((ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    rows = [l.split("\t") for l in (ROOT / "outputs/anndata/_schaefer_order.txt").read_text().splitlines() if l.strip()]
    net = np.array([{int(p[0]): p[1].split("_", 2)[2].split("_")[0] for p in rows}.get(int(k), "?") for k in nr])
    b = json.loads((ROOT / "outputs/logs/section5_evolution_battery.json").read_text())
    xu = dict(zip(np.asarray(b["Xu2020 mouse→human expansion"]["schaefer_ids"], int),
                  np.asarray(b["Xu2020 mouse→human expansion"]["map_values"], float)))

    cov_fc = recon_coverage(pi, Mfc, Hfc)
    cov_sc = recon_coverage(pi, Msc, Hsc)
    cov_comb = 0.7 * cov_fc + 0.3 * cov_sc
    masscov = np.log10(np.maximum(pi.sum(0), 1e-300))

    out = {"_def": "coverage_recon(j) = Pearson r(reconstructed human connectivity of j, true), per parcel"}

    def stats(name, cov):
        m = np.isfinite(cov) & np.isfinite(mye)
        r_iso = spearmanr(cov[m], iso[m]).statistic
        r_mass = spearmanr(cov[m], masscov[m]).statistic
        # L/R reliability: Schaefer k vs k+200, region means
        ids = [k for k in range(1, 201) if (nr == k).any() and (nr == k + 200).any()]
        L = [np.nanmean(cov[nr == k]) for k in ids]; R = [np.nanmean(cov[nr == k + 200]) for k in ids]
        rel = spearmanr(L, R, nan_policy="omit").statistic
        # expansion (region)
        eids = [k for k in range(1, 401) if (nr == k).any() and k in xu]
        cc = np.array([np.nanmean(cov[nr == k]) for k in eids]); ev = np.array([xu[k] for k in eids])
        r_exp = spearmanr(cc, ev, nan_policy="omit").statistic
        # ContB deficit (z)
        z = (cov[m] - np.nanmean(cov[m])) / np.nanstd(cov[m]); sel = net[m] == "ContB"
        contb = float(np.nanmean(z[sel]) - np.nanmean(z[~sel]))
        r = {"mean_recon_r": float(np.nanmean(cov[m])), "vs_spatial_isolation": float(r_iso),
             "vs_mass_coverage": float(r_mass), "LR_reliability": float(rel),
             "vs_Xu_expansion_region": float(r_exp), "ContB_deficit_SD": contb}
        out[name] = r
        print(f"{name:10s} mean_r={r['mean_recon_r']:+.3f}  vs_iso={r_iso:+.3f}  vs_mass={r_mass:+.3f}  "
              f"LR_rel={rel:+.3f}  vs_exp={r_exp:+.3f}  ContB={contb:+.2f}SD")
        return cov

    print("reconstruction-fidelity coverage (vs mass-coverage baselines iso=-0.47, LR=0.22):")
    stats("FC", cov_fc); stats("SC", cov_sc); stats("combined", cov_comb)
    out["mass_coverage_ref"] = {"vs_spatial_isolation": -0.468, "LR_reliability": 0.22}
    # store the combined map for figures
    out["coverage_recon_combined"] = [None if not np.isfinite(v) else round(float(v), 4) for v in cov_comb]

    dst = ROOT / "outputs/logs/section5_reconstruction_coverage.json"
    out.update(pi_provenance())   # which coupling produced these numbers
    dst.write_text(json.dumps(out, indent=2)); print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
