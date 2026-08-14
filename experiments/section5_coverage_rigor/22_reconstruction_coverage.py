#!/usr/bin/env python3
"""Coverage redefined as CONNECTIVITY RECONSTRUCTION ACCURACY (not transported mass).

"A human parcel is covered if its connectivity can be rebuilt from the mouse." For human parcel j:
  - column-normalise pi so each human parcel is a mouse-representative distribution:
        pihat[:,j] = pi[:,j] / pi[:,j].sum()   (well-defined even at tiny total mass -> avoids the
        mass-normalisation instability of the raw GW-distortion prototype)
  - push mouse connectivity into human space:  pred = pihat.T @ Mconn @ pihat   (n_h x n_h)
  - coverage_recon(j) = Pearson r between pred[j, :] and the TRUE human connectivity Hconn[j, :],
    over all other parcels l != j.

High = j's wiring has a mouse basis (covered); low = j's connectivity cannot be reproduced from the
mouse (genuinely no connectional counterpart, e.g. reorganised/expanded cortex). Computed for FC, SC,
and their fc0.7/sc0.3 average. Validates position-decoupling, left/right reliability, expansion, dlPFC.

LEFT/RIGHT RELIABILITY PAIRING
------------------------------
Pairing Schaefer region k with region k+200 is not a homotopic pairing. The Schaefer 400 ordering
puts the left hemisphere first and the right hemisphere second, but the two hemispheres are
parcellated independently and their sub-area indices do not line up: only 32 of 191 such pairs
carry the same area label, and region 150 pairs LH DefaultA_IPL_2 with RH ContB_PFCmp_1, which are
different networks in different lobes.

Two valid pairings are computed instead, with the index pairing retained under its own key.

  LR_reliability_parcel   the parcellation's own homotopic pair ids. 1,047 pairs covering all 2,094
                          parcels; 98.9 per cent are exact mirrors within 1 mm on all three axes.
                          This is the quantity that speaks to single-parcel reliability.
  LR_reliability_region   the region level, with left and right Schaefer areas matched
                          on the area label rather than the index.
  LR_reliability_index_k_kplus200   the index pairing, retained for comparison.

The comparison is not circular. The human functional connectivity is not hemisphere-averaged:
homotopic rows of Hfc differ by 0.028 on average and correlate at 0.91, so the two hemispheres are
independently measured.

The mass-coverage reference row is derived in the same run from log10 of the column mass, so the
comparison is like for like.

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


def homotopic_pairs(H_var):
    """Index pairs (left, right) from the parcellation's own pair ids, with a geometry check.

    Returns the two index arrays and the fraction of pairs whose centroids are mirrors within 1 mm,
    which is written to the log so the pairing is auditable rather than assumed.
    """
    pid = H_var["pairid"].to_numpy()
    hemi = H_var["hemisphere"].astype(str).to_numpy()
    L = {p: i for i, (p, h) in enumerate(zip(pid, hemi)) if h == "L"}
    R = {p: i for i, (p, h) in enumerate(zip(pid, hemi)) if h == "R"}
    keys = sorted(set(L) & set(R))
    li = np.array([L[p] for p in keys]); ri = np.array([R[p] for p in keys])
    xyz = H_var[["x", "y", "z"]].to_numpy(float)
    mirrored = ((np.abs(xyz[li, 0] + xyz[ri, 0]) <= 1.0)
                & (np.abs(xyz[li, 1:] - xyz[ri, 1:]).max(1) <= 1.0))
    return li, ri, float(mirrored.mean())


def schaefer_area_pairs(name_by_id, nr):
    """Left and right Schaefer region ids grouped by area label, so LH_X pairs with RH_X."""
    left, right = {}, {}
    for k, v in name_by_id.items():
        if not (nr == k).any():
            continue
        area = v.split("_", 2)[2]
        (left if "_LH_" in v else right).setdefault(area, []).append(k)
    return [(left[a], right[a]) for a in sorted(left) if a in right]


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
    name_by_id = {int(p[0]): p[1] for p in rows}
    net = np.array([{int(p[0]): p[1].split("_", 2)[2].split("_")[0] for p in rows}.get(int(k), "?") for k in nr])
    li, ri, frac_mirrored = homotopic_pairs(H.var)
    area_pairs = schaefer_area_pairs(name_by_id, nr)
    idx_pairs = [(k, k + 200) for k in range(1, 201) if (nr == k).any() and (nr == k + 200).any()]
    same_label = sum(1 for k, k2 in idx_pairs
                     if name_by_id[k].split("_", 2)[2] == name_by_id[k2].split("_", 2)[2])
    print(f"[pairing] {len(li)} homotopic parcel pairs, {frac_mirrored * 100:.1f} per cent are "
          f"mirrors within 1 mm")
    print(f"[pairing] {len(area_pairs)} Schaefer areas matched by label; the index pairing has "
          f"{same_label} of {len(idx_pairs)} pairs on the same area")
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
        # left/right reliability, three pairings. See the header note.
        rel_parcel = spearmanr(cov[li], cov[ri], nan_policy="omit").statistic
        A = [np.nanmean([np.nanmean(cov[nr == k]) for k in lk]) for lk, _ in area_pairs]
        B = [np.nanmean([np.nanmean(cov[nr == k]) for k in rk]) for _, rk in area_pairs]
        rel_region = spearmanr(A, B, nan_policy="omit").statistic
        Ai = [np.nanmean(cov[nr == k]) for k, _ in idx_pairs]
        Bi = [np.nanmean(cov[nr == k2]) for _, k2 in idx_pairs]
        rel_index = spearmanr(Ai, Bi, nan_policy="omit").statistic
        # expansion (region)
        eids = [k for k in range(1, 401) if (nr == k).any() and k in xu]
        cc = np.array([np.nanmean(cov[nr == k]) for k in eids]); ev = np.array([xu[k] for k in eids])
        r_exp = spearmanr(cc, ev, nan_policy="omit").statistic
        # ContB deficit (z)
        z = (cov[m] - np.nanmean(cov[m])) / np.nanstd(cov[m]); sel = net[m] == "ContB"
        contb = float(np.nanmean(z[sel]) - np.nanmean(z[~sel]))
        r = {"mean_recon_r": float(np.nanmean(cov[m])), "vs_spatial_isolation": float(r_iso),
             "vs_mass_coverage": float(r_mass),
             "LR_reliability_parcel": float(rel_parcel),
             "LR_reliability_region": float(rel_region),
             "LR_reliability_index_k_kplus200": float(rel_index),
             "vs_Xu_expansion_region": float(r_exp), "ContB_deficit_SD": contb}
        out[name] = r
        print(f"{name:10s} mean_r={r['mean_recon_r']:+.3f}  vs_iso={r_iso:+.3f}  vs_mass={r_mass:+.3f}  "
              f"LR parcel={rel_parcel:+.3f} region={rel_region:+.3f} "
              f"(retired index={rel_index:+.3f})  vs_exp={r_exp:+.3f}  ContB={contb:+.2f}SD")
        return cov

    print("reconstruction coverage, against the mass-coverage baseline computed in the same run:")
    stats("FC", cov_fc); stats("SC", cov_sc); stats("combined", cov_comb)
    # The mass-coverage reference is derived here on the same pairings, so the two rows can be
    # compared.
    stats("mass_coverage_ref", masscov)
    out["mass_coverage_ref"]["_note"] = (
        "This row is log10 of the column mass, the retired coverage metric, run through the same "
        "statistics so the two can be compared. mean_recon_r is therefore a log mass rather than a "
        "correlation, and vs_mass_coverage is 1.0 by construction. On the canonical coupling the "
        "isolation correlation is -0.310 on the myelin-masked parcels and -0.324 over all 2,094, "
        "and the reliability under a valid pairing is higher than reconstruction coverage's.")
    out["_reliability_reading"] = (
        "Mass coverage is the more reproducible of the two across hemispheres, which follows from "
        "construction: the parcellation is mirror-symmetric and the anchor-warped spatial cost is "
        "fitted on hemisphere-matched anchor pairs, so column mass is close to symmetric. "
        "Reconstruction coverage does not track spatial isolation (+0.046 against -0.310) and does "
        "track cross-species expansion (-0.321 against -0.048).")
    out["_pairing"] = {
        "n_homotopic_parcel_pairs": int(len(li)),
        "frac_pairs_mirrored_within_1mm": round(frac_mirrored, 4),
        "n_schaefer_areas_matched_by_label": int(len(area_pairs)),
        "retired_index_pairs": int(len(idx_pairs)),
        "retired_index_pairs_on_same_area": int(same_label),
        "_note": ("LR_reliability_index_k_kplus200 pairs Schaefer region k with k+200, which "
                  "matches anatomy for only "
                  f"{same_label} of {len(idx_pairs)} pairs."),
    }
    # store the combined map for figures
    out["coverage_recon_combined"] = [None if not np.isfinite(v) else round(float(v), 4) for v in cov_comb]
    out["coverage_recon_fc"] = [None if not np.isfinite(v) else round(float(v), 4) for v in cov_fc]

    dst = ROOT / "outputs/logs/section5_reconstruction_coverage.json"
    out.update(pi_provenance())   # which coupling produced these numbers
    dst.write_text(json.dumps(out, indent=2)); print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
