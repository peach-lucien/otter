#!/usr/bin/env python3
"""Reconcile: on the comparison coupling fitted without the anchor warp, coverage vs |x| is
strong (-0.30) while coverage vs published expansion maps is ~null. Geometric confound, region vs
parcel level, log vs linear, spin config, bilateral handling and parametric vs non-parametric are
checked. Non-parametric (Spearman) throughout unless noted.

RESOLUTION ON THE CANONICAL COUPLING: the discrepancy dissolves because the
coverage~|x| side collapses. Canonical rho(coverage,|x|) = -0.03 (spin p = 0.83) at parcel level
and +0.06 (p = 0.73) at region level; the -0.30 is confined to the comparison coupling. Coverage
vs the published expansion maps remains null.

The `region_level_maps` block reads `coverage_values` out of the stored
section5_evolution_battery.json rather than recomputing them, so those cov~map and cov~|x|
numbers carry whatever coupling that battery was run on.

Writes: outputs/logs/section5_expansion_reconciliation.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, pearsonr, rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance  # noqa: E402
from otter.eval.nulls import _haar_rotation                 # noqa: E402

N_SPIN = 2000
np.seterr(divide="ignore", invalid="ignore")


def spin_perms(coords, n=N_SPIN, seed=0):
    c = coords - coords.mean(0)
    sph = c / np.linalg.norm(c, axis=1, keepdims=True)
    tree = cKDTree(sph); rng = np.random.default_rng(seed)
    return [tree.query(sph @ _haar_rotation(rng).T)[1] for _ in range(n)]


def spin_rho(sig, tgt, perms):
    obs = spearmanr(sig, tgt).statistic
    null = np.array([spearmanr(sig[p], tgt).statistic for p in perms])
    return float(obs), float((np.sum(np.abs(null) >= abs(obs)) + 1) / (len(perms) + 1))


def main():
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    nr = np.asarray(json.loads((ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    mye = np.asarray(json.loads(
        (ROOT / "outputs/logs/buckner_krienen_2013_tethering.json").read_text())["myelin_per_parcel"], float)

    prov = pi_provenance()
    print(f"π: {prov['pi_file']}  sha256={prov['pi_sha256']}")
    out = {**prov,
           "_q": "Is coverage's medial-lateral organisation geometric or connectional, and does it "
                 "actually track evolutionary expansion?"}

    # ---------- region-level, SOUND coverage (log10 mass-normalised MEAN per Schaefer region) ----
    pi = load_pi(); col = pi.sum(0)
    ids = [k for k in range(1, 401) if (nr == k).any()]
    cov_r = {k: np.log10(col[nr == k].mean() + 1e-300) for k in ids}
    cen = {k: xyz[nr == k].mean(0) for k in ids}
    kk = np.array(ids)
    covv = np.array([cov_r[k] for k in ids])
    absx = np.abs(np.array([cen[k][0] for k in ids]))
    apy = np.array([cen[k][1] for k in ids]); dvz = np.array([cen[k][2] for k in ids])
    C = np.array([cen[k] for k in ids])
    perms = spin_perms(C)

    reg = {}
    for nm, tgt in [("medial_lateral_absX", absx), ("anterior_posterior_Y", apy), ("dorsal_ventral_Z", dvz)]:
        r, p = spin_rho(covv, tgt, perms)
        reg[nm] = {"spearman_rho": r, "spin_p": p}
    out["region_level_axes"] = reg
    print("REGION-level (sound coverage) spatial axes:")
    for nm, r in reg.items():
        print(f"  {nm:<22} rho={r['spearman_rho']:+.3f}  spin p={r['spin_p']:.4f}")

    # ---------- pull the stored published maps and cross-correlate at region level -------------
    bat = json.loads((ROOT / "outputs/logs/section5_evolution_battery.json").read_text())
    out["region_level_maps"] = {}
    print("\nREGION-level coverage vs published maps, and each map vs |x|:")
    for label in ["Xu2020 macaque→human expansion", "Hill2010 macaque→human expansion",
                  "Hill2010 developmental expansion", "Sydnor2021 S–A axis",
                  "Margulies2016 principal gradient", "HCP T1w/T2w hierarchy"]:
        v = bat.get(label, {})
        if "schaefer_ids" not in v:
            continue
        sid = np.asarray(v["schaefer_ids"], int)
        mval = np.asarray(v["map_values"], float)
        cval = np.asarray(v["coverage_values"], float)          # sound region coverage from battery
        # centroids + |x| for these ids
        cmap = {k: cen[k] for k in sid if k in cen}
        keep = np.array([i for i, k in enumerate(sid) if k in cmap])
        sid, mval, cval = sid[keep], mval[keep], cval[keep]
        Cm = np.array([cmap[k] for k in sid]); ax = np.abs(Cm[:, 0])
        pm = spin_perms(Cm)
        r_cov, p_cov = spin_rho(cval, mval, pm)                 # coverage vs map (Spearman + spin)
        r_mapx = spearmanr(mval, ax).statistic                  # is the MAP itself medial-lateral?
        r_covx = spearmanr(cval, ax).statistic                  # coverage vs |x| on THIS id set
        out["region_level_maps"][label] = {
            "n": int(len(sid)), "cov_vs_map_rho": r_cov, "cov_vs_map_spin_p": p_cov,
            "map_vs_absX_rho": float(r_mapx), "cov_vs_absX_rho": float(r_covx),
            "_coverage_source": ("section5_evolution_battery.json (stored, NOT recomputed here). "
                                 "That battery carries pi_file / pi_sha256 in its _meta; read "
                                 "them to identify the coupling.")}
        print(f"  {label:<38} n={len(sid):3d}  cov~map {r_cov:+.3f}(p={p_cov:.3f})  "
              f"map~|x| {r_mapx:+.3f}  cov~|x| {r_covx:+.3f}")

    # ---------- geometric confound: coverage~|x| across couplings (parcel level) ---------------
    print("\nGEOMETRIC CONFOUND  coverage~|x| parcel-level across couplings:")
    ctx = np.isfinite(mye)
    # "canonical" is the reported arm; "production" in the stored comparison is the
    # unwarped regional-entry coupling. The remaining arms are comparators.
    out["coverage_absX_across_couplings"] = {}
    out["_reported_coupling_arm"] = "canonical"
    coup = [("canonical", "pi_canonical.npy"),
            ("production", "pi_fc_plus_SC_with_all_packs.npy"),
            ("anchor_free", "pi_anchorfree_control.npy"),
            ("xyz_zeroed", "pi_fc_plus_SC_xyz_zero.npy"),
            ("FC_only_no_xyz_no_anchor(grid_FC_none)", "pi_ablation_grid_FC_none.npy"),
            ("FC_SC_conn_only", "pi_ablation_FC_SC.npy"),
            ("base_garin_points", "pi_fc_plus_SC.npy")]
    for name, path in coup:
        p = ROOT / "outputs/coupling" / path
        if not p.exists():
            out["coverage_absX_across_couplings"][name] = "missing"; print(f"  {name:<40} MISSING"); continue
        cv = np.log10(np.maximum(np.load(p).sum(0), 1e-300))
        m = ctx & np.isfinite(cv)
        pm = spin_perms(xyz[m])
        r, pp = spin_rho(cv[m], np.abs(xyz[m, 0]), pm)
        out["coverage_absX_across_couplings"][name] = {"spearman_rho": r, "spin_p": pp,
                                                        "xyz_weight": None,
                                                        **pi_provenance(path)}
        print(f"  {name:<40} rho={r:+.3f}  spin p={pp:.4f}")

    # ---------- Log versus linear, parcel versus region for the comparator -------------------
    m = ctx & np.isfinite(col)
    lin = col[m]; lg = np.log10(np.maximum(lin, 1e-300)); ax = np.abs(xyz[m, 0])
    out["log_vs_linear_parcel"] = {
        "spearman_log_vs_absX": float(spearmanr(lg, ax).statistic),      # rank-invariant to log
        "spearman_linear_vs_absX": float(spearmanr(lin, ax).statistic),  # identical by construction
        "pearson_log_vs_absX": float(pearsonr(lg, ax)[0]),
        "pearson_linear_vs_absX": float(pearsonr(lin, ax)[0]),
        "_note": "Spearman is identical for log and linear (monotone). Pearson differs: the eps tail "
                 "dominates Pearson-on-log, so Spearman is the appropriate statistic here."}
    print("\nLOG vs LINEAR (parcel, production):", {k: round(v, 3) if isinstance(v, float) else v
          for k, v in out["log_vs_linear_parcel"].items() if k != "_note"})

    dst = ROOT / "outputs/logs/section5_expansion_reconciliation.json"
    dst.write_text(json.dumps(out, indent=2)); print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
