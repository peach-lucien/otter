"""Translation, FC target: route the MOUSE ASD-model dysconnectivity
map through the canonical coupling into human space and test whether it predicts
the HUMAN ASD *functional-connectivity* deviation pattern (TransBrain 2025), a
non-null target — unlike the near-null ENIGMA cortical-thinning map used by
`27_translate_autism.py`.

HUMAN TARGET  data_external/transbrain_2025/z_autism_regress.csv
  233 ABIDE ASD individuals x 127 bilateral Brainnetome (BN) regions. TRAP: every
  COLUMN is z-scored across the 233 subjects (col mean == 0, one-sample t == 0),
  so the naive "mean z across subjects" carries NO group signal. The recoverable
  group pattern is the per-subject *consensus*: standardise each subject's row
  (across regions) and average across subjects -> G (non-zero, anatomically
  sensible). Equivalently, the mean per-subject Spearman(T, row_i) is the
  subject-level statistic; we report both.

ATLAS BRIDGE  transbrain's bn_atlas_2mm_symmetry.nii.gz (127 bilateral labels,
  MNI 2mm). Each OTTER human parcel (H.var xyz, MNI mm) is assigned to its nearest
  labelled atlas voxel; the routed mouse per-parcel map is averaged within each of
  the 127 BN regions. Region names are matched by NAME (the CSV column order and
  the atlas index order differ at the thalamic tail). Spin geometry uses the
  left-hemisphere voxel centroid per region (real lateral spread; a bilateral
  centroid sits on the midline and is degenerate).

MOUSE SOURCE  Pagani 2026 group occurrence maps (as in script 27):
  cluster1_pos = HYPER-connectivity occurrence, cluster2_neg = HYPO. Sampled at
  each 1864-parcel centre voxel (ns_center_ix).

NULLS
  - spin_null over region centroids (spatial-autocorrelation-preserving).
  - per-subject Wilcoxon across the 233 individuals.
  - pi-row-permutation (does the real coupling beat a scrambled one?).
  - translation spin (null B): spin the MOUSE input, route through the REAL pi
    (the fair null for a translation claim).
  - SPECIFICITY: shuffled / smooth-random / A-P-gradient mouse maps that are not
    ASD-specific should NOT predict the human ASD pattern.

Writes outputs/logs/section6_translate_asd_fc.json
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np, nibabel as nib, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import DATA_DIR, load_cached, load_pi, pi_provenance
from otter.eval.nulls import _route_normalized, spin_null

PAGANI = Path(DATA_DIR) / "pagani"
TB = ROOT / "data_external/transbrain_2025/z_autism_regress.csv"
# transbrain package atlas (installed in the sandbox pylib)
ATLAS_CANDIDATES = [
    Path("/var/tmp/pylibs/transbrain/atlas"),
    Path("/sessions/modest-tender-carson/mnt/outputs/.pylibs3/transbrain/atlas"),
]
N_SPIN = 2000
N_PIPERM = 1000
N_TRANSSPIN = 1000


def find_atlas():
    for d in ATLAS_CANDIDATES:
        if (d / "bn_atlas_2mm_symmetry.nii.gz").exists():
            return d
    raise FileNotFoundError("transbrain bn_atlas not found")


def load_human_target():
    df = pd.read_csv(TB)
    zcols = df.columns[1:].tolist()
    X = df[zcols].to_numpy(float)                     # 233 x 127
    col_mean_absmax = float(np.abs(X.mean(0)).max())  # ~0 (columns z-scored)
    Xr = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True, ddof=0)
    G = Xr.mean(0)                                     # consensus deviation, 127
    return zcols, X, G, col_mean_absmax


def build_bridge(H, atlas_dir, zcols):
    img = nib.load(str(atlas_dir / "bn_atlas_2mm_symmetry.nii.gz"))
    lab = np.asarray(img.get_fdata()).astype(int)
    bn = pd.read_csv(atlas_dir / "bn_atlas.csv")
    name2idx = {r["Anatomical Name"]: int(r["Atlas Index"]) for _, r in bn.iterrows()}
    assert set(zcols) <= set(name2idx), "unmatched region names"

    vox = np.argwhere(lab > 0)
    labs = lab[vox[:, 0], vox[:, 1], vox[:, 2]]
    mm = nib.affines.apply_affine(img.affine, vox)

    hxyz = H.var[["x", "y", "z"]].to_numpy(float)
    dist, jj = cKDTree(mm).query(hxyz)
    parcel_lab = labs[jj]

    # per-region parcel masks + left-hemisphere voxel centroid (spin geometry)
    regions, cen, dropped = {}, {}, []
    for nm in zcols:
        idx = name2idx[nm]
        m = parcel_lab == idx
        lvox = mm[(labs == idx) & (mm[:, 0] < 0)]
        if m.sum() == 0 or len(lvox) == 0:
            dropped.append(nm); continue
        regions[nm] = m
        cen[nm] = lvox.mean(0)
    return regions, cen, dropped, {
        "assign_dist_mean": float(dist.mean()),
        "assign_dist_p95": float(np.percentile(dist, 95)),
        "assign_dist_max": float(dist.max()),
        "n_regions_hit": int(len(set(parcel_lab) & set(name2idx.values()))),
    }


def agg(mvec, pi, regions, order):
    hp = _route_normalized(mvec, pi)
    return np.array([np.nanmean(hp[regions[nm]]) for nm in order])


def sample_occurrence(M):
    ix = M.var["ns_center_ix"].to_numpy(np.int64)
    d1 = nib.load(str(PAGANI / "cluster1_AMBA_occurrence_map_pos_cohens_d_0.8.nii.gz")).get_fdata()
    d2 = nib.load(str(PAGANI / "cluster2_AMBA_occurrence_map_neg_cohens_d_0.8.nii.gz")).get_fdata()
    return (d1.ravel(order="C")[ix].astype(float),
            d2.ravel(order="C")[ix].astype(float))


def smooth_random_mouse(mcoords, k=25, seed=0):
    """A spatially-smooth random mouse map: white noise averaged over each
    parcel's k nearest neighbours (matches the occurrence maps' rough smoothness).
    """
    rng = np.random.default_rng(seed)
    _, nn = cKDTree(mcoords).query(mcoords, k=k)
    z = rng.standard_normal(mcoords.shape[0])
    return z[nn].mean(1)


def per_subject(T, X, order_idx):
    """Mean per-subject Spearman(T, subject_row) + Wilcoxon across subjects."""
    Xs = X[:, order_idx]
    rho = np.array([spearmanr(T, Xs[i]).statistic for i in range(Xs.shape[0])])
    w = wilcoxon(rho, alternative="two-sided")
    return {"mean_rho": float(rho.mean()), "median_rho": float(np.median(rho)),
            "frac_pos": float((rho > 0).mean()),
            "wilcoxon_p": float(w.pvalue), "n_subj": int(len(rho))}


def pi_perm_null(mvec, pi, regions, order, G, rho_obs, n=N_PIPERM, seed=1):
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n):
        perm = rng.permutation(pi.shape[0])
        T = agg(mvec, pi[perm], regions, order)
        ok = np.isfinite(T) & np.isfinite(G)
        nulls.append(float(spearmanr(T[ok], G[ok]).statistic))
    nulls = np.asarray(nulls)
    return {"p_pi_perm": float((np.sum(np.abs(nulls) >= abs(rho_obs)) + 1) / (n + 1)),
            "null_abs_mean": float(np.abs(nulls).mean())}


def trans_spin_null(mvec, mcoords, pi, regions, order, G, rho_obs, n=N_TRANSSPIN, seed=2):
    """Null B: spin the MOUSE input, route through the REAL pi, aggregate, corr."""
    from otter.eval.nulls import _haar_rotation
    c = mcoords - mcoords.mean(0)
    sph = c / np.linalg.norm(c, axis=1, keepdims=True).clip(min=1e-12)
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n):
        _, perm = cKDTree(sph @ _haar_rotation(rng).T).query(sph)
        T = agg(mvec[perm], pi, regions, order)
        ok = np.isfinite(T) & np.isfinite(G)
        nulls.append(float(spearmanr(T[ok], G[ok]).statistic))
    nulls = np.asarray(nulls)
    return {"p_trans_spin": float((np.sum(np.abs(nulls) >= abs(rho_obs)) + 1) / (n + 1)),
            "null_abs_mean": float(np.abs(nulls).mean())}


def test_map(name, mvec, pi, regions, cen, order, G, X, order_idx,
             full=True, mcoords=None):
    T = agg(mvec, pi, regions, order)
    ok = np.isfinite(T) & np.isfinite(G)
    C = np.array([cen[nm] for nm in order])[ok]
    rho = float(spearmanr(T[ok], G[ok]).statistic)
    s = spin_null(rankdata(T[ok]), rankdata(G[ok]), C, n_trials=N_SPIN, seed=0)
    res = {"label": name, "n_regions": int(ok.sum()), "rho": rho,
           "spin_p": s["p_spin"], "spin_null_abs_p95": s["null_abs_p95"]}
    if full:
        res["per_subject"] = per_subject(T[ok], X, np.asarray(order_idx)[ok])
        res["pi_perm"] = pi_perm_null(mvec, pi, regions, order, G, rho)
        res["trans_spin"] = trans_spin_null(mvec, mcoords, pi, regions, order, G, rho)
    return res


def main():
    atlas_dir = find_atlas()
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    pi = load_pi()
    mcoords = M.var[["x", "y", "z"]].to_numpy(float)

    zcols, X, G, col_mean_absmax = load_human_target()
    regions, cen, dropped, bridge = build_bridge(H, atlas_dir, zcols)
    order = [nm for nm in zcols if nm in regions]
    order_idx = [zcols.index(nm) for nm in order]
    Go = G[order_idx]

    gs = pd.Series(G, index=zcols)
    sanity = {"top_pos_consensus": {k: round(float(v), 3) for k, v in gs.sort_values(ascending=False).head(8).items()},
              "top_neg_consensus": {k: round(float(v), 3) for k, v in gs.sort_values().head(8).items()},
              "dropped_regions": dropped}

    hyper, hypo = sample_occurrence(M)
    main_maps = {"total_dysconnectivity": hyper + hypo,
                 "hyper_occurrence": hyper,
                 "hypo_occurrence": hypo,
                 "signed_hyper_minus_hypo": hyper - hypo}
    spec_maps = {"shuffled_total": np.random.default_rng(7).permutation(hyper + hypo),
                 "smooth_random_s0": smooth_random_mouse(mcoords, seed=0),
                 "smooth_random_s1": smooth_random_mouse(mcoords, seed=1),
                 "smooth_random_s2": smooth_random_mouse(mcoords, seed=2),
                 "mouse_AP_gradient": mcoords[:, 1]}

    out = {"_note": "mouse ASD occurrence -> pi_canonical -> 127 BN regions vs human "
                    "ASD FC consensus (row-standardised, averaged across 233 ABIDE "
                    "subjects). Spin over left-hemi region centroids; ranks.",
           "human_target": {
               "n_subjects": int(X.shape[0]), "n_regions": int(X.shape[1]),
               "col_zscored_so_group_mean_is_zero": True,
               "naive_group_mean_absmax": col_mean_absmax,
               "consensus_G_std": float(G.std()),
               "consensus_G_abs_range": [float(G.min()), float(G.max())]},
           "bridge": bridge, "sanity": sanity, "n_regions_tested": len(order),
           "main_maps": {}, "specificity_maps": {}}

    print(f"bridge: {bridge}")
    print(f"consensus sanity top+: {list(sanity['top_pos_consensus'])[:5]}")

    for name, mvec in main_maps.items():
        r = test_map(name, mvec, pi, regions, cen, order, Go, X, order_idx,
                     full=True, mcoords=mcoords)
        out["main_maps"][name] = r
        print(f"  {name:26s} rho={r['rho']:+.3f} spin_p={r['spin_p']:.3f} "
              f"persubj_meanrho={r['per_subject']['mean_rho']:+.3f} "
              f"wilcox_p={r['per_subject']['wilcoxon_p']:.1e} "
              f"pi_perm_p={r['pi_perm']['p_pi_perm']:.3f} "
              f"trans_spin_p={r['trans_spin']['p_trans_spin']:.3f}")

    for name, mvec in spec_maps.items():
        r = test_map(name, mvec, pi, regions, cen, order, Go, X, order_idx, full=False)
        out["specificity_maps"][name] = r
        print(f"  [spec] {name:20s} rho={r['rho']:+.3f} spin_p={r['spin_p']:.3f}")

    out.update(pi_provenance())   # which coupling produced these numbers
    (ROOT / "outputs/logs/section6_translate_asd_fc.json").write_text(json.dumps(out, indent=2))
    print("wrote outputs/logs/section6_translate_asd_fc.json")


if __name__ == "__main__":
    main()
