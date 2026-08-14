"""Translation: route MOUSE autism-model brain phenotypes through the
canonical coupling into human space, and test whether the translated mouse ASD
phenotype recovers the HUMAN autism (ENIGMA cortical-thinning) pattern.

The disease question is posed as a translation validation: mouse ASD
dysconnectivity -> pi -> human parcels -> DK hemi-regions -> vs human ASD
thinning (Spearman + whole-brain spin null, L/R kept separate for valid
geometry).

MOUSE ASD phenotype source (Pagani 2026, data_crossspecies/pagani/):
  - cluster1_AMBA_occurrence_map_pos ...: per-voxel count (0-5) of models with a
    consistent HYPER-connectivity effect (Cohen's d>0.8).
  - cluster2_AMBA_occurrence_map_neg ...: same for HYPO-connectivity.
  Sampled at each of the 1864 mouse parcels' centre voxel (ns_center_ix, C-order
  flat index into the 456x320x528 AMBA volume). GROUP / subtype-level ASD
  phenotype; per-model per-parcel maps are NOT decodable from the shipped
  20x1491 feature matrix (see pagani_2026_per_model/README.md).

HUMAN target: ENIGMA ASD cortical thickness d_icv (68 DK L_/R_ rows). Near-null
(mean d~0.003, std 0.091), so power is low.

Writes outputs/logs/section6_translate_autism.json
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np, nibabel as nib, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import DATA_DIR, load_cached, load_pi, pi_provenance
from otter.eval.nulls import _route_normalized, spin_null

ENIGMA = ROOT / "data_external/enigma"
PAGANI = Path(DATA_DIR) / "pagani"
N_SPIN = 2000
N_PIPERM = 1000


def enig_hemi(p):
    out = {}
    for row in csv.DictReader(open(p)):
        s = row["Structure"]
        if "_" not in s:
            continue
        hemi, reg = s.split("_", 1)
        try:
            v = float(row["d_icv"])
        except Exception:
            continue
        if hemi in ("L", "R") and np.isfinite(v):
            out[(hemi, reg.strip().lower())] = v
    return out


def sample_occurrence(M):
    ix = M.var["ns_center_ix"].values.astype(np.int64)
    d1 = nib.load(str(PAGANI / "cluster1_AMBA_occurrence_map_pos_cohens_d_0.8.nii.gz")).get_fdata()
    d2 = nib.load(str(PAGANI / "cluster2_AMBA_occurrence_map_neg_cohens_d_0.8.nii.gz")).get_fdata()
    hyper = d1.ravel(order="C")[ix].astype(float)
    hypo = d2.ravel(order="C")[ix].astype(float)
    return hyper, hypo


def build_dk_regions(H):
    import abagen
    at = abagen.fetch_desikan_killiany()
    img = nib.load(at["image"]); info = pd.read_csv(at["info"])
    lab = np.asarray(img.get_fdata()).astype(int)
    ctx = info[info.structure == "cortex"]
    id_meta = {int(r.id): (str(r.hemisphere), str(r.label).lower()) for r in ctx.itertuples()}
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    vox = nib.affines.apply_affine(np.linalg.inv(img.affine), xyz); vi = np.rint(vox).astype(int)
    inb = np.all((vi >= 0) & (vi < np.array(lab.shape)), axis=1)
    pl = np.zeros(len(xyz), int); pl[inb] = lab[vi[inb, 0], vi[inb, 1], vi[inb, 2]]
    cvx = np.argwhere(np.isin(lab, list(id_meta))); need = np.where(pl == 0)[0]
    dd, jj = cKDTree(cvx).query(vox[need]); ok = dd <= 4; hit = cvx[jj[ok]]
    pl[need[ok]] = lab[hit[:, 0], hit[:, 1], hit[:, 2]]; pl[~np.isin(pl, list(id_meta))] = 0
    regions = {}
    for i, (hemi, reg) in id_meta.items():
        m = pl == i
        if m.sum() >= 8:
            regions[(hemi, reg)] = {"mask": m, "cen": xyz[m].mean(0)}
    return regions


def agg_regions(human_vec, regions):
    return {k: float(np.nanmean(human_vec[v["mask"]])) for k, v in regions.items()}


def test_map(mvec, pi, regions, target, label, seed=0):
    hpred = _route_normalized(mvec, pi)
    cov = agg_regions(hpred, regions)
    keys = [k for k in cov if k in target and np.isfinite(target[k]) and np.isfinite(cov[k])]
    c = np.array([cov[k] for k in keys]); d = np.array([target[k] for k in keys])
    C = np.array([regions[k]["cen"] for k in keys])
    rho = float(spearmanr(c, d).statistic)
    s = spin_null(rankdata(c), rankdata(d), C, n_trials=N_SPIN, seed=seed)
    return {"label": label, "n": len(keys), "rho": rho, "spin_p": s["p_spin"],
            "spin_null_abs_p95": s["null_abs_p95"]}, cov


def pi_perm_specificity(mvec, pi, regions, target, rho_obs, n=N_PIPERM, seed=1):
    rng = np.random.default_rng(seed)
    keys = None; d = None; n_ge = 0; nulls = []
    for t in range(n):
        perm = rng.permutation(pi.shape[0])
        hpred = _route_normalized(mvec, pi[perm])
        cov = agg_regions(hpred, regions)
        if keys is None:
            keys = [k for k in cov if k in target and np.isfinite(target[k])]
            d = np.array([target[k] for k in keys])
        c = np.array([cov[k] for k in keys])
        r = float(spearmanr(c, d).statistic)
        nulls.append(r)
        if abs(r) >= abs(rho_obs):
            n_ge += 1
    nulls = np.array(nulls)
    return {"p_pi_perm": (n_ge + 1) / (n + 1), "null_mean": float(nulls.mean()),
            "null_abs_mean": float(np.abs(nulls).mean())}


def main():
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    pi = load_pi()

    hyper, hypo = sample_occurrence(M)
    mouse_maps = {
        "hyper_occurrence": hyper,
        "hypo_occurrence": hypo,
        "total_dysconnectivity": hyper + hypo,
        "signed_hyper_minus_hypo": hyper - hypo,
    }
    print("mouse maps (1864 parcels): "
          + ", ".join(f"{k}: nnz={int((v!=0).sum())}" for k, v in mouse_maps.items()))

    regions = build_dk_regions(H)
    xs = [v["cen"][0] for v in regions.values()]
    print(f"DK: {len(regions)} hemi-regions, centroid-x [{min(xs):.0f},{max(xs):.0f}]")

    disorders = {p.stem.replace("cortical_thickness_", ""): enig_hemi(p)
                 for p in sorted(ENIGMA.glob("cortical_thickness_*.csv"))}
    asd = disorders["asd"]

    out = {"_note": "mouse ASD occurrence -> pi_canonical -> DK vs ENIGMA thinning; "
                    "L/R separate, whole-brain spin on ranks",
           "enigma_asd_target": {"n": len(asd),
                                 "mean_d": float(np.mean(list(asd.values()))),
                                 "std_d": float(np.std(list(asd.values()))),
                                 "abs_max_d": float(np.max(np.abs(list(asd.values()))))},
           "maps": {}}

    for name, mvec in mouse_maps.items():
        res, cov = test_map(mvec, pi, regions, asd, name)
        piperm = pi_perm_specificity(mvec, pi, regions, asd, res["rho"])
        res["pi_perm"] = piperm
        xdis = {}
        for dis, dm in disorders.items():
            keys = [k for k in cov if k in dm and np.isfinite(dm[k]) and np.isfinite(cov[k])]
            c = np.array([cov[k] for k in keys]); d = np.array([dm[k] for k in keys])
            xdis[dis] = round(float(spearmanr(c, d).statistic), 3)
        res["cross_disorder_rho"] = xdis
        out["maps"][name] = res
        print(f"  {name:26s} rho_vs_ASD={res['rho']:+.3f} spin_p={res['spin_p']:.3f} "
              f"pi_perm_p={piperm['p_pi_perm']:.3f}  xdis={xdis}")

    out.update(pi_provenance())   # which coupling produced these numbers
    (ROOT / "outputs/logs/section6_translate_autism.json").write_text(json.dumps(out, indent=2))
    print("wrote outputs/logs/section6_translate_autism.json")


if __name__ == "__main__":
    main()
