"""Disorder reachability redone at correct resolution + valid spin geometry.

BUG in the DK analysis: abagen DK labels are identical L/R (e.g. 'bankssts' = id 1 AND id 35), so
grouping by label pools BOTH hemispheres into 34 bilateral regions, and each region's centroid
becomes mean-over-both-hemispheres -> x ~ 0. The spin null then rotates a midline-collapsed point
cloud (degenerate, like the Hill half-sphere problem). This script keeps L and R SEPARATE (68 hemi-
regions, real centroids, valid whole-brain spin) and matches ENIGMA's own L_/R_ rows.

Compares mass-coverage, FC-recon, and FC+SC-recon coverage on the canonical pi, all disorders +
transdiagnostic, Spearman + spin + FDR. Also prints the centroid-x diagnostic proving the bug.

Run under: PYTHONPATH=/var/tmp/pylib:...:src ABAGEN_DATA=/var/tmp/abagen HOME=/var/tmp
Writes outputs/logs/section6_disease_68region.json
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np, nibabel as nib, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr, false_discovery_control

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi, pi_provenance  # noqa: E402
from otter.eval.nulls import spin_null                  # noqa: E402
ENIGMA = ROOT / "data_external/enigma"
N_SPIN = 2000


def recon(pi, Mc, Hc):
    ph = pi.sum(0); pit = pi / np.maximum(ph, 1e-300); pred = pit.T @ Mc @ pit
    n = pred.shape[0]; out = np.full(n, np.nan)
    for j in range(n):
        a = pred[j].copy(); b = Hc[j].copy(); a[j] = np.nan; b[j] = np.nan
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 10 and a[ok].std() > 1e-9: out[j] = np.corrcoef(a[ok], b[ok])[0, 1]
    return out


def enig_hemi(p):
    """{(hemi, region): d} keeping L/R separate."""
    out = {}
    for row in csv.DictReader(open(p)):
        s = row["Structure"]
        if "_" not in s: continue
        hemi, reg = s.split("_", 1)
        try: v = float(row["d_icv"])
        except: continue
        if hemi in ("L", "R") and np.isfinite(v): out[(hemi, reg.strip().lower())] = v
    return out


def main():
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    Mfc = np.asarray(M.uns["fc_mean"], float); Hfc = np.asarray(H.uns["fc_mean"], float)
    Msc = np.log1p(np.maximum(np.load(ROOT / "data_external/mouse_sc.npy").astype(float), 0))
    Hsc = np.log1p(np.maximum(np.load(ROOT / "data_external/human_sc.npy").astype(float), 0))
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    pi = load_pi(); col = pi.sum(0)

    at = abagen_fetch(); img = nib.load(at["image"]); info = pd.read_csv(at["info"])
    lab = np.asarray(img.get_fdata()).astype(int)
    ctx = info[info.structure == "cortex"]
    # (hemi, label) per DK id
    id_meta = {int(r.id): (str(r.hemisphere), str(r.label).lower()) for r in ctx.itertuples()}
    vox = nib.affines.apply_affine(np.linalg.inv(img.affine), xyz); vi = np.rint(vox).astype(int)
    inb = np.all((vi >= 0) & (vi < np.array(lab.shape)), axis=1)
    pl = np.zeros(len(xyz), int); pl[inb] = lab[vi[inb, 0], vi[inb, 1], vi[inb, 2]]
    cvx = np.argwhere(np.isin(lab, list(id_meta))); need = np.where(pl == 0)[0]
    dd, jj = cKDTree(cvx).query(vox[need]); ok = dd <= 4; hit = cvx[jj[ok]]
    pl[need[ok]] = lab[hit[:, 0], hit[:, 1], hit[:, 2]]; pl[~np.isin(pl, list(id_meta))] = 0

    # per HEMI-region (id): parcels, centroid (REAL x, no folding)
    regions = {}
    for i, (hemi, reg) in id_meta.items():
        m = pl == i
        if m.sum() >= 8:
            regions[(hemi, reg)] = {"mask": m, "cen": xyz[m].mean(0)}
    xs = [v["cen"][0] for v in regions.values()]
    print(f"68-region check: n={len(regions)} hemi-regions; centroid-x range [{min(xs):.0f}, {max(xs):.0f}] "
          f"(BUGGY pooled version had all x~0)")

    covmaps = {"mass": lambda: {k: float(np.log10(col[v['mask']].mean() + 1e-300)) for k, v in regions.items()},
               "FCrecon": None, "FCSCrecon": None}
    rc_fc = recon(pi, Mfc, Hfc); rc_fcsc = recon(pi, 0.7 * Mfc + 0.3 * Msc, 0.7 * Hfc + 0.3 * Hsc)
    covmaps["FCrecon"] = lambda: {k: float(np.nanmean(rc_fc[v['mask']])) for k, v in regions.items()}
    covmaps["FCSCrecon"] = lambda: {k: float(np.nanmean(rc_fcsc[v['mask']])) for k, v in regions.items()}

    disorders = {p.stem.replace("cortical_thickness_", ""): enig_hemi(p) for p in sorted(ENIGMA.glob("cortical_thickness_*.csv"))}
    allkeys = [k for k in regions if all(k in d for d in disorders.values())]
    disorders["transdiagnostic"] = {k: float(np.mean([disorders[x][k] for x in list(disorders)])) for k in allkeys}

    out = {"_note": "68 hemi-regions, real centroids, valid whole-brain spin"}
    for cname, fn in covmaps.items():
        cov = fn(); out[cname] = {}; ps = []; ks = []
        for dis, dm in disorders.items():
            keys = [k for k in cov if k in dm and np.isfinite(dm[k]) and np.isfinite(cov[k])]
            c = np.array([cov[k] for k in keys]); d = np.array([dm[k] for k in keys])
            C = np.array([regions[k]["cen"] for k in keys])
            s = spin_null(rankdata(c), rankdata(d), C, n_trials=N_SPIN, seed=0)
            out[cname][dis] = {"rho": float(spearmanr(c, d).statistic), "spin_p": s["p_spin"], "n": len(keys)}
            if dis != "transdiagnostic": ps.append(s["p_spin"]); ks.append(dis)
        for k, q in zip(ks, false_discovery_control(np.array(ps))): out[cname][k]["fdr_q"] = float(q)
        row = "  ".join(f"{d}={out[cname][d]['rho']:+.2f}(p{out[cname][d]['spin_p']:.3f})" for d in ['bipolar','schizophrenia','mdd','transdiagnostic'])
        print(f"{cname:11s} n={out[cname]['bipolar']['n']}  {row}")
    out.update(pi_provenance())   # which coupling produced these numbers
    (ROOT / "outputs/logs/section6_disease_68region.json").write_text(json.dumps(out, indent=2))
    print("wrote section6_disease_68region.json")


def abagen_fetch():
    import abagen
    return abagen.fetch_desikan_killiany()


if __name__ == "__main__":
    main()
