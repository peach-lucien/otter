"""Do human cortical disorders concentrate in the cortex HOMER leaves UNCOVERED?

Definitive version. Supersedes two earlier attempts, both of which were wrong:

  v1 (null result)  - SUMMED coverage per region instead of the mass-normalised MEAN.
                      This is the parcel-count confound already fixed elsewhere in S5.
                      Re-running the final pipeline with summed coverage reproduces the
                      false null exactly (bipolar rho=+0.08, p=0.74), confirming the bug
                      was the whole story.
  v2 (sphere ROIs)  - approximated each DK region by a 20 mm sphere around its centroid.
                      Overlapping ROIs, an arbitrary radius, and results that flipped
                      between 12 mm and 20 mm. Not trustworthy.

This version uses the REAL volumetric Desikan-Killiany atlas (abagen, MNI152 1 mm,
68 cortical regions) and assigns every HOMER human parcel to the DK label at its MNI
coordinate; parcels landing outside the labelled ribbon are rescued to the nearest
cortical voxel within 4 mm. Coverage per DK region is the MASS-NORMALISED MEAN of the
pi column-sums (log10). Statistics are rank-based (Spearman), because log-coverage is
heavy-tailed - one region (parsorbitalis) sits at ~1e-15 and single-handedly wrecks a
Pearson correlation. Significance comes from the repo's spin null over DK centroids.

ENIGMA d is negative for cortical thinning, so if disorders preferentially thin the
LOW-coverage cortex we expect a POSITIVE coverage-vs-d correlation.

Requires: pip install abagen
Run: cd homer && PYTHONPATH=src python experiments/section5_coverage_rigor/06_disorder_vs_coverage.py
Writes outputs/logs/section6_disorder_vs_coverage_DK.json
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr, false_discovery_control

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached, load_pi          # noqa: E402
from homer.eval.nulls import spin_null               # noqa: E402

N_SPIN = 2000
RESCUE_MM = 4.0        # parcels off the labelled ribbon snap to nearest cortical voxel
MIN_PARCELS = 10       # DK regions with fewer HOMER parcels are dropped
ENIGMA = ROOT / "data_external/enigma"
OUT = ROOT / "outputs/logs/section6_disorder_vs_coverage_DK.json"


def dk_parcel_labels(xyz):
    """Assign each HOMER human parcel to a volumetric Desikan-Killiany cortical region."""
    import abagen
    atlas = abagen.fetch_desikan_killiany()
    img = nib.load(atlas["image"])
    info = pd.read_csv(atlas["info"])
    lab = np.asarray(img.get_fdata()).astype(int)
    ctx_ids = set(info.loc[info.structure == "cortex", "id"])
    id2name = dict(zip(info.id, info.label))

    vox = nib.affines.apply_affine(np.linalg.inv(img.affine), xyz)
    vi = np.rint(vox).astype(int)
    inb = np.all((vi >= 0) & (vi < np.array(lab.shape)), axis=1)
    plab = np.zeros(len(xyz), int)
    plab[inb] = lab[vi[inb, 0], vi[inb, 1], vi[inb, 2]]

    cortical_vox = np.argwhere(np.isin(lab, list(ctx_ids)))
    need = np.where(plab == 0)[0]
    dist, j = cKDTree(cortical_vox).query(vox[need])
    ok = dist <= RESCUE_MM
    hit = cortical_vox[j[ok]]
    plab[need[ok]] = lab[hit[:, 0], hit[:, 1], hit[:, 2]]

    plab[~np.isin(plab, list(ctx_ids))] = 0
    return plab, ctx_ids, id2name


def enigma_d(path):
    """{dk_region: mean Cohen's d across hemispheres} from an ENIGMA cortical-thickness CSV."""
    acc = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            s = row["Structure"]
            if "_" not in s:
                continue
            reg = s.split("_", 1)[1].strip().lower()
            try:
                v = float(row["d_icv"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                acc.setdefault(reg, []).append(v)
    return {r: float(np.mean(v)) for r, v in acc.items() if v}


def partial_spearman(x, y, z):
    """Spearman(x, y) with z regressed out of both (all inputs already ranked)."""
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(spearmanr(rx, ry).statistic)


def main():
    pi = load_pi()
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    col = pi.sum(0)                                    # mouse mass landing on each human parcel
    xyz = H.var[["x", "y", "z"]].to_numpy(float)

    plab, ctx_ids, id2name = dk_parcel_labels(xyz)
    print(f"parcels assigned to a DK cortical region: {(plab > 0).sum()} / {len(xyz)}")

    # hierarchy covariate (T1w/T2w myelin), to show coverage is not just a hierarchy proxy
    bk = json.loads((ROOT / "outputs/logs/buckner_krienen_2013_tethering.json").read_text())
    myelin = np.asarray(bk["myelin_per_parcel"], float)

    cov, cen, mye = {}, {}, {}
    for name in sorted(set(id2name[i] for i in ctx_ids)):
        ids = [i for i in ctx_ids if id2name[i] == name]
        m = np.isin(plab, ids)
        if m.sum() < MIN_PARCELS:
            continue
        cov[name] = float(np.log10(col[m].mean() + 1e-300))     # MASS-NORMALISED MEAN
        c = xyz[m].mean(0)
        cen[name] = [abs(c[0]), c[1], c[2]]
        mm = m & np.isfinite(myelin)
        mye[name] = float(myelin[mm].mean()) if mm.any() else np.nan

    order = sorted(cov, key=cov.get)
    print("least covered:", ", ".join(order[:5]))
    print("most covered :", ", ".join(order[-5:]))

    disorders = {p.stem.replace("cortical_thickness_", ""): enigma_d(p)
                 for p in sorted(ENIGMA.glob("cortical_thickness_*.csv"))}
    common = [r for r in cov if all(r in d and np.isfinite(d[r]) for d in disorders.values())]
    tests = dict(disorders)
    tests["transdiagnostic"] = {r: float(np.mean([disorders[k][r] for k in disorders]))
                                for r in common}

    res, ps, keys = {}, [], []
    print(f"\n{'disorder':16s} {'n':>3s} {'mean|d|':>8s} {'rho':>8s} {'spin p':>8s} {'partial':>8s}")
    for name, dmap in tests.items():
        regs = sorted(r for r in cov if r in dmap and np.isfinite(dmap[r]))
        c = np.array([cov[r] for r in regs])
        d = np.array([dmap[r] for r in regs])
        C = np.array([cen[r] for r in regs])
        s = spin_null(rankdata(c), rankdata(d), C, n_trials=N_SPIN, seed=0)
        rho = float(spearmanr(c, d).statistic)
        ph = partial_spearman(rankdata(c), rankdata(d),
                              rankdata([mye[r] for r in regs]))
        res[name] = {"n_regions": len(regs), "spearman": rho, "spin_p": s["p_spin"],
                     "mean_abs_d": float(np.abs(d).mean()), "spearman_partial_hierarchy": ph}
        if name != "transdiagnostic":
            ps.append(s["p_spin"])
            keys.append(name)
        print(f"{name:16s} {len(regs):3d} {np.abs(d).mean():8.3f} "
              f"{rho:+8.3f} {s['spin_p']:8.4f} {ph:+8.3f}")

    for k, q in zip(keys, false_discovery_control(np.array(ps))):
        res[k]["fdr_q"] = float(q)
    print("\nFDR q:", {k: round(res[k]["fdr_q"], 4) for k in keys})

    res["_meta"] = {
        "atlas": "Desikan-Killiany volumetric (abagen, MNI152 1mm, 68 cortical regions)",
        "coverage": "log10 of mass-normalised MEAN pi column-sum over each region's HOMER parcels",
        "statistic": "Spearman; spin null over DK centroids",
        "n_spin": N_SPIN, "rescue_mm": RESCUE_MM, "min_parcels": MIN_PARCELS,
        "coverage_vs_myelin_spearman": float(spearmanr(
            [cov[r] for r in sorted(cov)], [mye[r] for r in sorted(cov)]).statistic),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
