"""Section 6 on the NEW coverage: does disorder cortical thinning concentrate where human
connectivity is least reconstructable from the mouse?

Same DK-atlas + ENIGMA pipeline as 06_disorder_vs_coverage.py, but coverage per DK region is the
reconstruction-fidelity coverage (mean per-parcel FC reconstruction r) on the CANONICAL pi, not the
mass-normalised column-sum. Low recon-coverage = connectivity has no mouse basis. ENIGMA d is
negative for thinning, so if thinning concentrates in low-recon cortex we expect POSITIVE rho.

Run: cd homer && PYTHONPATH=src python experiments/section5_coverage_rigor/24_disease_reconstruction.py
Writes outputs/logs/section6_disorder_vs_reconstruction_DK.json
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
from homer.data import load_cached, load_pi, pi_provenance  # noqa: E402
from homer.eval.nulls import spin_null                  # noqa: E402

N_SPIN = 2000
RESCUE_MM = 4.0
MIN_PARCELS = 10
ENIGMA = ROOT / "data_external/enigma"
OUT = ROOT / "outputs/logs/section6_disorder_vs_reconstruction_DK.json"


def recon_coverage(pi, Mfc, Hfc):
    ph = pi.sum(0); pit = pi / np.maximum(ph, 1e-300); pred = pit.T @ Mfc @ pit
    n = pred.shape[0]; out = np.full(n, np.nan)
    for j in range(n):
        a = pred[j].copy(); b = Hfc[j].copy(); a[j] = np.nan; b[j] = np.nan
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 10 and a[ok].std() > 1e-9:
            out[j] = np.corrcoef(a[ok], b[ok])[0, 1]
    return out


def dk_parcel_labels(xyz):
    import abagen
    atlas = abagen.fetch_desikan_killiany()
    img = nib.load(atlas["image"]); info = pd.read_csv(atlas["info"])
    lab = np.asarray(img.get_fdata()).astype(int)
    ctx_ids = set(info.loc[info.structure == "cortex", "id"]); id2name = dict(zip(info.id, info.label))
    vox = nib.affines.apply_affine(np.linalg.inv(img.affine), xyz); vi = np.rint(vox).astype(int)
    inb = np.all((vi >= 0) & (vi < np.array(lab.shape)), axis=1)
    plab = np.zeros(len(xyz), int); plab[inb] = lab[vi[inb, 0], vi[inb, 1], vi[inb, 2]]
    cortical_vox = np.argwhere(np.isin(lab, list(ctx_ids)))
    need = np.where(plab == 0)[0]; dist, j = cKDTree(cortical_vox).query(vox[need]); ok = dist <= RESCUE_MM
    hit = cortical_vox[j[ok]]; plab[need[ok]] = lab[hit[:, 0], hit[:, 1], hit[:, 2]]
    plab[~np.isin(plab, list(ctx_ids))] = 0
    return plab, ctx_ids, id2name


def enigma_d(path):
    acc = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            s = row["Structure"]
            if "_" not in s: continue
            reg = s.split("_", 1)[1].strip().lower()
            try: v = float(row["d_icv"])
            except (TypeError, ValueError): continue
            if np.isfinite(v): acc.setdefault(reg, []).append(v)
    return {r: float(np.mean(v)) for r, v in acc.items() if v}


def partial_spearman(x, y, z):
    rx = x - np.polyval(np.polyfit(z, x, 1), z); ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(spearmanr(rx, ry).statistic)


def main():
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    Mfc = np.asarray(M.uns["fc_mean"], float); Hfc = np.asarray(H.uns["fc_mean"], float)
    pi = load_pi()
    rc = recon_coverage(pi, Mfc, Hfc)
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    plab, ctx_ids, id2name = dk_parcel_labels(xyz)
    print(f"parcels assigned to DK cortex: {(plab > 0).sum()} / {len(xyz)}")
    myelin = np.asarray(json.loads((ROOT / "outputs/logs/buckner_krienen_2013_tethering.json").read_text())["myelin_per_parcel"], float)

    cov, cen, mye = {}, {}, {}
    for name in sorted(set(id2name[i] for i in ctx_ids)):
        ids = [i for i in ctx_ids if id2name[i] == name]
        m = np.isin(plab, ids)
        if m.sum() < MIN_PARCELS: continue
        cov[name] = float(np.nanmean(rc[m]))                    # RECONSTRUCTION coverage mean
        c = xyz[m].mean(0); cen[name] = [abs(c[0]), c[1], c[2]]
        mm = m & np.isfinite(myelin); mye[name] = float(myelin[mm].mean()) if mm.any() else np.nan

    order = sorted(cov, key=cov.get)
    print("least reconstructable:", ", ".join(order[:5]))
    print("most reconstructable :", ", ".join(order[-5:]))

    disorders = {p.stem.replace("cortical_thickness_", ""): enigma_d(p)
                 for p in sorted(ENIGMA.glob("cortical_thickness_*.csv"))}
    common = [r for r in cov if all(r in d and np.isfinite(d[r]) for d in disorders.values())]
    tests = dict(disorders)
    tests["transdiagnostic"] = {r: float(np.mean([disorders[k][r] for k in disorders])) for r in common}

    res, ps, keys = {}, [], []
    print(f"\n{'disorder':16s} {'n':>3s} {'rho':>8s} {'spin p':>8s} {'partial':>8s}")
    for name, dmap in tests.items():
        regs = sorted(r for r in cov if r in dmap and np.isfinite(dmap[r]))
        c = np.array([cov[r] for r in regs]); d = np.array([dmap[r] for r in regs]); C = np.array([cen[r] for r in regs])
        s = spin_null(rankdata(c), rankdata(d), C, n_trials=N_SPIN, seed=0)
        rho = float(spearmanr(c, d).statistic)
        ph = partial_spearman(rankdata(c), rankdata(d), rankdata([mye[r] for r in regs]))
        res[name] = {"n_regions": len(regs), "spearman": rho, "spin_p": s["p_spin"], "spearman_partial_hierarchy": ph}
        if name != "transdiagnostic": ps.append(s["p_spin"]); keys.append(name)
        print(f"{name:16s} {len(regs):3d} {rho:+8.3f} {s['p_spin']:8.4f} {ph:+8.3f}")

    for k, q in zip(keys, false_discovery_control(np.array(ps))):
        res[k]["fdr_q"] = float(q)
    print("\nFDR q:", {k: round(res[k]["fdr_q"], 4) for k in keys})
    res["_meta"] = {"coverage": "reconstruction-fidelity (mean per-parcel FC recon r) on pi_canonical",
                    "atlas": "Desikan-Killiany (abagen)", "n_spin": N_SPIN}
    res.update(pi_provenance())   # which coupling produced these numbers
    OUT.write_text(json.dumps(res, indent=2)); print("wrote", OUT)


if __name__ == "__main__":
    main()
