"""Search for a robust disease link on the PRINCIPLED (reconstruction) coverage.

Tests reconstruction-accuracy coverage (FC, and FC+SC combined) on the canonical (soft) and
canonical_sharp couplings, against all 6 ENIGMA disorders + transdiagnostic, DK region level,
Spearman + spin null + FDR. If nothing robust clears, the disorder-reachability claim has no
defensible home on the new metric.

Run under: PYTHONPATH=/var/tmp/pylib:...:src ABAGEN_DATA=/var/tmp/abagen HOME=/var/tmp
Writes outputs/logs/section6_disease_search.json
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
N_SPIN = 1000


def recon(pi, Mc, Hc):
    ph = pi.sum(0); pit = pi / np.maximum(ph, 1e-300); pred = pit.T @ Mc @ pit
    n = pred.shape[0]; out = np.full(n, np.nan)
    for j in range(n):
        a = pred[j].copy(); b = Hc[j].copy(); a[j] = np.nan; b[j] = np.nan
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 10 and a[ok].std() > 1e-9: out[j] = np.corrcoef(a[ok], b[ok])[0, 1]
    return out


def enig(p):
    acc = {}
    for row in csv.DictReader(open(p)):
        s = row["Structure"]
        if "_" not in s: continue
        r = s.split("_", 1)[1].strip().lower()
        try: v = float(row["d_icv"])
        except: continue
        if np.isfinite(v): acc.setdefault(r, []).append(v)
    return {r: float(np.mean(v)) for r, v in acc.items()}


def main():
    import abagen
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    Mfc = np.asarray(M.uns["fc_mean"], float); Hfc = np.asarray(H.uns["fc_mean"], float)
    Msc = np.log1p(np.maximum(np.load(ROOT / "data_external/mouse_sc.npy").astype(float), 0))
    Hsc = np.log1p(np.maximum(np.load(ROOT / "data_external/human_sc.npy").astype(float), 0))
    xyz = H.var[["x", "y", "z"]].to_numpy(float)

    at = abagen.fetch_desikan_killiany(); img = nib.load(at["image"]); info = pd.read_csv(at["info"])
    lab = np.asarray(img.get_fdata()).astype(int); ctx = set(info.loc[info.structure == "cortex", "id"])
    id2n = dict(zip(info.id, info.label))
    vox = nib.affines.apply_affine(np.linalg.inv(img.affine), xyz); vi = np.rint(vox).astype(int)
    inb = np.all((vi >= 0) & (vi < np.array(lab.shape)), axis=1)
    pl = np.zeros(len(xyz), int); pl[inb] = lab[vi[inb, 0], vi[inb, 1], vi[inb, 2]]
    cvx = np.argwhere(np.isin(lab, list(ctx))); need = np.where(pl == 0)[0]
    dd, jj = cKDTree(cvx).query(vox[need]); ok = dd <= 4; hit = cvx[jj[ok]]
    pl[need[ok]] = lab[hit[:, 0], hit[:, 1], hit[:, 2]]; pl[~np.isin(pl, list(ctx))] = 0

    names = sorted({id2n[i] for i in ctx})
    dk = {nm: np.isin(pl, [i for i in ctx if id2n[i] == nm]) for nm in names}
    dk = {nm: m for nm, m in dk.items() if m.sum() >= 10}
    cen = {nm: [abs(xyz[m, 0].mean()), xyz[m, 1].mean(), xyz[m, 2].mean()] for nm, m in dk.items()}
    disorders = {p.stem.replace("cortical_thickness_", ""): enig(p) for p in sorted(ENIGMA.glob("cortical_thickness_*.csv"))}
    common = [r for r in dk if all(r in d for d in disorders.values())]
    disorders["transdiagnostic"] = {r: float(np.mean([disorders[k][r] for k in list(disorders)])) for r in common}

    couplings = {"canonical": load_pi(),
                 "canonical_sharp": np.load(ROOT / "outputs/coupling/pi_canonical_sharp.npy")}
    out = {}
    for cpl, pi in couplings.items():
        for mod, (Mc, Hc) in {"FC": (Mfc, Hfc), "FCSC": (0.7 * Mfc + 0.3 * Msc, 0.7 * Hfc + 0.3 * Hsc)}.items():
            rc = recon(pi, Mc, Hc)
            cov = {nm: float(np.nanmean(rc[m])) for nm, m in dk.items()}
            tag = f"{cpl}/{mod}"; out[tag] = {}; ps = []; ks = []
            for dis, dm in disorders.items():
                regs = sorted(r for r in cov if r in dm and np.isfinite(dm[r]))
                c = np.array([cov[r] for r in regs]); d = np.array([dm[r] for r in regs]); C = np.array([cen[r] for r in regs])
                s = spin_null(rankdata(c), rankdata(d), C, n_trials=N_SPIN, seed=0)
                out[tag][dis] = {"rho": float(spearmanr(c, d).statistic), "spin_p": s["p_spin"]}
                if dis != "transdiagnostic": ps.append(s["p_spin"]); ks.append(dis)
            for k, q in zip(ks, false_discovery_control(np.array(ps))): out[tag][k]["fdr_q"] = float(q)
            best = min(out[tag].items(), key=lambda kv: kv[1]["spin_p"])
            print(f"{tag:22s} best: {best[0]} rho={best[1]['rho']:+.2f} spin_p={best[1]['spin_p']:.3f}  |  "
                  + " ".join(f"{d}={out[tag][d]['rho']:+.2f}({out[tag][d]['spin_p']:.2f})" for d in ['bipolar','schizophrenia','transdiagnostic']))
    out.update(pi_provenance())   # which coupling produced these numbers
    (ROOT / "outputs/logs/section6_disease_search.json").write_text(json.dumps(out, indent=2))
    print("wrote section6_disease_search.json")


if __name__ == "__main__":
    main()
