"""HOMER coverage vs a battery of published hierarchy / evolution maps.

Each map is fetched from neuromaps, resampled to fsLR-32k by nearest-neighbour on the
sphere (workbench-free), parcellated into Schaefer-400, and correlated against HOMER
coverage with the repo's spin null over parcel centroids.

Coverage per Schaefer region is the MASS-NORMALISED MEAN of the pi column-sums, not the
sum: summed coverage scales with the number of HOMER parcels inside a region (rho = 0.35
with parcel count), which is a size confound rather than a biological signal. Every
conclusion in the battery is unchanged under mass-normalisation, but the rho values move
slightly, so this is the version the figures must be built from.

Requires: neuromaps, netneurotools.
Run: cd homer && PYTHONPATH=src python experiments/section5_coverage_rigor/07_evolution_battery.py
Writes outputs/logs/section5_evolution_battery.json
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from neuromaps.datasets import fetch_annotation, fetch_atlas
from netneurotools.datasets import fetch_schaefer2018

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached, load_pi          # noqa: E402
from homer.eval.nulls import spin_null               # noqa: E402

N_SPIN = 1000
OUT = ROOT / "outputs/logs/section5_evolution_battery.json"

MAPS = {
    "Xu2020 mouse→human expansion":      ("xu2020", "evoexp"),
    "Xu2020 mouse–human FC homology":    ("xu2020", "FChomology"),
    "Hill2010 macaque→human expansion":  ("hill2010", "evoexp"),
    "Hill2010 developmental expansion":  ("hill2010", "devexp"),
    "Sydnor2021 S–A axis":               ("sydnor2021", "SAaxis"),
    "Margulies2016 principal gradient":  ("margulies2016", "fcgradient01"),
    "HCP T1w/T2w hierarchy":             ("hcps1200", "myelinmap"),
}


def _sphere(hemi, den):
    atl = fetch_atlas("fsLR", den)
    p = [f for f in atl["sphere"] if f"hemi-{hemi}" in str(f)][0]
    return np.asarray(nib.load(p).agg_data()[0])


def _values(path):
    a = nib.load(path).agg_data()
    return np.asarray(a[0] if isinstance(a, tuple) else a)


def schaefer_labels():
    # netneurotools ignores NNT_DATA_DIR; pass it explicitly so the cache is redirectable
    kw = {"data_dir": os.environ["NNT_DATA_DIR"]} if os.environ.get("NNT_DATA_DIR") else {}
    dl = fetch_schaefer2018("fslr32k", **kw)["400Parcels17Networks"]
    lab = np.asarray(nib.load(dl).get_fdata()).ravel().astype(int)
    n = lab.shape[0] // 2
    return {"L": lab[:n], "R": lab[n:]}


def map_to_schaefer(source, desc, labs):
    """{schaefer_id: mean value}, resampling each available hemisphere to fsLR-32k."""
    out = {}
    for hemi in ("L", "R"):
        try:
            f = fetch_annotation(source=source, desc=desc, space="fsLR", hemi=hemi)
        except Exception:
            continue
        f = f if isinstance(f, str) else (f[0] if isinstance(f, (list, tuple)) else f.get(hemi))
        if f is None or f"hemi-{hemi}" not in str(f):
            continue                       # e.g. Hill2010 is right-hemisphere only
        val = _values(f)
        if val.shape[0] > 60000:           # 164k native -> nearest-neighbour to 32k
            _, idx = cKDTree(_sphere(hemi, "164k")).query(_sphere(hemi, "32k"))
            val = val[idx]
        lab = labs[hemi]
        for k in np.unique(lab):
            if k <= 0:
                continue
            m = (lab == k) & np.isfinite(val) & (val != 0)
            if m.any():
                out[int(k)] = float(val[m].mean())
    return out


def main():
    labs = schaefer_labels()
    pi = load_pi()
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    node_region = np.asarray(json.loads(
        (ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    col = pi.sum(0)

    ids = [k for k in range(1, 401) if (node_region == k).any()]
    cov = {k: np.log10(col[node_region == k].mean() + 1e-300) for k in ids}   # MEAN, not sum
    cen = {k: xyz[node_region == k].mean(0) for k in ids}

    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for label, (src, desc) in MAPS.items():
        try:
            mp = map_to_schaefer(src, desc, labs)
            keep = [k for k in ids if k in mp]
            c = np.array([cov[k] for k in keep])
            m = np.array([mp[k] for k in keep])
            C = np.array([cen[k] for k in keep])
            s = spin_null(c, m, C, n_trials=N_SPIN, seed=0)
            res[label] = {"n": len(keep), "spearman": float(spearmanr(c, m).statistic),
                          "pearson_spin_r": s["r_observed"], "spin_p": s["p_spin"]}
            print(f"{label:38s} n={len(keep):3d} rho={res[label]['spearman']:+.3f} "
                  f"spin_p={s['p_spin']:.4f}")
        except Exception as e:                       # a missing annotation must not kill the run
            print("FAIL", label, repr(e)[:90])
            res[label] = {"error": repr(e)[:200]}
        OUT.write_text(json.dumps(res, indent=2))

    res["_meta"] = {"coverage": "log10 mass-normalised MEAN pi column-sum per Schaefer-400 region",
                    "n_spin": N_SPIN}
    OUT.write_text(json.dumps(res, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
