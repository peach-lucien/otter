"""Validation suite: every map that CLAIMS to be a published quantity must correlate
with that published quantity. Fails loudly if it does not.

WHY THIS EXISTS
---------------
For months the repo used `margulies_2016_gradient.json["human_gradient"]` as if it were
the Margulies principal gradient. It was not: principal_gradient() was returning the first
non-trivial diffusion component, which in this FC data is an ANTERIOR-POSTERIOR spatial
axis. Correlation with the actual published Margulies map: |rho| = 0.12.

The bug survived because nobody ever compared the map to the thing it was named after.
The identical check on the myelin map returns |rho| = 0.97 -- which is exactly why the
myelin pipeline was fine and the gradient was not. One thirty-second check, never run,
sitting upstream of three results sections.

This script runs that check for every published map the repo depends on, and asserts a
minimum correlation. If a map drifts, this fails before any figure or number does.

Also records the Fulcher structural-translation test under the null the repo's own
nulls.py designates as correct for TRANSLATION claims (null B: spin the mouse input and
route it through the real pi), so that all of section 3's claims are judged by one
standard.

Requires: neuromaps.
Run: cd homer && PYTHONPATH=src python experiments/validation/00_validate_published_maps.py
Writes outputs/logs/published_map_validation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached, load_pi              # noqa: E402
from homer.eval.nulls import _haar_rotation              # noqa: E402

OUT = ROOT / "outputs/logs/published_map_validation.json"
N_TRIALS = 1000

# map name -> (neuromaps source, desc, repo array, minimum |rho| we require)
MIN_RHO = {"margulies_gradient": 0.80, "hcp_myelin": 0.90}


def dk_sampler():
    """Sample any fsLR surface map into volumetric Desikan-Killiany regions."""
    import abagen
    atlas = abagen.fetch_desikan_killiany()
    img = nib.load(atlas["image"])
    info = pd.read_csv(atlas["info"])
    lab = np.asarray(img.get_fdata()).astype(int)
    inv = np.linalg.inv(img.affine)
    id2n = dict(zip(info.id, info.label))
    ctx_ids = set(info.loc[info.structure == "cortex", "id"])
    return img, lab, inv, id2n, ctx_ids


def surf_to_dk(source, desc, lab, inv, id2n, ctx_ids):
    from neuromaps.datasets import fetch_annotation, fetch_atlas
    atlas = fetch_atlas("fsLR", "32k")
    acc = {}
    for hemi in ("L", "R"):
        try:
            f = fetch_annotation(source=source, desc=desc, space="fsLR", hemi=hemi)
        except Exception:
            continue
        f = f if isinstance(f, str) else (f[0] if isinstance(f, (list, tuple)) else f.get(hemi))
        if f is None or f"hemi-{hemi}" not in str(f):
            continue
        val = nib.load(f).agg_data()
        val = np.asarray(val[0] if isinstance(val, tuple) else val)
        mt = [p for p in atlas["midthickness"] if f"hemi-{hemi}" in str(p)][0]
        xyz = np.asarray(nib.load(mt).agg_data()[0])
        if len(xyz) != len(val):
            continue
        vv = np.rint(nib.affines.apply_affine(inv, xyz)).astype(int)
        ok = np.all((vv >= 0) & (vv < np.array(lab.shape)), 1)
        lv = np.zeros(len(xyz), int)
        lv[ok] = lab[vv[ok, 0], vv[ok, 1], vv[ok, 2]]
        for k in np.unique(lv):
            if k not in ctx_ids:
                continue
            m = (lv == k) & np.isfinite(val) & (val != 0)
            if m.sum() > 20:
                acc.setdefault(id2n[k], []).append(float(val[m].mean()))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def parcels_to_dk(vec, plab, id2n, ctx_ids):
    out = {}
    for name in {id2n[i] for i in ctx_ids}:
        m = np.isin(plab, [i for i in id2n if id2n[i] == name]) & np.isfinite(vec)
        if m.sum() >= 10:
            out[name] = float(vec[m].mean())
    return out


def main():
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    img, lab, inv, id2n, ctx_ids = dk_sampler()

    vox = nib.affines.apply_affine(inv, xyz)
    vi = np.rint(vox).astype(int)
    inb = np.all((vi >= 0) & (vi < np.array(lab.shape)), 1)
    plab = np.zeros(len(xyz), int)
    plab[inb] = lab[vi[inb, 0], vi[inb, 1], vi[inb, 2]]
    lv = np.argwhere(lab > 0)
    need = np.where(plab == 0)[0]
    d, j = cKDTree(lv).query(vox[need])
    ok = d <= 4.0
    plab[need[ok]] = lab[lv[j[ok]][:, 0], lv[j[ok]][:, 1], lv[j[ok]][:, 2]]

    repo_grad = np.asarray(json.loads(
        (ROOT / "outputs/logs/margulies_2016_gradient.json").read_text())["human_gradient"], float)
    repo_myelin = np.asarray(json.loads(
        (ROOT / "outputs/logs/buckner_krienen_2013_tethering.json").read_text())["myelin_per_parcel"], float)

    res, failures = {}, []
    checks = [
        ("margulies_gradient", "margulies2016", "fcgradient01", repo_grad),
        ("hcp_myelin", "hcps1200", "myelinmap", repo_myelin),
    ]
    print("PUBLISHED-MAP VALIDATION (repo array vs the published source it is named after)")
    for name, src, desc, arr in checks:
        pub = surf_to_dk(src, desc, lab, inv, id2n, ctx_ids)
        rep = parcels_to_dk(arr, plab, id2n, ctx_ids)
        regs = sorted(set(pub) & set(rep))
        rho = float(spearmanr([rep[r] for r in regs], [pub[r] for r in regs]).statistic)
        need_rho = MIN_RHO[name]
        passed = abs(rho) >= need_rho
        res[name] = {"abs_spearman_vs_published": abs(rho), "n_dk_regions": len(regs),
                     "required_min": need_rho, "passed": passed}
        print(f"  {name:22s} |rho| = {abs(rho):.3f}  (require >= {need_rho})  "
              f"{'PASS' if passed else '*** FAIL ***'}")
        if not passed:
            failures.append(name)

    # ---- Fulcher structural translation, under the TRANSLATION null (null B) ----
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fu", ROOT / "experiments/fulcher_2019_multimodal_gradient/01_gradient_validation.py")
    fu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fu)
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    pi = load_pi()
    acr = fu.load_mouse_parcel_acronyms()
    mouse_xyz = M.var[["x", "y", "z"]].to_numpy(float)
    maps = {"t1w_t2w": np.array([fu.load_mouse_t1t2().get(a, np.nan) for a in acr], float),
            "cytoarchitecture": np.array([fu.load_mouse_cytoarch().get(a, np.nan) for a in acr], float)}

    def route(v):
        m = np.isfinite(v)
        num = v[m] @ pi[m]
        den = pi[m].sum(0)
        o = np.full(pi.shape[1], np.nan)
        good = den > 1e-12
        o[good] = num[good] / den[good]
        return o

    def obs_r(v):
        p = route(v)
        m = np.isfinite(p) & np.isfinite(repo_myelin)
        return abs(pearsonr(p[m], repo_myelin[m])[0]), int(m.sum())

    rng = np.random.default_rng(0)
    print("\nFULCHER STRUCTURAL TRANSLATION, null B (spin the mouse input, route through the real pi)")
    res["fulcher_translation_null_b"] = {}
    for name, v in maps.items():
        r, n = obs_r(v)
        idx = np.where(np.isfinite(v))[0]
        C = mouse_xyz[idx]
        tree = cKDTree(C)
        vals = v[idx]
        null = []
        for _ in range(N_TRIALS):
            R = _haar_rotation(rng)
            _, nn = tree.query(C @ R.T)
            vv = np.full(len(v), np.nan)
            vv[idx] = vals[nn]
            null.append(obs_r(vv)[0])
        null = np.array(null)
        p = float((np.sum(null >= r) + 1) / (N_TRIALS + 1))
        res["fulcher_translation_null_b"][name] = {
            "abs_r": float(r), "n_parcels": n, "null_abs_mean": float(null.mean()),
            "p_spin": p, "survives": bool(p < 0.05)}
        print(f"  {name:20s} |r| = {r:.3f}  null {null.mean():.3f}  p = {p:.4f}  "
              f"{'survives' if p < 0.05 else 'does NOT survive'}")

    res["_meta"] = {"n_trials": N_TRIALS,
                    "note": "null B is the null homer.eval.nulls designates for translation claims"}
    OUT.write_text(json.dumps(res, indent=2))
    print("\nwrote", OUT)
    if failures:
        raise SystemExit(f"VALIDATION FAILED for: {', '.join(failures)}")


if __name__ == "__main__":
    main()
