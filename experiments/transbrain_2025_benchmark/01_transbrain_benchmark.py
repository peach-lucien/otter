"""OTTER × TransBrain 2025, benchmark against a sibling method.

[Huang et al. 2025, Nature Methods](https://doi.org/10.1038/s41592-025-02961-3),
"TransBrain: a computational framework for translating brain-wide
phenotypes between humans and mice", is a published mouse↔human phenotype-
translation framework, a direct sibling/competitor to OTTER. It works at
region level (68-region mouse atlas; Brainnetome / DK / AAL human atlases)
via graph embeddings + dual regression.

This is a **methods-landscape** comparison, not a validation "pass".

**Part A. Homology benchmark.** TransBrain validated its mapping against a
literature-curated set of classic mouse↔human homologous region pairs
(`homo_cortex.csv`, `homo_subcortex.csv`), a set OTTER has never seen
(independent of the Garin anchors and the Beauchamp benchmark). For each
benchmarked mouse region we route OTTER's π and measure (1) whether the
literature-homolog human Brainnetome region is in the top-K, and (2) how
far OTTER's predicted human centroid is from the homolog (mm), the
resolution-fair metric.

**Part B. Head-to-head.** We translate the same mouse phenotype with BOTH
methods, on two phenotypes from TransBrain's own case studies:
  * a **smooth** one, the resting-fMRI principal gradient (expect convergence);
  * a **noisy** one, the Magel2 autism-model mutation pattern, with
    TransBrain's per-individual ASD risk-score workflow (expect divergence).

Requires `pip install transbrain` (Apache-2.0). The human Brainnetome atlas
is taken from the installed TransBrain package.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from otter.data import load_cached

try:
    import nibabel as nib
    from transbrain.config import Config
    from transbrain.trans import SpeciesTrans
except ImportError as e:  # pragma: no cover
    raise ImportError("This experiment needs TransBrain: `pip install transbrain`") from e

ANN = ROOT / "outputs" / "anndata"
DATA = ROOT / "data_external" / "transbrain_2025"
# Resolved through load_pi() rather than hardcoded: the name is right today, but a
# hardcoded path is how the July 2026 mix-up survived a re-run. pi_provenance() then
# records the sha256 actually loaded, so the log states which coupling it used rather
# than which one it intended to use.
N_NULL = 200
SEED = 42


# ---------------------------------------------------------------------------
# Brainnetome atlas
# ---------------------------------------------------------------------------
def load_bn_atlas(H_var):
    """Return (bn_id per OTTER human parcel, id→name, name→MNI centroid)."""
    nii = nib.load(Config.bnatlas_path)
    vol = np.asarray(nii.dataobj).astype(int)
    aff = nii.affine
    inv = np.linalg.inv(aff)
    lab = pd.read_csv(Config.bnatlas_label_path, index_col=0)
    id2name = dict(zip(lab["Atlas Index"].astype(int), lab["Anatomical Name"]))

    name2centroid = {}
    for rid, name in id2name.items():
        vox = np.argwhere(vol == rid)
        if len(vox):
            name2centroid[name] = ((aff[:3, :3] @ vox.T).T + aff[:3, 3]).mean(axis=0)

    xyz = H_var[["x", "y", "z"]].to_numpy(float)
    vox = np.round((inv[:3, :3] @ xyz.T).T + inv[:3, 3]).astype(int)
    shp = np.array(vol.shape)
    bn_id = np.zeros(len(xyz), int)
    for k in range(len(xyz)):
        v = vox[k]
        if not ((v >= 0).all() and (v < shp).all()):
            continue
        lbl = vol[v[0], v[1], v[2]]
        if lbl == 0:
            sl = tuple(slice(max(0, v[d] - 1), min(shp[d], v[d] + 2)) for d in range(3))
            nb = vol[sl][vol[sl] > 0]
            if nb.size:
                lbl = int(np.bincount(nb).argmax())
        bn_id[k] = lbl
    return bn_id, id2name, name2centroid


def aggregate_bn(vals, bn_id, id2name, reduce="mean"):
    """Reduce a per-parcel vector into BN regions (keyed by name)."""
    out = {}
    for rid in np.unique(bn_id):
        if not rid:
            continue
        v = vals[(bn_id == rid) & np.isfinite(vals)]
        if v.size:
            out[id2name[rid]] = float(v.sum() if reduce == "sum" else v.mean())
    return out


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------
def route_norm(value_by_acr, parcel_acr, pi):
    """Transport-weighted average of a mouse acronym→value map through π."""
    mvec = np.array([value_by_acr.get(a, np.nan) for a in parcel_acr])
    mask = np.isfinite(mvec)
    num = mvec[mask] @ pi[mask, :]
    den = pi[mask, :].sum(axis=0)
    out = np.full(pi.shape[1], np.nan)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def transbrain_translate(value_by_acr, region_type, col="phenotype"):
    """Translate a mouse acronym→value map to human BN via TransBrain."""
    df = pd.DataFrame({col: value_by_acr}).dropna()
    st = SpeciesTrans(atlas_type="bn")
    return st.mouse_to_human(df, region_type=region_type).iloc[:, 0].to_dict()


# ---------------------------------------------------------------------------
# Part A, homology benchmark
# ---------------------------------------------------------------------------
def homology_benchmark(pi, parcel_acr, H_var, bn_id, id2name, centroid,
                       csv_path, label):
    bench = pd.read_csv(csv_path, index_col=0)
    homologs: dict[str, set] = {}
    for _, row in bench.iterrows():
        homologs.setdefault(row["mouse_region"], set()).add(row["human_region"])
    xyz = H_var[["x", "y", "z"]].to_numpy(float)
    n_bn = len(set(id2name.values()))

    def evaluate(mouse_region, pi_use):
        idx = [i for i, a in enumerate(parcel_acr) if a == mouse_region]
        if not idx:
            return None
        pred = pi_use[idx].sum(axis=0)
        mass = aggregate_bn(pred, bn_id, id2name, reduce="sum")
        ranked = sorted(mass, key=mass.get, reverse=True)
        ranks = [ranked.index(h) + 1 for h in homologs[mouse_region] if h in ranked]
        best_rank = min(ranks) if ranks else n_bn
        w = pred / pred.sum()
        pred_centroid = (w[:, None] * xyz).sum(axis=0)
        dists = [np.linalg.norm(pred_centroid - centroid[h])
                 for h in homologs[mouse_region] if h in centroid]
        return best_rank, (min(dists) if dists else np.nan)

    rows, ranks, dmm = [], [], []
    for mr in sorted(homologs):
        res = evaluate(mr, pi)
        if res is None:
            continue
        ranks.append(res[0])
        dmm.append(res[1])
        rows.append({"mouse_region": mr, "n_homologs": len(homologs[mr]),
                     "best_rank": res[0], "centroid_dist_mm": res[1]})
    ranks = np.array(ranks)
    dmm = np.array([d for d in dmm if np.isfinite(d)])
    top1, top3, top5 = [float((ranks <= k).mean()) for k in (1, 3, 5)]
    chance1 = float(np.mean([r["n_homologs"] for r in rows]) / n_bn)

    rng = np.random.default_rng(SEED)
    mrs = [r["mouse_region"] for r in rows]
    null_top3, null_dist = [], []
    for _ in range(N_NULL):
        pin = pi[rng.permutation(pi.shape[0])]
        ev = [e for e in (evaluate(mr, pin) for mr in mrs) if e is not None]
        null_top3.append(np.mean([e[0] <= 3 for e in ev]))
        null_dist.append(np.nanmean([e[1] for e in ev]))
    null_top3 = np.array(null_top3)
    null_dist = np.array(null_dist)

    print(f"\n[Part A] Homology benchmark, {label}  ({len(rows)} mouse regions, "
          f"{n_bn} BN regions)")
    print(f"  top-1 {top1:.0%}  top-3 {top3:.0%}  top-5 {top5:.0%}   "
          f"(chance top-1 ≈ {chance1:.1%})")
    print(f"  permuted-π null top-3 = {null_top3.mean():.0%}   "
          f"empirical p = {(null_top3 >= top3).mean():.3f}")
    print(f"  centroid distance to literature homologue: {dmm.mean():.1f} mm  "
          f"(null {null_dist.mean():.1f} mm, p = {(null_dist <= dmm.mean()).mean():.3f})")
    return {
        "label": label, "n_regions": len(rows), "n_bn_regions": n_bn,
        "top1": top1, "top3": top3, "top5": top5, "chance_top1": chance1,
        "null_top3_mean": float(null_top3.mean()),
        "null_top3_empirical_p": float((null_top3 >= top3).mean()),
        "centroid_dist_mm": float(dmm.mean()),
        "null_centroid_dist_mm": float(null_dist.mean()),
        "centroid_empirical_p": float((null_dist <= dmm.mean()).mean()),
        "per_region": rows,
    }


# ---------------------------------------------------------------------------
# Part B, head-to-head
# ---------------------------------------------------------------------------
def head_to_head_gradient(pi, parcel_acr, bn_id, id2name):
    """Smooth phenotype: the resting-fMRI principal gradient."""
    mg = json.loads((ROOT / "outputs/logs/margulies_2016_gradient.json").read_text())
    mouse_grad = np.array(mg["mouse_gradient"])          # per OTTER mouse parcel
    human_grad = np.array(mg["human_gradient"])          # per OTTER human parcel

    # common mouse input: gradient averaged into the 39 mouse cortical acronyms.
    # 4 acronyms absent from OTTER's parcellation (PTLp/VISal/PERI/AUDpo) are
    # filled with the neutral mean so TransBrain receives its full region set.
    by_acr: dict[str, list] = {}
    for a, v in zip(parcel_acr, mouse_grad):
        if a in Config.MOUSE_CORTICAL:
            by_acr.setdefault(a, []).append(v)
    avail = {a: float(np.mean(v)) for a, v in by_acr.items()}
    fill = float(np.mean(list(avail.values())))
    mouse_in = {a: avail.get(a, fill) for a in Config.MOUSE_CORTICAL}

    otter = aggregate_bn(route_norm(mouse_in, parcel_acr, pi), bn_id, id2name)
    tb = transbrain_translate(mouse_in, "cortex", col="gradient")
    ref = aggregate_bn(human_grad, bn_id, id2name)

    shared = sorted(set(otter) & set(tb) & set(ref) & set(Config.BN_CORTICAL))
    h = np.array([otter[r] for r in shared])
    t = np.array([tb[r] for r in shared])
    rf = np.array([ref[r] for r in shared])
    # gradients are sign-ambiguous, report |r|
    r_h = abs(pearsonr(h, rf)[0])
    r_t = abs(pearsonr(t, rf)[0])
    r_ht = abs(pearsonr(h, t)[0])
    print(f"\n[Part B-1] Head-to-head, resting-fMRI gradient "
          f"({len(shared)} BN cortical regions)")
    print(f"  OTTER vs observed human gradient:      |r| = {r_h:.3f}")
    print(f"  TransBrain vs observed human gradient: |r| = {r_t:.3f}")
    print(f"  OTTER vs TransBrain (method agreement):|r| = {r_ht:.3f}")
    return {"n_bn_regions": len(shared), "otter_vs_reference": r_h,
            "transbrain_vs_reference": r_t, "otter_vs_transbrain": r_ht}


def head_to_head_autism(pi, parcel_acr, bn_id, id2name):
    """Noisy phenotype: the Magel2 autism-model mutation pattern."""
    magel2 = pd.read_csv(DATA / "magel2_mutation_pattern.csv", index_col=0)
    mouse_in = magel2["Magel2"].to_dict()

    otter = aggregate_bn(route_norm(mouse_in, parcel_acr, pi), bn_id, id2name)
    tb = transbrain_translate(mouse_in, "all", col="Magel2")

    z = pd.read_csv(DATA / "z_autism_regress.csv", index_col=0)
    shared = [r for r in z.columns if r in otter and r in tb]
    h = np.array([otter[r] for r in shared])
    t = np.array([tb[r] for r in shared])
    zmat = z[shared].to_numpy()

    map_agree = pearsonr(h, t)[0]
    risk_h = np.array([pearsonr(zmat[i], h)[0] for i in range(len(zmat))])
    risk_t = np.array([pearsonr(zmat[i], t)[0] for i in range(len(zmat))])
    concord = pearsonr(risk_h, risk_t)[0]
    print(f"\n[Part B-2] Head-to-head. Magel2 autism mutation pattern "
          f"({len(shared)} BN regions, {len(zmat)} ASD individuals)")
    print(f"  OTTER vs TransBrain translated maps:   r = {map_agree:+.3f}")
    print(f"  per-individual ASD risk-score concord: r = {concord:+.3f}")
    return {"n_bn_regions": len(shared), "n_individuals": int(len(zmat)),
            "map_agreement": float(map_agree),
            "risk_score_concordance": float(concord),
            "mean_risk_otter": float(risk_h.mean()),
            "mean_risk_transbrain": float(risk_t.mean()),
            "risk_otter": risk_h.tolist(),
            "risk_transbrain": risk_t.tolist()}


def main():
    print("=" * 80)
    print("OTTER × TransBrain 2025, sibling-method benchmark")
    print("=" * 80)

    from otter.data import load_pi, pi_provenance
    pi = load_pi()
    mm = json.loads((ROOT / "data_external" / "mouse_sc_meta.json").read_text())
    parcel_acr = [mm["structure_acronyms"][i] for i in mm["node_struct_idx"]]
    H, _ = load_cached("human", cache_dir=ANN)
    bn_id, id2name, centroid = load_bn_atlas(H.var)
    print(f"  π: {pi.shape}   OTTER human parcels with a BN label: "
          f"{int((bn_id > 0).sum())} / {len(bn_id)}")

    out = {
        **pi_provenance(),
        "homology_benchmark_cortex": homology_benchmark(
            pi, parcel_acr, H.var, bn_id, id2name, centroid,
            DATA / "homo_cortex.csv", "cortex"),
        "homology_benchmark_subcortex": homology_benchmark(
            pi, parcel_acr, H.var, bn_id, id2name, centroid,
            DATA / "homo_subcortex.csv", "subcortex"),
        "head_to_head_gradient": head_to_head_gradient(pi, parcel_acr, bn_id, id2name),
        "head_to_head_autism": head_to_head_autism(pi, parcel_acr, bn_id, id2name),
    }
    out_path = ROOT / "outputs" / "logs" / "transbrain_2025_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
