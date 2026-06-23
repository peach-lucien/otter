"""HOMER × TransBrain 2025, advanced comparison (four follow-up analyses).

Building on `01_transbrain_benchmark.py`, this digs past the head-line
"moderate agreement" number into *where* and *why* HOMER and TransBrain agree.

  1. Trust-stratified agreement, does HOMER↔TransBrain agreement on *where a
     mouse region maps* track HOMER's own per-parcel trust score?
  2. Optogenetic → human cognition. TransBrain's Case 2: route a mouse
     optogenetic circuit through π, decode it with Neurosynth cognitive-term
     maps, compare the cognitive annotation against TransBrain's.
  3. Bidirectional cycle-consistency, round-trip a phenotype
     mouse→human→mouse with each method; a fair, ground-truth-free metric.
  4. Consensus + disagreement map, rank mouse regions by HOMER↔TransBrain
     agreement: a consensus set and a flagged set of contested homologies.

Agreement between the two methods is measured concentration-robustly: the
distance between HOMER's and TransBrain's top human Brainnetome region for a
given mouse region (a correlation of the full maps is dominated by HOMER's
sharper concentration, not by where the peak lands).

Requires `pip install transbrain`. The Neurosynth term maps come from the
TransBrain repo's tutorial folder (set NEUROSYNTH_DIR; not redistributed here).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached

try:
    import nibabel as nib
    from transbrain.config import Config
    from transbrain.trans import SpeciesTrans
except ImportError as e:  # pragma: no cover
    raise ImportError("This experiment needs TransBrain: `pip install transbrain`") from e

ANN = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
DATA = ROOT / "data_external" / "transbrain_2025"
PI_FILE = COUP / "pi_fc_plus_SC_with_all_packs.npy"
# Neurosynth cognitive-term maps for the optogenetic decoding. Not bundled with
# the repo, place the TransBrain tutorial's neurosynth_data here (or point
# NEUROSYNTH_DIR at it); if absent, the optogenetic analysis is skipped cleanly.
NEUROSYNTH_DIR = Path(os.environ.get(
    "NEUROSYNTH_DIR", str(DATA / "neurosynth_data")))
MOUSE_REGIONS = list(Config.MOUSE_CORTICAL) + list(Config.MOUSE_SUBCORTICAL)


def load_bn_atlas(H_var):
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
    return bn_id, id2name, name2centroid, vol


def aggregate_bn(vals, bn_id, id2name, reduce="sum"):
    out = {}
    for rid in np.unique(bn_id):
        if not rid:
            continue
        v = vals[(bn_id == rid) & np.isfinite(vals)]
        if v.size:
            out[id2name[rid]] = float(v.sum() if reduce == "sum" else v.mean())
    return out


def route_fwd(m, pi, mask):
    num = m[mask] @ pi[mask, :]
    den = pi[mask, :].sum(axis=0)
    out = np.full(pi.shape[1], np.nan)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def route_rev(h, pi, mask):
    num = pi[:, mask] @ h[mask]
    den = pi[:, mask].sum(axis=1)
    out = np.full(pi.shape[0], np.nan)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4 or np.ptp(a[m]) == 0 or np.ptp(b[m]) == 0:
        return np.nan
    return float(pearsonr(a[m], b[m])[0])


def main():
    print("=" * 80)
    print("HOMER × TransBrain 2025, advanced comparison")
    print("=" * 80)
    pi = np.load(PI_FILE)
    mm = json.loads((ROOT / "data_external" / "mouse_sc_meta.json").read_text())
    parcel_acr = np.array([mm["structure_acronyms"][i] for i in mm["node_struct_idx"]])
    H, _ = load_cached("human", cache_dir=ANN)
    bn_id, id2name, centroid, bn_vol = load_bn_atlas(H.var)
    st = SpeciesTrans(atlas_type="bn")
    trust = np.load(COUP / "trust_multisource_all_packs.npz", allow_pickle=True)
    trust_score = trust["trust"].astype(float)
    tier = trust["evidence_tier"]

    # TransBrain's region→region operator (translate the 68×68 identity)
    ident = pd.DataFrame(np.eye(len(MOUSE_REGIONS)), index=MOUSE_REGIONS,
                         columns=MOUSE_REGIONS)
    tb_mat = st.mouse_to_human(ident, region_type="all")

    # HOMER's region→region operator
    homer_map, region_trust, region_tier = {}, {}, {}
    for X in MOUSE_REGIONS:
        idx = np.where(parcel_acr == X)[0]
        if len(idx) == 0:
            continue
        homer_map[X] = aggregate_bn(pi[idx].sum(axis=0), bn_id, id2name, "sum")
        region_trust[X] = float(np.mean(trust_score[idx]))
        vals, cnts = np.unique(tier[idx], return_counts=True)
        region_tier[X] = str(vals[cnts.argmax()])

    # ===== 1 + 4, per-region agreement (top-region distance) ===============
    print("\n[1+4] Per-region HOMER↔TransBrain agreement (top-region distance)")
    dist = {}
    for X in homer_map:
        tb_col = {r: tb_mat.loc[r, X] for r in tb_mat.index if r in centroid}
        hm = {r: v for r, v in homer_map[X].items() if r in centroid}
        if not tb_col or not hm:
            continue
        h_top = max(hm, key=hm.get)
        t_top = max(tb_col, key=tb_col.get)
        dist[X] = float(np.linalg.norm(centroid[h_top] - centroid[t_top]))
    regs = sorted(dist, key=dist.get)            # ascending: consensus first
    tr = np.array([region_trust[X] for X in dist])
    dd = np.array([dist[X] for X in dist])
    r_trust = corr(tr, dd)
    rho_trust = float(spearmanr(tr, dd)[0])
    print(f"  top-region distance vs HOMER trust: r = {r_trust:+.3f}  ρ = {rho_trust:+.3f}"
          f"  ({len(dist)} mouse regions), negative = methods agree more where HOMER is confident")
    tier_means = {}
    for tname in ["anchored_and_validated", "anchored_only", "validated_only",
                  "structural", "low_evidence"]:
        xs = [dist[X] for X in dist if region_tier[X] == tname]
        if xs:
            tier_means[tname] = float(np.mean(xs))
            print(f"    {tname:24s} mean distance = {np.mean(xs):5.1f} mm  (n={len(xs)})")
    print(f"  consensus (closest 6): {', '.join(regs[:6])}")
    print(f"  contested (farthest 6): {', '.join(regs[-6:])}")

    # ===== 3, bidirectional cycle-consistency =============================
    print("\n[3] Bidirectional cycle-consistency (mouse→human→mouse)")
    mg = json.loads((ROOT / "outputs/logs/margulies_2016_gradient.json").read_text())
    grad = {}
    for a, v in zip(parcel_acr, np.array(mg["mouse_gradient"])):
        if a in MOUSE_REGIONS:
            grad.setdefault(a, []).append(v)
    phenos = {
        "gradient": {a: float(np.mean(v)) for a, v in grad.items()},
        "AI_opto": pd.read_csv(DATA / "ai_opto.csv", index_col=0).iloc[:, 0].to_dict(),
        "Magel2": pd.read_csv(DATA / "magel2_mutation_pattern.csv",
                              index_col=0)["Magel2"].to_dict(),
    }
    cycle = {}
    for name, ph in phenos.items():
        mvec = np.array([ph.get(a, np.nan) for a in parcel_acr])
        m_back = route_rev(route_fwd(mvec, pi, np.isfinite(mvec)), pi,
                           np.isfinite(route_fwd(mvec, pi, np.isfinite(mvec))))
        bb = {}
        for a, v in zip(parcel_acr, m_back):
            if np.isfinite(v):
                bb.setdefault(a, []).append(v)
        common = [a for a in ph if a in bb]
        homer_cyc = corr(np.array([ph[a] for a in common]),
                         np.array([np.mean(bb[a]) for a in common]))
        full = {a: ph.get(a, np.nan) for a in MOUSE_REGIONS}
        fv = np.nanmean(list(full.values()))
        full = {a: (v if np.isfinite(v) else fv) for a, v in full.items()}
        df = pd.DataFrame({name: full})
        tb_back = st.human_to_mouse(st.mouse_to_human(df, region_type="all"),
                                    region_type="all")
        ci = [a for a in df.index if a in tb_back.index]
        tb_cyc = corr(df.loc[ci, name].to_numpy(), tb_back.loc[ci].iloc[:, 0].to_numpy())
        cycle[name] = {"homer": homer_cyc, "transbrain": tb_cyc}
        print(f"  {name:10s} HOMER {homer_cyc:+.3f}   TransBrain {tb_cyc:+.3f}")

    # ===== 2, optogenetic circuit → human cognition (Neurosynth) ===========
    print("\n[2] Optogenetic AI circuit → human cognition (Neurosynth decode)")
    opto = {}
    if NEUROSYNTH_DIR.exists():
        ai = pd.read_csv(DATA / "ai_opto.csv", index_col=0).iloc[:, 0].to_dict()
        mvec = np.array([ai.get(a, np.nan) for a in parcel_acr])
        homer_bn = aggregate_bn(route_fwd(mvec, pi, np.isfinite(mvec)),
                                bn_id, id2name, "mean")
        tb_bn = st.mouse_to_human(pd.DataFrame({"AI": ai}), region_type="all").iloc[:, 0].to_dict()
        terms = sorted(NEUROSYNTH_DIR.glob("*association-test*.nii.gz"))
        sh, st_ = {}, {}
        for tp in terms:
            term = tp.name.split("_association-test")[0]
            tvol = np.asarray(nib.load(tp).dataobj)
            if tvol.shape != bn_vol.shape:
                continue
            tv = {id2name[r]: float(tvol[bn_vol == r].mean())
                  for r in np.unique(bn_vol) if r and r in id2name}
            kh = [r for r in tv if r in homer_bn]
            kt = [r for r in tv if r in tb_bn]
            ch = corr(np.array([homer_bn[r] for r in kh]),
                      np.array([tv[r] for r in kh]))
            ct = corr(np.array([tb_bn[r] for r in kt]),
                      np.array([tv[r] for r in kt]))
            if np.isfinite(ch):
                sh[term] = ch
            if np.isfinite(ct):
                st_[term] = ct
        top_h = sorted(sh, key=sh.get, reverse=True)[:10]
        top_t = sorted(st_, key=st_.get, reverse=True)[:10]
        common = sorted(set(sh) & set(st_))
        rank_r = float(spearmanr([sh[t] for t in common],
                                 [st_[t] for t in common])[0])
        overlap = len(set(top_h) & set(top_t))
        print(f"  {len(terms)} Neurosynth association-test terms decoded")
        print(f"  HOMER top-10:      {', '.join(top_h)}")
        print(f"  TransBrain top-10: {', '.join(top_t)}")
        print(f"  top-10 overlap = {overlap}/10   term-rank ρ = {rank_r:+.3f}")
        opto = {"n_terms": len(terms), "homer_top10": top_h,
                "transbrain_top10": top_t, "top10_overlap": overlap,
                "term_rank_spearman": rank_r,
                "homer_scores": sh, "transbrain_scores": st_}
    else:
        print(f"  SKIPPED. Neurosynth maps not found at {NEUROSYNTH_DIR}")

    out = {
        "trust_stratified": {
            "topdist_vs_trust_pearson": r_trust,
            "topdist_vs_trust_spearman": rho_trust,
            "tier_mean_topdist_mm": tier_means, "n_regions": len(dist),
        },
        "consensus_disagreement": {
            "topdist_per_region": {X: dist[X] for X in regs},
            "consensus_top": regs[:8], "contested_bottom": regs[-8:],
        },
        "cycle_consistency": cycle,
        "optogenetic": opto,
    }
    (ROOT / "outputs" / "logs" / "transbrain_2025_advanced.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nWrote outputs/logs/transbrain_2025_advanced.json")


if __name__ == "__main__":
    main()
