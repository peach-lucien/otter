#!/usr/bin/env python3
"""The coverage result, built around a discrete dlPFC coverage hole.

Coverage is roughly flat across cortex (every myelin decile within +-0.25 SD) with one
exception: Yeo-17 network Control B = dorsolateral / rostrolateral prefrontal cortex, which
is under-covered by ~1.2 SD. It is not a smooth sensorimotor->association gradient:
log-column-mass is an eps-amplified transport cost, coverage L/R reliability is 0.22, and the
tertile contrast is carried by a single myelin decile corresponding to this same dlPFC
territory (08_anchorfree_control.py).

This script establishes four things, and writes one log:

1. NETWORK SPECIFICITY. Coverage per Yeo-17 network, spin-tested. On the canonical coupling
   Control B stands alone (mean coverage -0.97 SD, deficit -1.03 SD, spin p = 0.0005,
   Bonferroni p = 0.008 across 16 networks). It is NOT "granular cortex": V1 (granular
   koniocortex) is well covered.

2. IT IS CONNECTIONAL, NOT CURATIONAL. The deficit GROWS on the anchor-free coupling (no
   supervision at all, -1.44 SD) and is much weaker in the base coupling (Garin points, no
   FC/SC packs: -0.33 SD, spin p = 0.06).

COUPLINGS: canonical (-1.03 SD) is the reported arm. The retired pre-warp coupling
"production" gives -1.20 SD; direction and significance are unchanged. The medial->lateral
gradient (script 12) and the sensorimotor/association tertile gap (script 01) are null on the
canonical coupling.

3. IT IS CONNECTIONAL, NOT MOLECULAR. Transcriptomic similarity to the mouse does NOT dip in
   Control B, so the territory is molecularly mouse-like but connectionally unreachable.

4. THE MECHANISM. Mouse frontal cortex is entirely agranular (Goulas eulamination type 1);
   granular type-4 cortex exists in the mouse only in primary sensorimotor areas. Human dlPFC
   is granular association cortex, so it has no granular-prefrontal counterpart in the rodent to
   receive mouse mass. This is the connectional face of the absent rodent granular-PFC homologue
   (cf. the Schaeffer dlPFC control: mouse medial-frontal cortex routes ~0% to dlPFC).

Spin-null note: the signal spun here is COVERAGE (asymmetric); it is correlated against network
membership / myelin (symmetric). That configuration is calibrated (5.5% FPR). Do NOT bilaterally
average coverage before spinning: that makes both maps symmetric and the whole-brain spin null
then over-rejects (37% FPR).

Run:  cd otter && PYTHONPATH=src python experiments/section5_coverage_rigor/11_dlpfc_deficit.py
Writes: outputs/logs/section5_dlpfc_deficit.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from otter.data import load_cached, load_pi, pi_provenance        # noqa: E402
from otter.data.anchors import get_anchor_index                   # noqa: E402
from otter.eval.nulls import _haar_rotation                       # noqa: E402

N_SPIN = 2000
SEED = 0


def coverage(pi):
    return np.log10(np.maximum(pi.sum(0), 1e-300))


def coverage_recon(pi, Mfc, Hfc):
    """Reconstruction-coverage: how well each human parcel's FC fingerprint is rebuilt by
    routing mouse FC through pi. This is the reported metric; the log-column-mass
    above is the older, position-confounded one, kept for comparison."""
    pit = pi / np.maximum(pi.sum(0), 1e-300)
    pred = pit.T @ Mfc @ pit
    n = pred.shape[0]
    out = np.full(n, np.nan)
    for j in range(n):
        a, b = pred[j].copy(), Hfc[j].copy()
        a[j] = np.nan; b[j] = np.nan
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 10 and a[ok].std() > 1e-9:
            out[j] = np.corrcoef(a[ok], b[ok])[0, 1]
    return out


def yeo17(nr):
    rows = [l.split("\t") for l in
            (ROOT / "outputs/anndata/_schaefer_order.txt").read_text().splitlines() if l.strip()]
    nmap = {int(p[0]): p[1] for p in rows}
    return np.array([nmap.get(k, "?_?_?").split("_")[2] if k in nmap else "?" for k in nr])


def molecular_similarity():
    D = ROOT / "data_external"
    mg, hg = np.load(D / "mouse_genes_aligned.npy"), np.load(D / "human_genes_aligned.npy")

    def zc(A):
        A = A.astype(float)
        mu, sd = np.nanmean(A, 0), np.nanstd(A, 0)
        sd[sd < 1e-9] = 1.0
        return (A - mu) / sd

    Mz, Hz = zc(mg), zc(hg)
    mok, hok = np.isfinite(Mz).all(1), np.isfinite(Hz).all(1)
    Mc = Mz[mok] - Mz[mok].mean(1, keepdims=True)
    Mn = Mc / np.linalg.norm(Mc, axis=1, keepdims=True)
    Hc = Hz - np.nanmean(Hz, 1, keepdims=True)
    Hn = np.full_like(Hc, np.nan)
    Hn[hok] = Hc[hok] / np.linalg.norm(Hc[hok], axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        return np.nanmax(Hn @ Mn.T, axis=1)


def spin_perms(coords, n=N_SPIN, seed=SEED):
    c = coords - coords.mean(0)
    sph = c / np.linalg.norm(c, axis=1, keepdims=True)
    tree = cKDTree(sph)
    rng = np.random.default_rng(seed)
    return [tree.query(sph @ _haar_rotation(rng).T)[1] for _ in range(n)]


def block_gap(sig, sel, perms):
    """(mean over sel) - (mean over rest), spin p. sig is z-scored coverage."""
    f = lambda s: s[sel].mean() - s[~sel].mean()          # noqa: E731
    obs = f(sig)
    null = np.abs([f(sig[p]) for p in perms])
    return {"gap_sd": float(obs),
            "spin_p": float((np.sum(null >= abs(obs)) + 1) / (len(perms) + 1)),
            "null_abs_p95": float(np.percentile(null, 95))}


def main():
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    mye = np.asarray(json.loads(
        (ROOT / "outputs/logs/buckner_krienen_2013_tethering.json").read_text())["myelin_per_parcel"], float)
    nr = np.asarray(json.loads(
        (ROOT / "data_external/human_sc_meta.json").read_text())["node_region"], int)
    net = yeo17(nr)

    pi = load_pi()
    prov = pi_provenance()
    print(f"π: {prov['pi_file']}  sha256={prov['pi_sha256']}")
    cov = coverage(pi)
    m = np.isfinite(cov) & np.isfinite(mye)               # cortex with a hierarchy value
    z = (cov[m] - cov[m].mean()) / cov[m].std()
    nt = net[m]
    perms = spin_perms(xyz[m])

    out = {**prov,
           "_finding": ("Coverage is roughly flat across cortex except a discrete hole over "
                        "Yeo-17 Control B (dorsolateral/rostrolateral PFC). Not a gradient."),
           "n_cortical_parcels": int(m.sum()),
           "spin_config": "signal=coverage (asymmetric) vs network/myelin (symmetric): calibrated at 5.5% FPR"}

    # ---- 1. per-network coverage, spin-tested; Bonferroni across networks tested -------
    nets = sorted({u for u in set(nt) if (nt == u).sum() >= 30})
    net_res = {}
    for u in nets:
        r = block_gap(z, nt == u, perms)
        r["mean_coverage_sd"] = float(z[nt == u].mean())
        r["n"] = int((nt == u).sum())
        r["bonferroni_p"] = min(1.0, r["spin_p"] * len(nets))
        net_res[u] = r
    out["n_networks_tested"] = len(nets)
    out["per_network"] = net_res
    dl = net_res["ContB"]
    print(f"Control B (dlPFC): {dl['mean_coverage_sd']:+.2f} SD, deficit {dl['gap_sd']:+.2f}, "
          f"spin p={dl['spin_p']:.4f}, Bonferroni p={dl['bonferroni_p']:.4f}")
    ranked = sorted(net_res.items(), key=lambda kv: kv[1]["mean_coverage_sd"])
    print("  most deficient:", ", ".join(f"{k} {v['mean_coverage_sd']:+.2f}" for k, v in ranked[:3]))

    # ---- 2. connectional not curational: base / production / anchor-free ---------------
    # "canonical" is the reported arm (== the main analysis above); the others are kept
    # for cross-coupling comparison. "production" is the RETIRED pre-warp coupling.
    out["control_b_across_couplings"] = {}
    out["_reported_coupling_arm"] = "canonical"
    for name, path in [("canonical", "pi_canonical.npy"),
                       ("base_garin_points_only", "pi_fc_plus_SC.npy"),
                       ("production", "pi_fc_plus_SC_with_all_packs.npy"),
                       ("anchor_free", "pi_anchorfree_control.npy")]:
        p = ROOT / "outputs/coupling" / path
        if not p.exists():
            out["control_b_across_couplings"][name] = "coupling not found"
            continue
        cv = coverage(np.load(p))
        mm = np.isfinite(cv) & np.isfinite(mye)
        zc_ = (cv[mm] - cv[mm].mean()) / cv[mm].std()
        r = block_gap(zc_, net[mm] == "ContB", spin_perms(xyz[mm]))
        out["control_b_across_couplings"][name] = {"gap_sd": r["gap_sd"], "spin_p": r["spin_p"],
                                                    "mean_coverage_sd": float(zc_[net[mm] == "ContB"].mean()),
                                                    **pi_provenance(path)}
        print(f"  {name:<24} ContB deficit {r['gap_sd']:+.2f} SD  spin p={r['spin_p']:.3f}")

    # ---- 3. connectional not molecular: does transcriptomic similarity dip in ContB? ----
    best = molecular_similarity()
    mm = np.isfinite(cov) & np.isfinite(best)
    zc_cov = (cov[mm] - cov[mm].mean()) / cov[mm].std()
    zc_mol = (best[mm] - best[mm].mean()) / best[mm].std()
    perms2 = spin_perms(xyz[mm])
    sel = net[mm] == "ContB"
    Mm, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    Mfc = np.asarray(Mm.uns["fc_mean"], float); Hfc = np.asarray(H.uns["fc_mean"], float)
    rec = coverage_recon(pi, Mfc, Hfc)
    mr = np.isfinite(rec) & np.isfinite(best)
    zr_rec = (rec[mr] - rec[mr].mean()) / rec[mr].std()
    zr_mol = (best[mr] - best[mr].mean()) / best[mr].std()
    perms3 = spin_perms(xyz[mr])
    sel_r = net[mr] == "ContB"
    out["molecular_control"] = {
        "connectivity_coverage": block_gap(zc_cov, sel, perms2),
        "transcriptomic_similarity": block_gap(zc_mol, sel, perms2),
        "reconstruction_coverage": block_gap(zr_rec, sel_r, perms3),
        "transcriptomic_similarity_recon_support": block_gap(zr_mol, sel_r, perms3),
        "n_parcels_recon": int(mr.sum()),
        "_metric_note": ("connectivity_coverage is log-column-mass; reconstruction_coverage is "
                         "the reported metric. Both are tested against the same ContB block gap."),
        "n_parcels": int(mm.sum()),
        "_reading": ("Coverage dips in Control B; transcriptomic similarity does not. The dlPFC "
                     "territory is molecularly mouse-like but connectionally unreachable."),
    }
    cc = out["molecular_control"]
    print(f"  molecular control: coverage {cc['connectivity_coverage']['gap_sd']:+.2f} "
          f"(p={cc['connectivity_coverage']['spin_p']:.3f}) vs transcriptomic "
          f"{cc['transcriptomic_similarity']['gap_sd']:+.2f} (p={cc['transcriptomic_similarity']['spin_p']:.2f})")

    # ---- 4. the mechanism: mouse frontal cortex is agranular ---------------------------
    euot = {}
    for line in (ROOT / "data_external/fulcher_2019_gradients/CytoarchitectureTypes.txt").read_text().splitlines():
        p = line.split()
        if len(p) >= 2 and not line.startswith("#"):
            try:
                euot[p[0]] = float(p[1])
            except ValueError:
                pass
    frontal = ["MOs", "ACAd", "ACAv", "PL", "ILA", "ORBl", "ORBm", "ORBvl", "FRP", "AId", "AIv", "AIp"]
    fr_types = {a: euot[a] for a in frontal if a in euot}
    granular = sorted(a for a, v in euot.items() if v >= 4)
    out["mechanism_mouse_agranular_frontal"] = {
        "mouse_frontal_eulamination": fr_types,
        "mean_frontal_eulamination": float(np.mean(list(fr_types.values()))),
        "mouse_granular_type4_regions": granular,
        "_reading": ("Mouse frontal cortex is agranular (eulamination 1). Granular type-4 cortex "
                     "exists in the mouse only in primary sensorimotor areas. Human granular dlPFC "
                     "therefore has no granular-prefrontal counterpart in the rodent."),
    }
    print(f"  mechanism: mouse frontal eulamination mean = "
          f"{out['mechanism_mouse_agranular_frontal']['mean_frontal_eulamination']:.1f} "
          f"(agranular); granular type-4 only in {granular}")

    # per-parcel coverage percentile for the figure (scale-free)
    from scipy.stats import rankdata
    pct = np.full(len(xyz), np.nan)
    ctx = np.isfinite(mye)
    pct[ctx] = rankdata(pi.sum(0)[ctx]) / ctx.sum() * 100.0
    out["coverage_percentile_cortex"] = [None if not np.isfinite(v) else round(float(v), 3) for v in pct]
    out["control_b_parcel_mask"] = [bool(x) for x in (net == "ContB")]

    dst = ROOT / "outputs/logs/section5_dlpfc_deficit.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
