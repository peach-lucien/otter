"""Does OTTER's coverage map separate the REACHABLE part of a human
psychiatric disorder from the unreachable part?

The cortical result (from section5_coverage_rigor/06): case-control cortical THINNING in
bipolar disorder and schizophrenia concentrates where OTTER's coverage is LOWEST
(Spearman +0.63 and +0.51, spin p < 0.005). Positive rho = more thinning (more negative
Cohen's d) where coverage is lower.

The control this script adds is the SUBCORTICAL signature of the same two disorders. OTTER
covers subcortex well (S5 catalogue), so if coverage indexes what a mouse model can reach,
rather than simply tracking where disease is severe, the subcortical correlation should not
show the same deficit. It does not; it reverses. The three
structures with the largest schizophrenia effects (hippocampus, amygdala, thalamus) are
OTTER's three best-covered structures, giving rho = -0.79 (SCZ) and -0.68 (BD).

The finding is the INTERACTION, tested with a Fisher z on the two independent correlations
(the subcortical arm alone is only n = 7 and is not independently spin-testable).

Cortical d is thickness and subcortical d is volume, so the reversal is across ENIGMA's
two standard metrics rather than within one.

Data: ENIGMA summary statistics. Cortical thickness ships in data_external/enigma/.
Subcortical volume comes from the ENIGMA toolbox:
    pip install git+https://github.com/MICA-MNI/ENIGMA.git
and is read from enigmatoolbox/datasets/summary_statistics/ (override with ENIGMA_SUMSTATS).
Atlas: volumetric Desikan-Killiany via `pip install abagen` (also supplies the 7 subcortical
structures, which are exactly ENIGMA's).

Run: cd otter && PYTHONPATH=src python experiments/section6_disorder_reachability/01_double_dissociation.py
Writes outputs/logs/section6_double_dissociation.json
"""
from __future__ import annotations
import csv, json, os, sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, rankdata, norm, t as tdist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import load_cached, load_pi          # noqa: E402
from otter.data.anchors import get_anchor_index      # noqa: E402
from otter.eval.nulls import spin_null               # noqa: E402

N_SPIN = 2000
RESCUE_MM = 4.0
MIN_PARCELS = 10
OUT = ROOT / "outputs/logs/section6_double_dissociation.json"

# ENIGMA subcortical structure codes -> Desikan-Killiany label
SUBCORTEX = {
    "thal": "thalamusproper", "caud": "caudate", "put": "putamen", "pal": "pallidum",
    "accumb": "accumbensarea", "hippo": "hippocampus", "amyg": "amygdala",
}


def sumstats_dir():
    if os.environ.get("ENIGMA_SUMSTATS"):
        return Path(os.environ["ENIGMA_SUMSTATS"])
    import enigmatoolbox
    return Path(enigmatoolbox.__file__).parent / "datasets/summary_statistics"


def parcel_labels(xyz):
    """Assign each OTTER human parcel to a DK region (cortical or subcortical)."""
    import abagen
    atlas = abagen.fetch_desikan_killiany()
    img = nib.load(atlas["image"])
    info = pd.read_csv(atlas["info"])
    lab = np.asarray(img.get_fdata()).astype(int)

    vox = nib.affines.apply_affine(np.linalg.inv(img.affine), xyz)
    vi = np.rint(vox).astype(int)
    inb = np.all((vi >= 0) & (vi < np.array(lab.shape)), axis=1)
    plab = np.zeros(len(xyz), int)
    plab[inb] = lab[vi[inb, 0], vi[inb, 1], vi[inb, 2]]

    labelled = np.argwhere(lab > 0)
    need = np.where(plab == 0)[0]
    dist, j = cKDTree(labelled).query(vox[need])
    ok = dist <= RESCUE_MM
    hit = labelled[j[ok]]
    plab[need[ok]] = lab[hit[:, 0], hit[:, 1], hit[:, 2]]

    id2name = dict(zip(info.id, info.label))
    cortical = {id2name[i] for i in info.loc[info.structure == "cortex", "id"]}
    return plab, id2name, cortical


def read_enigma(path, delimiter=","):
    """{region: mean Cohen's d across hemispheres}. Handles cortical (L_bankssts) and
    subcortical (Lhippo) structure naming in one pass."""
    acc = {}
    with open(path) as fh:
        for row in csv.DictReader(fh, delimiter=delimiter):
            s = (row.get("Structure") or "").strip()
            try:
                v = float(row["d_icv"])
            except (TypeError, ValueError, KeyError):
                continue
            if not np.isfinite(v):
                continue
            key = None
            if len(s) > 1 and s[0] in "LR" and s[1:] in SUBCORTEX:      # Lhippo
                key = SUBCORTEX[s[1:]]
            elif "_" in s:                                              # L_bankssts
                key = s.split("_", 1)[1].strip().lower()
            if key:
                acc.setdefault(key, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def partial_spearman(x, y, z):
    """Spearman(x, y) with z regressed out of both (inputs already ranked)."""
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(spearmanr(rx, ry).statistic)


def williams(r12, r13, r23, n):
    """Williams' test comparing two DEPENDENT correlations r12 and r13 that share
    variable 1 (here: coverage-vs-d against hierarchy-vs-d, both sharing d).
    Without this, 'coverage predicts thinning better than the hierarchy' is an
    unsupported comparison of two p-values rather than a test of their difference."""
    R = 1 - r12 ** 2 - r13 ** 2 - r23 ** 2 + 2 * r12 * r13 * r23
    t = (r12 - r13) * np.sqrt((n - 1) * (1 + r23) /
                              (2 * R * (n - 1) / (n - 3) + ((r12 + r13) ** 2 / 4) * (1 - r23) ** 3))
    return float(t), float(2 * tdist.sf(abs(t), n - 3))


def main():
    pi = load_pi()
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    col = pi.sum(0)
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    plab, id2name, cortical = parcel_labels(xyz)

    # confound 1: T1w/T2w hierarchy.  confound 2: distance from the 42 curated anchors
    # (low coverage might merely index distance from the curated anchors).
    myelin = np.asarray(json.loads(
        (ROOT / "outputs/logs/buckner_krienen_2013_tethering.json").read_text())["myelin_per_parcel"], float)
    anchor_dist, _ = cKDTree(xyz[get_anchor_index(H.var).pos]).query(xyz)

    def coverage(name):
        m = np.isin(plab, [i for i in id2name if id2name[i] == name])
        if m.sum() < (MIN_PARCELS if name in cortical else 3):
            return np.nan, np.nan, 0
        c = xyz[m].mean(0)
        return float(np.log10(col[m].mean() + 1e-300)), [abs(c[0]), c[1], c[2]], int(m.sum())

    cov, cen, mye, adist = {}, {}, {}, {}
    for name in set(id2name.values()):
        v, c, n = coverage(name)
        if np.isfinite(v):
            cov[name], cen[name] = v, c
            m = np.isin(plab, [i for i in id2name if id2name[i] == name])
            mm = m & np.isfinite(myelin)
            mye[name] = float(myelin[mm].mean()) if mm.any() else np.nan
            adist[name] = float(anchor_dist[m].mean())

    S = sumstats_dir()
    disorders = {
        "schizophrenia": (ROOT / "data_external/enigma/cortical_thickness_schizophrenia.csv",
                          S / "scz_case-controls_SubVol.csv"),
        "bipolar":       (ROOT / "data_external/enigma/cortical_thickness_bipolar.csv",
                          S / "bd_case-controls_SubVol_typeI.csv"),
    }

    res = {}
    for name, (ctx_csv, sub_csv) in disorders.items():
        d_ctx, d_sub = read_enigma(ctx_csv), read_enigma(sub_csv)

        ck = sorted(r for r in d_ctx if r in cov and r in cortical)
        cc = np.array([cov[r] for r in ck]); cd = np.array([d_ctx[r] for r in ck])
        C = np.array([cen[r] for r in ck])
        spin = spin_null(rankdata(cc), rankdata(cd), C, n_trials=N_SPIN, seed=0)
        rho_ctx = float(spearmanr(cc, cd).statistic)

        sk = sorted(r for r in d_sub if r in cov and r not in cortical)
        sc = np.array([cov[r] for r in sk]); sd = np.array([d_sub[r] for r in sk])
        rho_sub = float(spearmanr(sc, sd).statistic)

        # the finding is the interaction; the subcortical arm alone is n = 7
        z = ((np.arctanh(rho_ctx) - np.arctanh(rho_sub))
             / np.sqrt(1 / (len(ck) - 3) + 1 / (len(sk) - 3)))
        p_int = float(2 * norm.sf(abs(z)))

        # burden concentration in cortex: share of |d| mass by coverage tertile
        t = np.argsort(cc); lo, hi = t[:len(t) // 3], t[-len(t) // 3:]
        burden_lo = float(np.abs(cd)[lo].sum() / np.abs(cd).sum())
        burden_hi = float(np.abs(cd)[hi].sum() / np.abs(cd).sum())

        # --- confound controls on the cortical arm ---
        rk_cov, rk_d = rankdata(cc), rankdata(cd)
        rk_mye = rankdata([mye[r] for r in ck])
        rk_ad = rankdata([adist[r] for r in ck])
        rho_mye = float(spearmanr([mye[r] for r in ck], cd).statistic)
        rho_ad = float(spearmanr([adist[r] for r in ck], cd).statistic)
        t_w, p_w = williams(rho_ctx, rho_mye, float(spearmanr(cc, [mye[r] for r in ck]).statistic), len(ck))

        res[name] = {
            "cortex": {"n": len(ck), "spearman": rho_ctx, "spin_p": spin["p_spin"],
                       "burden_low_coverage_tertile": burden_lo,
                       "burden_high_coverage_tertile": burden_hi},
            "subcortex": {"n": len(sk), "spearman": rho_sub,
                          "coverage": {r: cov[r] for r in sk},
                          "cohens_d": {r: d_sub[r] for r in sk}},
            "interaction": {"fisher_z": float(z), "p": p_int},
            "controls": {
                "hierarchy_T1wT2w": {
                    "spearman_with_d": rho_mye,
                    "coverage_partialling_hierarchy": partial_spearman(rk_cov, rk_d, rk_mye),
                    "williams_t": t_w, "williams_p": p_w,
                    "note": "coverage is NOT significantly better than T1w/T2w at this n",
                },
                "anchor_distance": {
                    "spearman_coverage_vs_anchor_distance":
                        float(spearmanr(cc, [adist[r] for r in ck]).statistic),
                    "spearman_with_d": rho_ad,
                    "coverage_partialling_anchor_distance": partial_spearman(rk_cov, rk_d, rk_ad),
                },
            },
        }
        print(f"{name:15s} cortex rho={rho_ctx:+.2f} (n={len(ck)}, spin p={spin['p_spin']:.4f})   "
              f"subcortex rho={rho_sub:+.2f} (n={len(sk)})   interaction z={z:.2f} p={p_int:.4f}")
        print(f"{'':15s} burden: {burden_lo:.0%} in the least-covered tertile vs {burden_hi:.0%} in the best")
        print(f"{'':15s} control | vs T1w/T2w ({rho_mye:+.2f}): Williams p={p_w:.3f} (NOT distinguishable); "
              f"partialled rho={res[name]['controls']['hierarchy_T1wT2w']['coverage_partialling_hierarchy']:+.2f}")
        print(f"{'':15s} control | anchor distance vs d {rho_ad:+.2f}; "
              f"coverage partialling anchor distance rho="
              f"{res[name]['controls']['anchor_distance']['coverage_partialling_anchor_distance']:+.2f}")

    res["_meta"] = {
        "atlas": "Desikan-Killiany volumetric (abagen), 68 cortical + 7 subcortical",
        "coverage": "log10 mass-normalised mean pi column-sum per region",
        "cortical_metric": "ENIGMA cortical thickness Cohen's d",
        "subcortical_metric": "ENIGMA subcortical volume Cohen's d (different metric; see text)",
        "note": "positive rho = disorder effect larger where coverage is lower",
        "n_spin": N_SPIN,
    }
    OUT.write_text(json.dumps(res, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
