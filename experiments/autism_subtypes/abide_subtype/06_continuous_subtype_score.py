"""Continuous HOMER cross-species subtype score for every ABIDE individual.

Pagani assign a *binary* subtype (hypo / hyper / unsubtyped) by a hard ±1 s.d.
threshold, which leaves ~75–78 % of individuals unclassified — yet they note
autism connectivity "exists along a subtle continuum". The mask-definition method
isn't the bottleneck (HOMER vs name-matched masks subtype the same ~22 %, 93 %
agreement — see 05); the *hard threshold* is.

This script instead gives EVERY individual a continuous position on a HOMER-defined
hyper↔hypo axis, so the whole sample is placed on the continuum:

  • From π (04_homer_human_masks.py) we have human hypo and hyper coupling maps.
    Their contrast (hyper − hypo) defines a per-region weight emphasising where the
    two subtypes diverge (the maps are distinct: coupling r=0.41, mask Jaccard 0.28).
  • For each individual, axis = Σ_region contrast_weight · (z-scored regional global
    connectivity vs controls). Positive ⇒ hyper-like, negative ⇒ hypo-like.

Then we test the genuinely new question Pagani's binary scheme can't:
  (1) does the continuous axis recover their hard labels? (sanity)
  (2) does ASD differ from controls on the axis? (population)
  (3) **does the axis track ADOS symptom severity across ALL individuals**, not just
      the ~22 % hard-subtyped — i.e. is the hyper↔hypo continuum dose-responsive?

Needs the ABIDE download (run like 05). Prereqs: 04 (masks JSON) + ideally 05
(hard labels, for the sanity check).
    HOMER_ALLOW_INSECURE_SSL=1 PYTHONPATH=src python \
        experiments/autism_subtypes/abide_subtype/06_continuous_subtype_score.py \
        --abide-data-dir /tmp/abide_cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass
if os.environ.get("HOMER_ALLOW_INSECURE_SSL", "").lower() in {"1", "true", "yes"}:
    import ssl, requests, urllib3
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _o = requests.Session.send
    requests.Session.send = lambda self, r, **k: _o(self, r, **{**k, "verify": False})

import nibabel as nib  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached  # noqa: E402

LOG = ROOT / "outputs" / "logs"
ADOS_FIELDS = ["ADOS_TOTAL", "ADOS_GOTHAM_TOTAL", "ADOS_GOTHAM_SEVERITY",
               "ADOS_2_SEVERITY_TOTAL", "ADOS_SOCIAL", "ADOS_STEREO_BEHAV",
               "ADOS_COMM", "ADOS_RRB"]


def _num(x):
    try:
        v = float(x)
        return np.nan if v in (-9999, -999, 9999) else v
    except (TypeError, ValueError):
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abide-data-dir", default=str(Path.home() / "abide_cache"))
    ap.add_argument("--pipeline", default="cpac")
    args = ap.parse_args()

    masks = json.loads((LOG / "pagani_homer_human_masks.json").read_text())
    hypo_c = np.array(masks["masks"]["hypo"]["coupling_vector"], float)
    hyper_c = np.array(masks["masks"]["hyper"]["coupling_vector"], float)
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    H_xyz = H.var[["x", "y", "z"]].to_numpy(float)

    # ---- ABIDE + AAL ----
    from nilearn import datasets as ndatasets
    print("Fetching ABIDE (rois_aal) + AAL ...")
    ab = ndatasets.fetch_abide_pcp(data_dir=args.abide_data_dir, derivatives=["rois_aal"],
                                   pipeline=args.pipeline, quality_checked=True)
    rois, pheno = ab["rois_aal"], ab["phenotypic"]
    dx = np.array([int(p["DX_GROUP"]) for p in pheno])
    aal = ndatasets.fetch_atlas_aal(data_dir=args.abide_data_dir)
    aarr = nib.load(str(aal["maps"])).get_fdata().astype(int)
    aaff = nib.load(str(aal["maps"])).affine
    idxs = [int(i) for i in aal["indices"]]
    cents = np.array([(aaff @ np.r_[np.array(np.where(aarr == lab)).mean(1), 1])[:3] for lab in idxs])
    n_aal = len(idxs)

    # map HOMER per-parcel coupling -> per-AAL region (nearest HOMER parcel)
    d2 = (cents ** 2).sum(1, keepdims=True) + (H_xyz ** 2).sum(1)[None, :] - 2 * cents @ H_xyz.T
    nearest = d2.argmin(1)
    contrast_w = (hyper_c - hypo_c)[nearest]                 # per-AAL contrast weight
    contrast_w = contrast_w / (np.abs(contrast_w).sum() + 1e-12)

    # ---- per-subject regional global connectivity, z vs controls ----
    G = []
    for ts in rois:
        ts = np.asarray(ts, float)
        m = min(ts.shape[1], n_aal); ts = ts[:, :m]
        fc = np.corrcoef(ts.T); np.fill_diagonal(fc, np.nan)
        gc = np.nanmean(fc, 1)
        if gc.shape[0] < n_aal:
            gc = np.r_[gc, np.full(n_aal - gc.shape[0], np.nan)]
        G.append(gc)
    G = np.array(G); ctrl = dx == 2; asd = dx == 1
    Gz = (G - np.nanmean(G[ctrl], 0)) / (np.nanstd(G[ctrl], 0) + 1e-9)

    # ---- continuous axis (positive = hyper-like) ----
    axis = np.nansum(Gz * contrast_w[None, :], axis=1)

    # (1) recover hard labels?
    sanity = {}
    hl = LOG / "abide_homer_subtyping.json"
    # (re)compute hard labels here if 05's per-subject labels not stored
    # quick HOMER-mask hard labels for the sanity check:
    hypo_aal = np.isin(nearest, np.where(hypo_c >= masks["masks"]["hypo"]["coupling_threshold_pct80"])[0])
    hyper_aal = np.isin(nearest, np.where(hyper_c >= masks["masks"]["hyper"]["coupling_threshold_pct80"])[0])
    hard = np.full(len(Gz), "uns", object)
    hard[np.nanmean(Gz[:, hypo_aal], 1) < -1] = "hypo"
    hard[np.nanmean(Gz[:, hyper_aal], 1) > +1] = "hyper"
    for lab in ("hypo", "hyper"):
        sel = asd & (hard == lab)
        sanity[lab] = {"n": int(sel.sum()), "axis_mean": float(np.nanmean(axis[sel])) if sel.any() else None}
    print(f"Sanity — hard-hypo axis mean {sanity['hypo']['axis_mean']} (n={sanity['hypo']['n']}); "
          f"hard-hyper axis mean {sanity['hyper']['axis_mean']} (n={sanity['hyper']['n']})  "
          f"(expect hyper > hypo)")

    # (2) ASD vs control on the axis
    u, p = mannwhitneyu(axis[asd], axis[ctrl], alternative="two-sided")
    print(f"ASD vs control axis: ASD mean {np.nanmean(axis[asd]):+.3f}, "
          f"CTRL mean {np.nanmean(axis[ctrl]):+.3f}, Mann-Whitney p={p:.3g}")

    # (3) dose-response: continuous axis vs ADOS across ALL ASD with a score
    keys = list(pheno.dtype.names) if getattr(pheno, "dtype", None) is not None and pheno.dtype.names \
        else (list(pheno[0].keys()) if len(pheno) else [])
    avail = [f for f in ADOS_FIELDS if f in keys]
    ados_res = {}
    for f in avail:
        vals = np.array([_num(p[f]) for p in pheno])
        m = asd & np.isfinite(vals) & np.isfinite(axis)
        if m.sum() >= 30:
            rho, pp = spearmanr(axis[m], vals[m])
            ados_res[f] = {"n": int(m.sum()), "spearman_rho": float(rho), "p": float(pp)}
            print(f"  axis vs {f:<22} n={int(m.sum()):4d}  Spearman ρ={rho:+.3f}  p={pp:.3g}")
    if not ados_res:
        print("  (no ADOS field with ≥30 scored ASD individuals found in phenotypic table)")

    out = {"n_asd": int(asd.sum()), "n_ctrl": int(ctrl.sum()),
           "axis_definition": "sum(contrast_weight * z_global_connectivity), contrast=hyper-hypo coupling",
           "sanity_hard_label_axis_means": sanity,
           "asd_vs_ctrl": {"asd_mean": float(np.nanmean(axis[asd])), "ctrl_mean": float(np.nanmean(axis[ctrl])),
                            "mannwhitney_p": float(p)},
           "ados_doseresponse": ados_res,
           "axis_per_subject": {str(pheno[i]["SUB_ID"] if "SUB_ID" in keys else i): float(axis[i])
                                 for i in range(len(pheno))}}
    (LOG / "abide_continuous_subtype.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {LOG/'abide_continuous_subtype.json'}")


if __name__ == "__main__":
    main()
