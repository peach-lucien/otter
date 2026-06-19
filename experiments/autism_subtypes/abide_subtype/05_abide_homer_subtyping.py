"""ABIDE re-subtyping with HOMER-derived masks vs Pagani's name-matched masks.

This implements Pagani 2026's ACTUAL human-subtyping procedure (their Methods:
score each individual's regional global connectivity, classify hypo if < −1 s.d.
in the hypo mask, hyper if > +1 s.d. in the hyper mask), but swaps the
mouse→human mask-definition step:

  • Pagani: human mask = the SAME-NAMED human regions as the mouse prominent
    regions (name-matching).
  • HOMER: human mask = the human regions HOMER's π actually routes the mouse
    prominent regions to (data-driven; from 04_homer_human_masks.py).

It runs BOTH and compares: does the learned coupling subtype MORE than Pagani's
~25 % of individuals, and how much do the two subtypings agree?

Needs the ABIDE-pcp download (nilearn) — run on a machine with the data, e.g.:
    HOMER_ALLOW_INSECURE_SSL=1 PYTHONPATH=src python \
        experiments/autism_subtypes/abide_subtype/05_abide_homer_subtyping.py \
        --abide-data-dir /tmp/abide_cache
(see 01's header for the SSL/atlas notes). Prerequisite:
`04_homer_human_masks.py` must have been run (writes pagani_homer_human_masks.json).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# SSL workaround (same opt-in as abide_subtype_prediction.py)
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass
if os.environ.get("HOMER_ALLOW_INSECURE_SSL", "").lower() in {"1", "true", "yes"}:
    import ssl
    import requests
    import urllib3
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _orig = requests.Session.send
    requests.Session.send = lambda self, r, **k: _orig(self, r, **{**k, "verify": False})

import nibabel as nib  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import load_cached  # noqa: E402

LOG = ROOT / "outputs" / "logs"

# Pagani name-matched human regions → AAL-116 label substrings.
NAME_AAL = {
    "hypo":  ["Cingulum_Ant", "Cingulum_Mid", "Insula", "Precentral",
              "Supp_Motor", "Caudate", "Putamen"],
    "hyper": ["Amygdala", "Hippocampus", "Caudate", "Putamen"],
}


def aal_regions_for_homer_mask(mask_parcel_idx, H_xyz, aal_centroids):
    """An AAL region is 'in' the HOMER mask if its nearest HOMER human parcel is."""
    maskset = set(int(i) for i in mask_parcel_idx)
    sq_a = (H_xyz ** 2).sum(1)
    sq_b = (aal_centroids ** 2).sum(1, keepdims=True)
    d2 = sq_b + sq_a[None, :] - 2.0 * aal_centroids @ H_xyz.T
    nearest = d2.argmin(axis=1)                      # AAL region -> HOMER parcel
    return np.array([int(n) in maskset for n in nearest])  # bool over AAL regions


def classify(global_conn_z, hypo_aal, hyper_aal):
    """Pagani rule: hypo if mean z over hypo-mask AAL regions < −1; hyper if > +1.
    global_conn_z: (n_subjects, n_aal) z-scored regional global connectivity."""
    hypo_score = np.nanmean(global_conn_z[:, hypo_aal], axis=1)
    hyper_score = np.nanmean(global_conn_z[:, hyper_aal], axis=1)
    label = np.full(len(global_conn_z), "unsubtyped", dtype=object)
    label[hypo_score < -1.0] = "hypo"
    label[hyper_score > +1.0] = "hyper"   # hyper takes precedence if both (rare)
    return label, hypo_score, hyper_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abide-data-dir", default=str(Path.home() / "abide_cache"))
    ap.add_argument("--pipeline", default="cpac")
    args = ap.parse_args()

    masks = json.loads((LOG / "pagani_homer_human_masks.json").read_text())
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs/anndata"))
    H_xyz = H.var[["x", "y", "z"]].to_numpy(float)

    # ---- fetch ABIDE + AAL (mirror abide_subtype_prediction.py) ----
    from nilearn import datasets as ndatasets
    print("Fetching ABIDE (rois_aal) + AAL atlas ...")
    abide = ndatasets.fetch_abide_pcp(data_dir=args.abide_data_dir,
                                      derivatives=["rois_aal"], pipeline=args.pipeline,
                                      quality_checked=True)
    rois = abide["rois_aal"]; pheno = abide["phenotypic"]
    dx = np.array([int(p["DX_GROUP"]) for p in pheno])   # 1=autism, 2=control
    aal = ndatasets.fetch_atlas_aal(data_dir=args.abide_data_dir)
    aal_img = nib.load(str(aal["maps"])); aal_arr = aal_img.get_fdata().astype(int)
    labels = list(aal["labels"])
    idxs = [int(i) for i in aal["indices"]]
    cents = []
    for lab in idxs:
        ii, jj, kk = np.where(aal_arr == lab)
        cents.append((aal_img.affine @ np.array([ii.mean(), jj.mean(), kk.mean(), 1]))[:3])
    aal_centroids = np.array(cents)
    n_aal = len(labels)

    # ---- per-subject regional global connectivity ----
    profiles = []
    for ts in rois:
        ts = np.asarray(ts, float)                       # (T, n_aal)
        if ts.shape[1] != n_aal:                          # align if ABIDE dropped ROIs
            m = min(ts.shape[1], n_aal)
            ts = ts[:, :m]
        fc = np.corrcoef(ts.T)
        np.fill_diagonal(fc, np.nan)
        gc = np.nanmean(fc, axis=1)                       # global connectivity per region
        if gc.shape[0] < n_aal:
            gc = np.concatenate([gc, np.full(n_aal - gc.shape[0], np.nan)])
        profiles.append(gc)
    G = np.array(profiles)                                # (n_subj, n_aal)
    ctrl = dx == 2
    mu, sd = np.nanmean(G[ctrl], 0), np.nanstd(G[ctrl], 0) + 1e-9
    Gz = (G - mu) / sd                                    # z vs controls
    asd = dx == 1

    def name_mask(substrs):
        return np.array([any(s.lower() in lab.lower() for s in substrs) for lab in labels])

    results = {}
    for scheme in ("homer", "name"):
        if scheme == "homer":
            hypo_aal = aal_regions_for_homer_mask(masks["masks"]["hypo"]["parcel_indices"], H_xyz, aal_centroids)
            hyper_aal = aal_regions_for_homer_mask(masks["masks"]["hyper"]["parcel_indices"], H_xyz, aal_centroids)
        else:
            hypo_aal = name_mask(NAME_AAL["hypo"]); hyper_aal = name_mask(NAME_AAL["hyper"])
        lab, _, _ = classify(Gz[asd], hypo_aal, hyper_aal)
        n = len(lab)
        results[scheme] = {
            "n_hypo": int((lab == "hypo").sum()), "n_hyper": int((lab == "hyper").sum()),
            "n_subtyped": int((lab != "unsubtyped").sum()), "n_total": n,
            "pct_subtyped": round(100 * (lab != "unsubtyped").sum() / n, 1),
            "hypo_aal_regions": int(hypo_aal.sum()), "hyper_aal_regions": int(hyper_aal.sum()),
            "labels": lab.tolist(),
        }
        print(f"[{scheme:5}] subtyped {results[scheme]['pct_subtyped']}% "
              f"(hypo {results[scheme]['n_hypo']}, hyper {results[scheme]['n_hyper']}) of {n} ASD"
              f"  | mask AAL regions: hypo {int(hypo_aal.sum())}, hyper {int(hyper_aal.sum())}")

    # agreement between the two subtypings
    lh = np.array(results["homer"]["labels"]); ln = np.array(results["name"]["labels"])
    agree = float((lh == ln).mean())
    print(f"\nPagani reports ~25% subtyped (7.9% hypo / 17.2% hyper). "
          f"HOMER-mask subtyped {results['homer']['pct_subtyped']}%, name-mask {results['name']['pct_subtyped']}%.")
    print(f"Label agreement HOMER vs name-matched: {agree:.2f}")
    out = {"results": {k: {kk: vv for kk, vv in v.items() if kk != 'labels'}
                       for k, v in results.items()},
           "label_agreement": agree, "pagani_reference_pct": 25.1}
    (LOG / "abide_homer_subtyping.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {LOG/'abide_homer_subtyping.json'}")


if __name__ == "__main__":
    main()
