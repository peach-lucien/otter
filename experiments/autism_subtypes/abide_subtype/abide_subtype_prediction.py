"""ABIDE per-subject HOMER-template subtype scoring.

Tests whether HOMER's mouse → human translation of Pagani's per-subtype
network maps can distinguish ASD subjects from controls at the individual
level, and whether ASD subjects split into hyper/hypo subtypes by their
projection onto the HOMER templates.

This does NOT re-implement Pagani's clustering pipeline. We just:
  1. Build two HOMER-translated human templates (one per Pagani subtype) by
     routing the mouse 9×9 network perturbation matrices through π.
  2. For each ABIDE subject, compute a per-parcel FC strength profile,
     subtract site-matched control mean to get a perturbation pattern.
  3. Score subject against (hyper - hypo) HOMER template.
  4. Test: ASD vs control on that score; within-ASD bimodality.

Disk / time:
  - ABIDE preprocessed CC400 timeseries: ~3-8 GB (one-time download via nilearn).
  - Craddock atlas: ~10 MB.
  - Wall-clock first run: 1-2 hours after download finishes.
  - Cached on subsequent runs.

Usage:
    PYTHONPATH=src python experiments/autism_subtypes/abide_subtype/abide_subtype_prediction.py
    PYTHONPATH=src python ... --n-subjects 50           # smoke test
    PYTHONPATH=src python ... --abide-data-dir /path/to/cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# ---- SSL/CA fix for macOS Anaconda environments where the system CA store is
# out of date and downloads via `requests` fail with "unable to get local
# issuer certificate". Point requests/urllib to certifi's bundled CAs, which
# are always current. Set BEFORE importing requests/nilearn so they pick it up.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

# If env-level CA fix isn't enough (very old certifi), allow opting in to
# skipping SSL verification via env var. This is OPT-IN — the user has to
# explicitly set it, never on by default. Must monkey-patch BOTH stdlib ssl
# (used by urllib) AND requests.Session (used by nilearn), since they have
# independent SSL configurations.
if os.environ.get("HOMER_ALLOW_INSECURE_SSL", "").lower() in {"1", "true", "yes"}:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    # Patch requests.Session BEFORE nilearn imports — set verify=False as
    # default on every Session instance constructed thereafter.
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _orig_session_init = requests.Session.__init__
    def _patched_session_init(self, *a, **kw):
        _orig_session_init(self, *a, **kw)
        self.verify = False
    requests.Session.__init__ = _patched_session_init
    # Also patch already-constructed default adapters
    _orig_send = requests.Session.send
    def _patched_send(self, request, **kw):
        kw.setdefault("verify", False)
        return _orig_send(self, request, **kw)
    requests.Session.send = _patched_send
    warnings.warn("HOMER_ALLOW_INSECURE_SSL is set — SSL verification "
                  "is DISABLED for this run. Re-enable for production use.")

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, mannwhitneyu, gaussian_kde

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "autism_subtypes"))

from homer.data import load_cached

ABIDE_DERIVATIVE = "rois_aal"   # was rois_cc400; CC400 atlas host (NITRC) has
                                 # an SSL/hostname problem as of 2026. AAL-116
                                 # is universally available + nilearn's fetcher
                                 # works against a different (GIN/CNRS) host.


def build_homer_templates(mouse_M_hypo, mouse_M_hyper, mouse_pagani_net,
                          n_mouse_nets, pi, kept_mask):
    """Per-human-parcel HOMER templates for each subtype."""
    def _intensity(M):
        Ma = np.abs(M)
        return Ma.sum(axis=0) + Ma.sum(axis=1) - np.diag(Ma)
    hypo_int = _intensity(mouse_M_hypo)
    hyper_int = _intensity(mouse_M_hyper)
    def _to_parcel(intensity_per_net):
        v = np.zeros(pi.shape[0])
        for i in range(n_mouse_nets):
            v[(mouse_pagani_net == i) & kept_mask] = intensity_per_net[i]
        return v
    return _to_parcel(hypo_int) @ pi, _to_parcel(hyper_int) @ pi


def map_cc400_to_homer_parcels(H_var, cc400_centroids):
    """Nearest-HOMER-parcel index per CC400 parcel."""
    H_xyz = H_var[["x", "y", "z"]].to_numpy()
    sq_a = (H_xyz ** 2).sum(1, keepdims=True)
    sq_b = (cc400_centroids ** 2).sum(1, keepdims=True)
    d2 = sq_b + sq_a.T - 2.0 * cc400_centroids @ H_xyz.T
    return d2.argmin(axis=1)


def _phenotypic_field(pheno, name):
    """Robust phenotype access — handles structured array, DataFrame, or recarray."""
    if hasattr(pheno, "columns") and name in pheno.columns:
        return pheno[name].values
    if hasattr(pheno, "dtype") and name in pheno.dtype.names:
        return np.array([row[name] for row in pheno])
    raise KeyError(f"Phenotype field {name} not found")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--abide-data-dir", default=str(Path.home() / "abide_cache"),
                   help="nilearn cache directory for ABIDE + Craddock data")
    p.add_argument("--n-subjects", type=int, default=None,
                   help="Limit to first N (smoke test). Default: all available.")
    p.add_argument("--pipeline", default="cpac",
                   choices=["cpac", "ccs", "dparsf", "niak"],
                   help="ABIDE pcp preprocessing pipeline")
    p.add_argument("--out-dir", default=str(ROOT / "outputs" / "logs"))
    args = p.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ABIDE per-subject HOMER-template subtype scoring")
    print("=" * 80)

    # ---- Step 1: HOMER templates ----
    print("\nStep 1: building HOMER-translated templates from Pagani mouse matrices...")
    H, _ = load_cached("human", cache_dir=str(ROOT / "outputs" / "anndata"))
    M, _ = load_cached("mouse", cache_dir=str(ROOT / "outputs" / "anndata"))
    pi = np.load(str(ROOT / "outputs" / "coupling" / "pi_fc_plus_SC.npy"))

    from importlib import import_module
    st = import_module("04_subtype_translation")
    fm = import_module("07_full_matrix_translation")

    data = st.load_pagani_subtype_matrices()
    mouse_pagani_net, mouse_pagani_names = fm.assign_mouse_pagani_networks(M.var)
    keep = mouse_pagani_net >= 0
    print(f"  using {keep.sum()}/{len(keep)} mouse parcels")

    M_hypo  = 0.5 * (data["mouse_hypo"]  + data["mouse_hypo"].T)
    M_hyper = 0.5 * (data["mouse_hyper"] + data["mouse_hyper"].T)
    template_hypo, template_hyper = build_homer_templates(
        M_hypo, M_hyper, mouse_pagani_net, len(mouse_pagani_names), pi, keep)
    template_delta = template_hyper - template_hypo
    template_z = (template_delta - template_delta.mean()) / (template_delta.std() + 1e-9)
    print(f"  template Δ over 2,094 human parcels: "
          f"mean {template_delta.mean():+.4f}, sd {template_delta.std():.4f}")

    # ---- Step 2: fetch ABIDE ----
    print(f"\nStep 2: fetching ABIDE preprocessed FC ({args.pipeline} / {ABIDE_DERIVATIVE})...")
    print(f"  cache dir: {args.abide_data_dir}")
    from nilearn import datasets as ndatasets

    abide = ndatasets.fetch_abide_pcp(
        data_dir=args.abide_data_dir, derivatives=[ABIDE_DERIVATIVE],
        pipeline=args.pipeline, band_pass_filtering=True,
        global_signal_regression=False, quality_checked=True,
    )
    pheno = abide.phenotypic
    rois = getattr(abide, ABIDE_DERIVATIVE)
    print(f"  ABIDE subjects fetched: {len(rois)}")
    if args.n_subjects:
        rois = rois[: args.n_subjects]
        if hasattr(pheno, "iloc"):
            pheno = pheno.iloc[: args.n_subjects].reset_index(drop=True)
        else:
            pheno = pheno[: args.n_subjects]
        print(f"  (limiting to first {args.n_subjects} for smoke test)")

    # ---- Step 3: AAL atlas → HOMER mapping ----
    # (Switched from CC400 because cluster_roi.projects.nitrc.org has an SSL
    #  cert mismatch as of 2026 and nilearn's Craddock fetcher fails. AAL is
    #  hosted at www.gin.cnrs.fr which works reliably.)
    print(f"\nStep 3: aligning AAL atlas → HOMER parcels...")
    import nibabel as nib
    aal = ndatasets.fetch_atlas_aal(data_dir=args.abide_data_dir)
    aal_path = aal["maps"] if "maps" in aal else aal.maps
    aal_img = nib.load(str(aal_path))
    aal_arr = aal_img.get_fdata().astype(int)
    aal_affine = aal_img.affine
    # AAL labels are non-contiguous integers; "labels" + "indices" come from the bunch
    aal_labels = aal["labels"]
    aal_indices = [int(i) for i in aal["indices"]] if "indices" in aal else \
                   sorted(int(x) for x in np.unique(aal_arr) if x != 0)
    centroids = []
    valid_indices = []
    for lab in aal_indices:
        ii, jj, kk = np.where(aal_arr == lab)
        if len(ii) == 0: continue
        ijk_mean = np.array([ii.mean(), jj.mean(), kk.mean(), 1.0])
        centroids.append((aal_affine @ ijk_mean)[:3])
        valid_indices.append(lab)
    aal_centroids = np.array(centroids)
    print(f"  AAL parcels resolved: {len(aal_centroids)} (expected ~116)")
    cc_to_homer = map_cc400_to_homer_parcels(H.var, aal_centroids)
    cc_centroids = aal_centroids   # variable reused below
    n_h = pi.shape[1]

    # ---- Step 4: per-subject FC + template scoring ----
    print("\nStep 4: per-subject scoring against HOMER templates...")
    site = _phenotypic_field(pheno, "SITE_ID").astype(str)
    dx = _phenotypic_field(pheno, "DX_GROUP").astype(int)
    subj_id = _phenotypic_field(pheno, "SUB_ID") if "SUB_ID" in (
        pheno.columns if hasattr(pheno, "columns") else pheno.dtype.names) else np.arange(len(rois))

    # First pass: discover what column count the timeseries actually have.
    # nilearn 0.13 returns rois as an iterable that yields either:
    #   - numpy.ndarray (already loaded) — most common
    #   - str (file path to .1D) — older versions
    # Handle both.
    from collections import Counter

    def _coerce_to_array(entry):
        """Return ndarray or None. Accepts either np.ndarray or path-like."""
        if isinstance(entry, np.ndarray):
            return entry
        if isinstance(entry, (str, os.PathLike)):
            p = str(entry)
            if not os.path.exists(p):
                return None
            try:
                with open(p) as fh:
                    first = fh.readline()
                sep = "\t" if "\t" in first else None
                arr = np.loadtxt(p, delimiter=sep, comments="#")
                return arr if arr.ndim == 2 else None
            except Exception:
                return None
        return None

    print(f"  type(rois) = {type(rois).__name__}, len = {len(rois)}")
    print(f"    rois[0] type = {type(rois[0]).__name__}")
    col_counts = []
    sample_shape = None
    for ts in rois[:30]:
        arr = _coerce_to_array(ts)
        if arr is None: continue
        col_counts.append(arr.shape[1])
        if sample_shape is None:
            sample_shape = arr.shape
            print(f"  first valid timeseries shape: {sample_shape}, first row[:5]: {arr[0, :5]}")
    if not col_counts:
        print("  ERROR: no valid timeseries found among the first 30 subjects.")
        sys.exit(1)
    modal_cols = Counter(col_counts).most_common(1)[0][0]
    print(f"  modal column count across timeseries: {modal_cols} "
          f"(atlas has {len(cc_centroids)} centroids)")

    # Reconcile centroid count with actual timeseries column count
    if modal_cols < len(cc_centroids):
        print(f"  trimming centroids to first {modal_cols} (ABIDE dropped some ROIs)")
        cc_centroids = cc_centroids[:modal_cols]
        cc_to_homer = cc_to_homer[:modal_cols]
    elif modal_cols > len(cc_centroids):
        print(f"  WARN: timeseries has more columns ({modal_cols}) than atlas "
              f"centroids ({len(cc_centroids)}); this is unexpected.")

    # Compute TWO per-subject features in parallel:
    #   "abs"    — mean(|FC|) per parcel (weighted degree; loses sign — same for hyper/hypo)
    #   "signed" — mean(FC) per parcel (signed degree; positive = locally hyperconnected,
    #             negative = locally hypoconnected). This is what Pagani's signal lives in.
    # Audit M1: original abs-only feature destroyed the hyper/hypo distinction.
    subject_pat_abs    = np.full((len(rois), n_h), np.nan, dtype=np.float32)
    subject_pat_signed = np.full((len(rois), n_h), np.nan, dtype=np.float32)
    skipped = 0
    skipped_reasons: Counter = Counter()
    for s, ts_entry in enumerate(rois):
        ts = _coerce_to_array(ts_entry)
        if ts is None:
            skipped += 1; skipped_reasons["unreadable"] += 1; continue
        if ts.shape[1] != modal_cols:
            skipped += 1; skipped_reasons[f"col_mismatch_{ts.shape[1]}"] += 1; continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fc = np.corrcoef(ts.T)
            # Mask diagonal (self-correlation = 1; not informative)
            np.fill_diagonal(fc, np.nan)
            per_cc_abs    = np.nanmean(np.abs(fc), axis=1)
            per_cc_signed = np.nanmean(fc, axis=1)
        per_homer_abs    = np.zeros(n_h); per_homer_signed = np.zeros(n_h); cnt = np.zeros(n_h)
        for i, hp in enumerate(cc_to_homer):
            if i >= len(per_cc_abs): break
            per_homer_abs[hp]    += per_cc_abs[i]
            per_homer_signed[hp] += per_cc_signed[i]
            cnt[hp] += 1
        denom = np.maximum(cnt, 1)
        subject_pat_abs[s]    = per_homer_abs / denom
        subject_pat_signed[s] = per_homer_signed / denom
        if (s + 1) % 100 == 0:
            print(f"    {s+1}/{len(rois)} processed (skipped: {skipped})")
    print(f"  total skipped: {skipped}/{len(rois)}")
    if skipped:
        print(f"  skip reasons: {dict(skipped_reasons)}")

    # Site-matched control mean, computed separately for each feature variant.
    # Audit B3 fix: track sites with control coverage and exclude subjects from
    # sites without any controls (otherwise they get uncorrected = biased).
    def _site_correct(pattern):
        valid = np.isfinite(pattern).all(axis=1)
        sites_with_ctrl = set()
        site_unique = np.unique(site[valid])
        site_mean = np.zeros((len(site_unique), n_h))
        for k, st_id in enumerate(site_unique):
            m = (site == st_id) & (dx == 2) & valid
            if m.any():
                site_mean[k] = pattern[m].mean(axis=0)
                sites_with_ctrl.add(st_id)
        site_idx = np.array([list(site_unique).index(s) if s in site_unique else -1
                              for s in site])
        has_ctrl = np.array([s in sites_with_ctrl for s in site])
        pert = np.full_like(pattern, np.nan)
        ok = valid & has_ctrl
        pert[ok] = pattern[ok] - site_mean[site_idx[ok]]
        return pert, ok

    # Test for both features. The audit M1 fix predicts that "signed" should be
    # informative whereas "abs" is not (loses sign).
    results_by_feature = {}
    for feat_name, pattern in [("abs", subject_pat_abs), ("signed", subject_pat_signed)]:
        pert, ok = _site_correct(pattern)
        score = np.nanmean(pert * template_z, axis=1)
        asd = (dx == 1) & ok
        ctrl = (dx == 2) & ok
        asd_s, ctrl_s = score[asd], score[ctrl]
        u, p = mannwhitneyu(asd_s, ctrl_s, alternative="two-sided")
        n1, n2 = len(asd_s), len(ctrl_s)
        cliffs = float(2 * u / (n1 * n2) - 1)
        print(f"\n--- Feature: {feat_name} ({asd.sum()} ASD, {ctrl.sum()} control valid) ---")
        print(f"  ASD:  mean={asd_s.mean():+.5e}, sd={asd_s.std():.5e}")
        print(f"  CTRL: mean={ctrl_s.mean():+.5e}, sd={ctrl_s.std():.5e}")
        print(f"  Mann-Whitney U = {u:.0f}, two-sided p = {p:.4f}")
        print(f"  Cliff's δ = {cliffs:+.4f}  (>0 means ASD > CTRL)")
        results_by_feature[feat_name] = {
            "pattern": pattern, "pert": pert, "ok": ok, "score": score,
            "asd_mean": float(asd_s.mean()), "asd_sd": float(asd_s.std()),
            "ctrl_mean": float(ctrl_s.mean()), "ctrl_sd": float(ctrl_s.std()),
            "u": float(u), "p": float(p), "cliffs": cliffs,
            "n_asd": int(asd.sum()), "n_ctrl": int(ctrl.sum()),
        }

    # Use the "signed" feature for downstream within-ASD bimodality
    score = results_by_feature["signed"]["score"]
    valid = results_by_feature["signed"]["ok"]
    asd  = (dx == 1) & valid
    ctrl = (dx == 2) & valid
    asd_s, ctrl_s = score[asd], score[ctrl]
    u = results_by_feature["signed"]["u"]
    p = results_by_feature["signed"]["p"]
    cliffs = results_by_feature["signed"]["cliffs"]
    print(f"\n[Using 'signed' feature for the bimodality + summary panels below.]")

    # ---- Step 6: within-ASD bimodality ----
    print(f"\nWithin-ASD bimodality check (kernel density):")
    kde = gaussian_kde(asd_s)
    xs = np.linspace(asd_s.min(), asd_s.max(), 200)
    density = kde(xs)
    peaks = [i for i in range(2, len(xs)-2)
             if density[i] > density[i-1] > density[i-2]
             and density[i] > density[i+1] > density[i+2]]
    print(f"  density peaks within ASD: {len(peaks)} (≥2 suggests bimodal)")
    # Hartigan's dip test would be more rigorous but isn't in scipy; use AIC of
    # 1-component vs 2-component GMM as a heuristic
    try:
        from sklearn.mixture import GaussianMixture
        g1 = GaussianMixture(n_components=1).fit(asd_s.reshape(-1, 1))
        g2 = GaussianMixture(n_components=2).fit(asd_s.reshape(-1, 1))
        aic1, aic2 = g1.aic(asd_s.reshape(-1, 1)), g2.aic(asd_s.reshape(-1, 1))
        bic1, bic2 = g1.bic(asd_s.reshape(-1, 1)), g2.bic(asd_s.reshape(-1, 1))
        print(f"  GMM 1-comp AIC = {aic1:.1f}, BIC = {bic1:.1f}")
        print(f"  GMM 2-comp AIC = {aic2:.1f}, BIC = {bic2:.1f}")
        print(f"  Δ AIC (2-1) = {aic2-aic1:+.1f}  ({'2-comp preferred' if aic2 < aic1 else '1-comp preferred'})")
        print(f"  Δ BIC (2-1) = {bic2-bic1:+.1f}  ({'2-comp preferred' if bic2 < bic1 else '1-comp preferred'})")
        gmm_verdict = {
            "delta_aic_2_minus_1": float(aic2 - aic1),
            "delta_bic_2_minus_1": float(bic2 - bic1),
            "two_comp_preferred_aic": bool(aic2 < aic1),
            "two_comp_preferred_bic": bool(bic2 < bic1),
        }
    except ImportError:
        gmm_verdict = None
        print("  sklearn not available; skipped GMM bimodality test")

    # ---- Output ----
    out = {
        "n_subjects":    int(len(rois)),
        "n_valid":       int(valid.sum()),
        "n_asd":         int(asd.sum()),
        "n_control":     int(ctrl.sum()),
        "homer_template_dim": int(n_h),
        "cc400_parcels": int(len(cc_centroids)),
        "by_feature": {k: {kk: vv for kk, vv in v.items()
                            if kk in {"asd_mean","asd_sd","ctrl_mean","ctrl_sd",
                                      "u","p","cliffs","n_asd","n_ctrl"}}
                       for k, v in results_by_feature.items()},
        # Keep legacy keys mirroring the "signed" feature (the audit-fixed one)
        "asd_score_mean":  float(asd_s.mean()),
        "asd_score_sd":    float(asd_s.std()),
        "ctrl_score_mean": float(ctrl_s.mean()),
        "ctrl_score_sd":   float(ctrl_s.std()),
        "mann_whitney_u":  float(u),
        "mann_whitney_p":  float(p),
        "cliffs_delta":    float(cliffs),
        "within_asd_density_peaks": int(len(peaks)),
        "gmm_bimodality":  gmm_verdict,
    }
    (out_dir / "autism_subtypes_abide.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_dir/'autism_subtypes_abide.json'}")

    # Also save raw per-subject scores for figure-making
    if hasattr(pheno, "iloc"):
        df_out = pheno.copy()
    else:
        df_out = pd.DataFrame({k: pheno[k] for k in pheno.dtype.names})
    df_out["homer_score"] = score
    df_out["valid"] = valid
    df_out.to_csv(out_dir / "abide_per_subject_scores.csv", index=False)
    print(f"Wrote {out_dir/'abide_per_subject_scores.csv'}")


if __name__ == "__main__":
    main()
