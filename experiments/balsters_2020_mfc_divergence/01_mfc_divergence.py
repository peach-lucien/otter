"""HOMER × Balsters 2020 — rodent medial frontal cortex divergence (falsification test).

[Balsters, Zerbi, Sallet, Wenderoth & Mars 2020, PNAS](https://doi.org/10.1073/pnas.2003181117)
— "Divergence of rodent and primate medial frontal cortex functional
connectivity" — compared whole-brain FC of the medial frontal cortex (MFC)
across rodent, marmoset and human. Their headline, data-backed claim:

  * Rodent MFC does **NOT** correspond to primate **lateral / dorsolateral
    prefrontal cortex** — contradicting the common proposal that rat MFC is
    the functional analogue of primate LFC.
  * Rodent MFC connectivity instead most resembles **premotor** cortex.

This is a *falsification* test, not a confirmation test. It states, with a
specific direction, where a faithful mouse↔human mapping should and should
NOT send mouse MFC:

  PASS  — mouse MFC routes to human medial-frontal / cingulate / premotor
          cortex, and **avoids** dorsolateral PFC (BA9/46).
  FAIL  — mouse MFC routes confidently onto human dlPFC.

HOMER already encodes a falsifiable design choice here: the Garin point
anchor for mPFC pairs mouse mPFC with human *medial* frontal cortex, and
the contested mouse-Prelimbic ↔ human-dlPFC homology (Carlén 2017 vs
Preuss 1995) is shipped as the **opt-in** `lateral_pfc` pack, not in the
recommended π. Balsters 2020 is independent FC evidence adjudicating that
choice. We test three couplings:

  * `pi_fc_plus_SC.npy`              — Garin anchors only (strict baseline)
  * `pi_fc_plus_SC_with_all_packs`  — recommended (no lateral_pfc pack)
  * `pi_fc_plus_SC_with_lateral_pfc`— adds the contested Prelimbic→dlPFC anchor

Note on species: Balsters used rat + marmoset + human; HOMER is mouse +
human. Rodent MFC (anterior cingulate + prelimbic + infralimbic) is the
comparable structure. The test compares HOMER's π against Balsters'
*published directional conclusion*, not their FC matrices — the rat/mouse
and marmoset/human mismatches make re-routing their data unjustified.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from homer.data import load_cached

ANN = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
N_NULL = 200
SEED = 42

# Human target ROIs — bilateral MNI spheres (|x|, y, z, radius_mm).
# dlPFC reuses the lateral_pfc anchor pack's own BA9/46 centroid.
ROIS = {
    "dlPFC":         (40, 25, 35, 12),   # BA9/46 — the contested "should NOT"
    "premotor":      (28,  0, 54, 14),   # BA6 / PMd — Balsters' "instead"
    "medial_PFC":    ( 7, 34, 22, 16),   # mPFC / pregenual ACC — conventional homologue
    "mid_cingulate": ( 5,  8, 40, 14),   # mid-cingulate — conventional homologue
}
# Rodent MFC = the medial frontal wall (Balsters' "rat MFC").
MFC_ACRONYMS = ["ACAd", "ACAv", "PL", "ILA"]


def assign_human_rois(H_var) -> np.ndarray:
    """Label each human parcel with its nearest ROI (within radius), else 'other'."""
    xyz = H_var[["x", "y", "z"]].to_numpy(dtype=float)
    labels = np.array(["other"] * len(xyz), dtype=object)
    best = np.full(len(xyz), np.inf)
    for name, (cx, cy, cz, r) in ROIS.items():
        for sx in (-1, 1):
            d = np.linalg.norm(xyz - np.array([sx * cx, cy, cz]), axis=1)
            sel = (d <= r) & (d < best)
            labels[sel] = name
            best[sel] = d[sel]
    return labels


def mass_fractions(pi: np.ndarray, rows: np.ndarray,
                   roi_labels: np.ndarray) -> dict[str, float]:
    """Fraction of π mass from `rows` landing in each ROI."""
    sub = pi[rows]
    total = sub.sum()
    out = {}
    for name in list(ROIS) + ["other"]:
        out[name] = float(sub[:, roi_labels == name].sum() / total)
    return out


def main():
    print("=" * 80)
    print("HOMER × Balsters 2020 — rodent MFC divergence (falsification test)")
    print("=" * 80)

    # ---- human ROIs ---------------------------------------------------------
    H, _ = load_cached("human", cache_dir=ANN)
    roi_labels = assign_human_rois(H.var)
    print("\nHuman target ROIs (bilateral MNI spheres):")
    for name in ROIS:
        print(f"  {name:14s} {int((roi_labels == name).sum()):3d} parcels")

    # ---- mouse MFC parcels --------------------------------------------------
    mm = json.loads((ROOT / "data_external" / "mouse_sc_meta.json").read_text())
    acr = [mm["structure_acronyms"][i] for i in mm["node_struct_idx"]]
    mfc = np.array([i for i, a in enumerate(acr) if a in MFC_ACRONYMS])
    pl = np.array([i for i, a in enumerate(acr) if a == "PL"])
    print(f"\nMouse rodent-MFC parcels (ACAd/ACAv/PL/ILA): {len(mfc)}")
    print(f"Mouse Prelimbic (PL) parcels:                {len(pl)}")

    # ---- recommended π: where does mouse MFC land? -------------------------
    pi = np.load(COUP / "pi_fc_plus_SC_with_all_packs.npy")
    frac = mass_fractions(pi, mfc, roi_labels)
    print("\n[Recommended π] mouse-MFC coupling mass by human territory:")
    for name in list(ROIS) + ["other"]:
        print(f"  {name:14s} {frac[name] * 100:5.1f} %")

    # argmax landing of each MFC parcel
    argmax_idx = [int(pi[i].argmax()) for i in mfc]
    argmax_roi = [roi_labels[j] for j in argmax_idx]
    argmax_counts = {n: int(argmax_roi.count(n)) for n in list(ROIS) + ["other"]}
    print(f"\n  argmax (top-1 human partner) of the {len(mfc)} MFC parcels:")
    for n, c in argmax_counts.items():
        print(f"    {n:14s} {c}")

    # ---- permuted-π null ----------------------------------------------------
    rng = np.random.default_rng(SEED)
    null = {n: [] for n in ROIS}
    for _ in range(N_NULL):
        perm = rng.permutation(pi.shape[0])
        f = mass_fractions(pi[perm], mfc, roi_labels)
        for n in ROIS:
            null[n].append(f[n])
    print(f"\n[Permuted-π null, {N_NULL} trials] observed vs chance (enrichment):")
    null_stats = {}
    for n in ROIS:
        arr = np.array(null[n])
        enr = frac[n] / arr.mean() if arr.mean() > 0 else float("nan")
        # one-sided p: dlPFC tests for NON-enrichment, others for enrichment
        emp_p = float((arr >= frac[n]).mean())
        null_stats[n] = {"mean": float(arr.mean()),
                         "ci95": [float(np.percentile(arr, 2.5)),
                                  float(np.percentile(arr, 97.5))],
                         "enrichment": enr, "empirical_p": emp_p}
        print(f"  {n:14s} obs {frac[n] * 100:5.1f} %   null {arr.mean() * 100:5.1f} %"
              f"   enrichment ×{enr:4.1f}   p={emp_p:.3f}")

    # ---- contrast: what the contested lateral_pfc anchor does --------------
    print("\n[Contrast] mouse-MFC and mouse-Prelimbic mass onto human dlPFC:")
    dl = roi_labels == "dlPFC"
    contrast = {}
    for tag, fname in [("baseline (Garin only)", "pi_fc_plus_SC.npy"),
                       ("recommended", "pi_fc_plus_SC_with_all_packs.npy"),
                       ("+lateral_pfc pack", "pi_fc_plus_SC_with_lateral_pfc.npy")]:
        p = np.load(COUP / fname)
        mfc_dl = float(p[mfc][:, dl].sum() / p[mfc].sum())
        pl_dl = float(p[pl][:, dl].sum() / p[pl].sum())
        contrast[tag] = {"file": fname, "mfc_to_dlpfc": mfc_dl, "pl_to_dlpfc": pl_dl}
        print(f"  {tag:24s} MFC→dlPFC {mfc_dl * 100:5.1f} %   "
              f"Prelimbic→dlPFC {pl_dl * 100:5.1f} %")

    # ---- dlPFC as an orphan territory --------------------------------------
    dlpfc_global = float(pi[:, dl].sum() / pi.sum())
    dlpfc_share = float(dl.sum() / pi.shape[1])
    print(f"\n[Orphan check] human dlPFC receives {dlpfc_global * 100:.1f}% of ALL "
          f"mouse→human mass; its parcel share is {dlpfc_share * 100:.1f}%.")

    # ---- verdict ------------------------------------------------------------
    balsters_consistent = (frac["medial_PFC"] + frac["mid_cingulate"]
                           + frac["premotor"])
    passed = (null_stats["dlPFC"]["enrichment"] <= 1.5
              and frac["dlPFC"] < 0.10
              and balsters_consistent > frac["dlPFC"])
    verdict = "PASS — Balsters-consistent" if passed else "FAIL"
    print(f"\nVERDICT: {verdict}")
    print(f"  mouse MFC → dlPFC = {frac['dlPFC'] * 100:.1f}% "
          f"(enrichment ×{null_stats['dlPFC']['enrichment']:.1f}, not favoured)")
    print(f"  mouse MFC → medial-frontal + cingulate + premotor = "
          f"{balsters_consistent * 100:.1f}%")

    # ---- save ---------------------------------------------------------------
    out = {
        "rois": {n: {"center_mni": ROIS[n][:3], "radius_mm": ROIS[n][3],
                     "n_parcels": int((roi_labels == n).sum())} for n in ROIS},
        "n_mouse_mfc_parcels": int(len(mfc)),
        "n_mouse_prelimbic_parcels": int(len(pl)),
        "recommended_pi": {
            "mass_fraction": frac,
            "argmax_counts": argmax_counts,
            "argmax_human_idx": argmax_idx,
            "null": null_stats,
        },
        "contrast": contrast,
        "dlpfc_orphan": {"global_mass_fraction": dlpfc_global,
                         "parcel_share": dlpfc_share},
        "verdict": verdict,
        "roi_labels": roi_labels.tolist(),
    }
    out_path = ROOT / "outputs" / "logs" / "balsters_2020_mfc_divergence.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
