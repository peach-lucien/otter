"""Estimate the rigid transform from the colleague's bregma-centred mouse
coordinate system into Allen CCFv3 voxel space.

The colleague's mouse mask is in a bregma-centred stereotaxic convention
(x symmetric around midline, y/z origin near a stereotaxic landmark) — not
in raw CCFv3 coordinates. This script:

  1. Downloads CCFv3 100 µm annotation via AllenSDK.
  2. Tries all 48 possible signed axis permutations of the colleague's
     coordinates (6 permutations × 8 sign flips), each followed by a translation
     that aligns the brain centroids.
  3. For each candidate transform, computes the fraction of mouse-node centres
     that land inside the CCFv3 brain.
  4. Reports the best transform and saves it for use by 01_mouse_sc.py /
     02_mouse_genes.py.

If the best transform achieves >90% brain coverage, you're done. If 50–90%,
the transform is approximately right but a fine-tuning ANTs step is recommended.
If <50%, the colleague's space is not a simple permutation of CCFv3 — proper
registration required.

Output: data_external/_diagnostics/mouse_to_ccf_transform.json
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import DATA_DIR, load_metadata, parse_t_table        # noqa: E402

OUT = ROOT / "data_external" / "_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".reference_templates"


def main(min_coverage_pass: float = 0.9):
    try:
        from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    except ImportError:
        print("ERROR: pip install allensdk (then re-run)")
        sys.exit(1)

    print("downloading Allen CCFv3 100 µm annotation...")
    mcc = MouseConnectivityCache(resolution=100, manifest_file=str(CACHE / "allen_manifest.json"))
    ann, _ = mcc.get_annotation_volume()
    res_mm = 0.1
    ccf_brain = (ann > 0)
    ix, iy, iz = np.where(ccf_brain)
    ccf_brain_voxels = np.stack([ix, iy, iz], axis=1)
    ccf_centroid = ccf_brain_voxels.mean(axis=0) * res_mm
    print(f"  CCFv3 100 µm shape={ann.shape}; brain centroid (mm) = {ccf_centroid.round(2).tolist()}")

    # Colleague mouse mask centroid in its own world coords
    rsmask = nib.load(DATA_DIR / "_mouse_mask" / "rsmask.nii")
    mask_data = (np.asarray(rsmask.dataobj) > 0)
    ix, iy, iz = np.where(mask_data)
    ones = np.ones(len(ix))
    world = (rsmask.affine @ np.stack([ix, iy, iz, ones])).T[:, :3]
    rs_centroid = world.mean(axis=0)
    print(f"  rsmask brain centroid (mm) = {rs_centroid.round(2).tolist()}")

    # Per-node centres
    meta = load_metadata("mouse"); df = parse_t_table(meta["t"], meta["ht"])
    centres = df[["x","y","z"]].values

    # Try all signed axis permutations
    best = {"coverage": -1.0}
    permutations = list({(i, j, k) for i, j, k in product(range(3), repeat=3) if len({i, j, k}) == 3})
    sign_combos = list(product([1, -1], repeat=3))
    for perm in permutations:
        for signs in sign_combos:
            # Apply: new_coord[axis] = signs[axis] * centres[:, perm[axis]]
            transformed = np.column_stack([
                signs[0] * centres[:, perm[0]],
                signs[1] * centres[:, perm[1]],
                signs[2] * centres[:, perm[2]],
            ])
            # Translate so transformed centroid lands at CCFv3 brain centroid
            transformed_centroid = transformed.mean(axis=0)
            shift = ccf_centroid - transformed_centroid
            transformed = transformed + shift
            # Convert to CCFv3 voxel index
            ccf_ijk = (transformed / res_mm).astype(int)
            in_bounds = ((ccf_ijk[:, 0] >= 0) & (ccf_ijk[:, 0] < ann.shape[0]) &
                         (ccf_ijk[:, 1] >= 0) & (ccf_ijk[:, 1] < ann.shape[1]) &
                         (ccf_ijk[:, 2] >= 0) & (ccf_ijk[:, 2] < ann.shape[2]))
            if not in_bounds.any():
                continue
            region_ids = np.zeros(len(centres), dtype=np.int32)
            ok = np.where(in_bounds)[0]
            region_ids[ok] = ann[ccf_ijk[ok, 0], ccf_ijk[ok, 1], ccf_ijk[ok, 2]]
            coverage = float((region_ids > 0).mean())
            if coverage > best["coverage"]:
                best = {
                    "perm":         list(perm),
                    "signs":        list(signs),
                    "shift_mm":     shift.tolist(),
                    "coverage":     coverage,
                    "n_in_brain":   int((region_ids > 0).sum()),
                    "n_total":      int(len(centres)),
                }

    print(f"\nBest transform found:")
    print(f"  axis permutation : {best['perm']}  (colleague's [x,y,z] → CCFv3 axis order)")
    print(f"  sign flips       : {best['signs']}")
    print(f"  translation (mm) : {[round(v, 2) for v in best['shift_mm']]}")
    print(f"  brain coverage   : {best['coverage']:.1%}  ({best['n_in_brain']} / {best['n_total']})")

    if best["coverage"] >= min_coverage_pass:
        verdict = "ok_use_directly"
        print("  ✓ transform is good — saved for use by 01_mouse_sc.py")
    elif best["coverage"] >= 0.5:
        verdict = "approximate_recommend_ants"
        print("  ⚠ transform is approximate — usable but ANTs registration would be cleaner")
    else:
        verdict = "needs_ants"
        print("  ✗ no simple transform works — proper registration required")

    out = {**best, "verdict": verdict, "ccf_centroid_mm": ccf_centroid.tolist(),
           "rs_centroid_mm": rs_centroid.tolist()}
    out_path = OUT / "mouse_to_ccf_transform.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved → {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-coverage", type=float, default=0.9)
    main(ap.parse_args().min_coverage)
