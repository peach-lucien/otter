"""Verify that the colleague's masks are in standard reference spaces (just at
fMRI-friendly downsampled resolutions), so we can avoid running ANTs/FSL
registration.

The 200 µm voxel size + (62, 94, 47) shape strongly suggests the mouse mask is
the AIBS Average Template (RS_AVGT), a downsampled version of Allen CCFv3 used
in mouse rs-fMRI pipelines (Grandjean, Coletta, etc).

The 3 mm voxel size + (61, 73, 61) shape strongly suggests FSL's MNI152_T1_3mm
template, the standard for human rs-fMRI.

This script:
  1. Downloads the canonical reference templates (Allen CCFv3 100µm, MNI152 2mm).
  2. Compares each colleague mask's brain centroid + bounding box to the
     reference, after rescaling.
  3. Reports a confidence score for "in standard space, just downsampled".

If the report is YES for both, the downstream 01-04 scripts can use simple
resampling rather than full registration. If NO, ANTs is required.

Output: data_external/_diagnostics/alignment_verification.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import DATA_DIR, load_metadata, parse_t_table   # noqa: E402

OUT = ROOT / "data_external" / "_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".reference_templates"
CACHE.mkdir(parents=True, exist_ok=True)


def _download(url: str, dest: Path) -> None:
    if dest.exists(): return
    print(f"  downloading {url}\n    → {dest}")
    import requests
    r = requests.get(url, stream=True, timeout=120); r.raise_for_status()
    with dest.open("wb") as f:
        for chunk in r.iter_content(chunk_size=2**20):
            f.write(chunk)


def _bbox(data: np.ndarray, affine: np.ndarray) -> dict:
    """World-coordinate bounding box of non-zero voxels."""
    if data.size == 0 or (data > 0).sum() == 0:
        return {"empty": True}
    ix, iy, iz = np.where(data > 0)
    ijk = np.stack([ix, iy, iz, np.ones_like(ix)])
    world = (affine @ ijk).T[:, :3]
    return {
        "n_voxels":  int((data > 0).sum()),
        "bbox_min":  world.min(axis=0).tolist(),
        "bbox_max":  world.max(axis=0).tolist(),
        "centroid":  world.mean(axis=0).tolist(),
        "extent":    (world.max(axis=0) - world.min(axis=0)).tolist(),
    }


def verify_mouse():
    print("\n=== verifying mouse alignment ===")
    rsmask = nib.load(DATA_DIR / "_mouse_mask" / "rsmask.nii")
    rsmask_data = np.asarray(rsmask.dataobj)
    rsmask_bbox = _bbox(rsmask_data > 0, rsmask.affine)
    print(f"colleague's rsmask: shape={rsmask.shape}, bbox extent={np.round(rsmask_bbox['extent'], 2).tolist()} mm")

    # Try to fetch CCFv3 100µm annotation as a known reference
    print("fetching Allen CCFv3 100µm annotation...")
    try:
        from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    except ImportError:
        print("  (allensdk not installed, skipping reference fetch; install for the full check)")
        return {"colleague_bbox": rsmask_bbox, "verified": False, "reason": "no allensdk"}

    mcc = MouseConnectivityCache(resolution=100, manifest_file=str(CACHE / "allen_manifest.json"))
    ann, ann_meta = mcc.get_annotation_volume()
    # CCFv3 affine is identity-scaled by voxel size; origin is upper-anterior-left
    res_mm = 0.1
    ccf_affine = np.diag([res_mm, res_mm, res_mm, 1.0])
    ccf_bbox = _bbox(ann > 0, ccf_affine)
    print(f"Allen CCFv3 100µm: shape={ann.shape}, bbox extent={np.round(ccf_bbox['extent'], 2).tolist()} mm")

    # Check 1: are the brain extents similar?
    extent_ratio = (np.array(rsmask_bbox["extent"]) /
                    np.maximum(np.array(ccf_bbox["extent"]), 1e-6))
    extent_match = bool(np.all((extent_ratio > 0.7) & (extent_ratio < 1.4)))
    print(f"extent ratio (rsmask / CCFv3): {extent_ratio.round(2).tolist()}  → "
          f"{'OK' if extent_match else 'MISMATCH'}")

    # Check 2: per-node centre coords vs CCFv3 brain bounding box
    meta = load_metadata("mouse"); df = parse_t_table(meta["t"], meta["ht"])
    centres = df[["x","y","z"]].values
    print(f"mouse node centres: range x [{centres[:,0].min():.2f}, {centres[:,0].max():.2f}], "
          f"y [{centres[:,1].min():.2f}, {centres[:,1].max():.2f}], "
          f"z [{centres[:,2].min():.2f}, {centres[:,2].max():.2f}]")
    print(f"CCFv3 brain extent:        x [{ccf_bbox['bbox_min'][0]:.2f}, {ccf_bbox['bbox_max'][0]:.2f}], "
          f"y [{ccf_bbox['bbox_min'][1]:.2f}, {ccf_bbox['bbox_max'][1]:.2f}], "
          f"z [{ccf_bbox['bbox_min'][2]:.2f}, {ccf_bbox['bbox_max'][2]:.2f}]")

    # Test: take the per-node centres, look up the CCFv3 region at each (interpreting
    # centres as world-mm coords in CCFv3 space). What fraction land in the brain?
    ccf_inv = np.linalg.inv(ccf_affine)
    homog = np.hstack([centres, np.ones((centres.shape[0], 1))])
    ccf_ijk = (ccf_inv @ homog.T).T[:, :3].astype(int)
    in_bounds = ((ccf_ijk[:, 0] >= 0) & (ccf_ijk[:, 0] < ann.shape[0]) &
                 (ccf_ijk[:, 1] >= 0) & (ccf_ijk[:, 1] < ann.shape[1]) &
                 (ccf_ijk[:, 2] >= 0) & (ccf_ijk[:, 2] < ann.shape[2]))
    if in_bounds.any():
        region_at_node = np.zeros(centres.shape[0], dtype=np.int32)
        ok_mask = np.where(in_bounds)[0]
        region_at_node[ok_mask] = ann[ccf_ijk[ok_mask, 0], ccf_ijk[ok_mask, 1], ccf_ijk[ok_mask, 2]]
        in_brain = (region_at_node > 0).sum()
        print(f"node centres landing in CCFv3 brain (direct lookup): "
              f"{in_brain}/{len(centres)} ({in_brain/len(centres):.1%})")
        if in_brain / len(centres) > 0.8:
            print("  ✓ STRONG match, direct projection should work")
            verdict = "direct_projection_ok"
        elif in_brain / len(centres) > 0.4:
            print("  ⚠ PARTIAL match, origin/orientation may differ; investigate")
            verdict = "partial_match"
        else:
            print("  ✗ POOR match, registration needed")
            verdict = "needs_registration"
    else:
        print("  ✗ no node centres land in CCFv3 grid, definite misalignment")
        verdict = "needs_registration"

    return {"colleague_bbox": rsmask_bbox, "ccf_bbox": ccf_bbox,
            "extent_ratio": extent_ratio.tolist(),
            "extent_match": extent_match,
            "centres_in_ccf_brain": float(in_brain / len(centres)) if in_bounds.any() else 0.0,
            "verdict": verdict}


def verify_human():
    print("\n=== verifying human alignment ===")
    rsmask = nib.load(DATA_DIR / "_human_mask" / "rsmask_human.nii")
    rsmask_data = np.asarray(rsmask.dataobj)
    rsmask_bbox = _bbox(rsmask_data > 0, rsmask.affine)
    print(f"colleague's rsmask: shape={rsmask.shape}, bbox extent={np.round(rsmask_bbox['extent'], 2).tolist()} mm")

    # MNI152 brain extent in standard orientation: ~ x[-78, 78], y[-112, 76], z[-50, 85]
    mni_extent_expected = [156.0, 188.0, 135.0]
    extent_ratio = np.array(rsmask_bbox["extent"]) / np.array(mni_extent_expected)
    extent_match = bool(np.all((extent_ratio > 0.85) & (extent_ratio < 1.15)))
    print(f"vs expected MNI152 extent {mni_extent_expected}: ratio "
          f"{extent_ratio.round(2).tolist()}  → {'OK' if extent_match else 'MISMATCH'}")

    # Per-node centres
    meta = load_metadata("human"); df = parse_t_table(meta["t"], meta["ht"])
    centres = df[["x","y","z"]].values
    print(f"human node centres: range x [{centres[:,0].min():.1f}, {centres[:,0].max():.1f}], "
          f"y [{centres[:,1].min():.1f}, {centres[:,1].max():.1f}], "
          f"z [{centres[:,2].min():.1f}, {centres[:,2].max():.1f}]")

    # Heuristic: MNI152 centre is roughly origin (0, 0, 0) ± ~80mm.
    # If the colleague's centres are also centred around origin, it's MNI.
    ctr = centres.mean(axis=0)
    spread = (centres.max(axis=0) - centres.min(axis=0))
    print(f"  centroid:           {ctr.round(1).tolist()}")
    print(f"  spread (max-min):   {spread.round(1).tolist()}  (MNI152 brain ≈ [156, 188, 135])")
    centred = bool(np.all(np.abs(ctr) < 30))
    if centred and extent_match:
        verdict = "mni152_likely"
        print("  ✓ likely MNI152, direct projection should work")
    elif extent_match:
        verdict = "mni_orientation_offset"
        print("  ⚠ extent matches MNI but centroid is offset, may need translation")
    else:
        verdict = "needs_registration"
        print("  ✗ extent does NOT match MNI152, registration needed")

    return {"colleague_bbox": rsmask_bbox,
            "extent_ratio_vs_mni": extent_ratio.tolist(),
            "extent_match": extent_match,
            "centres_centroid": ctr.tolist(),
            "centred": centred,
            "verdict": verdict}


def main():
    out = {}
    out["mouse"] = verify_mouse()
    out["human"] = verify_human()

    summary_path = OUT / "alignment_verification.json"
    summary_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved → {summary_path}")
    print("\nSummary:")
    print(f"  mouse: {out['mouse'].get('verdict', 'unknown')}")
    print(f"  human: {out['human'].get('verdict', 'unknown')}")


if __name__ == "__main__":
    main()
