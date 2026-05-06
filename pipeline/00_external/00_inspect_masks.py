"""Inspect the colleague's mask files to determine coordinate systems.

This is the first thing to run. The downstream alignment of public datasets
(Allen mouse atlas, MNI152 human atlas) depends on knowing:
  - What space is rsmask.nii in?           (CCFv3? RBM? other?)
  - What space is rsmask_human.nii in?     (MNI152? Talairach? other?)
  - What's the voxel resolution?
  - Do the per-node `voxel_indices` from the t-table actually index into these
    masks correctly (i.e., 1-based MATLAB indexing or 0-based)?

Output: data_external/_diagnostics/mask_info.json — a structured report that
the other scripts can read to choose the right alignment strategy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from homer.data import DATA_DIR, load_cached                       # noqa: E402

OUT = ROOT / "data_external" / "_diagnostics"
OUT.mkdir(parents=True, exist_ok=True)


def _classify_human_space(affine: np.ndarray, shape: tuple) -> str:
    """Heuristic to identify common human atlases by their affine + shape."""
    # MNI152 typical resolutions: 1mm (182, 218, 182), 2mm (91, 109, 91)
    if shape == (182, 218, 182): return "MNI152 1mm (likely)"
    if shape == (91, 109, 91):   return "MNI152 2mm (likely)"
    if shape == (193, 229, 193): return "ICBM 1mm (likely)"
    # voxel size from affine
    vox = np.abs(np.diag(affine)[:3])
    return f"unknown human space (vox≈{vox.round(2).tolist()}, shape={shape})"


def _classify_mouse_space(affine: np.ndarray, shape: tuple) -> str:
    """Heuristic to identify common mouse atlases."""
    vox = np.abs(np.diag(affine)[:3])
    # Allen CCFv3 standard resolutions:
    #   10  µm: (1320, 800, 1140)
    #   25  µm: (528, 320, 456)
    #   50  µm: (264, 160, 228)
    #  100  µm: (132, 80, 114)
    #  200  µm: (66, 40, 57)
    if shape == (528, 320, 456):  return "Allen CCFv3 25µm"
    if shape == (264, 160, 228):  return "Allen CCFv3 50µm"
    if shape == (132, 80, 114):   return "Allen CCFv3 100µm"
    # SIGMA / Dorr atlases have different shapes; likely the colleague
    # rescaled to mm
    return f"unknown mouse space (vox≈{vox.round(3).tolist()} mm, shape={shape})"


def inspect_mask(path: Path, species: str) -> dict:
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    affine = img.affine
    shape = img.shape
    print(f"\n=== {species}: {path.name} ===")
    print(f"  shape:    {shape}")
    print(f"  dtype:    {data.dtype}")
    print(f"  voxel mm: {np.abs(np.diag(affine)[:3]).round(3).tolist()}")
    print(f"  affine:")
    for row in affine:
        print(f"    [{row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f} {row[3]:10.4f}]")

    if species == "mouse":
        space = _classify_mouse_space(affine, shape)
    else:
        space = _classify_human_space(affine, shape)
    print(f"  best-guess: {space}")

    # Brain mask coverage
    if data.dtype != bool and data.max() <= 1.5:
        n_in_mask = int((data > 0).sum())
        print(f"  voxels in mask: {n_in_mask}")
    else:
        n_in_mask = -1
    return {
        "species": species, "path": str(path),
        "shape": list(shape), "voxel_mm": np.abs(np.diag(affine)[:3]).tolist(),
        "affine": affine.tolist(),
        "best_guess_space": space,
        "n_voxels_in_mask": n_in_mask,
    }


def check_voxel_indices(species: str, mask_data: np.ndarray) -> dict:
    """Verify that the t-table's voxel_indices actually point into the mask.

    The colleague stored voxel indices as 1-based linear MATLAB indices into
    the 3D mask volume. We need to confirm:
    - They're 1-based (so we subtract 1 to use as 0-based numpy indices)
    - They fit in the mask volume
    - When reshaped, they pick out non-zero mask voxels
    """
    A, _ = load_cached(species, cache_dir=ROOT / "outputs" / "anndata")
    if "voxel_indices" not in A.uns:
        # voxel_indices were stripped before cache; re-extract from raw t-table
        from homer.data import load_metadata, parse_t_table
        meta = load_metadata(species)
        df = parse_t_table(meta["t"], meta["ht"])
        all_indices = np.concatenate([np.asarray(v) for v in df["voxel_indices"]])
    else:
        all_indices = np.concatenate([np.asarray(v) for v in A.uns["voxel_indices"]])

    n_total = mask_data.size
    n_idx = len(all_indices)
    out = {"n_indices_total": int(n_idx)}

    # Range checks (1-based vs 0-based)
    out["min_index"] = int(all_indices.min())
    out["max_index"] = int(all_indices.max())
    out["mask_voxel_count"] = int(n_total)
    out["likely_one_based"] = bool(all_indices.min() >= 1 and all_indices.max() <= n_total)
    out["likely_zero_based"] = bool(all_indices.min() >= 0 and all_indices.max() < n_total)

    # If 1-based, decrement; otherwise keep as-is
    use = all_indices - (1 if out["likely_one_based"] else 0)
    use = use[(use >= 0) & (use < n_total)]
    flat = mask_data.ravel(order="F")     # MATLAB column-major
    in_mask_F = (flat[use] > 0).mean() if len(use) else 0.0
    flat_C = mask_data.ravel(order="C")
    in_mask_C = (flat_C[use] > 0).mean() if len(use) else 0.0
    out["frac_in_mask_F_order"] = float(in_mask_F)
    out["frac_in_mask_C_order"] = float(in_mask_C)
    out["recommended_order"] = "F" if in_mask_F > in_mask_C else "C"
    print(f"  voxel_indices range: [{out['min_index']}, {out['max_index']}]")
    print(f"  1-based: {out['likely_one_based']}, 0-based: {out['likely_zero_based']}")
    print(f"  fraction inside mask (Fortran order): {in_mask_F:.3f}")
    print(f"  fraction inside mask (C order):       {in_mask_C:.3f}")
    print(f"  recommended unravel order: {out['recommended_order']}")
    return out


def main():
    info = {}
    mouse_mask_path = DATA_DIR / "_mouse_mask" / "rsmask.nii"
    human_mask_path = DATA_DIR / "_human_mask" / "rsmask_human.nii"

    if mouse_mask_path.exists():
        info["mouse_mask"] = inspect_mask(mouse_mask_path, "mouse")
        try:
            mask_data = np.asarray(nib.load(mouse_mask_path).dataobj)
            info["mouse_voxel_index_check"] = check_voxel_indices("mouse", mask_data)
        except Exception as e:
            print(f"voxel-index check failed: {e}")
    else:
        print(f"missing: {mouse_mask_path}")

    if human_mask_path.exists():
        info["human_mask"] = inspect_mask(human_mask_path, "human")
        try:
            mask_data = np.asarray(nib.load(human_mask_path).dataobj)
            info["human_voxel_index_check"] = check_voxel_indices("human", mask_data)
        except Exception as e:
            print(f"voxel-index check failed: {e}")
    else:
        print(f"missing: {human_mask_path}")

    out = OUT / "mask_info.json"
    out.write_text(json.dumps(info, indent=2, default=float))
    print(f"\nsaved → {out}")
    print("\nNext steps:")
    if "mouse_mask" in info and "Allen CCFv3" in info["mouse_mask"]["best_guess_space"]:
        print("  ✓ mouse mask is in Allen CCFv3 — script 01 should work directly")
    elif "mouse_mask" in info:
        print(f"  ⚠ mouse mask is NOT identifiably CCFv3 ({info['mouse_mask']['best_guess_space']})")
        print("    → may need ANTs/FSL registration to CCFv3 before running 01")
    if "human_mask" in info and "MNI152" in info["human_mask"]["best_guess_space"]:
        print("  ✓ human mask is in MNI152 — scripts 03/04 should work directly")
    elif "human_mask" in info:
        print(f"  ⚠ human mask is NOT identifiably MNI152 ({info['human_mask']['best_guess_space']})")
        print("    → may need registration to MNI152 before running 03/04")


if __name__ == "__main__":
    main()
