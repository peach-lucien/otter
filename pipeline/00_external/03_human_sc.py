"""Project the Domhof et al. 2022 group-averaged human structural connectome
onto the colleague's 2094-node parcellation.

Default behaviour: looks for the Domhof bundle at
  data_external/p6ebec-hbp-d000038_SC-FC_HCP_eNKI_pub/Schaefer2018_400Parcels_17Networks.zip
and uses:
  - Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz  (parcellation)
  - Averaged_SC_..._HCP_10M_count_MEAN.tsv                          (SC matrix, HCP)

Pipeline:
  1. Extract the Schaefer 2 mm parcellation NIfTI + the SC tsv from the zip.
  2. Resample the parcellation onto the colleague's rsmask grid (3 mm MNI152).
  3. For each of the 2094 human nodes, take the modal Schaefer parcel ID across
     its voxels (using the colleague's voxel_indices and rsmask affine).
  4. Build the (2094, 2094) SC matrix by indexing the (400, 400) Schaefer matrix.
     Nodes outside Schaefer cortex get all-zero rows/cols.

Output: data_external/human_sc.npy + human_sc_meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import DATA_DIR, load_metadata, parse_t_table       # noqa: E402

OUT  = ROOT / "data_external"; OUT.mkdir(parents=True, exist_ok=True)
DIAG = OUT / "_diagnostics"
MASK = DATA_DIR / "_human_mask" / "rsmask_human.nii"

DOMHOF_ROOT = OUT / "p6ebec-hbp-d000038_SC-FC_HCP_eNKI_pub"
DEFAULT_PARC_ZIP = DOMHOF_ROOT / "Schaefer2018_400Parcels_17Networks.zip"


def _extract_from_zip(zip_path: Path, name_contains: str, dest: Path) -> Path:
    """Extract the first member whose name contains the given substring."""
    with zipfile.ZipFile(zip_path) as z:
        match = next((n for n in z.namelist() if name_contains in n), None)
        if match is None:
            raise FileNotFoundError(f"{name_contains} not found in {zip_path}")
        # Preserve just the basename
        out = dest / Path(match).name
        if not out.exists():
            with z.open(match) as src, out.open("wb") as f:
                f.write(src.read())
        return out


def main(args):
    cache = OUT / "_domhof_extracted"
    cache.mkdir(parents=True, exist_ok=True)
    parc_zip = Path(args.parc_zip) if args.parc_zip else DEFAULT_PARC_ZIP
    if not parc_zip.exists():
        print(f"ERROR: Domhof zip not found at {parc_zip}")
        print("       download from EBRAINS or pass --parc-zip PATH")
        sys.exit(1)

    # 1. Extract the parcellation NIfTI + SC matrix --------------------------
    print(f"extracting parcellation + SC from {parc_zip.name} ...")
    parc_path = _extract_from_zip(parc_zip, "FSLMNI152_2mm.nii.gz", cache)
    sc_match = "HCP_10M_count_MEAN.tsv" if args.cohort == "HCP" else "eNKI_10M_count_MEAN.tsv"
    sc_path = _extract_from_zip(parc_zip, sc_match, cache)
    print(f"  parcellation : {parc_path.name}")
    print(f"  SC matrix    : {sc_path.name}")

    # 2. Load both -----------------------------------------------------------
    parc = nib.load(parc_path)
    parc_data = np.asarray(parc.dataobj).astype(np.int32)
    parc_affine = parc.affine
    n_regions = int(parc_data.max())
    print(f"  parcellation: shape={parc_data.shape}  n_regions={n_regions}")

    SC = pd.read_csv(sc_path, sep="\t", header=None).values
    print(f"  SC matrix shape: {SC.shape}")
    if SC.shape != (n_regions, n_regions):
        # Some Domhof tsvs have a header row/column with parcel labels, try skipping
        SC = pd.read_csv(sc_path, sep="\t").values
        print(f"  retried with header → shape: {SC.shape}")
        if SC.shape != (n_regions, n_regions):
            raise ValueError(f"SC shape {SC.shape} ≠ parcellation regions {n_regions}")

    # Domhof's count matrix is symmetric streamline counts.
    # Sanity: check symmetric, log-distribute, etc.
    asym = np.abs(SC - SC.T).max()
    print(f"  SC symmetry: max |M - M.T| = {asym:.3f}")
    if asym > 1e-3:
        print("  symmetrising via 0.5*(M + M.T)")
        SC = 0.5 * (SC + SC.T)

    # 3. Resample parcellation onto the colleague's rsmask grid -------------
    print("loading colleague's human mask + node table...")
    rsmask = nib.load(MASK)
    rsmask_data = np.asarray(rsmask.dataobj)
    rsmask_affine = rsmask.affine
    print(f"  rsmask shape: {rsmask_data.shape} (vs Schaefer {parc_data.shape})")

    if rsmask_data.shape != parc_data.shape:
        from scipy.ndimage import map_coordinates
        ijk = np.indices(rsmask_data.shape).reshape(3, -1).T
        world = (rsmask_affine @ np.vstack([ijk.T, np.ones(ijk.shape[0])])).T[:, :3]
        parc_ijk = (np.linalg.inv(parc_affine) @
                    np.vstack([world.T, np.ones(world.shape[0])])).T[:, :3]
        parc_at_rsmask = map_coordinates(parc_data.astype(np.float64), parc_ijk.T,
                                          order=0, mode="constant", cval=0).reshape(rsmask_data.shape)
        parc_at_rsmask = parc_at_rsmask.astype(np.int32)
        print(f"  resampled parcellation onto rsmask grid")
    else:
        parc_at_rsmask = parc_data

    # 4. Per-node Schaefer region (modal) -----------------------------------
    diagnostics = json.loads((DIAG / "mask_info.json").read_text())
    one_based = diagnostics["human_voxel_index_check"]["likely_one_based"]
    order = diagnostics["human_voxel_index_check"]["recommended_order"]
    flat_parc = parc_at_rsmask.ravel(order=order)

    meta = load_metadata("human"); df = parse_t_table(meta["t"], meta["ht"])
    n_nodes = len(df)
    node_region = np.zeros(n_nodes, dtype=np.int64)
    n_unmapped = 0
    for i, vox in enumerate(df["voxel_indices"]):
        idx = (np.asarray(vox).astype(np.int64) - (1 if one_based else 0))
        idx = idx[(idx >= 0) & (idx < flat_parc.size)]
        if len(idx) == 0:
            n_unmapped += 1; continue
        regs = flat_parc[idx]
        regs = regs[regs > 0]
        if len(regs) == 0:
            n_unmapped += 1; continue
        node_region[i] = Counter(regs.tolist()).most_common(1)[0][0]
    print(f"  {n_unmapped}/{n_nodes} nodes had no Schaefer cortex voxels "
          f"(expected for subcortical nodes)")

    # 5. Build (2094, 2094) SC matrix ----------------------------------------
    print("assembling per-node SC matrix...")
    SC_node = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for i in range(n_nodes):
        ri = node_region[i] - 1
        if ri < 0: continue
        for j in range(n_nodes):
            rj = node_region[j] - 1
            if rj < 0: continue
            SC_node[i, j] = SC[ri, rj]
    np.save(OUT / "human_sc.npy", SC_node)

    info = {
        "source":      "Domhof et al. 2022 Sci Data. Schaefer400 17Networks averaged SC",
        "cohort":      args.cohort,
        "metric":      "streamline_count",
        "n_nodes":     int(n_nodes),
        "n_regions":   int(n_regions),
        "n_unmapped":  int(n_unmapped),
        "frac_subcortical_unmapped": float(n_unmapped / n_nodes),
        "rsmask_shape": list(rsmask_data.shape),
        "node_region": node_region.tolist(),
    }
    (OUT / "human_sc_meta.json").write_text(json.dumps(info, indent=2, default=str))

    n_mapped = (node_region > 0).sum()
    print(f"\n  done: {n_mapped}/{n_nodes} nodes have SC ({n_mapped/n_nodes:.1%})")
    print(f"  saved → {OUT / 'human_sc.npy'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parc-zip", default=None,
                    help="path to Domhof Schaefer400 zip (default: data_external/p6ebec.../Schaefer2018_400Parcels_17Networks.zip)")
    ap.add_argument("--cohort", choices=["HCP", "eNKI"], default="HCP",
                    help="which Domhof cohort's averaged matrix to use (default HCP)")
    main(ap.parse_args())
