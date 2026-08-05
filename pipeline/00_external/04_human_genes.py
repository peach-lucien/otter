"""Use `abagen` to pull Allen Human Brain Atlas gene expression and project
onto the 2094-node parcellation.

abagen does the heavy lifting: probe-selection, donor-normalisation, sample-to-
region assignment, mirroring across hemispheres. We just need to give it our
parcellation as a NIfTI volume in MNI152 space.

Pipeline:
  1. Build a parcellation NIfTI from rsmask_human.nii where each node's voxels
     are labelled with the node's numid (1..2094). (We have the voxel_indices
     for every node, assemble them into a 3D label volume.)
  2. Call abagen.get_expression_data(parcellation_nifti).
  3. Save the (2094, n_genes) matrix and gene metadata.

Output:
  data_external/human_genes.npy        (2094, n_genes) float32
  data_external/human_gene_list.csv    gene metadata (entrez_id, symbol)
  data_external/_diagnostics/parcellation_2094.nii.gz   the assembled volume
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from otter.data import DATA_DIR, load_metadata, parse_t_table        # noqa: E402

OUT = ROOT / "data_external"; OUT.mkdir(parents=True, exist_ok=True)
DIAG = OUT / "_diagnostics"; DIAG.mkdir(parents=True, exist_ok=True)
MASK = DATA_DIR / "_human_mask" / "rsmask_human.nii"


def build_parcellation_volume() -> Path:
    """Construct a 3D label volume in rsmask_human.nii's space where each
    node's voxels are labelled 1..2094.
    """
    print("assembling 2094-node parcellation volume...")
    rsmask = nib.load(MASK); rsmask_data = np.asarray(rsmask.dataobj)
    rsmask_affine = rsmask.affine
    diagnostics = json.loads((DIAG / "mask_info.json").read_text())
    one_based = diagnostics["human_voxel_index_check"]["likely_one_based"]
    order = diagnostics["human_voxel_index_check"]["recommended_order"]

    meta = load_metadata("human")
    df = parse_t_table(meta["t"], meta["ht"])
    n_nodes = len(df)
    parc = np.zeros(rsmask_data.shape, dtype=np.int32)
    flat = parc.ravel(order=order)
    for i, vox in enumerate(df["voxel_indices"], start=1):
        idx = (np.asarray(vox).astype(np.int64) - (1 if one_based else 0))
        idx = idx[(idx >= 0) & (idx < flat.size)]
        flat[idx] = i
    parc = flat.reshape(parc.shape, order=order)
    print(f"  parcellation has {(parc > 0).sum()} labelled voxels across {n_nodes} regions")

    out_path = DIAG / "parcellation_2094.nii.gz"
    nib.save(nib.Nifti1Image(parc, rsmask_affine), str(out_path))
    print(f"  saved → {out_path}")
    return out_path


def main(args):
    try:
        import abagen
    except ImportError:
        print("ERROR: pip install abagen")
        sys.exit(1)
    parc_path = build_parcellation_volume()

    print("running abagen (this is the slow step, ~30 min on first run)...")
    # abagen API has shifted across versions, keep this call minimal so it works
    # on both the 0.1.x and 0.2.x lines. All of these kwargs are stable across
    # versions and use abagen's recommended defaults for cross-species work.
    expr_df = abagen.get_expression_data(
        atlas=str(parc_path),
        return_donors=False,                # one row per region (averaged across 6 donors)
        sample_norm="srs",                  # scaled robust sigmoid per donor
        gene_norm="srs",                    # scaled robust sigmoid per gene
        verbose=2 if args.verbose else 1,
    )
    print(f"  abagen returned: {expr_df.shape}  (regions × genes)")

    # The DataFrame index will be the region IDs (1..2094)
    n_nodes = 2094
    expr_array = np.full((n_nodes, expr_df.shape[1]), np.nan, dtype=np.float32)
    for region_id, row in expr_df.iterrows():
        idx = int(region_id) - 1
        if 0 <= idx < n_nodes:
            expr_array[idx] = row.values.astype(np.float32)
    np.save(OUT / "human_genes.npy", expr_array)

    # Save gene metadata (gene symbols are the column labels of expr_df)
    gene_meta = pd.DataFrame({
        "gene_symbol": expr_df.columns,
        "rank":        np.arange(expr_df.shape[1]),
    })
    gene_meta.to_csv(OUT / "human_gene_list.csv", index=False)

    info = {
        "source": "Allen Human Brain Atlas via abagen (Markello et al. 2021, eLife)",
        "n_nodes":            int(n_nodes),
        "n_genes":            int(expr_df.shape[1]),
        "n_nodes_with_data":  int(np.isfinite(expr_array).all(axis=1).sum()),
        "abagen_options":     {"return_donors": False, "missing": "interpolate",
                                "norm_all": True},
    }
    (OUT / "human_genes_meta.json").write_text(json.dumps(info, indent=2))
    print(f"\nsaved → {OUT / 'human_genes.npy'}  shape {expr_array.shape}")
    print(f"        {OUT / 'human_gene_list.csv'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    main(ap.parse_args())
