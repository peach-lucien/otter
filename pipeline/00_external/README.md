# External-data scripts

Scripts that download and project public neuroscience datasets onto the
colleague's specific 1864 (mouse) / 2094 (human) parcellation. Run these on a
machine with internet access and the colleague's `rsmask.nii` /
`rsmask_human.nii` files at the locations expected by `homer.data.DATA_DIR`.

All outputs land in `homer/data_external/`:

```
data_external/
├── README.md                    # auto-generated, lists what's there
├── _diagnostics/
│   └── mask_info.json           # output of 00_inspect_masks.py
├── mouse_sc.npy                 # (1864, 1864) float32 — Allen mouse SC
├── mouse_sc_meta.json
├── mouse_genes.npy              # (1864, n_genes) — Allen ISH expression
├── mouse_genes_meta.json
├── human_sc.npy                 # (2094, 2094) — group connectome
├── human_sc_meta.json
├── human_genes.npy              # (2094, n_genes) — abagen
├── human_genes_meta.json
└── orthologs.csv                # mouse_gene ↔ human_gene pairs
```

## Install dependencies

```bash
pip install -r scripts/external/requirements.txt
```

## Run order

1. **`00_inspect_masks.py`** — Examines `rsmask.nii` / `rsmask_human.nii` and
   reports their affine, dims, and likely coordinate system. **Run this first**
   so we know what we're working with before downloading datasets.

2. **`01_mouse_sc.py`** — Downloads the Allen Mouse Connectivity Atlas
   (Oh et al. 2014, Nature) summary-structure connectivity matrix (~290 regions)
   plus the CCFv3 annotation volume. For each of the 1864 mouse nodes, looks up
   its CCFv3 region (using the node's voxel indices and `rsmask.nii`'s affine),
   then assigns the corresponding row/column of the structure-level SC matrix.
   Output: `mouse_sc.npy`.

   Caveats: assumes `rsmask.nii` is in CCFv3 coordinates (or a linearly-related
   space). `00_inspect_masks.py` will flag if not — you may need a registration
   step with ANTs/FSL first.

3. **`02_mouse_genes.py`** — Downloads Allen Mouse Brain ISH atlas
   (Lein et al. 2007, Nature). Uses a curated set of ~4000 well-characterised
   genes (configurable). Per-node expression by averaging voxels.
   Output: `mouse_genes.npy`.

4. **`03_human_sc.py`** — Downloads a public group-averaged human structural
   connectome. Default source is the **Domhof et al. 2022 Scientific Data**
   release, which is hosted on EBRAINS with no credentials required (the HCP-
   based versions all need credentialed access). For each of the 2094 human
   nodes, takes the dMRI streamline count to/from each parcellation region
   that contains the node's voxels.
   Output: `human_sc.npy`.

   Alternative: if you have HCP credentials, the script has a `--source hcp1065`
   flag that uses Yeh's HCP1065 atlas via DSI Studio's published files.

5. **`04_human_genes.py`** — Uses `abagen` (Markello et al. 2021, eLife) to
   pull Allen Human Brain Atlas microarray data and average it within each of
   the 2094 voxel-defined regions. abagen handles all the donor-normalisation
   and probe-selection internally.
   Output: `human_genes.npy`.

6. **`05_orthologs.py`** — Aligns the mouse and human gene sets via NCBI
   Homologene. Outputs `orthologs.csv` with the mouse-gene ↔ human-gene pairs
   that exist in both `mouse_genes.npy` and `human_genes.npy`. Re-saves both
   matrices restricted to orthologs only.

## What to do if things break

The most likely failure mode: `00_inspect_masks.py` reports that `rsmask.nii` is
**not** aligned to Allen CCFv3 (or `rsmask_human.nii` is not in MNI152). In that
case you'll need to:

- Register the colleague's mask to the standard space (ANTs `antsRegistration`
  or FSL `flirt`)
- Apply the resulting transformation to the per-node centre coordinates
- Re-run the projection

Each script is idempotent — re-running just rebuilds the npy without
re-downloading raw data (which is cached by allensdk / abagen).
