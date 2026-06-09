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

2. **`01b_mouse_sc_v2.py`** — Downloads the Allen Mouse Connectivity Atlas
   (Oh et al. 2014, Nature) summary-structure connectivity matrix (~290 regions).
   For each of the 1864 mouse nodes it reads the pre-warped CCFv3 voxel centre
   `ns_center_ix` directly from `corrs_mouse_v2.mat` (Paul's nonlinear
   DSURQE→CCFv3 elastix warp), looks up the CCFv3 region there, and assigns the
   corresponding row/column of the structure-level SC matrix.
   Output: `mouse_sc.npy`.

   No registration step is needed under v2 — the mouse→CCFv3 warp is applied
   upstream by Paul and ships inside the .mat file. (The legacy v1 heuristic
   transform path was removed; it lives on the `archive/v1-pipeline` branch.)

3. **`02c_mouse_genes_v2.py`** — Downloads Allen Mouse Brain ISH atlas
   (Lein et al. 2007, Nature) energy volumes and samples them at 200 µm over
   each node's pre-warped CCFv3 voxel set (`AS_ix` from `corrs_mouse_v2.mat`).
   Reads its gene list from the immutable `mouse_gene_list_master.csv` and writes
   the NaN-pruned list (aligned 1:1 with the matrix) to `mouse_gene_list.csv`.
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

On the **mouse** side, the v2 path requires no mask registration — voxel indices
are pre-warped into CCFv3 (NS) and DSURQE (SS) space inside `corrs_mouse_v2.mat`.
If `01b`/`02c` warn that the loaded schema is not `v2`, the resolver fell back to
the legacy `corrs_mouse.mat`; point `homer.data.DATA_DIR` at the
`updated_connectom_0906_26/` package.

On the **human** side, the most likely failure mode is `00_inspect_masks.py`
reporting that `rsmask_human.nii` is not in MNI152 — register it to the standard
space (ANTs `antsRegistration` / FSL `flirt`) and re-run.

Each script is idempotent — re-running just rebuilds the npy without
re-downloading raw data (which is cached by allensdk / abagen).
