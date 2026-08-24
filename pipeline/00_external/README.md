# External-data preparation

These scripts prepare public neuroscience datasets for OTTER's 1,864-parcel mouse and
2,094-parcel human representations. They require the raw data tier and network access for sources
that are fetched from their original provider.

```bash
python scripts/fetch_data.py --tier raw
pip install -r pipeline/00_external/requirements.txt
```

Run the scripts from the repository root.

## Required order

| Script | Output or purpose |
|---|---|
| `00_inspect_masks.py` | Inspect the supplied mouse and human masks and write alignment diagnostics |
| `00b_verify_alignment.py` | Verify mouse CCFv3 and human MNI152 alignment against reference templates |
| `01_mouse_sc.py` | Project Allen Mouse Connectivity Atlas data to mouse parcels |
| `02_mouse_genes.py` | Sample Allen Mouse Brain Atlas ISH expression over mouse parcels |
| `03_human_sc.py` | Project the Domhof et al. group structural connectome to human parcels |
| `04_human_genes.py` | Process Allen Human Brain Atlas expression with `abagen` |
| `05_orthologs.py` | Align the mouse and human expression matrices through gene orthology |

The preparation scripts write matrices and metadata beneath `data_external/`, including:

```text
mouse_sc.npy
mouse_sc_meta.json
mouse_genes.npy
mouse_gene_list.csv
human_sc.npy
human_sc_meta.json
human_genes.npy
human_gene_list.csv
orthologs.csv
orthologs_meta.json
```

`00c_align_mouse_to_ccf.py` is available when the alignment check indicates that a mouse-space
transform must be regenerated. `06_knox_sc.py` prepares the optional Knox structural-connectivity
comparison input; neither script is required when the corresponding prepared files are present.

The scripts preserve a metadata sidecar for each derived matrix. Dataset citations,
redistribution constraints and archive contents are documented in [`DATA.md`](../../DATA.md).
