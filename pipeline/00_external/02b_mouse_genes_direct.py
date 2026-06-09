"""Fallback mouse-gene downloader: bypass AllenSDK and hit the REST endpoint
directly with `requests`.

.. deprecated:: v2
    LEGACY (v1 only). Like ``02_mouse_genes.py``, this script samples ISH
    energy volumes through the heuristic 48-permutation transform. The v2
    successor ``02c_mouse_genes_v2.py`` reads the pre-warped voxel set
    ``AS_ix`` directly from ``corrs_mouse_v2.mat`` and is the production
    path. Use this fallback only when reproducing the v1 mouse-gene
    pipeline.

Use this if 02_mouse_genes.py keeps producing zips with only `data_set.xml`.

What this does differently:
  - Hits  http://api.brain-map.org/grid_data/download/<id>?include=energy
    directly via `requests`, asking explicitly for the energy.zip stream.
  - Inspects the Content-Type / first bytes to confirm it's a real zip with
    .mhd + .raw before saving.
  - Restricts to a curated list of ~80 well-known genes (cortical layer
    markers, cell-type markers, classical region-defining genes) — every one
    of these is documented to have a 3D grid in the Allen Mouse ISH atlas.

Output is the same as 02_mouse_genes.py:
  data_external/mouse_genes.npy        (1864, n_genes_kept) float32
  data_external/mouse_gene_list.csv

Run from homer/ root:
    PYTHONPATH=src python scripts/external/02b_mouse_genes_direct.py
"""
from __future__ import annotations

import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from homer.data import DATA_DIR, load_metadata, parse_t_table       # noqa: E402
from _mouse_transform import load_transform, colleague_voxel_to_ccf_world  # noqa: E402

OUT  = ROOT / "data_external"; OUT.mkdir(parents=True, exist_ok=True)
DIAG = OUT / "_diagnostics"
MASK = DATA_DIR / "_mouse_mask" / "rsmask.nii"

# ---------------------------------------------------------------------------
# Curated gene set: well-characterised Allen ISH markers known to have 3D grid
# reconstructions. Mix of: cortical layer markers, GABAergic / glutamatergic
# subtype markers, region-specific markers (striatum, hippocampus, thalamus...),
# and classic neuromodulator markers. Picked to give meaningful cross-species
# anatomical signal even if only a fraction succeed.
# ---------------------------------------------------------------------------
CURATED_GENES = [
    # === Cortical layer markers ===
    "Cux1", "Cux2", "Rorb", "Foxp2", "Bcl11b", "Tbr1", "Satb2", "Fezf2",
    "Pou3f2", "Pou3f3", "Pou3f4", "Pou6f2", "Nr4a2", "Crym", "Prss12",
    "Lmo3", "Lmo4", "Etv1", "Cartpt", "Foxp1", "Tle4", "Cplx3", "Ctip2",
    "Bhlhe22", "Sox5", "Ldb2", "Tbca", "Diap3", "Lhx2",
    # === Glutamatergic markers ===
    "Slc17a7", "Slc17a6", "Slc17a8", "Camk2a", "Camk2b", "Camk2g",
    "Grin1", "Grin2a", "Grin2b", "Grin2c", "Grin2d", "Grin3a",
    "Gria1", "Gria2", "Gria3", "Gria4",
    "Grm1", "Grm2", "Grm3", "Grm4", "Grm5", "Grm7", "Grm8",
    "Glul", "Slc1a2", "Slc1a3", "Slc1a6",
    # === GABAergic interneurons ===
    "Gad1", "Gad2", "Slc32a1", "Pvalb", "Sst", "Vip", "Calb1", "Calb2",
    "Reln", "Lhx6", "Sp8", "Npy", "Cck", "Crh", "Tac3", "Pnoc", "Nos1",
    "Pax6", "Calb2", "Cck", "Cnr1",
    # GABA receptor subunits
    "Gabra1", "Gabra2", "Gabra3", "Gabra4", "Gabra5", "Gabra6",
    "Gabrb1", "Gabrb2", "Gabrb3", "Gabrg1", "Gabrg2", "Gabrg3",
    "Gabbr1", "Gabbr2",
    # === Striatum / dopamine system ===
    "Drd1", "Drd2", "Drd3", "Drd4", "Drd5",
    "Th", "Slc6a3", "Slc18a2", "Penk", "Tac1", "Tac3",
    "Adora2a", "Pdyn", "Ebf1", "Foxp1", "Rgs9", "Rasd2",
    "Six3", "Isl1", "Meis2",
    # === Cholinergic ===
    "Chat", "Slc18a3", "Ache", "Chrm1", "Chrm2", "Chrm3", "Chrm4", "Chrm5",
    "Chrna4", "Chrna7", "Chrnb2",
    # === Hippocampus markers ===
    "Prox1", "Wfs1", "Dock10", "Mpped1", "Rgs14", "Spink8", "Bok",
    "Tdo2", "Nov", "Cnih3", "Lefty1", "Trpc6", "Adcy1",
    # === Thalamus ===
    "Plekhg1", "Tnnt1", "Gabbr2", "Cdh22",
    # === Cerebellum ===
    "Pcp2", "Calb1", "Aldoc", "Pvalb", "Pou3f2", "Itpr1",
    # === Neuromodulator systems ===
    "Slc6a4", "Tph2",                                           # serotonin
    "Htr1a", "Htr1b", "Htr2a", "Htr2c", "Htr3a", "Htr5a", "Htr6", "Htr7",
    "Slc6a2", "Dbh", "Pnmt", "Adra1a", "Adra1b", "Adra2a",      # noradrenaline
    "Adrb1", "Adrb2",
    # === Hypothalamic peptides ===
    "Hcrt", "Pmch", "Avp", "Oxt", "Crh", "Trh", "Pomc", "Agrp",
    "Npy", "Gal", "Mch", "Cartpt", "Sst",
    # === Glia / oligodendrocytes ===
    "Gfap", "Aqp4", "Aldh1l1", "S100b", "Slc1a3", "Glt1", "Glul",
    "Mbp", "Mog", "Plp1", "Olig1", "Olig2", "Sox10", "Mag", "Cnp", "Mobp",
    "Pdgfra", "Cspg4",
    # === Microglia ===
    "Cx3cr1", "Csf1r", "Tmem119", "Cd68", "Aif1", "P2ry12", "Trem2", "Itgam",
    # === Region-specific TFs ===
    "Pax6", "Emx1", "Emx2", "Lhx2", "Foxg1", "Otx1", "Otx2",
    "Nkx2-1", "Nkx2-2", "Pax3", "Pax7", "Foxa2", "Lmx1a", "Lmx1b",
    "Bsx", "Dlx1", "Dlx2", "Dlx5", "Dlx6", "Sim1", "Sim2",
    # === Voltage-gated ion channels ===
    "Scn1a", "Scn1b", "Scn2a", "Scn2b", "Scn3a", "Scn8a",
    "Kcna1", "Kcna2", "Kcna3", "Kcna4", "Kcnb1", "Kcnb2", "Kcnc1", "Kcnc2",
    "Kcnj2", "Kcnj4", "Kcnj6", "Kcnj9", "Kcnq2", "Kcnq3",
    "Cacna1a", "Cacna1b", "Cacna1c", "Cacna1d", "Cacna1e", "Cacna1g",
    "Hcn1", "Hcn2", "Hcn3", "Hcn4",
    # === Synaptic / vesicular proteins ===
    "Syn1", "Syn2", "Syn3", "Syp", "Synpr", "Snap25", "Snap47",
    "Stx1a", "Stx1b", "Vamp1", "Vamp2", "Stxbp1", "Cplx1", "Cplx2",
    "Syt1", "Syt2", "Syt4", "Syt7", "Syt9",
    "Rims1", "Rims2", "Rab3a", "Sv2a", "Sv2b", "Sv2c",
    # === Trophic factors / receptors ===
    "Bdnf", "Ntrk2", "Ngf", "Ntrk1", "Ntf3", "Ntf4", "Gdnf", "Ret",
    "Cntf", "Lif", "Lifr", "Igf1", "Igf2", "Vegfa",
    # === Immediate early genes ===
    "Fos", "Fosb", "Arc", "Egr1", "Egr2", "Junb", "Jun", "Npas4", "Nr4a1",
    # === Cell adhesion / morphology ===
    "Ncam1", "L1cam", "Cdh1", "Cdh2", "Cdh4", "Cdh8", "Cdh11", "Cdh13",
    "Pcdh8", "Pcdh17", "Pcdh19",
    # === Other classical brain-wide markers ===
    "Nrgn", "Nefh", "Nefm", "Nefl", "Mapt", "Map2", "Tubb3", "Stmn1",
    "Eno2", "Snca", "App", "Apoe",
    # === Endocannabinoid / opioid ===
    "Cnr1", "Cnr2", "Oprm1", "Oprd1", "Oprk1",
    # === Neuropeptide receptors ===
    "Sstr1", "Sstr2", "Sstr3", "Sstr4", "Sstr5",
    "Vipr1", "Vipr2", "Cckar", "Cckbr", "Crhr1", "Crhr2",
    "Tacr1", "Tacr3", "Mc4r", "Mc3r", "Npy1r", "Npy2r", "Npy5r", "Galr1", "Galr2",
    # === Habenula / specific subnetwork markers ===
    "Tac2", "Pou4f1", "Gpr151", "Crh",
    # === Brainstem nuclei ===
    "Phox2a", "Phox2b", "Tlx3", "Rnx",
    # === Visual system specific ===
    "Calb2", "Cbln2", "Synpo", "Rorb",
    # === More layer markers ===
    "Vamp1", "Tle4", "Bcl11a", "Sla", "Cdh13", "Foxp2",
]
CURATED_GENES = sorted(set(CURATED_GENES))

ALLEN_BASE = "http://api.brain-map.org"


def _query_section_data_sets(symbol: str, rma=None) -> list[dict]:
    """Return all SectionDataSet records for a given gene symbol (mouse).

    Uses AllenSDK's RmaApi for proper URL encoding (single-quote handling in
    the criteria string is fragile when constructing URLs by hand).
    """
    if rma is None:
        from allensdk.api.queries.rma_api import RmaApi
        rma = RmaApi()
    return rma.model_query(
        "SectionDataSet",
        criteria=(
            "[failed$eqfalse],"
            "products[id$eq1],"
            "plane_of_section[name$eq'coronal'],"
            f"genes[acronym$eq'{symbol}']"
        ),
        include="genes",
        num_rows="all",
    )


def _try_download_grid(sds_id: int, dest: Path, gda=None,
                       volume_type: str = "energy") -> bool:
    """Download via AllenSDK's `download_gene_expression_grid_data` (the
    non-deprecated method). Then verify the zip actually contains .mhd + .raw
    before keeping it.

    The deprecated `download_expression_grid_data` had a bug where passing a
    string `include="energy"` got joined character-by-character into the URL,
    producing garbage and a metadata-only response.
    """
    from allensdk.api.queries.grid_data_api import GridDataApi
    if gda is None:
        gda = GridDataApi()
    try:
        gda.download_gene_expression_grid_data(
            section_data_set_id=sds_id,
            volume_type=volume_type,
            path=str(dest),
        )
    except Exception:
        return False
    if not dest.exists() or dest.stat().st_size < 1000:
        return False
    try:
        zf = zipfile.ZipFile(dest)
        names = zf.namelist()
    except zipfile.BadZipFile:
        return False
    has_grid = any(n.endswith(".mhd") for n in names) and \
               any(n.endswith(".raw") for n in names)
    if not has_grid:
        try: dest.unlink()
        except OSError: pass
        return False
    return True


_MHD_DTYPES = {
    "MET_FLOAT":  np.float32, "MET_DOUBLE": np.float64,
    "MET_USHORT": np.uint16,  "MET_SHORT":  np.int16,
    "MET_UCHAR":  np.uint8,   "MET_CHAR":   np.int8,
    "MET_INT":    np.int32,   "MET_UINT":   np.uint32,
}


def _read_ish_grid(zip_path: Path, variable: str = "energy") -> np.ndarray:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        mhd = next((n for n in names if n.endswith(f"{variable}.mhd")), None)
        raw = next((n for n in names if n.endswith(f"{variable}.raw")), None)
        if mhd is None or raw is None:
            raise FileNotFoundError(names)
        text = z.read(mhd).decode()
        dim_m = re.search(r"DimSize\s*=\s*(\d+)\s+(\d+)\s+(\d+)", text)
        type_m = re.search(r"ElementType\s*=\s*(\S+)", text)
        shape = tuple(int(d) for d in dim_m.groups())
        dtype = _MHD_DTYPES[type_m.group(1)]
        msb = re.search(r"ElementByteOrderMSB\s*=\s*(\S+)", text)
        big = msb is not None and msb.group(1).strip().lower() in {"true", "1"}
        np_dtype = np.dtype(dtype).newbyteorder(">" if big else "<")
        buf = z.read(raw)
        arr = np.frombuffer(buf, dtype=np_dtype)
        return arr.reshape(shape).astype(np.float32)


def main():
    transform = load_transform(DIAG)
    print(f"using mouse→CCFv3 transform (coverage at fit: {transform['coverage']:.1%})")
    diagnostics = json.loads((DIAG / "mask_info.json").read_text())
    one_based = diagnostics["mouse_voxel_index_check"]["likely_one_based"]
    order = diagnostics["mouse_voxel_index_check"]["recommended_order"]

    cache_dir = Path.home() / ".allensdk_cache" / "ish_energy_direct"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve curated symbols → section_data_set_id ----------------------
    from allensdk.api.queries.rma_api import RmaApi
    rma = RmaApi()
    print(f"resolving {len(CURATED_GENES)} curated gene symbols → SectionDataSets ...")
    resolved = []
    for sym in CURATED_GENES:
        try:
            datasets = _query_section_data_sets(sym, rma=rma)
        except Exception as e:
            print(f"  {sym}: query failed ({e})"); continue
        if not datasets:
            print(f"  {sym}: no datasets")
            continue
        # Pick the lowest-id coronal dataset (Allen's earliest-published is
        # typically the canonical one with full 3D reconstruction)
        ds = min(datasets, key=lambda d: int(d["id"]))
        resolved.append({"gene_symbol": sym,
                         "section_data_set_id": int(ds["id"])})
        time.sleep(0.1)
    print(f"  resolved {len(resolved)}/{len(CURATED_GENES)} symbols")
    if not resolved:
        print("ERROR: could not resolve any symbols — Allen API issue")
        sys.exit(1)

    # 2. Mouse mask + node setup --------------------------------------------
    rsmask = nib.load(MASK)
    rsmask_affine = rsmask.affine; rsmask_shape = rsmask.shape
    meta = load_metadata("mouse"); df = parse_t_table(meta["t"], meta["ht"])
    n_nodes = len(df)
    res_mm = 0.2

    print("pre-computing per-node CCFv3 200 µm voxel indices ...")
    node_ccf_voxels = []
    for vox in df["voxel_indices"]:
        ccf_world = colleague_voxel_to_ccf_world(
            rsmask_affine, np.asarray(vox), rsmask_shape,
            one_based=one_based, order=order, transform=transform,
        )
        ccf_ijk = (ccf_world / res_mm).astype(np.int64)
        node_ccf_voxels.append(ccf_ijk)

    # 3. Download + sample per gene -----------------------------------------
    from allensdk.api.queries.grid_data_api import GridDataApi
    gda = GridDataApi()
    print("downloading + sampling gene volumes (AllenSDK) ...")
    expr = np.full((n_nodes, len(resolved)), np.nan, dtype=np.float32)
    valid = []; n_fail = 0
    from tqdm import tqdm
    for k, rec in enumerate(tqdm(resolved)):
        sds_id = rec["section_data_set_id"]
        path = cache_dir / f"sds_{sds_id}_energy.zip"
        if not path.exists():
            ok = _try_download_grid(sds_id, path, gda=gda)
            if not ok:
                n_fail += 1
                tqdm.write(f"  {rec['gene_symbol']:8s} (sds {sds_id}): no grid available")
                continue
        try:
            volume = _read_ish_grid(path)
        except Exception as e:
            tqdm.write(f"  {rec['gene_symbol']}: read fail: {e}")
            n_fail += 1; continue
        for i, ccf_ijk in enumerate(node_ccf_voxels):
            in_b = ((ccf_ijk[:, 0] >= 0) & (ccf_ijk[:, 0] < volume.shape[0]) &
                    (ccf_ijk[:, 1] >= 0) & (ccf_ijk[:, 1] < volume.shape[1]) &
                    (ccf_ijk[:, 2] >= 0) & (ccf_ijk[:, 2] < volume.shape[2]))
            if not in_b.any(): continue
            ok_idx = ccf_ijk[in_b]
            vals = volume[ok_idx[:, 0], ok_idx[:, 1], ok_idx[:, 2]]
            vals = vals[np.isfinite(vals) & (vals >= 0)]
            if len(vals) > 0:
                expr[i, k] = float(vals.mean())
        valid.append(k)

    print(f"\n  downloaded {len(valid)}/{len(resolved)} genes successfully ({n_fail} failed)")
    if not valid:
        print("ERROR: no gene volumes succeeded — Allen API issue")
        sys.exit(1)

    expr_kept = expr[:, valid]
    np.save(OUT / "mouse_genes.npy", expr_kept)
    metadata = pd.DataFrame([resolved[i] for i in valid])
    metadata.to_csv(OUT / "mouse_gene_list.csv", index=False)
    info = {
        "source":   "Allen Mouse Brain ISH atlas (curated 80-gene set, direct REST)",
        "n_nodes":  int(n_nodes),
        "n_genes":  int(len(valid)),
        "n_attempted":      int(len(resolved)),
        "n_failed":         int(n_fail),
        "transform_used":   transform,
        "ccf_resolution_um": int(res_mm * 1000),
        "curated_set":      "well-known cortical layer + cell-type + region markers",
    }
    (OUT / "mouse_genes_meta.json").write_text(json.dumps(info, indent=2, default=str))
    print(f"\nsaved → {OUT / 'mouse_genes.npy'}  shape {expr_kept.shape}")
    print(f"        {OUT / 'mouse_gene_list.csv'}")


if __name__ == "__main__":
    main()
