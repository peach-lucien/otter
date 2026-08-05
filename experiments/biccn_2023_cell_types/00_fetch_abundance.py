"""Build the mouse + human region x cell-subclass abundance tables that
`04_abundance_composition.py` consumes, from the Allen Brain Cell (ABC) atlas.

Output CSVs (one row per region):
    region, x, y, z, <subclass_1>, <subclass_2>, ...
x,y,z = region centroids IN OTTER'S COORDINATE FRAME (mouse: Allen CCF, matching
M.var; human: MNI mm, matching H.var). Subclass columns = FRACTION of cells of each
subclass in the region (rows ~sum to 1). Only subclasses whose column names match
across species are used by 04, so keep the label level comparable.

MOUSE  : MERFISH-C57BL6J-638850 (Zhuang/Yao 2023 spatial) — has per-cell CCF coords.
         join: cluster-annotation (subclass)  x  CCF coords  x  parcellation region.
HUMAN  : WHB-10Xv3 (Siletti 2023, via the ABC atlas — no CELLxGENE dataset id needed).
         cells carry a region-of-interest but NO MNI coordinate, so supply an
         ROI->MNI centroid lookup with --human-roi-mni (roi_name,x,y,z).

Install + run (multi-GB download; needs disk + time, so run locally not in a sandbox):
    pip install "abc_atlas_access @ git+https://github.com/AllenInstitute/abc_atlas_access.git"
    python experiments/biccn_2023_cell_types/00_fetch_abundance.py --list          # inspect names
    python experiments/biccn_2023_cell_types/00_fetch_abundance.py --species mouse
    python experiments/biccn_2023_cell_types/00_fetch_abundance.py --species human --human-roi-mni roi_mni.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd


def _cache(out_dir: Path):
    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
    return AbcProjectCache.from_cache_dir(out_dir / "abc_cache")


def _pick(cols, *cands, contains=None):
    for c in cands:
        if c in cols:
            return c
    if contains:
        for c in cols:
            if all(t in c.lower() for t in contains):
                return c
    raise KeyError(f"none of {cands} / contains={contains} in {list(cols)[:20]}")


def _frac_and_centroid(df, region_col, sub_col, xyz_cols):
    df = df.dropna(subset=[region_col, sub_col, *xyz_cols])
    frac = df.groupby([region_col, sub_col]).size().unstack(fill_value=0)
    frac = frac.div(frac.sum(1), axis=0)
    cent = df.groupby(region_col)[list(xyz_cols)].mean()
    cent.columns = ["x", "y", "z"]
    out = cent.join(frac).reset_index().rename(columns={region_col: "region"})
    return out


def build_mouse(out_dir: Path) -> Path:
    c = _cache(out_dir)
    D0, D1 = "MERFISH-C57BL6J-638850", "MERFISH-C57BL6J-638850-CCF"
    ann = c.get_metadata_dataframe(directory=D0, file_name="cell_metadata_with_cluster_annotation")
    ann = ann.set_index("cell_label") if "cell_label" in ann.columns else ann
    ccf = c.get_metadata_dataframe(directory=D1, file_name="ccf_coordinates")
    ccf = ccf.set_index("cell_label") if "cell_label" in ccf.columns else ccf
    par = c.get_metadata_dataframe(directory=D1, file_name="cell_metadata_with_parcellation_annotation")
    par = par.set_index("cell_label") if "cell_label" in par.columns else par

    sub_col = _pick(ann.columns, "subclass", contains=["subclass"])
    reg_col = _pick(par.columns, "parcellation_structure", "parcellation_division",
                    "parcellation_substructure", contains=["parcellation", "structure"])
    xcol = _pick(ccf.columns, "x", contains=["x"]); ycol = _pick(ccf.columns, "y", contains=["y"]); zcol = _pick(ccf.columns, "z", contains=["z"])
    df = ann[[sub_col]].join(par[[reg_col]], how="inner").join(ccf[[xcol, ycol, zcol]], how="inner")
    out = _frac_and_centroid(df, reg_col, sub_col, (xcol, ycol, zcol))
    print(f"[MOUSE] {out.shape[0]} CCF regions x {out.shape[1]-4} subclasses. "
          f"CCF x-range {out.x.min():.2f}..{out.x.max():.2f} — verify vs M.var frame/units.")
    p = out_dir / "mouse_yao2023_region_by_subclass.csv"; out.to_csv(p, index=False); return p


def build_human(out_dir: Path, roi_mni_csv: str | None) -> Path:
    c = _cache(out_dir)
    meta = c.get_metadata_dataframe(directory="WHB-10Xv3", file_name="cell_metadata")
    sub_col = _pick(meta.columns, "subclass", "supercluster_term", contains=["subclass"])
    roi_col = _pick(meta.columns, "region_of_interest_acronym", "roi", "dissection",
                    contains=["region", "interest"])
    df = meta.dropna(subset=[roi_col, sub_col])
    frac = df.groupby([roi_col, sub_col]).size().unstack(fill_value=0)
    frac = frac.div(frac.sum(1), axis=0)
    if roi_mni_csv is None:
        print("[HUMAN] no --human-roi-mni: writing FRACTIONS ONLY. Add x,y,z (MNI mm) per "
              "ROI before running 04. ROIs to map:", list(frac.index)[:8], "...")
        out = frac.reset_index().rename(columns={roi_col: "region"})
    else:
        mni = pd.read_csv(roi_mni_csv); mni = mni.set_index(mni.columns[0])
        cent = pd.DataFrame([mni.loc[r, ["x", "y", "z"]].values if r in mni.index else [np.nan]*3
                             for r in frac.index], index=frac.index, columns=["x", "y", "z"])
        out = cent.join(frac).reset_index().rename(columns={roi_col: "region"})
        print(f"[HUMAN] {out.shape[0]} ROIs x {frac.shape[1]} subclasses; "
              f"{cent.notna().all(1).sum()} matched to MNI centroids.")
    p = out_dir / "human_siletti2023_region_by_subclass.csv"; out.to_csv(p, index=False); return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data_external/cell_atlases")
    ap.add_argument("--human-roi-mni", default=None, help="CSV: roi_name,x,y,z (MNI mm)")
    ap.add_argument("--species", choices=["mouse", "human", "both"], default="both")
    ap.add_argument("--list", action="store_true", help="print ABC directories/files and exit")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.list:
        c = _cache(out)
        for d in ["MERFISH-C57BL6J-638850", "MERFISH-C57BL6J-638850-CCF", "WHB-10Xv3"]:
            print(d, "->", c.list_metadata_files(d))
        return
    if args.species in ("mouse", "both"):
        print("mouse ->", build_mouse(out))
    if args.species in ("human", "both"):
        print("human ->", build_human(out, args.human_roi_mni))
    print("\nThen: python experiments/biccn_2023_cell_types/04_abundance_composition.py "
          "--mouse-abundance <mouse_csv> --human-abundance <human_csv>")


if __name__ == "__main__":
    main()
