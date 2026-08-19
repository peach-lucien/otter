"""Step 0. Build the ABIDE bundle the later steps read.

Downloads the ABIDE Preprocessed rois_ho derivative through nilearn, reads the Harvard-Oxford
ROI ids from each .1D header, and writes one connectivity bundle, one phenotype table and the
three Harvard-Oxford atlases the centroid steps need. The per-subject feature is the mean
absolute Fisher-z correlation of each ROI with every other ROI. The grand matrix is the mean
Fisher-z correlation matrix over subjects.

The ABIDE data are distributed by the 1000 Functional Connectomes Project and are available
after registration. Neither the download nor anything derived from it is redistributed with
this repository.

Outputs, written to $OTTER_ABIDE_BUNDLE or data_external/abide_ho/
    abide_G.npz                          roi_ids, G, grand_fc_z, file_ids
    abide_pheno_small.csv                one row per subject, in the column order of G
    ho_labels.json                       Harvard-Oxford subcortical label names
    HO-cort-maxprob-thr25-2mm.nii.gz     bilateral cortical atlas
    HO-cortl-maxprob-thr25-2mm.nii.gz    hemisphere-split cortical atlas
    HO-sub-maxprob-thr25-2mm.nii.gz      subcortical atlas

Run from the repository root:
    python experiments/autism_subtypes/abide_subtype/abide_ho_s0_bundle.py
"""
import json, os, re, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root():
    p = Path.cwd().resolve()
    for q in [p, *p.parents]:
        if (q / "src" / "otter").is_dir():
            return q
    raise SystemExit("run this from inside the OTTER repository")


ROOT = _repo_root()
BUNDLE = Path(os.environ.get("OTTER_ABIDE_BUNDLE", ROOT / "data_external/abide_ho"))
CACHE = Path(os.environ.get("OTTER_ABIDE_DIR", Path.home() / "abide_cache"))
PIPELINE = os.environ.get("OTTER_ABIDE_PIPELINE", "cpac")

# Phenotype columns the later steps read. qc_ok is derived below.
PHENO_COLS = ["FILE_ID", "SITE_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX", "func_mean_fd", "ADOS_TOTAL"]
QC_RATERS = ["qc_rater_1", "qc_anat_rater_2", "qc_func_rater_2", "qc_func_rater_3"]


def read_1d(path):
    """ROI ids from the header line and the (t x n_roi) time series beneath it."""
    with open(path) as fh:
        head = fh.readline()
    ids = [int(t) for t in re.findall(r"-?\d+", head)]
    ts = np.loadtxt(path, skiprows=1, ndmin=2)
    if ts.shape[1] != len(ids):
        raise ValueError("%s: %d header ids against %d columns" % (path, len(ids), ts.shape[1]))
    return ids, ts


def fisher_z(ts):
    """Fisher-z correlation matrix of the columns of ts, diagonal set to nan."""
    keep = ts.std(0) > 1e-9
    r = np.full((ts.shape[1], ts.shape[1]), np.nan)
    if keep.sum() > 1:
        sub = np.corrcoef(ts[:, keep].T)
        r[np.ix_(keep, keep)] = np.clip(sub, -0.999999, 0.999999)
    np.fill_diagonal(r, np.nan)
    return np.arctanh(r)


def main():
    from nilearn import datasets

    BUNDLE.mkdir(parents=True, exist_ok=True)
    print("bundle directory:", BUNDLE)

    print("fetching ABIDE Preprocessed rois_ho (pipeline=%s); this downloads several GB once" % PIPELINE)
    datasets.fetch_abide_pcp(data_dir=str(CACHE), pipeline=PIPELINE, band_pass_filtering=True,
                             global_signal_regression=False, derivatives=["rois_ho"], quality_checked=False)

    files = sorted(CACHE.rglob("*_rois_ho.1D"))
    if not files:
        raise SystemExit("no *_rois_ho.1D under %s; check the download" % CACHE)
    print("found %d rois_ho time series" % len(files))

    pheno_path = next(CACHE.rglob("Phenotypic_V1_0b_preprocessed1.csv"))
    pheno = pd.read_csv(pheno_path)
    pheno["FILE_ID"] = pheno["FILE_ID"].astype(str)

    ref_ids = None
    file_ids, rows, zsum, zn = [], [], None, None
    for k, path in enumerate(files):
        fid = path.name[: -len("_rois_ho.1D")]
        ids, ts = read_1d(path)
        if ref_ids is None:
            ref_ids = ids
        elif ids != ref_ids:
            print("  skipping %s: ROI id set differs from the first subject" % fid)
            continue
        z = fisher_z(ts)
        if zsum is None:
            zsum = np.zeros_like(z)
            zn = np.zeros_like(z)
        ok = np.isfinite(z)
        zsum[ok] += z[ok]
        zn[ok] += 1
        with np.errstate(invalid="ignore"):
            rows.append(np.nanmean(np.abs(z), axis=1))
        file_ids.append(fid)
        if (k + 1) % 100 == 0:
            print("  %d/%d" % (k + 1, len(files)))

    G = np.asarray(rows, float)
    with np.errstate(invalid="ignore"):
        grand = zsum / np.where(zn > 0, zn, np.nan)
    roi_ids = np.asarray(ref_ids, int)
    print("G", G.shape, "| grand_fc_z", grand.shape, "| roi ids", len(roi_ids))

    ph = pheno.set_index("FILE_ID").reindex(file_ids).reset_index()
    missing = int(ph["SITE_ID"].isna().sum())
    if missing:
        raise SystemExit("%d subjects have no phenotype row; the two sources disagree" % missing)

    for c in PHENO_COLS:
        if c not in ph.columns:
            raise SystemExit("phenotype table has no column %r" % c)
    present = [c for c in QC_RATERS if c in ph.columns]
    qc = np.ones(len(ph), bool)
    for c in present:
        qc &= ph[c].astype(str).str.strip().str.lower().ne("fail").to_numpy()
    small = ph[PHENO_COLS].copy()
    small["qc_ok"] = qc.astype(int)
    print("qc raters used: %s -> %d of %d subjects pass" % (present, int(qc.sum()), len(qc)))

    small.to_csv(BUNDLE / "abide_pheno_small.csv", index=False)
    np.savez_compressed(BUNDLE / "abide_G.npz", roi_ids=roi_ids, G=G,
                        grand_fc_z=grand, file_ids=np.asarray(file_ids))

    for key, name in [("cort-maxprob-thr25-2mm", "HO-cort-maxprob-thr25-2mm.nii.gz"),
                      ("cortl-maxprob-thr25-2mm", "HO-cortl-maxprob-thr25-2mm.nii.gz"),
                      ("sub-maxprob-thr25-2mm", "HO-sub-maxprob-thr25-2mm.nii.gz")]:
        atlas = datasets.fetch_atlas_harvard_oxford(key, data_dir=str(CACHE))
        src = atlas.filename if isinstance(atlas.filename, str) else atlas.maps
        shutil.copyfile(src, BUNDLE / name)
        if key.startswith("sub"):
            json.dump({"sub": list(atlas.labels)}, open(BUNDLE / "ho_labels.json", "w"))
        print("  wrote", name)

    print("\nbundle complete in", BUNDLE)


if __name__ == "__main__":
    main()
