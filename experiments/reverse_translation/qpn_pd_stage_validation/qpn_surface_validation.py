#!/usr/bin/env python3
"""Validate the QPN-NC Parkinson stage result at native surface resolution.

Restricted FreeSurfer archives are read in place. The script writes only
cohort-level statistics and never writes participant identifiers, maps or scores.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from neuromaps.nulls.spins import gen_spinsamples
from nilearn import datasets, surface
from scipy.spatial import cKDTree
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = ROOT.parent
HERE = Path(__file__).resolve().parent
SEED = 20260903
N_ROT = 10_000
N_BOOT = 10_000
N_CLINICAL_PERM = 10_000
MAX_SURFACE_DISTANCE_MM = 8.0
VOLUME_RESCUE_MM = 4.0


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


qpn = _load_module("qpn_stage_validation", HERE / "qpn_stage_validation.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_subject(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold()).removeprefix("sub")


def parse_aparc_stats(payload: bytes) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in payload.decode("utf-8", errors="strict").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            continue
        values[fields[0]] = float(fields[4])
    return values


def parcel_labels(xyz: np.ndarray) -> tuple[np.ndarray, dict, set[str]]:
    """Assign OTTER human parcels to volumetric Desikan--Killiany regions."""
    import abagen

    atlas = abagen.fetch_desikan_killiany()
    image = nib.load(atlas["image"])
    info = pd.read_csv(atlas["info"])
    labels = np.asarray(image.dataobj).astype(int)
    voxels = nib.affines.apply_affine(np.linalg.inv(image.affine), xyz)
    ijk = np.rint(voxels).astype(int)
    inside = np.all((ijk >= 0) & (ijk < np.array(labels.shape)), axis=1)
    assigned = np.zeros(len(xyz), int)
    assigned[inside] = labels[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]]

    labelled = np.argwhere(labels > 0)
    missing = np.where(assigned == 0)[0]
    distance, nearest = cKDTree(labelled).query(voxels[missing])
    use = distance <= VOLUME_RESCUE_MM
    hits = labelled[nearest[use]]
    assigned[missing[use]] = labels[hits[:, 0], hits[:, 1], hits[:, 2]]

    id_to_name = dict(zip(info.id, info.label))
    cortical = {
        id_to_name[index] for index in info.loc[info.structure == "cortex", "id"]
    }
    return assigned, id_to_name, cortical


def destrieux_labels(atlas_dir: Path) -> tuple[list[str], object]:
    atlas = datasets.fetch_atlas_surf_destrieux(data_dir=str(atlas_dir), verbose=0)
    labels = [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in atlas.labels
    ]
    if labels[0].casefold() != "unknown" or labels.count("Medial_wall") != 1:
        raise ValueError("unexpected Destrieux label ordering")
    regions = [label for label in labels[1:] if label != "Medial_wall"]
    if len(regions) != 74:
        raise ValueError(f"expected 74 Destrieux regions, found {len(regions)}")
    return regions, atlas


def load_surface_thickness(
    freesurfer_dir: Path, regions: list[str]
) -> tuple[pd.DataFrame, dict]:
    archives = sorted(freesurfer_dir.glob("freesurfer_v7.3.2_*.tar"))
    if len(archives) != 21:
        raise ValueError(f"expected 21 FreeSurfer archives, found {len(archives)}")
    rows: list[dict] = []
    files_per_subject: list[int] = []
    success_logs = 0
    missing_region_counts: dict[str, int] = {}
    for archive in archives:
        with tarfile.open(archive) as stream:
            members = stream.getmembers()
            names = {member.name for member in members}
            subjects = sorted(
                {
                    member.name.split("/")[1]
                    for member in members
                    if len(member.name.split("/")) > 2
                    and member.name.split("/")[1].startswith("sub-")
                }
            )
            for subject in subjects:
                prefix = f"{archive.stem}/{subject}"
                record: dict[str, object] = {"_subject_key": normalize_subject(subject)}
                files_per_subject.append(
                    sum(name.startswith(f"{prefix}/") and not name.endswith("/") for name in names)
                )
                log_member = f"{prefix}/scripts/recon-all.log"
                log = stream.extractfile(log_member)
                if log is None:
                    raise ValueError("missing recon-all log")
                recon_success = (
                    "finished without error"
                    in log.read().decode(errors="replace").casefold()
                )
                record["_recon_success"] = recon_success
                success_logs += int(recon_success)
                for hemi in ("L", "R"):
                    member = f"{prefix}/stats/{hemi.casefold()}h.aparc.a2009s.stats"
                    handle = stream.extractfile(member)
                    if handle is None:
                        raise ValueError(f"missing {member}")
                    values = parse_aparc_stats(handle.read())
                    missing = sorted(set(regions) - set(values))
                    extra = sorted(set(values) - set(regions))
                    if extra or len(missing) > 2:
                        raise ValueError(
                            f"Destrieux stats mismatch: missing={missing}, extra={extra}"
                        )
                    for region in missing:
                        key = f"{hemi}_{region}"
                        missing_region_counts[key] = missing_region_counts.get(key, 0) + 1
                    record.update(
                        {
                            f"ct_{hemi}_{region}": values.get(region, np.nan)
                            for region in regions
                        }
                    )
                rows.append(record)
    frame = pd.DataFrame(rows)
    if frame["_subject_key"].duplicated().any() or len(frame) != 271:
        raise ValueError("unexpected duplicate or missing FreeSurfer subjects")
    region_columns = [f"ct_{hemi}_{region}" for hemi in ("L", "R") for region in regions]
    thickness = frame[region_columns].to_numpy(float)
    if np.any(np.isinf(thickness)):
        raise ValueError("infinite Destrieux thickness")
    audit = {
        "n_archives": len(archives),
        "n_subjects": int(len(frame)),
        "n_regions": len(region_columns),
        "files_per_subject_range": [min(files_per_subject), max(files_per_subject)],
        "recon_all_success": int(success_logs),
        "missing_parcel_values": int(np.isnan(thickness).sum()),
        "subjects_with_complete_148_region_surface": int(
            np.isfinite(thickness).all(axis=1).sum()
        ),
        "subjects_eligible_after_surface_qc": int(
            (np.isfinite(thickness).all(axis=1) & frame["_recon_success"].to_numpy(bool)).sum()
        ),
        "missing_region_counts": missing_region_counts,
    }
    return frame, audit


def attach_surface(frame: pd.DataFrame, surface_frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["_subject_key"] = work["participant_id"].map(normalize_subject)
    joined = work.merge(surface_frame, on="_subject_key", validate="one_to_one")
    if len(joined) != len(frame):
        raise ValueError(f"surface join lost participants: {len(frame)} -> {len(joined)}")
    region_columns = [column for column in joined if column.startswith("ct_")]
    joined = joined.loc[
        joined[region_columns].notna().all(axis=1) & joined["_recon_success"]
    ].copy()
    return joined.drop(columns=["participant_id", "_subject_key", "_recon_success"])


def surface_geometry_and_weight(
    H,
    pi: np.ndarray,
    acronyms: np.ndarray,
    regions: list[str],
    atlas,
) -> tuple[list[tuple[str, str]], np.ndarray, np.ndarray, np.ndarray, dict]:
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
    row = pi / pi.sum(axis=1, keepdims=True).clip(1e-12)
    human_weight = np.mean(
        [row[acronyms == value].mean(axis=0) for value in qpn.INTEROCEPTIVE], axis=0
    ) - np.mean(
        [row[acronyms == value].mean(axis=0) for value in qpn.PRIMARY_MOTOR], axis=0
    )
    parcel_label_ids, id_to_name, cortical_names = parcel_labels(
        H.var[["x", "y", "z"]].to_numpy(float)
    )
    cortical_ids = [index for index, name in id_to_name.items() if name in cortical_names]
    is_cortical = np.isin(parcel_label_ids, cortical_ids)
    keys = [(hemi, region) for hemi in ("L", "R") for region in regions]
    weight = np.zeros(len(keys), float)
    centroids = np.empty((len(keys), 3), float)
    hemi_id = np.repeat([0, 1], len(regions))
    valid_nodes = 0
    mapped_by_hemi = {}
    represented_by_hemi = {}
    distance_quantiles = {}
    atlas_labels = [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in atlas.labels
    ]
    atlas_ids = [atlas_labels.index(region) for region in regions]
    atlas_id_to_region = {label: index for index, label in enumerate(atlas_ids)}
    for hemi, side, offset in (("L", "left", 0), ("R", "right", len(regions))):
        pial = surface.load_surf_mesh(fsaverage[f"pial_{side}"]).coordinates
        sphere = surface.load_surf_mesh(fsaverage[f"sphere_{side}"]).coordinates
        annotation = np.asarray(atlas[f"map_{side}"], int)
        for region_index, atlas_id in enumerate(atlas_ids):
            centroid = sphere[annotation == atlas_id].mean(axis=0)
            centroids[offset + region_index] = centroid / np.linalg.norm(centroid)
        indices = np.where(is_cortical & (H.var["hemisphere"].to_numpy() == hemi))[0]
        distances, nearest = cKDTree(pial).query(
            H.var.iloc[indices][["x", "y", "z"]].to_numpy(float)
        )
        labels = annotation[nearest]
        keep = (
            (distances <= MAX_SURFACE_DISTANCE_MM)
            & np.isin(labels, atlas_ids)
        )
        for node, label in zip(indices[keep], labels[keep]):
            weight[offset + atlas_id_to_region[int(label)]] += human_weight[node]
        valid_nodes += int(keep.sum())
        mapped_by_hemi[hemi] = int(keep.sum())
        represented_by_hemi[hemi] = int(len(np.unique(labels[keep])))
        distance_quantiles[hemi] = np.quantile(
            distances, [0, 0.5, 0.95, 0.99, 1]
        ).tolist()
    if valid_nodes != 1580:
        raise ValueError(f"unexpected valid cortical OTTER node count: {valid_nodes}")
    audit = {
        "max_surface_distance_mm": MAX_SURFACE_DISTANCE_MM,
        "mapped_OTTER_nodes": valid_nodes,
        "mapped_by_hemisphere": mapped_by_hemi,
        "represented_Destrieux_regions_by_hemisphere": represented_by_hemi,
        "all_cortical_distance_quantiles_mm": distance_quantiles,
        "weight_sum": float(weight.sum()),
        "weight_l1": float(np.abs(weight).sum()),
    }
    return keys, weight, centroids, hemi_id, audit


def surface_rotations(
    centroids: np.ndarray, hemi_id: np.ndarray, n_rot: int, seed: int
) -> np.ndarray:
    spins = gen_spinsamples(
        centroids,
        hemi_id,
        n_rotate=n_rot,
        method="hungarian",
        check_duplicates=False,
        seed=seed,
    )
    rotations = np.asarray(spins.T, dtype=np.int32)
    if rotations.shape != (n_rot, len(centroids)):
        raise ValueError("unexpected surface spin shape")
    expected = np.arange(len(centroids))
    if not all(np.array_equal(np.sort(rotation), expected) for rotation in rotations):
        raise ValueError("surface spin is not bijective")
    left = hemi_id == 0
    if np.any(rotations[:, left] >= left.sum()) or np.any(rotations[:, ~left] < left.sum()):
        raise ValueError("surface spin crosses hemispheres")
    return rotations


def rank_residuals(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    ranked = rankdata(np.asarray(values, float), method="average")
    ranked_covariates = np.column_stack(
        [rankdata(covariates[:, i], method="average") for i in range(covariates.shape[1])]
    )
    design = np.column_stack([np.ones(len(values)), ranked_covariates])
    return ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]


def participant_surface_analysis(
    controls,
    patients,
    region_columns,
    weight,
    rotations,
    n_clinical_perm,
) -> dict:
    model = qpn.fit_normative(controls, region_columns)
    maps = qpn.control_normative_atrophy(patients, region_columns, model)
    maps = (maps - maps.mean(axis=1, keepdims=True)) / maps.std(axis=1, ddof=1)[:, None]
    outcome = patients["hy_ordinal"].to_numpy(float)
    covariates = qpn.covariate_matrix(patients, "hy_lag_months")
    scores = maps @ weight
    x_residual = rank_residuals(scores, covariates)
    y_residual = rank_residuals(outcome, covariates)
    observed = float(np.corrcoef(x_residual, y_residual)[0, 1])
    rng = np.random.default_rng(SEED + 1)
    clinical_null = np.array(
        [np.corrcoef(x_residual, y_residual[rng.permutation(len(y_residual))])[0, 1]
         for _ in range(n_clinical_perm)]
    )
    spatial_null = np.empty(len(rotations), float)
    for start in range(0, len(rotations), 100):
        use = rotations[start : start + 100]
        rotated_scores = np.einsum("nkr,r->nk", maps[:, use], weight, optimize=True)
        for column in range(rotated_scores.shape[1]):
            spatial_null[start + column] = qpn.partial_spearman(
                rotated_scores[:, column], outcome, covariates
            )
    return {
        "partial_spearman_rho": observed,
        "clinical_residual_permutation_p_one_sided": float(
            (np.sum(clinical_null >= observed) + 1) / (len(clinical_null) + 1)
        ),
        "spatial_p_one_sided": float(
            (np.sum(spatial_null >= observed) + 1) / (len(spatial_null) + 1)
        ),
        "spatial_null_ci95": np.quantile(spatial_null, [0.025, 0.975]).tolist(),
        "n": int(len(patients)),
    }


def supporting_regional(
    controls,
    patients,
    region_columns,
    weight,
    rotations,
    keys,
) -> dict:
    frame = patients.copy()
    frame["advanced_HY3plus"] = (frame["hy_raw"] >= 3).astype(float)
    definitions = {
        "HY4_5_Huber": ("hy_ordinal", True),
        "raw_HY1_5_OLS": ("hy_raw", False),
        "advanced_HY3plus_vs_HY1_2": ("advanced_HY3plus", False),
    }
    results = {}
    for name, (outcome, robust) in definitions.items():
        summary, _ = qpn.endpoint_once(
            controls,
            frame,
            region_columns,
            outcome,
            "hy_lag_months",
            weight,
            rotations,
            keys,
            robust=robust,
        )
        summary["n"] = int(len(frame))
        results[name] = summary
    qc, _ = qpn.endpoint_once(
        controls,
        frame,
        region_columns,
        "hy_ordinal",
        "hy_lag_months",
        weight,
        rotations,
        keys,
        include_qc=True,
    )
    qc["n"] = int(len(frame))
    results["SurfaceHoles_adjusted"] = qc
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive-dir", type=Path, default=PROJECT_ROOT / "data_external/qpn-nc-r01"
    )
    parser.add_argument(
        "--atlas-dir", type=Path, default=ROOT / "data_external/nilearn"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/logs/reverse_translation_qpn_pd_stage_surface.json",
    )
    parser.add_argument("--n-rot", type=int, default=N_ROT)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-clinical-perm", type=int, default=N_CLINICAL_PERM)
    args = parser.parse_args()

    regions, atlas = destrieux_labels(args.atlas_dir)
    surface_frame, archive_audit = load_surface_thickness(
        args.archive_dir / "freesurfer_v7.3.2", regions
    )
    controls, patients, _ = qpn.load_qpn(args.archive_dir)
    controls = attach_surface(controls, surface_frame)
    patients = attach_surface(patients, surface_frame)
    region_columns = [f"ct_{hemi}_{region}" for hemi in ("L", "R") for region in regions]
    H, _ = qpn.load_cached("human", cache_dir=ROOT / "outputs/anndata")
    pi_path = ROOT / "outputs/coupling/pi_canonical.npy"
    keys, weight, centroids, hemi_id, mapping_audit = surface_geometry_and_weight(
        H, np.load(pi_path), qpn.mouse_acronyms(), regions, atlas
    )
    rotations = surface_rotations(centroids, hemi_id, args.n_rot, SEED)

    primary, _ = qpn.endpoint_once(
        controls,
        patients,
        region_columns,
        "hy_ordinal",
        "hy_lag_months",
        weight,
        rotations,
        keys,
    )
    boot = qpn.bootstrap_D(
        controls,
        patients,
        region_columns,
        "hy_ordinal",
        "hy_lag_months",
        weight,
        args.n_boot,
        SEED + 3,
    )
    primary.update(
        {
            "n": int(len(patients)),
            "bootstrap_n": int(len(boot)),
            "bootstrap_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "bootstrap_median": float(np.median(boot)),
            "influence": qpn.leave_one_out_ranges(
                controls,
                patients,
                region_columns,
                "hy_ordinal",
                "hy_lag_months",
                weight,
            ),
        }
    )
    participant = participant_surface_analysis(
        controls,
        patients,
        region_columns,
        weight,
        rotations,
        args.n_clinical_perm,
    )
    supporting = supporting_regional(
        controls, patients, region_columns, weight, rotations, keys
    )

    result = {
        "schema_version": "1.0.0",
        "status": "completed",
        "analysis": "QPN-NC Parkinson stage surface-resolution validation",
        "hypothesis": (
            "Increasing Parkinson stage preferentially increases OTTER-translated "
            "mouse interoceptive-versus-primary-motor bias."
        ),
        "directional_alternative": "interoceptive_minus_primary_motor > 0",
        "dataset": {
            "name": "Quebec Parkinson Network Neuroimaging Cohort (QPN-NC)",
            "doi": "10.5281/zenodo.17246063",
            "access": "restricted; obtained directly from the data owners",
        },
        "seed": SEED,
        "n_surface_rotations": int(args.n_rot),
        "n_bootstrap": int(args.n_boot),
        "n_clinical_permutations": int(args.n_clinical_perm),
        "sample": {"controls": int(len(controls)), "PD": int(len(patients))},
        "coupling": {
            "pi_file": str(pi_path.relative_to(ROOT)),
            "pi_sha256": sha256(pi_path),
        },
        "surface_resolution": {
            "n_regions": len(keys),
            "n_left": int(np.sum(hemi_id == 0)),
            "n_right": int(np.sum(hemi_id == 1)),
            "eligible_reconstructions": archive_audit[
                "subjects_eligible_after_surface_qc"
            ],
            "mapped_OTTER_nodes": mapping_audit["mapped_OTTER_nodes"],
        },
        "primary_HY_surface": primary,
        "participant_level_support": participant,
        "supporting_model_forms": supporting,
        "input_hashes": {
            "left.aparc.a2009s.annot": sha256(
                args.atlas_dir / "destrieux_surface/left.aparc.a2009s.annot"
            ),
            "right.aparc.a2009s.annot": sha256(
                args.atlas_dir / "destrieux_surface/right.aparc.a2009s.annot"
            ),
        },
        "privacy": {
            "participant_identifiers_written": False,
            "participant_maps_or_scores_written": False,
            "regional_QPN_maps_written": False,
            "restricted_file_manifest_written": False,
            "small_stage_cell_counts_written": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "primary_HY_surface": primary,
        "participant_level_support": participant,
        "supporting_model_forms": supporting,
    }, indent=2))


if __name__ == "__main__":
    main()
