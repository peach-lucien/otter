#!/usr/bin/env python3
"""Discovery analysis of clinical Alzheimer disease phenotypes.

This analysis asks whether severity-matched, amyloid-positive
clinical phenotypes imply different mouse experiments.  The human source is
La Joie et al. (Neurology 2021; NeuroVault collection 8546): unthresholded
tau-PET and VBM T maps for amnestic AD, posterior cortical atrophy (PCA), and
logopenic primary progressive aphasia (lvPPA), each contrasted against the
other two phenotypes while controlling age, CDR-SB, and global signal.

The target sets and statistic below were frozen before the first run:

* amnestic AD -> mouse entorhinal/hippocampal circuit
* PCA         -> mouse visual cortex
* lvPPA       -> mouse auditory/temporal-association cortex

The threshold-free primary statistic sums, over phenotypes, the translated
score of the matched target minus the mean score of the two mismatched target
families.  Its null applies the same mirrored, one-to-one rotation jointly to
all three maps, separately within each human hemisphere.  The primary mouse
score weights parcels equally.  Equal-structure weighting and VBM are frozen
sensitivities.  A leakage control refits OTTER after removing only the eight
directly relevant entorhinal/hippocampal, visual, and auditory anchor packs.

This discovery analysis was frozen before its first run. The unchanged
assignments are evaluated in the largely independent LEADS cohort by
``10_ad_external_leads_confirmation.py``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import abagen
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from otter.data import load_cached  # noqa: E402
from otter.eval.nulls import _haar_rotation  # noqa: E402
from otter.repro import (  # noqa: E402
    CANONICAL,
    anchor_warped_xyz,
    fit_coupling,
    load_inputs,
)

N_ROT = 10_000
SEED = 20260826
PHENOTYPES = ["AD", "PCA", "lvPPA"]
MAPDIR = ROOT / "experiments/reverse_translation/ad_phenotype_maps"
MAPS = {
    "tau": {phenotype: MAPDIR / f"tau_{phenotype}.nii.gz" for phenotype in PHENOTYPES},
    "vbm": {phenotype: MAPDIR / f"vbm_{phenotype}.nii.gz" for phenotype in PHENOTYPES},
}
TARGETS = {
    "AD": ["ENTl", "ENTm", "CA1", "CA3", "DG", "SUB"],
    "PCA": ["VISp", "VISl", "VISal", "VISam", "VISpm", "VISrl", "VISa", "VISpor"],
    "lvPPA": ["AUDp", "AUDd", "AUDv", "TEa"],
}
DROP_PACK_PREFIXES = (
    "Subiculum",
    "CA1 ",
    "CA3 ",
    "Dentate gyrus",
    "Entorhinal cortex",
    "Lateral visual area",
    "Primary auditory",
    "Auditory belt",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RT01 = load_module("rt01_ad_phenotypes", ROOT / "experiments/reverse_translation/01_validate.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    return (values - values.mean()) / values.std(ddof=1)


def desikan_masks(H):
    """Assign OTTER human parcels to bilateral Desikan cortical regions."""
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    atlas = abagen.fetch_desikan_killiany()
    image = nib.load(atlas["image"])
    info = pd.read_csv(atlas["info"])
    labels = np.asarray(image.dataobj).astype(int)
    cortex = info[info.structure == "cortex"]
    metadata = {
        int(row.id): (str(row.hemisphere), str(row.label).lower())
        for row in cortex.itertuples()
    }

    voxels = nib.affines.apply_affine(np.linalg.inv(image.affine), xyz)
    ijk = np.rint(voxels).astype(int)
    inside = np.all((ijk >= 0) & (ijk < np.array(labels.shape)), axis=1)
    parcel_label = np.zeros(len(xyz), int)
    parcel_label[inside] = labels[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]]

    atlas_voxels = np.argwhere(np.isin(labels, list(metadata)))
    missing = np.where(parcel_label == 0)[0]
    distance, nearest = cKDTree(atlas_voxels).query(voxels[missing])
    use = distance <= 4
    hits = atlas_voxels[nearest[use]]
    parcel_label[missing[use]] = labels[hits[:, 0], hits[:, 1], hits[:, 2]]
    parcel_label[~np.isin(parcel_label, list(metadata))] = 0

    masks = {key: parcel_label == label for label, key in metadata.items()}
    masks = {key: mask for key, mask in masks.items() if mask.sum() >= 8}
    centroids = {key: xyz[mask].mean(axis=0) for key, mask in masks.items()}
    return masks, centroids


def mirrored_bijective_rotations(keys, centroids):
    """Mirrored Hungarian rotations that preserve hemisphere and parcel count."""
    xyz = np.array([centroids[key] for key in keys])
    groups = {
        hemi: np.array([i for i, key in enumerate(keys) if key[0] == hemi])
        for hemi in ("L", "R")
    }
    sphere = {}
    for hemi, indices in groups.items():
        centered = xyz[indices] - xyz[indices].mean(axis=0)
        sphere[hemi] = centered / np.linalg.norm(centered, axis=1, keepdims=True)

    mirror = np.diag([-1.0, 1.0, 1.0])
    rng = np.random.default_rng(SEED)
    permutations = np.empty((N_ROT, len(keys)), dtype=np.int32)
    for draw in range(N_ROT):
        rotation = _haar_rotation(rng)
        permutations[draw] = np.arange(len(keys))
        for hemi, transform in (("L", rotation), ("R", mirror @ rotation @ mirror)):
            indices = groups[hemi]
            source = sphere[hemi]
            rotated = source @ transform.T
            cost = np.sum((rotated[:, None, :] - source[None, :, :]) ** 2, axis=2)
            rows, columns = linear_sum_assignment(cost)
            assignment = np.empty(len(indices), int)
            assignment[rows] = columns
            permutations[draw, indices] = indices[assignment]
    return permutations


def load_dk_maps(H, masks, keys):
    parcel_image, numids = RT01.build_node_voxel_map(H.var)
    result = {}
    for modality, paths in MAPS.items():
        maps = []
        for phenotype in PHENOTYPES:
            node_values = RT01.sample_to_nodes(paths[phenotype], parcel_image, numids)
            finite = np.isfinite(node_values)
            if not finite.any():
                raise ValueError(f"no finite values sampled from {paths[phenotype]}")
            node_values[~finite] = np.nanmean(node_values[finite])
            maps.append(zscore([node_values[masks[key]].mean() for key in keys]))
        result[modality] = np.asarray(maps)
    return result


def target_weights(pi, acronyms, masks, keys, weighting):
    row = pi / np.clip(pi.sum(axis=1, keepdims=True), 1e-12, None)
    weights = []
    present = {}
    for phenotype in PHENOTYPES:
        structures = [structure for structure in TARGETS[phenotype] if np.any(acronyms == structure)]
        present[phenotype] = structures
        if weighting == "parcel_balanced":
            human_weight = row[np.isin(acronyms, structures)].mean(axis=0)
        elif weighting == "structure_balanced":
            human_weight = np.mean(
                [row[acronyms == structure].mean(axis=0) for structure in structures],
                axis=0,
            )
        else:
            raise ValueError(weighting)
        weights.append([human_weight[masks[key]].sum() for key in keys])
    return np.asarray(weights), present


def exceedance_p(null: np.ndarray, observed: float) -> float:
    """Monte Carlo upper-tail p, counting floating-point ties conservatively."""
    tolerance = 1e-12 * max(1.0, abs(observed))
    return float((np.sum(null >= observed - tolerance) + 1) / (len(null) + 1))


def score_maps(maps, weights, permutations):
    score_matrix = maps @ weights.T
    diagonal = np.diag(score_matrix)
    row_selectivity = diagonal - (score_matrix.sum(axis=1) - diagonal) / 2
    observed = float(row_selectivity.sum())

    permuted_maps = maps[:, permutations]
    null_scores = np.einsum("tnk,jk->ntj", permuted_maps, weights)
    null_rows = np.empty((N_ROT, len(PHENOTYPES)))
    for index in range(len(PHENOTYPES)):
        null_rows[:, index] = (
            null_scores[:, index, index]
            - (null_scores[:, index].sum(axis=1) - null_scores[:, index, index]) / 2
        )
    null = null_rows.sum(axis=1)
    max_null = null_rows.max(axis=1)
    return {
        "score_matrix_rows_phenotypes_cols_targets": score_matrix.tolist(),
        "row_selectivity": dict(zip(PHENOTYPES, row_selectivity.tolist())),
        "row_p_max_corrected": {
            phenotype: exceedance_p(max_null, value)
            for phenotype, value in zip(PHENOTYPES, row_selectivity)
        },
        "joint_selectivity": observed,
        "p_one_sided": exceedance_p(null, observed),
        "z_null": float((observed - null.mean()) / null.std(ddof=1)),
        "null_ci95": np.quantile(null, [0.025, 0.975]).tolist(),
    }


def refit_without_relevant_packs():
    M, H, costs, entries = load_inputs(ROOT)
    kept = [
        entry
        for entry in entries
        if not str(getattr(entry, "label", "")).startswith(DROP_PACK_PREFIXES)
    ]
    dropped = [
        str(getattr(entry, "label", ""))
        for entry in entries
        if entry not in kept
    ]
    pi = fit_coupling(M, H, costs, kept, anchor_warped_xyz(M, H), **CANONICAL)
    return pi, dropped


def main():
    for paths in MAPS.values():
        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(path)

    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    acronyms = RT01.mouse_acr()
    masks, centroids = desikan_masks(H)
    keys = sorted(masks)
    dk_maps = load_dk_maps(H, masks, keys)
    permutations = mirrored_bijective_rotations(keys, centroids)

    canonical_path = ROOT / "outputs/coupling/pi_canonical.npy"
    canonical = np.load(canonical_path)
    no_relevant, dropped = refit_without_relevant_packs()
    couplings = {
        "canonical": canonical,
        "no_relevant_anchor_packs": no_relevant,
    }

    result = {
        "status": "discovery_frozen_before_first_run",
        "source": {
            "study": "La Joie et al., Neurology 2021",
            "doi": "10.1212/WNL.0000000000011270",
            "neurovault_collection": "https://identifiers.org/neurovault.collection:8546",
            "sample": {"AD": 69, "PCA": 21, "lvPPA": 29},
            "maps": {
                modality: {
                    phenotype: {
                        "file": str(path.relative_to(ROOT)),
                        "sha256": sha256_file(path),
                    }
                    for phenotype, path in paths.items()
                }
                for modality, paths in MAPS.items()
            },
        },
        "targets": TARGETS,
        "primary": "tau/canonical/parcel_balanced joint_selectivity",
        "n_human_regions": len(keys),
        "n_rotations": N_ROT,
        "seed": SEED,
        "spatial_null": "joint mirrored one-to-one rotations within hemisphere",
        "couplings": {
            "canonical": {
                "pi_file": str(canonical_path.relative_to(ROOT)),
                "pi_sha256": sha256_file(canonical_path),
            },
            "no_relevant_anchor_packs": {
                "sha256_array": sha256_array(no_relevant),
                "dropped_packs": dropped,
                "n_dropped": len(dropped),
            },
        },
        "results": {},
    }

    for coupling_name, pi in couplings.items():
        result["results"][coupling_name] = {}
        for weighting in ("parcel_balanced", "structure_balanced"):
            weights, present = target_weights(pi, acronyms, masks, keys, weighting)
            result["results"][coupling_name][weighting] = {
                "present_targets": present,
                **{
                    modality: score_maps(maps, weights, permutations)
                    for modality, maps in dk_maps.items()
                },
            }

    output = ROOT / "outputs/logs/reverse_translation_ad_phenotypes.json"
    output.write_text(json.dumps(result, indent=2))

    print("\nALZHEIMER PHENOTYPE DISSOCIATION")
    for coupling_name in couplings:
        for weighting in ("parcel_balanced", "structure_balanced"):
            values = result["results"][coupling_name][weighting]
            print(
                f"{coupling_name:28s} {weighting:18s} "
                f"tau={values['tau']['joint_selectivity']:+.3f} "
                f"p={values['tau']['p_one_sided']:.4f}; "
                f"VBM={values['vbm']['joint_selectivity']:+.3f} "
                f"p={values['vbm']['p_one_sided']:.4f}"
            )
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
