#!/usr/bin/env python3
"""Test a directional hypothesis about Parkinson disease stage translation.

ENIGMA cortical-thickness effects for Hoehn--Yahr stages 1, 2, 3, and 4/5
are routed through OTTER. The investigator-defined contrast compares mouse
interoceptive cortex (VISC/GU/AId/AIv/AIp) with primary motor cortex (MOp/MOs).
The analysis tests whether increasing clinical stage is associated with a
relative shift towards the interoceptive set. The source maps are
cross-sectional estimates for separate stage groups and do not establish
within-person progression.

The spatial null uses one-to-one, mirrored rotations within the two human
hemispheres.  It never exchanges hemispheres and never duplicates parcels.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import abagen
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from otter.data import load_cached  # noqa: E402
from otter.eval.nulls import _haar_rotation  # noqa: E402

N_ROT = 10_000
SEED = 20260825
INTEROCEPTIVE = ["VISC", "GU", "AId", "AIv", "AIp"]
PRIMARY_MOTOR = ["MOp", "MOs"]
STAGES = ["HY1", "HY2", "HY3", "HY45"]
STAGE_LABELS = ["HY1", "HY2", "HY3", "HY4/5"]
STAGE_COEFFICIENTS = np.array([-1.5, -0.5, 0.5, 1.5])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_stage(path: Path) -> dict[tuple[str, str], float]:
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            match = re.match(r"^([LR])_?(.*)$", row["Structure"].strip())
            if match:
                out[(match.group(1), match.group(2).strip().lower())] = float(row["d_icv"])
    return out


def mouse_acronyms() -> np.ndarray:
    meta = json.loads((ROOT / "data_external/mouse_sc_meta.json").read_text())
    return np.array([
        meta["structure_acronyms"][i] if i >= 0 else "NA"
        for i in meta["node_struct_idx"]
    ])


def desikan_masks(H):
    """Map OTTER human parcels to the nearest Desikan cortical parcel."""
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    atlas = abagen.fetch_desikan_killiany()
    img = nib.load(atlas["image"])
    info = pd.read_csv(atlas["info"])
    labels = np.asarray(img.dataobj).astype(int)
    cortex = info[info.structure == "cortex"]
    id_meta = {
        int(r.id): (str(r.hemisphere), str(r.label).lower())
        for r in cortex.itertuples()
    }
    vox = nib.affines.apply_affine(np.linalg.inv(img.affine), xyz)
    ijk = np.rint(vox).astype(int)
    inside = np.all((ijk >= 0) & (ijk < np.array(labels.shape)), axis=1)
    parcel_label = np.zeros(len(xyz), int)
    parcel_label[inside] = labels[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]]

    atlas_voxels = np.argwhere(np.isin(labels, list(id_meta)))
    missing = np.where(parcel_label == 0)[0]
    distance, nearest = cKDTree(atlas_voxels).query(vox[missing])
    use = distance <= 4
    hits = atlas_voxels[nearest[use]]
    parcel_label[missing[use]] = labels[hits[:, 0], hits[:, 1], hits[:, 2]]
    parcel_label[~np.isin(parcel_label, list(id_meta))] = 0

    masks = {key: parcel_label == lab for lab, key in id_meta.items()}
    masks = {key: mask for key, mask in masks.items() if mask.sum() >= 8}
    centroids = {key: xyz[mask].mean(0) for key, mask in masks.items()}
    return masks, centroids


def zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / values.std(ddof=1)


def mirrored_bijective_rotations(keys, centroids, n_rot=N_ROT, seed=SEED):
    """Return permutations made by mirrored, within-hemisphere Hungarian spins."""
    xyz = np.array([centroids[key] for key in keys])
    groups = {
        hemi: np.array([i for i, key in enumerate(keys) if key[0] == hemi])
        for hemi in ("L", "R")
    }
    sphere = {}
    for hemi, idx in groups.items():
        centered = xyz[idx] - xyz[idx].mean(0)
        sphere[hemi] = centered / np.linalg.norm(centered, axis=1, keepdims=True)
    mirror = np.diag([-1.0, 1.0, 1.0])
    rng = np.random.default_rng(seed)
    permutations = np.empty((n_rot, len(keys)), dtype=np.int32)
    for draw in range(n_rot):
        Q = _haar_rotation(rng)
        permutations[draw] = np.arange(len(keys))
        for hemi, rotation in (("L", Q), ("R", mirror @ Q @ mirror)):
            idx = groups[hemi]
            source = sphere[hemi]
            rotated = source @ rotation.T
            cost = np.sum((rotated[:, None, :] - source[None, :, :]) ** 2, axis=2)
            rows, cols = linear_sum_assignment(cost)
            local = np.empty(len(idx), int)
            local[rows] = cols
            permutations[draw, idx] = idx[local]
    return permutations


def p_two_sided(null: np.ndarray, observed: float) -> float:
    return float((np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1))


def bijective_rotations(xyz: np.ndarray, n_rot: int, seed: int) -> np.ndarray:
    centered = xyz - xyz.mean(0)
    sphere = centered / np.linalg.norm(centered, axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    permutations = np.empty((n_rot, len(xyz)), dtype=np.int32)
    for draw in range(n_rot):
        rotated = sphere @ _haar_rotation(rng).T
        cost = np.sum((rotated[:, None, :] - sphere[None, :, :]) ** 2, axis=2)
        rows, cols = linear_sum_assignment(cost)
        permutations[draw, rows] = cols
    return permutations


def main() -> None:
    M, _ = load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    acronyms = mouse_acronyms()
    masks, centroids = desikan_masks(H)

    source_paths = {
        "HY1": ROOT / "data_external/enigma/cortical_thickness_parkinsons_HY1.csv",
        "HY2": ROOT / "data_external/enigma/cortical_thickness_parkinsons_HY2.csv",
        "HY3": ROOT / "data_external/enigma/cortical_thickness_parkinsons_HY3.csv",
        "HY45": ROOT / "data_external/enigma/cortical_thickness_parkinsons_HY4and5.csv",
    }
    source = {stage: parse_stage(path) for stage, path in source_paths.items()}
    keys = [key for key in masks if all(key in source[stage] for stage in STAGES)]
    # Negate Cohen's d so larger values denote greater cortical thinning.
    burden = {stage: np.array([-source[stage][key] for key in keys]) for stage in STAGES}
    stage_maps = {stage: zscore(burden[stage]) for stage in STAGES}
    slope_map = sum(
        STAGE_COEFFICIENTS[i] * stage_maps[stage] for i, stage in enumerate(STAGES)
    ) / np.sum(STAGE_COEFFICIENTS ** 2)
    permutations = mirrored_bijective_rotations(keys, centroids)

    result = {
        "status": "completed_directional_hypothesis_test",
        "hypothesis": (
            "increasing Parkinson stage preferentially increases translated "
            "interoceptive relative to primary-motor burden"
        ),
        "directional_alternative": "interoceptive_minus_primary_motor > 0",
        "cross_sectional": True,
        "pi_file": "outputs/coupling/pi_canonical.npy",
        "pi_sha256": sha256(ROOT / "outputs/coupling/pi_canonical.npy"),
        "n_human_regions": len(keys),
        "n_rotations": N_ROT,
        "seed": SEED,
        "human_null": "mirrored one-to-one rotations within hemisphere",
        "mouse_sets": {"interoceptive": INTEROCEPTIVE, "primary_motor": PRIMARY_MOTOR},
        "sources": {
            stage: {"file": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for stage, path in source_paths.items()
        },
        "couplings": {},
    }

    coupling_files = [
        "pi_canonical.npy",
        "pi_fc_plus_SC.npy",
        "pi_anchorfree_control.npy",
        "pi_ladder_1_connectivity_only.npy",
        "pi_ladder_2_+spatial.npy",
        "pi_ladder_3_+anchors.npy",
        "pi_ladder_5_NOCONN_spatial_only.npy",
        "pi_ladder_6_NOCONN_spatial+anch+packs.npy",
    ]
    canonical_bias = None
    canonical_dk_weight = None
    coupling_nulls = {}
    for filename in coupling_files:
        path = ROOT / "outputs/coupling" / filename
        pi = np.load(path)
        row = pi / pi.sum(1, keepdims=True).clip(1e-12)
        human_weight = (
            np.mean([row[acronyms == s].mean(0) for s in INTEROCEPTIVE], axis=0)
            - np.mean([row[acronyms == s].mean(0) for s in PRIMARY_MOTOR], axis=0)
        )
        dk_weight = np.array([human_weight[masks[key]].sum() for key in keys])
        observed = float(dk_weight @ slope_map)
        null = np.array([float(dk_weight @ slope_map[p]) for p in permutations])
        coupling_nulls[filename] = null
        stage_bias = {stage: float(dk_weight @ stage_maps[stage]) for stage in STAGES}
        result["couplings"][filename] = {
            "sha256": sha256(path),
            "stage_bias": stage_bias,
            "linear_trend": observed,
            "p_two_sided": p_two_sided(null, observed),
            "z_null": float((observed - null.mean()) / null.std(ddof=1)),
            "null_ci95": np.quantile(null, [0.025, 0.975]).tolist(),
        }
        if filename == "pi_canonical.npy":
            canonical_bias = stage_bias
            canonical_dk_weight = dk_weight

    result["coupling_contrasts"] = {}
    canonical_observed = result["couplings"]["pi_canonical.npy"]["linear_trend"]
    for comparator in [
        "pi_ladder_1_connectivity_only.npy",
        "pi_ladder_5_NOCONN_spatial_only.npy",
        "pi_ladder_6_NOCONN_spatial+anch+packs.npy",
    ]:
        observed = canonical_observed - result["couplings"][comparator]["linear_trend"]
        null = coupling_nulls["pi_canonical.npy"] - coupling_nulls[comparator]
        result["coupling_contrasts"][f"canonical_minus_{comparator}"] = {
            "delta_linear_trend": observed,
            "p_one_sided": float((np.sum(null >= observed) + 1) / (len(null) + 1)),
            "null_ci95": np.quantile(null, [0.025, 0.975]).tolist(),
        }

    # Bilateral consistency check; both hemispheres derive from the same meta-analysis.
    result["hemisphere_consistency"] = {}
    all_xyz = np.array([centroids[key] for key in keys])
    for hemi, offset in (("L", 11), ("R", 29)):
        idx = np.array([i for i, key in enumerate(keys) if key[0] == hemi])
        hemi_stage_maps = {stage: zscore(burden[stage][idx]) for stage in STAGES}
        hemi_slope = sum(
            STAGE_COEFFICIENTS[i] * hemi_stage_maps[stage]
            for i, stage in enumerate(STAGES)
        ) / np.sum(STAGE_COEFFICIENTS ** 2)
        hemi_permutations = bijective_rotations(all_xyz[idx], N_ROT, SEED + offset)
        weight = canonical_dk_weight[idx]
        observed = float(weight @ hemi_slope)
        null = np.array([float(weight @ hemi_slope[p]) for p in hemi_permutations])
        result["hemisphere_consistency"][hemi] = {
            "n_regions": len(idx),
            "linear_trend": observed,
            "p_two_sided": p_two_sided(null, observed),
            "z_null": float((observed - null.mean()) / null.std(ddof=1)),
        }

    # Coarse source-atlas benchmark: human insula minus precentral cortex.
    direct = np.array([
        1.0 if key[1] == "insula" else -1.0 if key[1] == "precentral" else 0.0
        for key in keys
    ])
    direct[direct > 0] /= np.sum(direct > 0)
    direct[direct < 0] /= np.sum(direct < 0)
    direct_observed = float(direct @ slope_map)
    direct_null = np.array([float(direct @ slope_map[p]) for p in permutations])
    result["direct_human_benchmark"] = {
        "contrast": "insula minus precentral",
        "linear_trend": direct_observed,
        "p_two_sided": p_two_sided(direct_null, direct_observed),
        "z_null": float((direct_observed - direct_null.mean()) / direct_null.std(ddof=1)),
    }

    out = ROOT / "outputs/logs/reverse_translation_pd_stage_progression.json"
    out.write_text(json.dumps(result, indent=2) + "\n")

    values = [canonical_bias[stage] for stage in STAGES]
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    ax.plot(range(4), values, marker="o", linewidth=2.2, color="#6657d9")
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xticks(range(4), STAGE_LABELS)
    ax.set_ylabel("Interoceptive − primary motor bias")
    ax.set_title("OTTER reverse translation of Parkinson stage maps")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig_path = ROOT / "outputs/figures/reverse_translation_pd_stage_progression.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    print(json.dumps({"log": str(out), "figure": str(fig_path), **result["couplings"]["pi_canonical.npy"]}, indent=2))


if __name__ == "__main__":
    main()
