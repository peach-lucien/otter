#!/usr/bin/env python3
"""Frozen external confirmation of the Alzheimer phenotype dissociation.

The discovery analysis (09_ad_phenotype_dissociation.py) used La Joie et al.
(2021).  This script tests its unchanged target mapping in public, multisite
LEADS early-onset Alzheimer disease tau-PET maps (NeuroVault collection
23001; Lin et al., 2026):

    S1 / Typical       -> AD medial-temporal target
    S3 / Posterior     -> PCA visual target
    S2 / Left Temporal -> lvPPA auditory/temporal-association target

The frozen primary input is the study-provided subtype-versus-rest T map,
adjusted for SuStaIn stage, age, sex, education, and Centiloid.  The gate is:
all three row selectivities positive, joint one-sided p < .05, and joint
p < .05 after refitting OTTER without the directly relevant anchor packs.

A frozen sensitivity removes the thresholded-map concern.  Within each of the
three adequately populated SuStaIn bins (9-11, 12-14, 15-20), it contrasts
each subtype's unthresholded mean SUVR map against the equal mean of the other
two and then averages the z-scored contrasts across bins.  Stage 0-8 was
excluded before running because S2 contains only six participants.

Equal-structure target weighting is reported as a sensitivity, not a gate.
The public NIfTI files are downloaded on demand into ignored data_external/.
"""
from __future__ import annotations

import importlib.util
import json
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DISCOVERY_SCRIPT = ROOT / "experiments/reverse_translation/09_ad_phenotype_dissociation.py"
DATA_DIR = ROOT / "data_external/ad_leads_confirmation"
BASE_URL = "https://neurovault.org/media/images/23001"

PRIMARY_IMAGES = {
    "AD": (1017479, "baseline_tau_1vr_compare_3x_s1.nii.gz"),
    "PCA": (1017481, "baseline_tau_1vr_compare_3x_s3.nii.gz"),
    "lvPPA": (1017480, "baseline_tau_1vr_compare_3x_s2.nii.gz"),
}
STAGE_IMAGES = {
    1: {
        "9-11": (1017486, "baseline_tau_mean_3x_s1_stage9-11.nii.gz"),
        "12-14": (1017484, "baseline_tau_mean_3x_s1_stage12-14.nii.gz"),
        "15-20": (1017485, "baseline_tau_mean_3x_s1_stage15-20.nii.gz"),
    },
    2: {
        "9-11": (1017491, "baseline_tau_mean_3x_s2_stage9-11.nii.gz"),
        "12-14": (1017489, "baseline_tau_mean_3x_s2_stage12-14.nii.gz"),
        "15-20": (1017490, "baseline_tau_mean_3x_s2_stage15-20.nii.gz"),
    },
    3: {
        "9-11": (1017496, "baseline_tau_mean_3x_s3_stage9-11.nii.gz"),
        "12-14": (1017494, "baseline_tau_mean_3x_s3_stage12-14.nii.gz"),
        "15-20": (1017495, "baseline_tau_mean_3x_s3_stage15-20.nii.gz"),
    },
}
STAGES = ("9-11", "12-14", "15-20")
SOURCE_SUBTYPE = {"AD": 1, "PCA": 3, "lvPPA": 2}


def load_discovery_module():
    spec = importlib.util.spec_from_file_location("ad09_confirmation", DISCOVERY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_map(filename: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    if not path.exists():
        print(f"downloading {filename}")
        urllib.request.urlretrieve(f"{BASE_URL}/{filename}", path)
    return path


def sample_map(ad, path, parcel_image, numids, masks, keys):
    values = ad.RT01.sample_to_nodes(path, parcel_image, numids)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(f"no finite values sampled from {path}")
    values[~finite] = np.nanmean(values[finite])
    return np.asarray([values[masks[key]].mean() for key in keys])


def primary_maps(ad, parcel_image, numids, masks, keys):
    maps = []
    for phenotype in ad.PHENOTYPES:
        _, filename = PRIMARY_IMAGES[phenotype]
        maps.append(ad.zscore(sample_map(ad, ensure_map(filename), parcel_image, numids, masks, keys)))
    return np.asarray(maps)


def stage_matched_maps(ad, parcel_image, numids, masks, keys):
    stage_contrasts = []
    for stage in STAGES:
        raw = []
        for phenotype in ad.PHENOTYPES:
            subtype = SOURCE_SUBTYPE[phenotype]
            _, filename = STAGE_IMAGES[subtype][stage]
            raw.append(sample_map(ad, ensure_map(filename), parcel_image, numids, masks, keys))
        raw = np.asarray(raw)
        contrasts = []
        for index in range(len(ad.PHENOTYPES)):
            contrast = raw[index] - np.mean(np.delete(raw, index, axis=0), axis=0)
            contrasts.append(ad.zscore(contrast))
        stage_contrasts.append(np.asarray(contrasts))
    averaged = np.mean(stage_contrasts, axis=0)
    return np.asarray([ad.zscore(row) for row in averaged])


def passes_gate(result):
    return (
        all(value > 0 for value in result["row_selectivity"].values())
        and result["p_one_sided"] < 0.05
    )


def map_provenance(ad):
    records = {}
    all_images = list(PRIMARY_IMAGES.values())
    all_images.extend(item for subtype in STAGE_IMAGES.values() for item in subtype.values())
    for image_id, filename in all_images:
        path = ensure_map(filename)
        records[filename] = {
            "neurovault_image_id": image_id,
            "url": f"{BASE_URL}/{filename}",
            "sha256": ad.sha256_file(path),
        }
    return records


def main():
    ad = load_discovery_module()
    _, _ = ad.load_cached("mouse", cache_dir=ROOT / "outputs/anndata")
    H, _ = ad.load_cached("human", cache_dir=ROOT / "outputs/anndata")
    acronyms = ad.RT01.mouse_acr()
    masks, centroids = ad.desikan_masks(H)
    keys = sorted(masks)
    parcel_image, numids = ad.RT01.build_node_voxel_map(H.var)

    maps = {
        "primary_thresholded_subtype_vs_rest": primary_maps(
            ad, parcel_image, numids, masks, keys
        ),
        "unthresholded_stage_matched": stage_matched_maps(
            ad, parcel_image, numids, masks, keys
        ),
    }
    permutations = ad.mirrored_bijective_rotations(keys, centroids)

    canonical_path = ROOT / "outputs/coupling/pi_canonical.npy"
    canonical = np.load(canonical_path)
    no_relevant, dropped = ad.refit_without_relevant_packs()
    couplings = {
        "canonical": canonical,
        "no_relevant_anchor_packs": no_relevant,
    }

    result = {
        "status": "external_confirmation_frozen_before_first_run",
        "source": {
            "study": "Lin et al., Brain Communications 2026",
            "doi": "10.1093/braincomms/fcag176",
            "pmid": "42255923",
            "pmcid": "PMC13234610",
            "neurovault_collection": "https://neurovault.org/collections/23001/",
            "cohort": "LEADS multisite amyloid-positive sporadic early-onset AD",
            "sample": {"total_assigned": 359, "S1_typical": 144, "S2_left_temporal": 111, "S3_posterior": 104},
            "images": map_provenance(ad),
        },
        "frozen_mapping": {
            "S1_typical": "AD medial-temporal target",
            "S3_posterior": "PCA visual target",
            "S2_left_temporal": "lvPPA auditory/temporal-association target",
        },
        "targets": ad.TARGETS,
        "primary": "thresholded subtype-vs-rest/canonical/parcel_balanced joint_selectivity",
        "confirmation_gate": "all row selectivities > 0 and joint p < .05 for canonical and relevant-packs-removed parcel-balanced analyses",
        "stage_sensitivity": "unthresholded mean SUVR, within-stage subtype contrasts averaged over stages 9-11, 12-14, and 15-20",
        "n_human_regions": len(keys),
        "n_rotations": ad.N_ROT,
        "seed": ad.SEED,
        "spatial_null": "joint mirrored one-to-one rotations within hemisphere",
        "couplings": {
            "canonical": {
                "pi_file": str(canonical_path.relative_to(ROOT)),
                "pi_sha256": ad.sha256_file(canonical_path),
            },
            "no_relevant_anchor_packs": {
                "sha256_array": ad.sha256_array(no_relevant),
                "dropped_packs": dropped,
                "n_dropped": len(dropped),
            },
        },
        "results": {},
    }

    for coupling_name, pi in couplings.items():
        result["results"][coupling_name] = {}
        for weighting in ("parcel_balanced", "structure_balanced"):
            weights, present = ad.target_weights(pi, acronyms, masks, keys, weighting)
            result["results"][coupling_name][weighting] = {
                "present_targets": present,
                **{
                    analysis: ad.score_maps(values, weights, permutations)
                    for analysis, values in maps.items()
                },
            }

    primary_key = "primary_thresholded_subtype_vs_rest"
    primary_canonical = result["results"]["canonical"]["parcel_balanced"][primary_key]
    primary_no_relevant = result["results"]["no_relevant_anchor_packs"]["parcel_balanced"][primary_key]
    result["confirmation_gate_passed"] = passes_gate(primary_canonical) and passes_gate(primary_no_relevant)

    output = ROOT / "outputs/logs/reverse_translation_ad_leads_confirmation.json"
    output.write_text(json.dumps(result, indent=2))

    print("\nLEADS EXTERNAL CONFIRMATION")
    for coupling_name in couplings:
        for weighting in ("parcel_balanced", "structure_balanced"):
            values = result["results"][coupling_name][weighting]
            primary = values[primary_key]
            stage = values["unthresholded_stage_matched"]
            print(
                f"{coupling_name:28s} {weighting:18s} "
                f"primary={primary['joint_selectivity']:+.3f}, p={primary['p_one_sided']:.4f}; "
                f"stage-matched={stage['joint_selectivity']:+.3f}, p={stage['p_one_sided']:.4f}"
            )
    print(f"confirmation gate passed: {result['confirmation_gate_passed']}")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
