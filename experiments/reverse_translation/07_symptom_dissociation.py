#!/usr/bin/env python3
"""Final audited reverse translation of symptom-specific TMS circuits.

Negative values in the Siddiqi et al. atlas define the dysphoric circuit and
positive values define the anxiosomatic circuit.  The frozen mouse targets are
medial-prefrontal/cingulate versus amygdala/insula.  A positive statistic

    C = bias(dysphoric map) - bias(anxiosomatic map)

supports their dissociation.  This implementation uses joint bilateral-pair
Moran spectral randomization stratified by cortex/non-cortex; maximum-statistic
correction across the declared 5/10/20/30% tails; threshold-free and
conditional-axis sensitivities; a refit without relevant regional anchor packs;
and a paired comparison with a matched no-connectivity coupling.

The primary mouse score is parcel mass: every mouse parcel has equal status, so
large structures do not receive the same weight as a one-parcel structure.
Equal-acronym (structure-balanced) scores are reported as a sensitivity.  All
declared targets are included, including ILA (4 parcels) and LA (1 parcel).

Run from ``otter/``::

    PYTHONPATH=src python experiments/reverse_translation/07_symptom_dissociation.py

Writes ``outputs/logs/reverse_translation_symptom_dissociation.json`` and the
pack-out refit in ``outputs/coupling/``.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree, distance

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from otter.data.anchor_packs.registry import DEFAULT_PACK_NAMES, PACKS  # noqa: E402
from otter.repro import CANONICAL, anchor_warped_xyz, fit_coupling, load_inputs  # noqa: E402

MAP_PATH = ROOT / "experiments/reverse_translation/clinical_maps/tms_anxdys.nii.gz"
PARCELLATION_PATH = ROOT / "data_external/_diagnostics/parcellation_2094.nii.gz"
HO_CORTEX_PATH = (
    ROOT / "data_external/_domhof_extracted/HarvardOxford-cortl-maxprob-thr25-2mm.nii.gz"
)
CANONICAL_PATH = ROOT / "outputs/coupling/pi_canonical.npy"
NO_CONNECTIVITY_PATH = ROOT / "outputs/coupling/pi_ladder_6_NOCONN_spatial+anch+packs.npy"
PACK_OUT_PATH = ROOT / "outputs/coupling/pi_tms_no_pfc_cingulate_amygdala_packs.npy"
OUTPUT_PATH = ROOT / "outputs/logs/reverse_translation_symptom_dissociation.json"

FRACTIONS = (0.05, 0.10, 0.20, 0.30)
PRIMARY_FRACTION = 0.10
WEIGHTINGS = ("parcel_mass", "structure_balanced")
PRIMARY_WEIGHTING = "parcel_mass"
N_ROTATIONS = 10_000
SEED = 20260826
BATCH_SIZE = 256
HO_RADIUS_MM = 6.0
TOP_K = 10

EXCLUDED_PACKS = frozenset({"amygdala", "cingulate", "lateral_pfc"})
DYS_SET = ("PL", "ILA", "ACAd", "ACAv", "ORBm", "FRP")
ANX_SET = ("BLA", "BMA", "CEA", "LA", "AId", "AIv", "AIp")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RT01 = load_module("rt01_tms_final", ROOT / "experiments/reverse_translation/01_validate.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def clean_json(value):
    """Convert NumPy values and fail rather than write non-standard NaN JSON."""
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        item = float(value)
        if not np.isfinite(item):
            raise ValueError(f"refusing to write non-finite JSON value {item}")
        return item
    return value


def conservative_p(null: np.ndarray, observed: float) -> float:
    """Upper-tail Monte Carlo p; floating-point ties count as exceedances."""
    null = np.asarray(null, float)
    if not np.all(np.isfinite(null)) or not np.isfinite(observed):
        raise ValueError("non-finite statistic")
    tolerance = 1e-12 * max(1.0, abs(float(observed)))
    return float((np.sum(null >= observed - tolerance) + 1) / (len(null) + 1))


def null_summary(null: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(null)),
        "sd": float(np.std(null, ddof=1)),
        "ci95": np.quantile(null, [0.025, 0.975]).tolist(),
    }


def row_normalize(coupling: np.ndarray) -> np.ndarray:
    coupling = np.asarray(coupling, dtype=np.float64)
    if coupling.ndim != 2 or not np.all(np.isfinite(coupling)):
        raise ValueError("coupling must be a finite matrix")
    if coupling.min() < -1e-14:
        raise ValueError(f"coupling contains a negative entry: {coupling.min()}")
    coupling = np.clip(coupling, 0.0, None)
    totals = coupling.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("coupling contains an empty mouse row")
    return coupling / totals


def sample_sanitized_source(H) -> tuple[np.ndarray, dict]:
    """Finite-mask-normalized interpolation, with centroid fallback.

    Resampling the NaN-containing source directly spreads NaNs, while replacing
    them by zero attenuates values near the invalid-data boundary.  We therefore
    resample both a zero-filled numerator and a finite-data mask and divide the
    two.  Human anchor nodes absent from the voxel parcellation are sampled at
    their MNI centroids using the same normalized interpolation.
    """
    source = nib.load(MAP_PATH)
    raw = np.asarray(source.dataobj, dtype=np.float64)
    finite = np.isfinite(raw)
    if not finite.any():
        raise ValueError("TMS NIfTI has no finite voxels")
    clean = raw.copy()
    clean[~finite] = 0.0
    clean_image = nib.Nifti1Image(clean, source.affine, source.header)
    mask_image = nib.Nifti1Image(finite.astype(np.float64), source.affine, source.header)

    parcellation = nib.load(PARCELLATION_PATH)
    numerator_image = resample_to_img(clean_image, parcellation, interpolation="linear")
    with warnings.catch_warnings():
        # Linear interpolation is intentional: this is a continuous finite-data
        # fraction used to normalize the numerator, not a resampled label mask.
        warnings.filterwarnings("ignore", message="Resampling binary images.*")
        denominator_image = resample_to_img(mask_image, parcellation, interpolation="linear")
    numerator = np.asarray(numerator_image.dataobj, float)
    denominator = np.asarray(denominator_image.dataobj, float)
    values = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 1e-8,
    )
    labels = np.asarray(parcellation.dataobj).astype(int)
    numids = np.asarray(H.var["numid"], int)
    means = {
        int(label): float(np.nanmean(values[labels == label]))
        for label in np.unique(labels)
        if label > 0 and np.any(np.isfinite(values[labels == label]))
    }
    sampled = np.array([means.get(int(label), np.nan) for label in numids], float)
    missing = ~np.isfinite(sampled)
    missing_indices = np.flatnonzero(missing)
    centroid_xyz = H.var[["x", "y", "z"]].to_numpy(float)[missing]
    centroid_ijk = nib.affines.apply_affine(np.linalg.inv(source.affine), centroid_xyz).T
    centroid_num = map_coordinates(clean, centroid_ijk, order=1, mode="constant", cval=0.0)
    centroid_den = map_coordinates(
        finite.astype(float), centroid_ijk, order=1, mode="constant", cval=0.0
    )
    centroid_values = np.divide(
        centroid_num,
        centroid_den,
        out=np.full_like(centroid_num, np.nan),
        where=centroid_den > 1e-8,
    )
    sampled[missing] = centroid_values
    if not np.all(np.isfinite(sampled)):
        bad = np.flatnonzero(~np.isfinite(sampled))
        raise ValueError(f"centroid fallback did not resolve human nodes {bad.tolist()}")
    if not np.any(sampled != 0):
        raise ValueError("sampled TMS map is identically zero")
    return sampled, {
        "source_shape": list(raw.shape),
        "source_n_voxels": int(raw.size),
        "source_n_nonfinite_replaced_with_zero_before_resampling": int((~finite).sum()),
        "source_finite_min": float(raw[finite].min()),
        "source_finite_max": float(raw[finite].max()),
        "sampled_n_nodes": int(sampled.size),
        "parcel_overlap_n_nodes": int(len(sampled) - len(missing_indices)),
        "centroid_fallback_n_nodes": int(len(missing_indices)),
        "centroid_fallback_node_indices": missing_indices.tolist(),
        "sampled_n_nonfinite_after_fallback": 0,
        "sampled_min": float(sampled.min()),
        "sampled_max": float(sampled.max()),
        "sampled_n_negative": int(np.sum(sampled < 0)),
        "sampled_n_positive": int(np.sum(sampled > 0)),
        "sampled_n_zero": int(np.sum(sampled == 0)),
    }


def cortex_strata(H) -> tuple[np.ndarray, dict]:
    """Harvard-Oxford cortex/non-cortex crossed with supplied hemispheres."""
    if not HO_CORTEX_PATH.exists():
        raise FileNotFoundError(HO_CORTEX_PATH)
    image = nib.load(HO_CORTEX_PATH)
    data = np.asarray(image.dataobj)
    atlas_ijk = np.argwhere(data > 0)
    atlas_xyz = nib.affines.apply_affine(image.affine, atlas_ijk)
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    distances, _ = cKDTree(atlas_xyz).query(xyz)
    tissue = np.where(distances <= HO_RADIUS_MM, "cortex", "noncortex")
    hemisphere = np.asarray(H.var["hemisphere"].astype(str).to_numpy(), dtype=str)
    if not np.all(np.isin(hemisphere, ["L", "R"])):
        raise ValueError("all human nodes require L/R hemisphere labels")
    strata = np.char.add(np.char.add(hemisphere, "_"), tissue)
    return strata, {
        "atlas": str(HO_CORTEX_PATH.relative_to(ROOT)),
        "definition": f"within {HO_RADIUS_MM:g} mm of a labelled cortical voxel",
        "radius_mm": HO_RADIUS_MM,
        "n_cortex": int(np.sum(tissue == "cortex")),
        "n_noncortex": int(np.sum(tissue == "noncortex")),
        "counts": {label: int(np.sum(strata == label)) for label in sorted(set(strata))},
        "distance_mm": {
            "median": float(np.median(distances)),
            "p95": float(np.quantile(distances, 0.95)),
            "max": float(distances.max()),
        },
    }


def moran_basis(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Moran eigenvectors from a double-centred inverse-distance matrix.

    This follows BrainSpace ``compute_mem`` for a dense symmetric weight
    matrix and the ``spectrum='all'`` case.  Complete inverse-distance weights
    have a single constant/zero mode, which is removed explicitly.
    """
    distances = distance.squareform(distance.pdist(np.asarray(xyz, float)))
    weights = np.zeros_like(distances)
    nonzero = distances > 0
    weights[nonzero] = 1.0 / distances[nonzero]
    column_mean = weights.mean(axis=0, keepdims=True)
    centered = weights.mean() - column_mean - column_mean.T + weights
    eigenvalues, eigenvectors = np.linalg.eigh(centered.astype(np.float32))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order].astype(float)
    eigenvectors = eigenvectors[:, order].astype(float)
    zero_index = int(np.argmin(np.abs(eigenvalues)))
    if abs(eigenvalues[zero_index]) > 1e-5:
        raise ValueError("Moran matrix has no numerical zero/constant mode")
    keep = np.arange(len(eigenvalues)) != zero_index
    return eigenvectors[:, keep], eigenvalues[keep]


def stratified_moran_surrogates(
    values: np.ndarray,
    H,
    strata: np.ndarray,
    n_surrogates: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Joint singleton Moran randomisation of bilateral pair features.

    Each ``pairid`` supplies one spatial location and two features (left and
    right map values).  A pair is cortical if either member passes the fixed
    Harvard-Oxford mask (an a priori inclusive OR rule); otherwise it is
    non-cortical.  Right coordinates are reflected before the bilateral pair
    centroid is calculated.  Within each tissue, the same random sign is
    applied to a Moran component for both features (BrainSpace ``joint=True``),
    preserving each hemisphere's mean/variance and their cross-hemisphere
    multiscale dependence.
    """
    values = np.asarray(values, float)
    xyz = H.var[["x", "y", "z"]].to_numpy(float)
    hemispheres = np.asarray(H.var["hemisphere"].astype(str), dtype=str)
    pair_ids = np.asarray(H.var["pairid"], int)
    strata = np.asarray(strata)
    node_tissue = np.array([str(label).split("_", 1)[1] for label in strata])

    pairs = {}
    for pair_id in sorted(set(pair_ids.tolist())):
        indices = np.flatnonzero(pair_ids == pair_id)
        lookup = {hemispheres[index]: int(index) for index in indices}
        if len(indices) != 2 or set(lookup) != {"L", "R"}:
            raise ValueError(f"pairid {pair_id} is not an exact bilateral pair")
        pairs[int(pair_id)] = lookup

    groups = {"cortex": [], "noncortex": []}
    discordant = []
    for pair_id, lookup in pairs.items():
        tissues = {node_tissue[lookup["L"]], node_tissue[lookup["R"]]}
        label = "cortex" if "cortex" in tissues else "noncortex"
        groups[label].append(pair_id)
        if len(tissues) > 1:
            discordant.append(pair_id)

    bases = {}
    details = {}
    for label, ids in groups.items():
        indices = np.array([[pairs[pair_id]["L"], pairs[pair_id]["R"]] for pair_id in ids])
        right_xyz = xyz[indices[:, 1]].copy()
        right_xyz[:, 0] *= -1
        pair_xyz = (xyz[indices[:, 0]] + right_xyz) / 2
        pair_values = values[indices]
        basis, eigenvalues = moran_basis(pair_xyz)
        feature_means = pair_values.mean(axis=0)
        centered = pair_values - feature_means
        coefficients = basis.T @ centered
        reconstruction = feature_means + basis @ coefficients
        error = float(np.max(np.abs(reconstruction - pair_values)))
        if error > 2e-5:
            raise ValueError(f"Moran basis reconstruction failed for {label}: {error}")
        bases[label] = (indices, basis, coefficients, feature_means)
        details[str(label)] = {
            "n_bilateral_pairs": int(len(indices)),
            "n_components": int(basis.shape[1]),
            "eigenvalues_sha256": sha256_array(eigenvalues),
            "basis_sha256": sha256_array(basis),
            "reconstruction_max_abs_error": error,
        }

    rng = np.random.default_rng(seed)
    surrogates = np.empty((n_surrogates, len(values)), dtype=np.float64)
    sign_hashes = {}
    for tissue in ("cortex", "noncortex"):
        indices, basis, coefficients, feature_means = bases[tissue]
        n_components = basis.shape[1]
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_surrogates, n_components))
        sign_hashes[tissue] = sha256_array(signs)
        for start in range(0, n_surrogates, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, n_surrogates)
            # One sign per component/draw, broadcast over L/R features: joint=True.
            signed_coefficients = signs[start:stop, :, None] * coefficients[None, :, :]
            simulated = feature_means + np.einsum("pc,bch->bph", basis, signed_coefficients)
            surrogates[start:stop, indices[:, 0]] = simulated[:, :, 0]
            surrogates[start:stop, indices[:, 1]] = simulated[:, :, 1]
    details["pair_definition"] = {
        "pair_column": "H.var['pairid']",
        "n_bilateral_pairs": int(len(pairs)),
        "tissue_rule": "cortex if either member is cortical (inclusive OR), else noncortex",
        "pair_counts": {label: int(len(ids)) for label, ids in groups.items()},
        "discordant_node_tissue_pairids_resolved_by_or": discordant,
        "coordinate": "mean of left coordinate and x-reflected right coordinate",
    }
    details["joint_sign_stream_sha256"] = sign_hashes
    invariants = {}
    for tissue, ids in groups.items():
        indices = np.array([[pairs[pair_id]["L"], pairs[pair_id]["R"]] for pair_id in ids])
        observed = values[indices]
        simulated = np.stack(
            [surrogates[:, indices[:, 0]], surrogates[:, indices[:, 1]]], axis=2
        )
        observed_mean = observed.mean(axis=0)
        observed_std = observed.std(axis=0, ddof=1)
        observed_covariance = float(np.cov(observed.T, ddof=1)[0, 1])
        observed_correlation = float(np.corrcoef(observed.T)[0, 1])
        simulated_mean = simulated.mean(axis=1)
        simulated_std = simulated.std(axis=1, ddof=1)
        centered_simulated = simulated - simulated_mean[:, None, :]
        simulated_covariance = np.sum(
            centered_simulated[:, :, 0] * centered_simulated[:, :, 1], axis=1
        ) / (len(indices) - 1)
        simulated_correlation = simulated_covariance / (
            simulated_std[:, 0] * simulated_std[:, 1]
        )
        invariants[tissue] = {
            "max_abs_hemisphere_mean_error": float(
                np.max(np.abs(simulated_mean - observed_mean))
            ),
            "max_abs_hemisphere_sd_error": float(
                np.max(np.abs(simulated_std - observed_std))
            ),
            "max_abs_lr_covariance_error": float(
                np.max(np.abs(simulated_covariance - observed_covariance))
            ),
            "max_abs_lr_correlation_error": float(
                np.max(np.abs(simulated_correlation - observed_correlation))
            ),
        }
    details["verified_invariants_all_draws"] = invariants
    return surrogates, details


def refit_without_relevant_packs() -> tuple[np.ndarray, dict]:
    M, H, costs, _ = load_inputs(ROOT)
    kept_entries = []
    kept_names = []
    dropped_labels = []
    for name in DEFAULT_PACK_NAMES:
        entries = PACKS[name].builder(M.var, H.var, atlas_root=str(ROOT))
        if name in EXCLUDED_PACKS:
            dropped_labels.extend(str(getattr(entry, "label", "")) for entry in entries)
        else:
            kept_names.append(name)
            kept_entries.extend(entries)
    coupling = fit_coupling(
        M, H, costs, kept_entries, anchor_warped_xyz(M, H), **CANONICAL
    )
    PACK_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(PACK_OUT_PATH, coupling)
    return coupling, {
        "excluded_pack_names": sorted(EXCLUDED_PACKS),
        "excluded_entry_labels": dropped_labels,
        "kept_pack_names": kept_names,
        "n_kept_entries": len(kept_entries),
        "recipe": dict(CANONICAL),
        "saved_file": str(PACK_OUT_PATH.relative_to(ROOT)),
        "array_sha256": sha256_array(coupling),
        "file_sha256": sha256_file(PACK_OUT_PATH),
    }


def make_signed_weights(values: np.ndarray, sign: float, fraction: float | None) -> np.ndarray:
    signed = np.clip(sign * np.asarray(values, float), 0.0, None)
    if fraction is None:
        return signed
    threshold = np.quantile(signed, 1.0 - fraction, axis=-1, keepdims=True)
    return np.clip(signed - threshold, 0.0, None)


class Scorer:
    """Reduced linear routes needed for parcel- and structure-balanced axes."""

    def __init__(self, coupling: np.ndarray, acronyms: np.ndarray, weighting: str):
        row = row_normalize(coupling)
        acronyms = np.asarray(acronyms)
        structures = sorted(s for s in set(acronyms.tolist()) if s != "NA" and np.any(acronyms == s))
        missing = (set(DYS_SET) | set(ANX_SET)) - set(structures)
        if missing:
            raise ValueError(f"declared target acronyms absent: {sorted(missing)}")
        self.row = row
        self.acronyms = acronyms
        self.structures = structures
        self.weighting = weighting
        self.dys = list(DYS_SET)
        self.anx = list(ANX_SET)
        self.parcel_counts = {s: int(np.sum(acronyms == s)) for s in self.dys + self.anx}

        if weighting == "parcel_mass":
            named = acronyms != "NA"
            dys_weights = np.isin(acronyms, self.dys).astype(float)
            anx_weights = np.isin(acronyms, self.anx).astype(float)
            all_weights = named.astype(float)
        elif weighting == "structure_balanced":
            dys_weights = sum((acronyms == s) / np.sum(acronyms == s) for s in self.dys)
            anx_weights = sum((acronyms == s) / np.sum(acronyms == s) for s in self.anx)
            all_weights = sum((acronyms == s) / np.sum(acronyms == s) for s in structures)
        else:
            raise ValueError(weighting)
        self.numerator_route = (dys_weights - anx_weights) @ row
        self.all_route = all_weights @ row
        self.target_route = (dys_weights + anx_weights) @ row

    def bias_many(self, weights: np.ndarray, conditional: bool) -> np.ndarray:
        weights = np.asarray(weights, float)
        numerator = weights @ self.numerator_route
        denominator_route = self.target_route if conditional else self.all_route
        denominator = weights @ denominator_route
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)

    def target_mass_fraction(self, weights: np.ndarray) -> float:
        target = float(np.asarray(weights) @ self.target_route)
        total = float(np.asarray(weights) @ self.all_route)
        return target / total if total > 0 else 0.0

    def contrast(self, values: np.ndarray, fraction: float | None, conditional: bool = False):
        dys_weights = make_signed_weights(values, -1.0, fraction)
        anx_weights = make_signed_weights(values, +1.0, fraction)
        dys_bias = float(self.bias_many(dys_weights, conditional))
        anx_bias = float(self.bias_many(anx_weights, conditional))
        return dys_bias - anx_bias, dys_bias, anx_bias, dys_weights, anx_weights

    def null_contrast(
        self,
        observed_map: np.ndarray,
        surrogate_maps: np.ndarray,
        fraction: float | None,
        conditional: bool = False,
    ) -> np.ndarray:
        result = np.empty(len(surrogate_maps), float)
        for start in range(0, len(surrogate_maps), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(surrogate_maps))
            values = surrogate_maps[start:stop]
            dys = make_signed_weights(values, -1.0, fraction)
            anx = make_signed_weights(values, +1.0, fraction)
            result[start:stop] = self.bias_many(dys, conditional) - self.bias_many(anx, conditional)
        return result

    def top_structures(self, human_weights: np.ndarray, k: int = TOP_K) -> list[dict]:
        mouse = self.row @ human_weights
        raw = []
        for structure in self.structures:
            values = mouse[self.acronyms == structure]
            score = float(values.sum() if self.weighting == "parcel_mass" else values.mean())
            raw.append(max(score, 0.0))
        raw = np.asarray(raw)
        distribution = raw / raw.sum() if raw.sum() > 0 else raw
        order = np.argsort(distribution)[::-1][:k]
        return [
            {"structure": self.structures[index], "mass": float(distribution[index])}
            for index in order
        ]


def analyse_scorer(
    scorer: Scorer, observed_map: np.ndarray, surrogate_maps: np.ndarray
) -> tuple[dict, dict[str, np.ndarray]]:
    observed_by_threshold = []
    null_by_threshold = []
    thresholded = {}
    null_arrays = {}
    for fraction in FRACTIONS:
        key = f"top_{int(round(fraction * 100)):02d}"
        observed = scorer.contrast(observed_map, fraction)
        null = scorer.null_contrast(observed_map, surrogate_maps, fraction)
        observed_by_threshold.append(observed[0])
        null_by_threshold.append(null)
        null_arrays[key] = null
        thresholded[key] = {
            "fraction": fraction,
            "contrast_C": observed[0],
            "bias_dysphoric": observed[1],
            "bias_anxiosomatic": observed[2],
            "active_human_nodes": {
                "dysphoric": int(np.count_nonzero(observed[3])),
                "anxiosomatic": int(np.count_nonzero(observed[4])),
            },
            "p_one_sided_uncorrected": conservative_p(null, observed[0]),
            "null": null_summary(null),
            "top_dysphoric": scorer.top_structures(observed[3]),
            "top_anxiosomatic": scorer.top_structures(observed[4]),
        }

    observed_by_threshold = np.asarray(observed_by_threshold)
    null_by_threshold = np.column_stack(null_by_threshold)
    max_null = null_by_threshold.max(axis=1)
    for index, fraction in enumerate(FRACTIONS):
        key = f"top_{int(round(fraction * 100)):02d}"
        thresholded[key]["p_one_sided_max_fwer"] = conservative_p(
            max_null, observed_by_threshold[index]
        )
    best = int(np.argmax(observed_by_threshold))
    null_arrays["threshold_max"] = max_null

    threshold_free_observed = scorer.contrast(observed_map, None)
    threshold_free_null = scorer.null_contrast(observed_map, surrogate_maps, None)
    null_arrays["threshold_free_signed"] = threshold_free_null

    conditional = {}
    for label, fraction in (("primary_top_10", PRIMARY_FRACTION), ("threshold_free_signed", None)):
        observed = scorer.contrast(observed_map, fraction, conditional=True)
        null = scorer.null_contrast(observed_map, surrogate_maps, fraction, conditional=True)
        null_arrays[f"conditional_{label}"] = null
        conditional[label] = {
            "contrast_C": observed[0],
            "bias_dysphoric": observed[1],
            "bias_anxiosomatic": observed[2],
            "target_union_mass_fraction": {
                "dysphoric_input": scorer.target_mass_fraction(observed[3]),
                "anxiosomatic_input": scorer.target_mass_fraction(observed[4]),
            },
            "p_one_sided": conservative_p(null, observed[0]),
            "null": null_summary(null),
        }

    result = {
        "definition": (
            "mouse parcel mass" if scorer.weighting == "parcel_mass" else
            "mean within each acronym, followed by equal weighting of acronyms"
        ),
        "target_parcel_counts": scorer.parcel_counts,
        "thresholded": thresholded,
        "threshold_omnibus": {
            "statistic": "max C across top 5/10/20/30 percent",
            "observed_max_C": float(observed_by_threshold[best]),
            "selected_threshold": float(FRACTIONS[best]),
            "p_one_sided": conservative_p(max_null, observed_by_threshold[best]),
            "null": null_summary(max_null),
        },
        "threshold_free_signed": {
            "contrast_C": threshold_free_observed[0],
            "bias_dysphoric": threshold_free_observed[1],
            "bias_anxiosomatic": threshold_free_observed[2],
            "p_one_sided": conservative_p(threshold_free_null, threshold_free_observed[0]),
            "null": null_summary(threshold_free_null),
            "top_dysphoric": scorer.top_structures(threshold_free_observed[3]),
            "top_anxiosomatic": scorer.top_structures(threshold_free_observed[4]),
        },
        "conditional_axis": conditional,
    }
    return result, null_arrays


def paired_tests(canonical, no_connectivity, canonical_null, no_connectivity_null) -> dict:
    thresholded = {}
    observed_deltas = []
    null_deltas = []
    for fraction in FRACTIONS:
        key = f"top_{int(round(fraction * 100)):02d}"
        delta = canonical["thresholded"][key]["contrast_C"] - no_connectivity["thresholded"][key]["contrast_C"]
        null = canonical_null[key] - no_connectivity_null[key]
        observed_deltas.append(delta)
        null_deltas.append(null)
        thresholded[key] = {
            "delta_C": delta,
            "p_one_sided_uncorrected": conservative_p(null, delta),
            "null": null_summary(null),
        }
    observed_deltas = np.asarray(observed_deltas)
    max_null = np.column_stack(null_deltas).max(axis=1)
    for index, fraction in enumerate(FRACTIONS):
        key = f"top_{int(round(fraction * 100)):02d}"
        thresholded[key]["p_one_sided_max_fwer"] = conservative_p(max_null, observed_deltas[index])
    best = int(np.argmax(observed_deltas))

    sensitivities = {}
    definitions = {
        "threshold_free_signed": (("threshold_free_signed",), "threshold_free_signed"),
        "conditional_primary_top_10": (("conditional_axis", "primary_top_10"), "conditional_primary_top_10"),
        "conditional_threshold_free_signed": (("conditional_axis", "threshold_free_signed"), "conditional_threshold_free_signed"),
    }
    for label, (path, null_key) in definitions.items():
        left, right = canonical, no_connectivity
        for item in path:
            left, right = left[item], right[item]
        delta = left["contrast_C"] - right["contrast_C"]
        null = canonical_null[null_key] - no_connectivity_null[null_key]
        sensitivities[label] = {
            "delta_C": delta,
            "p_one_sided": conservative_p(null, delta),
            "null": null_summary(null),
        }
    return {
        "interpretation": "positive delta-C means canonical exceeds matched no-connectivity",
        "thresholded": thresholded,
        "threshold_omnibus": {
            "statistic": "max paired delta-C across top 5/10/20/30 percent",
            "observed_max_delta_C": float(observed_deltas[best]),
            "selected_threshold": float(FRACTIONS[best]),
            "p_one_sided": conservative_p(max_null, observed_deltas[best]),
            "null": null_summary(max_null),
        },
        "sensitivities": sensitivities,
    }


def legacy_reproduction(coupling: np.ndarray, observed_map: np.ndarray) -> dict:
    """Reproduce the old top-10 score, explicitly marking its omitted targets."""
    row = row_normalize(coupling)
    acronyms = RT01.mouse_acr()
    structures = sorted(s for s in set(acronyms.tolist()) if s != "NA" and np.sum(acronyms == s) >= 5)
    dys = [s for s in DYS_SET if s in structures]
    anx = [s for s in ANX_SET if s in structures]
    axis = np.array([1 if s in dys else -1 if s in anx else 0 for s in structures])
    group = np.column_stack([(acronyms == s) / np.sum(acronyms == s) for s in structures])

    def bias(weights):
        score = np.maximum((row @ weights) @ group, 0.0)
        return float(axis @ (score / score.sum()))

    dys_weights = make_signed_weights(observed_map, -1, PRIMARY_FRACTION)
    anx_weights = make_signed_weights(observed_map, +1, PRIMARY_FRACTION)
    return {
        "contrast_C": bias(dys_weights) - bias(anx_weights),
        "original_legacy_output_C_before_finite_mask_normalized_loading": 0.5850492371403845,
        "note": "contrast_C recomputes the legacy formula using the corrected finite-data loader",
        "included_dys_targets": dys,
        "included_anx_targets": anx,
        "silently_omitted_by_old_minimum_5_rule": sorted((set(DYS_SET) | set(ANX_SET)) - set(dys) - set(anx)),
        "purpose": "audit only; not used for inference",
    }


def main():
    for path in (MAP_PATH, PARCELLATION_PATH, HO_CORTEX_PATH, CANONICAL_PATH, NO_CONNECTIVITY_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    _, H, _, _ = load_inputs(ROOT)
    observed_map, sanitation = sample_sanitized_source(H)
    strata, strata_details = cortex_strata(H)
    surrogate_maps, moran_details = stratified_moran_surrogates(
        observed_map,
        H,
        strata,
        N_ROTATIONS,
        SEED,
    )

    pack_out, pack_provenance = refit_without_relevant_packs()
    coupling_arrays = {
        "canonical": np.load(CANONICAL_PATH),
        "no_relevant_anchor_packs": pack_out,
        "matched_no_connectivity": np.load(NO_CONNECTIVITY_PATH),
    }
    coupling_paths = {
        "canonical": CANONICAL_PATH,
        "no_relevant_anchor_packs": PACK_OUT_PATH,
        "matched_no_connectivity": NO_CONNECTIVITY_PATH,
    }
    acronyms = RT01.mouse_acr()

    results = {}
    nulls = {}
    for arm, coupling in coupling_arrays.items():
        results[arm] = {"weightings": {}}
        nulls[arm] = {}
        for weighting in WEIGHTINGS:
            scorer = Scorer(coupling, acronyms, weighting)
            result, arm_null = analyse_scorer(scorer, observed_map, surrogate_maps)
            results[arm]["weightings"][weighting] = result
            nulls[arm][weighting] = arm_null

    paired = {}
    for weighting in WEIGHTINGS:
        paired[weighting] = paired_tests(
            results["canonical"]["weightings"][weighting],
            results["matched_no_connectivity"]["weightings"][weighting],
            nulls["canonical"][weighting],
            nulls["matched_no_connectivity"][weighting],
        )

    coupling_provenance = {}
    for arm, coupling in coupling_arrays.items():
        path = coupling_paths[arm]
        coupling_provenance[arm] = {
            "file": str(path.relative_to(ROOT)),
            "file_sha256": sha256_file(path),
            "array_sha256": sha256_array(coupling),
            "shape": list(coupling.shape),
        }
    coupling_provenance["no_relevant_anchor_packs"]["refit"] = pack_provenance

    primary_key = "top_10"
    def compact(arm):
        value = results[arm]["weightings"][PRIMARY_WEIGHTING]["thresholded"][primary_key]
        return {
            "contrast_C": value["contrast_C"],
            "p_uncorrected": value["p_one_sided_uncorrected"],
            "p_max_fwer": value["p_one_sided_max_fwer"],
        }

    paired_primary = paired[PRIMARY_WEIGHTING]["thresholded"][primary_key]
    output = {
        "schema_version": "3.0.0",
        "analysis": {
            "name": "TMS symptom-specific circuit dissociation",
            "directional_hypothesis": "dysphoric -> mPFC/cingulate; anxiosomatic -> amygdala/insula",
            "primary_weighting": PRIMARY_WEIGHTING,
            "primary_fraction": PRIMARY_FRACTION,
            "threshold_family": list(FRACTIONS),
            "n_spatial_surrogates": N_ROTATIONS,
            "seed": SEED,
        },
        "target_sets": {
            "dysphoric_medial_prefrontal_cingulate": list(DYS_SET),
            "anxiosomatic_amygdala_insula": list(ANX_SET),
            "parcel_counts": Scorer(coupling_arrays["canonical"], acronyms, PRIMARY_WEIGHTING).parcel_counts,
            "minimum_parcels": 1,
        },
        "data": {
            "map_file": str(MAP_PATH.relative_to(ROOT)),
            "map_sha256": sha256_file(MAP_PATH),
            "map_sign": {"negative": "dysphoric", "positive": "anxiosomatic"},
            "sanitation": sanitation,
        },
        "null": {
            "role": "primary inferential null",
            "method": (
                "joint singleton Moran spectral randomization of left/right features "
                "at bilateral pair locations, stratified by cortex/noncortex"
            ),
            "spatial_weights": "symmetric inverse Euclidean distance, double centered",
            "spectrum": "all modes except the constant/zero mode",
            "strata": strata_details,
            "moran": moran_details,
            "surrogate_maps_sha256": sha256_array(surrogate_maps),
            "p_value": "upper-tail (exceedances+1)/(N+1); ties within 1e-12 relative tolerance exceed",
        },
        "provenance": {
            "script": str(Path(__file__).resolve().relative_to(ROOT)),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "nibabel", "nilearn")},
            "couplings": coupling_provenance,
        },
        "couplings": results,
        "paired_canonical_minus_no_connectivity": paired,
        "legacy_reproduction": legacy_reproduction(coupling_arrays["canonical"], observed_map),
        "headline": {
            "primary_weighting": PRIMARY_WEIGHTING,
            "canonical_primary_top_10": compact("canonical"),
            "canonical_threshold_omnibus": results["canonical"]["weightings"][PRIMARY_WEIGHTING]["threshold_omnibus"],
            "pack_out_primary_top_10": compact("no_relevant_anchor_packs"),
            "no_connectivity_primary_top_10": compact("matched_no_connectivity"),
            "paired_primary_top_10": {
                "delta_C": paired_primary["delta_C"],
                "p_uncorrected": paired_primary["p_one_sided_uncorrected"],
                "p_max_fwer": paired_primary["p_one_sided_max_fwer"],
            },
            "paired_threshold_omnibus": paired[PRIMARY_WEIGHTING]["threshold_omnibus"],
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(clean_json(output), indent=2) + "\n")

    print("TMS symptom dissociation final audit")
    print(f"map sha256 {output['data']['map_sha256']}")
    print(f"surrogate sha256 {output['null']['surrogate_maps_sha256']}")
    print(f"strata {strata_details['counts']}")
    for weighting in WEIGHTINGS:
        print(f"\n{weighting}")
        for arm in results:
            result = results[arm]["weightings"][weighting]
            primary = result["thresholded"][primary_key]
            print(
                f"  {arm}: top10 C={primary['contrast_C']:+.6f} "
                f"p={primary['p_one_sided_uncorrected']:.6f} "
                f"p_max={primary['p_one_sided_max_fwer']:.6f}; "
                f"omnibus p={result['threshold_omnibus']['p_one_sided']:.6f}; "
                f"TF p={result['threshold_free_signed']['p_one_sided']:.6f}; "
                f"conditional-TF p={result['conditional_axis']['threshold_free_signed']['p_one_sided']:.6f}"
            )
        delta = paired[weighting]["thresholded"][primary_key]
        print(
            f"  canonical-no-connectivity: top10 delta={delta['delta_C']:+.6f} "
            f"p={delta['p_one_sided_uncorrected']:.6f} "
            f"p_max={delta['p_one_sided_max_fwer']:.6f}; "
            f"omnibus p={paired[weighting]['threshold_omnibus']['p_one_sided']:.6f}"
        )
    print(f"\nwrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
