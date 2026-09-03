#!/usr/bin/env python3
"""Validate an OTTER Parkinson stage hypothesis in restricted QPN-NC data.

The script reads approved QPN-NC archives in place and writes an aggregate-only
JSON summary. It never writes participant identifiers, maps or scores.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from otter.data import load_cached  # noqa: E402

DISCOVERY_PATH = ROOT / "experiments/reverse_translation/08_pd_stage_progression.py"
spec = importlib.util.spec_from_file_location("pd_stage_progression", DISCOVERY_PATH)
pd_stage = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pd_stage)

INTEROCEPTIVE = ("VISC", "GU", "AId", "AIv", "AIp")
PRIMARY_MOTOR = ("MOp", "MOs")
EXPECTED_MISSING = {("L", "bankssts"), ("R", "bankssts")}
HY_COLUMN = "Hoehn and Yahr Stage (derived)"
THICKNESS_MEMBER = "structural_measures/ses-01/CTh_aparc_R1.csv"
GLOBAL_MEMBER = "structural_measures/ses-01/global_vol_aseg_R1.csv"
TABULAR_MEMBERS = {
    "demographics": "tabular/demographics.csv",
    "hy": "tabular/assessments/hy.csv",
    "mri": "tabular/mri_sessions.csv",
}
N_ROT = 10_000
N_BOOT = 10_000
SEED = 20260901


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_tar_csv(archive: Path, member: str) -> pd.DataFrame:
    with tarfile.open(archive, "r:*") as tf:
        handle = tf.extractfile(member)
        if handle is None:
            raise FileNotFoundError(f"{member!r} is absent from {archive}")
        with io.TextIOWrapper(handle, encoding="utf-8-sig", newline="") as text:
            return pd.read_csv(text)


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _hemi(value: object) -> str:
    token = _norm(value)
    if token in {"l", "lh", "left"}:
        return "L"
    if token in {"r", "rh", "right"}:
        return "R"
    raise ValueError(f"unrecognized hemisphere label: {value!r}")


def _is_baseline(values: pd.Series) -> pd.Series:
    return values.astype(str).str.casefold().str.contains("baseline", na=False)


def _metadata_row(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if "redcap_event_name" in work:
        work["_not_baseline"] = ~_is_baseline(work["redcap_event_name"])
        work = work.sort_values(
            ["participant_id", "_not_baseline", "redcap_event_name"], kind="stable"
        ).drop(columns="_not_baseline")
    return work.drop_duplicates("participant_id", keep="first")


def closest_assessment(frame: pd.DataFrame, outcome: str, lag: str) -> pd.DataFrame:
    required = {"participant_id", outcome, lag}
    if missing := required - set(frame):
        raise ValueError(f"assessment table missing {sorted(missing)}")
    work = frame.copy()
    work[outcome] = pd.to_numeric(work[outcome], errors="coerce")
    work[lag] = pd.to_numeric(work[lag], errors="coerce")
    work = work.loc[work[outcome].notna() & work[lag].notna()].copy()
    work["_distance"] = work[lag].abs()
    order = ["participant_id", "_distance"]
    if "redcap_event_name" in work:
        work["_not_baseline"] = ~_is_baseline(work["redcap_event_name"])
        order += ["_not_baseline", "redcap_event_name"]
    work = work.sort_values(order, kind="stable").drop_duplicates("participant_id")
    return work.drop(columns=[c for c in ("_distance", "_not_baseline") if c in work])


def load_thickness(archive: Path) -> tuple[pd.DataFrame, dict[tuple[str, str], str]]:
    long = _read_tar_csv(archive, THICKNESS_MEMBER)
    metadata = {"participant_id", "diagnosis_group_for_analysis", "hemi"}
    if missing := metadata - set(long):
        raise ValueError(f"thickness table missing {sorted(missing)}")
    long = long.copy()
    long["_hemi"] = long["hemi"].map(_hemi)
    if long.duplicated(["participant_id", "_hemi"]).any():
        raise ValueError("duplicate participant/hemisphere thickness rows")
    regions = [c for c in long if c not in metadata | {"_hemi"}]
    if len({_norm(c) for c in regions}) != len(regions):
        raise ValueError("QPN cortical labels collide after normalization")
    group = long.groupby("participant_id")["diagnosis_group_for_analysis"].agg(
        lambda x: x.dropna().astype(str).iloc[0]
    )
    pieces: list[pd.DataFrame] = []
    mapping: dict[tuple[str, str], str] = {}
    for hemi in ("L", "R"):
        side = long.loc[long["_hemi"] == hemi, ["participant_id", *regions]].copy()
        rename = {c: f"ct_{hemi}_{_norm(c)}" for c in regions}
        mapping.update({(hemi, _norm(c)): rename[c] for c in regions})
        pieces.append(side.rename(columns=rename).set_index("participant_id"))
    wide = pieces[0].join(pieces[1], how="inner").join(group.rename("diagnosis_group"))
    return wide.reset_index(), mapping


def load_qpn(archive_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    tabular = archive_dir / "tabular.tar"
    structural = archive_dir / "structural_measures.tar"
    if not tabular.exists() or not structural.exists():
        raise FileNotFoundError(f"required QPN archives are absent from {archive_dir}")
    tables = {name: _read_tar_csv(tabular, member) for name, member in TABULAR_MEMBERS.items()}
    thickness, mapping = load_thickness(structural)
    global_fs = _read_tar_csv(structural, GLOBAL_MEMBER)[
        ["participant_id", "SurfaceHoles"]
    ].drop_duplicates("participant_id")
    demo = _metadata_row(tables["demographics"])[
        ["participant_id", "sex", "diagnosis_group_for_analysis"]
    ]
    mri = _metadata_row(tables["mri"])[["participant_id", "MRI_age"]]
    base = thickness.merge(demo, on="participant_id", validate="one_to_one")
    base = base.merge(mri, on="participant_id", validate="one_to_one")
    base = base.merge(global_fs, on="participant_id", how="left", validate="one_to_one")
    if not (base["diagnosis_group"] == base["diagnosis_group_for_analysis"]).all():
        raise ValueError("diagnosis group mismatch between QPN tables")

    hy = closest_assessment(tables["hy"], HY_COLUMN, "hy_age_diff").rename(
        columns={HY_COLUMN: "hy_raw", "hy_age_diff": "hy_lag_months"}
    )
    hy_keep = ["participant_id", "hy_raw", "hy_lag_months", "hy_age"]
    if "redcap_event_name" in hy:
        hy = hy.rename(columns={"redcap_event_name": "hy_event"})
        hy_keep.append("hy_event")
    controls = base.loc[base["diagnosis_group"] == "control"].copy()
    patients = base.loc[base["diagnosis_group"] == "PD"].copy()
    patients = patients.merge(hy[hy_keep], on="participant_id", how="inner", validate="one_to_one")
    hy_values = set(pd.to_numeric(patients["hy_raw"]).unique())
    if hy_values != {1.0, 2.0, 3.0, 4.0, 5.0}:
        raise ValueError(f"unexpected H&Y values: {sorted(hy_values)}")
    patients["hy_ordinal"] = np.minimum(pd.to_numeric(patients["hy_raw"]), 4.0)
    hy_expected = 12.0 * (pd.to_numeric(patients["hy_age"]) - pd.to_numeric(patients["MRI_age"]))
    if not np.allclose(patients["hy_lag_months"], hy_expected, atol=0.02):
        raise ValueError("hy_age_diff is inconsistent with months from MRI")
    required_numeric = ["MRI_age", "hy_raw", "hy_lag_months", "SurfaceHoles"]
    if not np.isfinite(patients[required_numeric].apply(pd.to_numeric).to_numpy(float)).all():
        raise ValueError("primary QPN covariates contain non-finite values")
    if len(controls) != 69 or len(patients) != 141:
        raise ValueError(f"unexpected QPN release size: {len(controls)} controls, {len(patients)} PD")
    if patients["hy_lag_months"].abs().max() > 6:
        raise ValueError("clinical-MRI lag exceeds the frozen six-month release limit")
    audit = {
        "n_controls": int(len(controls)),
        "n_pd_with_thickness_and_clinical": int(len(patients)),
        "n_HY_ordinal_levels": int(patients["hy_ordinal"].nunique()),
        "small_stage_cells_suppressed": True,
        "max_abs_clinical_mri_lag_months": float(patients["hy_lag_months"].abs().max()),
        "n_within_one_month": int((patients["hy_lag_months"].abs() <= 1).sum()),
    }
    return controls.reset_index(drop=True), patients.reset_index(drop=True), {
        "mapping": mapping,
        "audit": audit,
    }


@dataclass(frozen=True)
class NormativeModel:
    beta: np.ndarray
    residual_sd: np.ndarray
    age_mean: float
    age_sd: float
    sex_levels: tuple[str, ...]
    qc_mean: float | None = None
    qc_sd: float | None = None


def _sex_matrix(values: pd.Series, levels: tuple[str, ...]) -> np.ndarray:
    tokens = values.astype(str).map(_norm)
    if unknown := set(tokens) - set(levels):
        raise ValueError(f"sex levels absent from controls: {sorted(unknown)}")
    return np.column_stack([(tokens == level).to_numpy(float) for level in levels[1:]])


def _normative_design(
    frame: pd.DataFrame,
    age_mean: float,
    age_sd: float,
    sex_levels: tuple[str, ...],
    qc_mean: float | None = None,
    qc_sd: float | None = None,
) -> np.ndarray:
    age = pd.to_numeric(frame["MRI_age"]).to_numpy(float)
    columns = [np.ones(len(frame)), (age - age_mean) / age_sd]
    sex = _sex_matrix(frame["sex"], sex_levels)
    if sex.size:
        columns.extend(sex.T)
    if qc_mean is not None and qc_sd is not None:
        qc = np.log1p(pd.to_numeric(frame["SurfaceHoles"]).to_numpy(float))
        columns.append((qc - qc_mean) / qc_sd)
    return np.column_stack(columns)


def fit_normative(
    controls: pd.DataFrame, region_columns: list[str], include_qc: bool = False
) -> NormativeModel:
    age = pd.to_numeric(controls["MRI_age"]).to_numpy(float)
    age_mean, age_sd = float(age.mean()), float(age.std(ddof=1))
    if len(controls) < 30 or not np.isfinite(age_sd) or age_sd <= 0:
        raise ValueError("invalid normative-control sample")
    sex_levels = tuple(sorted(controls["sex"].astype(str).map(_norm).unique()))
    qc_mean = qc_sd = None
    if include_qc:
        qc = np.log1p(pd.to_numeric(controls["SurfaceHoles"]).to_numpy(float))
        qc_mean, qc_sd = float(qc.mean()), float(qc.std(ddof=1))
        if not np.isfinite(qc_sd) or qc_sd <= 0:
            raise ValueError("invalid SurfaceHoles variance")
    X = _normative_design(controls, age_mean, age_sd, sex_levels, qc_mean, qc_sd)
    Y = controls[region_columns].apply(pd.to_numeric).to_numpy(float)
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    residual = Y - X @ beta
    dof = len(controls) - np.linalg.matrix_rank(X)
    residual_sd = np.sqrt(np.sum(residual**2, axis=0) / dof)
    if np.any(~np.isfinite(residual_sd)) or np.any(residual_sd <= 0):
        raise ValueError("invalid control residual variance")
    return NormativeModel(beta, residual_sd, age_mean, age_sd, sex_levels, qc_mean, qc_sd)


def control_normative_atrophy(
    frame: pd.DataFrame, region_columns: list[str], model: NormativeModel
) -> np.ndarray:
    X = _normative_design(
        frame, model.age_mean, model.age_sd, model.sex_levels, model.qc_mean, model.qc_sd
    )
    observed = frame[region_columns].apply(pd.to_numeric).to_numpy(float)
    return (X @ model.beta - observed) / model.residual_sd


def _severity_design(
    frame: pd.DataFrame, outcome: str, lag: str, include_qc: bool = False
) -> np.ndarray:
    age = pd.to_numeric(frame["MRI_age"]).to_numpy(float)
    lag_values = pd.to_numeric(frame[lag]).to_numpy(float)
    severity = pd.to_numeric(frame[outcome]).to_numpy(float)
    sex_levels = tuple(sorted(frame["sex"].astype(str).map(_norm).unique()))
    columns = [np.ones(len(frame)), severity, (age - age.mean()) / age.std(ddof=1)]
    sex = _sex_matrix(frame["sex"], sex_levels)
    if sex.size:
        columns.extend(sex.T)
    lag_sd = lag_values.std(ddof=1)
    columns.append((lag_values - lag_values.mean()) / lag_sd if lag_sd > 0 else np.zeros(len(frame)))
    if include_qc:
        qc = np.log1p(pd.to_numeric(frame["SurfaceHoles"]).to_numpy(float))
        columns.append((qc - qc.mean()) / qc.std(ddof=1))
    return np.column_stack(columns)


def severity_slope_map(
    atrophy: np.ndarray,
    frame: pd.DataFrame,
    outcome: str,
    lag: str,
    *,
    robust: bool = False,
    include_qc: bool = False,
) -> np.ndarray:
    X = _severity_design(frame, outcome, lag, include_qc)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("rank-deficient severity design")
    if not robust:
        return np.linalg.lstsq(X, atrophy, rcond=None)[0][1]
    return np.array(
        [
            sm.RLM(atrophy[:, j], X, M=sm.robust.norms.HuberT()).fit().params[1]
            for j in range(atrophy.shape[1])
        ]
    )


def zscore_map(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    sd = values.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("regional slope map has zero or invalid variance")
    return (values - values.mean()) / sd


def mouse_acronyms() -> np.ndarray:
    meta = json.loads((ROOT / "data_external/mouse_sc_meta.json").read_text())
    return np.array(
        [meta["structure_acronyms"][i] if i >= 0 else "NA" for i in meta["node_struct_idx"]]
    )


def contrast_weight(pi: np.ndarray, acronyms: np.ndarray, masks, keys) -> np.ndarray:
    row = pi / pi.sum(axis=1, keepdims=True).clip(1e-12)
    human = np.mean(
        [row[acronyms == structure].mean(axis=0) for structure in INTEROCEPTIVE], axis=0
    ) - np.mean(
        [row[acronyms == structure].mean(axis=0) for structure in PRIMARY_MOTOR], axis=0
    )
    return np.array([human[masks[key]].sum() for key in keys], float)


def spatial_summary(
    slope: np.ndarray,
    weight: np.ndarray,
    rotations: np.ndarray,
    keys: list[tuple[str, str]],
    *,
    scaled: bool = True,
) -> dict:
    slope = np.asarray(slope, float)
    use = zscore_map(slope) if scaled else slope - slope.mean()
    observed = float(weight @ use)
    null = use[rotations] @ weight
    p_upper = float((np.sum(null >= observed) + 1) / (len(null) + 1))
    p_lower = float((np.sum(null <= observed) + 1) / (len(null) + 1))
    hemispheres = {}
    for hemi in ("L", "R"):
        idx = np.array([key[0] == hemi for key in keys])
        hemi_values = slope[idx]
        hemi_map = zscore_map(hemi_values) if scaled else hemi_values - hemi_values.mean()
        hemispheres[hemi] = float(weight[idx] @ hemi_map)
    return {
        "D": observed,
        "spatial_p_one_sided": p_upper,
        "spatial_p_two_sided": min(1.0, 2.0 * min(p_upper, p_lower)),
        "spatial_null_ci95": np.quantile(null, [0.025, 0.975]).tolist(),
        "hemisphere_D": hemispheres,
    }


def endpoint_once(
    controls: pd.DataFrame,
    patients: pd.DataFrame,
    region_columns: list[str],
    outcome: str,
    lag: str,
    weight: np.ndarray,
    rotations: np.ndarray,
    keys: list[tuple[str, str]],
    *,
    robust: bool = False,
    include_qc: bool = False,
    scaled: bool = True,
) -> tuple[dict, np.ndarray]:
    normative = fit_normative(controls, region_columns, include_qc=include_qc)
    atrophy = control_normative_atrophy(patients, region_columns, normative)
    slope = severity_slope_map(
        atrophy, patients, outcome, lag, robust=robust, include_qc=include_qc
    )
    return spatial_summary(slope, weight, rotations, keys, scaled=scaled), slope


def bootstrap_D(
    controls: pd.DataFrame,
    patients: pd.DataFrame,
    region_columns: list[str],
    outcome: str,
    lag: str,
    weight: np.ndarray,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, float)
    completed = attempts = 0
    while completed < n_boot:
        attempts += 1
        if attempts > n_boot * 2:
            raise RuntimeError("too many degenerate bootstrap samples")
        c = controls.iloc[rng.integers(0, len(controls), len(controls))].reset_index(drop=True)
        p = patients.iloc[rng.integers(0, len(patients), len(patients))].reset_index(drop=True)
        try:
            model = fit_normative(c, region_columns)
            maps = control_normative_atrophy(p, region_columns, model)
            slope = severity_slope_map(maps, p, outcome, lag)
            draws[completed] = float(weight @ zscore_map(slope))
            completed += 1
        except (ValueError, np.linalg.LinAlgError):
            continue
    return draws


def leave_one_out_ranges(
    controls: pd.DataFrame,
    patients: pd.DataFrame,
    region_columns: list[str],
    outcome: str,
    lag: str,
    weight: np.ndarray,
) -> dict:
    model = fit_normative(controls, region_columns)
    maps = control_normative_atrophy(patients, region_columns, model)
    pd_values = []
    for i in range(len(patients)):
        keep = np.arange(len(patients)) != i
        frame = patients.loc[keep].reset_index(drop=True)
        slope = severity_slope_map(maps[keep], frame, outcome, lag)
        pd_values.append(float(weight @ zscore_map(slope)))
    control_values = []
    for i in range(len(controls)):
        c = controls.drop(index=i).reset_index(drop=True)
        model = fit_normative(c, region_columns)
        maps_i = control_normative_atrophy(patients, region_columns, model)
        slope = severity_slope_map(maps_i, patients, outcome, lag)
        control_values.append(float(weight @ zscore_map(slope)))
    return {
        "leave_one_pd": {
            "n": len(pd_values),
            "min_D": float(np.min(pd_values)),
            "max_D": float(np.max(pd_values)),
            "all_positive": bool(np.min(pd_values) > 0),
        },
        "leave_one_control": {
            "n": len(control_values),
            "min_D": float(np.min(control_values)),
            "max_D": float(np.max(control_values)),
            "all_positive": bool(np.min(control_values) > 0),
        },
    }


def analyze_endpoint(
    controls: pd.DataFrame,
    patients: pd.DataFrame,
    region_columns: list[str],
    outcome: str,
    lag: str,
    weight: np.ndarray,
    rotations: np.ndarray,
    keys: list[tuple[str, str]],
    n_boot: int,
    seed: int,
) -> tuple[dict, np.ndarray]:
    summary, slope = endpoint_once(
        controls, patients, region_columns, outcome, lag, weight, rotations, keys
    )
    boot = bootstrap_D(
        controls, patients, region_columns, outcome, lag, weight, n_boot, seed
    )
    summary.update(
        {
            "n": int(len(patients)),
            "bootstrap_n": int(len(boot)),
            "bootstrap_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "bootstrap_median": float(np.median(boot)),
            "influence": leave_one_out_ranges(
                controls, patients, region_columns, outcome, lag, weight
            ),
        }
    )
    return summary, slope


def covariate_matrix(frame: pd.DataFrame, lag: str) -> np.ndarray:
    """Return age, assessment lag and dummy-coded sex covariates."""
    age = pd.to_numeric(frame["MRI_age"]).to_numpy(float)
    lag_values = pd.to_numeric(frame[lag]).to_numpy(float)
    sex = frame["sex"].astype(str).map(_norm)
    levels = sorted(sex.unique())
    columns = [age, lag_values]
    columns.extend((sex == level).to_numpy(float) for level in levels[1:])
    return np.column_stack(columns)


def partial_spearman(x: np.ndarray, y: np.ndarray, covariates: np.ndarray) -> float:
    """Partial Spearman correlation using residualized ranks."""
    rx = rankdata(np.asarray(x, float), method="average")
    ry = rankdata(np.asarray(y, float), method="average")
    ranked_covariates = np.column_stack(
        [rankdata(covariates[:, i], method="average") for i in range(covariates.shape[1])]
    )
    design = np.column_stack([np.ones(len(x)), ranked_covariates])
    rx -= design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ry -= design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def participant_observed_null(
    controls: pd.DataFrame,
    patients: pd.DataFrame,
    region_columns: list[str],
    weight: np.ndarray,
    rotations: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Participant score association and its synchronized spatial null."""
    model = fit_normative(controls, region_columns)
    maps = control_normative_atrophy(patients, region_columns, model)
    maps = (maps - maps.mean(axis=1, keepdims=True)) / maps.std(
        axis=1, ddof=1, keepdims=True
    )
    outcome = patients["hy_ordinal"].to_numpy(float)
    covariates = covariate_matrix(patients, "hy_lag_months")
    observed = partial_spearman(maps @ weight, outcome, covariates)
    null = np.empty(len(rotations), float)
    for start in range(0, len(rotations), 250):
        use = rotations[start : start + 250]
        scores = np.einsum("nkr,r->nk", maps[:, use], weight, optimize=True)
        for column in range(scores.shape[1]):
            null[start + column] = partial_spearman(
                scores[:, column], outcome, covariates
            )
    return observed, null


def regional_observed_null(
    slope: np.ndarray, weight: np.ndarray, rotations: np.ndarray
) -> tuple[float, np.ndarray]:
    values = zscore_map(slope)
    return float(weight @ values), values[rotations] @ weight


def synchronized_stage_analysis(
    controls: pd.DataFrame,
    patients: pd.DataFrame,
    region_columns: list[str],
    weight: np.ndarray,
    rotations: np.ndarray,
    keys: list[tuple[str, str]],
) -> dict:
    """Evaluate related stage formulations with one synchronized spatial null."""
    model = fit_normative(controls, region_columns)
    atrophy = control_normative_atrophy(patients, region_columns, model)
    frame = patients.copy()
    frame["advanced_HY3plus"] = (frame["hy_raw"] >= 3).astype(float)
    definitions = [
        ("HY4_5_OLS", "hy_ordinal", False),
        ("HY1_5_OLS", "hy_raw", False),
        ("HY3plus_vs_HY1_2_OLS", "advanced_HY3plus", False),
        ("HY4_5_Huber", "hy_ordinal", True),
    ]
    names: list[str] = []
    observed: list[float] = []
    nulls: list[np.ndarray] = []
    components: dict[str, dict] = {}
    for name, outcome, robust in definitions:
        slope = severity_slope_map(
            atrophy, frame, outcome, "hy_lag_months", robust=robust
        )
        value, null = regional_observed_null(slope, weight, rotations)
        names.append(name)
        observed.append(value)
        nulls.append(null)
        summary = spatial_summary(slope, weight, rotations, keys)
        summary["n"] = int(len(frame))
        components[name] = summary

    participant, participant_null = participant_observed_null(
        controls, frame, region_columns, weight, rotations
    )
    names.append("participant_partial_spearman")
    observed.append(participant)
    nulls.append(participant_null)
    participant_upper = float(
        (np.sum(participant_null >= participant) + 1) / (len(participant_null) + 1)
    )
    components["participant_partial_spearman"] = {
        "rho": participant,
        "spatial_p_one_sided": participant_upper,
        "spatial_null_ci95": np.quantile(
            participant_null, [0.025, 0.975]
        ).tolist(),
        "n": int(len(frame)),
    }

    observed_array = np.asarray(observed, float)
    null_matrix = np.column_stack(nulls)
    means = null_matrix.mean(axis=0)
    sds = null_matrix.std(axis=0, ddof=1)
    observed_z = (observed_array - means) / sds
    null_z = (null_matrix - means) / sds
    observed_max = float(observed_z.max())
    return {
        "components": components,
        "synchronized_max_statistic": {
            "statistic": observed_max,
            "spatial_p_one_sided": float(
                (np.sum(null_z.max(axis=1) >= observed_max) + 1)
                / (len(null_z) + 1)
            ),
            "n_components": len(names),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive-dir", type=Path, default=PROJECT_ROOT / "data_external/qpn-nc-r01"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/logs/reverse_translation_qpn_pd_stage.json",
    )
    parser.add_argument("--n-rot", type=int, default=N_ROT)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    controls, patients, loaded = load_qpn(args.archive_dir)
    H, _ = load_cached("human", cache_dir=ROOT / "outputs/anndata")
    masks, centroids = pd_stage.desikan_masks(H)
    available = set(loaded["mapping"])
    missing = set(masks) - available
    if missing != EXPECTED_MISSING:
        raise ValueError(f"expected only bilateral bankssts missing; found {sorted(missing)}")
    keys = sorted(set(masks) & available)
    if len(keys) != 50:
        raise ValueError(f"frozen intersection must contain 50 regions, found {len(keys)}")
    region_columns = [loaded["mapping"][key] for key in keys]
    all_values = pd.concat([controls[region_columns], patients[region_columns]]).to_numpy(float)
    if not np.isfinite(all_values).all() or np.any((all_values < 0.1) | (all_values > 10)):
        raise ValueError("QPN thickness contains missing or implausible values")

    pi_path = ROOT / "outputs/coupling/pi_canonical.npy"
    weight = contrast_weight(np.load(pi_path), mouse_acronyms(), masks, keys)
    rotations = pd_stage.mirrored_bijective_rotations(keys, centroids, args.n_rot, SEED)
    result = {
        "schema_version": "1.0.0",
        "status": "validated_inputs_only" if args.validate_only else "completed",
        "analysis": "QPN-NC Parkinson stage validation",
        "hypothesis": (
            "Increasing Parkinson stage preferentially increases OTTER-translated "
            "mouse interoceptive-versus-primary-motor bias."
        ),
        "design": (
            "Control-normative cortical thinning with regional stage models adjusted "
            "for MRI age, sex and clinical-MRI interval."
        ),
        "directional_alternative": "interoceptive_minus_primary_motor > 0",
        "stage_encoding": "Hoehn-Yahr 1, 2, 3 and combined 4/5",
        "cross_sectional": True,
        "seed": SEED,
        "n_rotations": int(args.n_rot),
        "n_bootstrap": int(args.n_boot),
        "dataset": {
            "name": "Quebec Parkinson Network Neuroimaging Cohort (QPN-NC)",
            "doi": "10.5281/zenodo.17246063",
            "access": "restricted; obtained directly from the data owners",
        },
        "coupling": {
            "pi_file": str(pi_path.relative_to(ROOT)),
            "pi_sha256": sha256(pi_path),
        },
        "atlas": {
            "n_regions": len(keys),
            "n_left": sum(key[0] == "L" for key in keys),
            "n_right": sum(key[0] == "R" for key in keys),
        },
        "sample": {"controls": int(len(controls)), "PD": int(len(patients))},
        "privacy": {
            "participant_identifiers_written": False,
            "participant_maps_or_scores_written": False,
            "small_stage_cell_counts_written": False,
            "restricted_file_manifest_written": False,
        },
    }
    if args.validate_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    primary, primary_slope = analyze_endpoint(
        controls,
        patients,
        region_columns,
        "hy_ordinal",
        "hy_lag_months",
        weight,
        rotations,
        keys,
        args.n_boot,
        SEED + 1,
    )
    del primary_slope
    synchronized = synchronized_stage_analysis(
        controls, patients, region_columns, weight, rotations, keys
    )
    result.update(
        {
            "HY4_5_OLS": primary,
            "stage_formulations": synchronized["components"],
            "synchronized_max_statistic": synchronized[
                "synchronized_max_statistic"
            ],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
