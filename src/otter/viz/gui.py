"""Region-first static GUI for exploring OTTER couplings.

This module builds a self-contained HTML application plus a JSON sidecar.
It is intentionally static: all expensive work happens in Python, and the
browser only searches, aggregates top-K rows, and renders browser-side
3D views from precomputed mesh/point payloads.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from otter.data.anchors import get_anchor_index
from otter.data.networks import NETWORKS, assign_networks
from otter.viz.viewer import col_entropy, row_entropy, topk_per_col, topk_per_row


def _as_float_list(x) -> list[float]:
    return [float(v) for v in np.asarray(x, dtype=float).tolist()]


def _as_str_list(x) -> list[str]:
    return [str(v) for v in list(x)]


def _rounded_float_list(x, ndigits: int = 3) -> list[float]:
    arr = np.asarray(x, dtype=float).ravel()
    return [float(round(float(v), ndigits)) for v in arr.tolist()]


def _var_text(var, name: str, n: int, default: str = "") -> list[str]:
    if name not in var:
        return [default] * n
    # .astype(object) first: several var columns load from h5ad as pandas
    # Categorical, and .fillna() on a Categorical raises on modern pandas
    # (>=2.0) when the fill value is not already a category. Demoting to
    # object dtype makes fillna accept any value; behaviour is unchanged
    # when the column has no missing entries.
    return var[name].astype(object).fillna(default).astype(str).tolist()


def _short_text(text: str, max_len: int = 140) -> str:
    text = str(text or "")
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def _load_npz_array(path: Optional[str | Path], key: str, n: int, default) -> np.ndarray:
    if path is None or not Path(path).exists():
        return np.full(n, default, dtype=object if isinstance(default, str) else float)
    z = np.load(path, allow_pickle=True)
    if key not in z.files:
        return np.full(n, default, dtype=object if isinstance(default, str) else float)
    arr = z[key]
    if arr.shape != (n,):
        return np.full(n, default, dtype=object if isinstance(default, str) else float)
    return arr


def _node_records(ad, *, trust_path: Optional[str | Path] = None, species: str) -> dict:
    var = ad.var
    idx = get_anchor_index(var)
    nets_int = assign_networks(var, idx)
    networks = [NETWORKS[int(i)] for i in nets_int]
    n = len(var)

    records = {
        "ids": _as_str_list(var.index),
        "x": _as_float_list(var["x"]),
        "y": _as_float_list(var["y"]),
        "z": _as_float_list(var["z"]),
        "region": [_short_text(v, 180) for v in _var_text(var, "region", n)],
        "subregion": [_short_text(v, 220) for v in _var_text(var, "subregion", n)],
        "hemisphere": _as_str_list(var["hemisphere"]),
        "network": networks,
        "is_garin_anchor": var["garin_anchor"].astype(bool).tolist(),
        "pairid": [int(v) for v in var["pairid"].astype(int).tolist()],
        "anchor_pair_id": [
            int(v) if str(v) not in ("<NA>", "nan", "None") else 0
            # .astype(object) before fillna: anchor_pair_id loads as a
            # Categorical, and fillna() on a Categorical raises on pandas >=2.0
            # unless the fill value is already a category.
            for v in var["anchor_pair_id"].astype(object).fillna(0).astype(int).tolist()
        ],
    }
    if species == "mouse":
        records["evidence_tier"] = _as_str_list(
            _load_npz_array(trust_path, "evidence_tier", n, "unknown")
        )
        records["trust"] = [
            float(v) for v in _load_npz_array(trust_path, "trust", n, np.nan).astype(float)
        ]
        records["pack_anchored"] = [
            bool(v) for v in _load_npz_array(trust_path, "pack_anchored", n, False)
        ]
        records["garin_anchored"] = [
            bool(v) for v in _load_npz_array(trust_path, "garin_anchored", n, False)
        ]
        records["beauchamp_top1"] = [
            float(v) for v in _load_npz_array(trust_path, "beauchamp_top1", n, np.nan).astype(float)
        ]
    return records


def _group_by_labels(labels: Sequence[str], *, kind: str, prefix: str, min_size: int = 2) -> list[dict]:
    groups: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        label = str(label or "").strip()
        if not label or label.lower() == "nan":
            continue
        groups.setdefault(label, []).append(i)
    out = []
    for label, idxs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0].lower())):
        if len(idxs) < min_size:
            continue
        out.append({
            "id": f"{prefix}:{len(out)}",
            "kind": kind,
            "label": _short_text(label, 180),
            "indices": idxs,
            "n": len(idxs),
        })
    return out


def _anchor_entry_groups(entries: Optional[Sequence[Any]]) -> list[dict]:
    if not entries:
        return []
    groups = []
    for e in entries:
        groups.append({
            "id": f"anchor_pack:{int(e.pair_id)}",
            "kind": "anchor_pack",
            "label": _short_text(f"pid {int(e.pair_id)} - {e.label}", 180),
            "indices": [int(i) for i in e.mouse_indices],
            "human_indices": [int(i) for i in e.human_indices],
            "n": int(len(e.mouse_indices)),
        })
    return groups


def _build_groups(mouse_records: dict, human_records: dict, anchor_entries=None) -> dict:
    mouse_groups = [{"id": "all_mouse", "kind": "all", "label": "All mouse parcels",
                     "indices": list(range(len(mouse_records["ids"]))),
                     "n": len(mouse_records["ids"])}]
    mouse_groups.extend(_group_by_labels(mouse_records["network"], kind="network", prefix="mouse_network"))
    mouse_groups.extend(_group_by_labels(mouse_records["subregion"], kind="subregion", prefix="mouse_subregion"))
    mouse_groups.extend(_group_by_labels(mouse_records.get("evidence_tier", []),
                                         kind="evidence", prefix="mouse_evidence"))
    mouse_groups.extend(_anchor_entry_groups(anchor_entries))

    human_groups = [{"id": "all_human", "kind": "all", "label": "All human parcels",
                     "indices": list(range(len(human_records["ids"]))),
                     "n": len(human_records["ids"])}]
    human_groups.extend(_group_by_labels(human_records["network"], kind="network", prefix="human_network"))
    human_groups.extend(_group_by_labels(human_records["subregion"], kind="subregion", prefix="human_subregion"))

    return {"mouse": mouse_groups, "human": human_groups}


def _load_region_eval(path: Optional[str | Path]) -> dict:
    if path is None or not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text())
    return {
        "aggregate": data.get("aggregate", {}),
        "anchor_overlapping": data.get("anchor_overlapping", {}),
        "novel": data.get("novel", {}),
        "n_candidates": data.get("n_candidates"),
        "n_pairs_evaluated": data.get("n_pairs_evaluated"),
    }


def _normalise_model_spec(spec: dict, root: Path) -> dict:
    out = dict(spec)
    if "id" not in out:
        stem = Path(out.get("pi_file", "model")).stem
        out["id"] = stem.replace("pi_", "")
    if "label" not in out:
        out["label"] = out["id"].replace("_", " ")
    if "pi" not in out:
        if "pi_file" not in out:
            raise ValueError("model spec needs either 'pi' or 'pi_file'")
        path = Path(out["pi_file"])
        if not path.is_absolute():
            path = root / path
        out["pi_path"] = str(path)
        out["pi"] = np.load(path)
    return out


def _empty_visual_layers() -> dict:
    return {
        "human_surface": {
            "available": False,
            "source": "",
            "message": "Human surface layer was not built.",
        },
        "mouse_shell": {
            "available": False,
            "source": "",
            "message": "Mouse shell layer was not built.",
        },
    }


def _mesh_payload(coords: np.ndarray, faces: np.ndarray, *, ndigits: int = 3) -> dict:
    faces = np.asarray(faces, dtype=np.int64)
    coords = np.asarray(coords, dtype=float)
    return {
        "x": _rounded_float_list(coords[:, 0], ndigits),
        "y": _rounded_float_list(coords[:, 1], ndigits),
        "z": _rounded_float_list(coords[:, 2], ndigits),
        "i": [int(v) for v in faces[:, 0].tolist()],
        "j": [int(v) for v in faces[:, 1].tolist()],
        "k": [int(v) for v in faces[:, 2].tolist()],
    }


def _build_human_surface_layer(
    human_ad,
    *,
    stencil_k: int = 14,
    max_distance_mm: float = 18.0,
    sigma_mm: float = 7.0,
) -> dict:
    """Build a Nilearn fsaverage5 surface plus parcel-to-surface stencils.

    The GUI uses these sparse stencils to spread parcel-level coupling mass
    over nearby cortical mesh vertices in the browser. Parcels far from the
    surface are left as off-surface points.
    """
    try:
        from nilearn import datasets, surface
        from scipy.spatial import cKDTree
    except Exception as exc:  # pragma: no cover - depends on optional packages
        return {
            "available": False,
            "source": "nilearn fsaverage5",
            "message": f"Nilearn surface layer unavailable: {exc}",
        }

    try:
        fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
        left = surface.load_surf_mesh(fsavg["pial_left"])
        right = surface.load_surf_mesh(fsavg["pial_right"])
        coords_l = np.asarray(left.coordinates, dtype=float)
        faces_l = np.asarray(left.faces, dtype=np.int64)
        coords_r = np.asarray(right.coordinates, dtype=float)
        faces_r = np.asarray(right.faces, dtype=np.int64) + len(coords_l)
        coords = np.vstack([coords_l, coords_r])
        faces = np.vstack([faces_l, faces_r])

        trees = {
            "L": (cKDTree(coords_l), 0),
            "R": (cKDTree(coords_r), len(coords_l)),
        }
        xyz = human_ad.var[["x", "y", "z"]].to_numpy(dtype=float)
        hemi = human_ad.var["hemisphere"].astype(str).tolist()

        stencils: list[list[list[float]]] = []
        distances: list[float] = []
        for point, h in zip(xyz, hemi):
            tree, offset = trees.get(h, trees["L"])
            d, idx = tree.query(point, k=stencil_k)
            d = np.atleast_1d(d).astype(float)
            idx = np.atleast_1d(idx).astype(int)
            valid = np.isfinite(d) & (d <= max_distance_mm)
            if not np.any(valid):
                stencils.append([])
                distances.append(float(round(float(np.nanmin(d)), 3)))
                continue
            d = d[valid]
            idx = idx[valid]
            weights = np.exp(-(d ** 2) / (2.0 * sigma_mm ** 2))
            weights = weights / weights.sum()
            stencils.append([
                [int(i + offset), float(round(float(w), 5))]
                for i, w in zip(idx.tolist(), weights.tolist())
            ])
            distances.append(float(round(float(d.min()), 3)))

        return {
            "available": True,
            "source": "nilearn fsaverage5 pial surface",
            "n_vertices": int(len(coords)),
            "n_faces": int(len(faces)),
            "stencil_k": int(stencil_k),
            "max_distance_mm": float(max_distance_mm),
            **_mesh_payload(coords, faces, ndigits=2),
            "parcel_stencil": stencils,
            "surface_distance_mm": distances,
        }
    except Exception as exc:  # pragma: no cover - depends on local data/runtime
        return {
            "available": False,
            "source": "nilearn fsaverage5",
            "message": f"Nilearn surface layer failed: {exc}",
        }


def _build_parcel_stencils(
    parcel_xyz: np.ndarray,
    vertex_coords: np.ndarray,
    *,
    stencil_k: int,
    max_distance_mm: float,
    sigma_mm: float,
    vertex_index_offset: int = 0,
) -> tuple[list[list[list[float]]], list[float]]:
    """Build sparse per-parcel vertex stencils using a kd-tree of mesh vertices.

    Each stencil is a list of ``[vertex_index, weight]`` pairs covering the
    nearest ``stencil_k`` vertices that lie within ``max_distance_mm``. Weights
    are Gaussian in distance and normalised to sum to one. Parcels whose
    nearest mesh vertex exceeds ``max_distance_mm`` get an empty stencil and
    are rendered as off-surface points in the GUI.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(np.asarray(vertex_coords, dtype=float))
    stencils: list[list[list[float]]] = []
    distances: list[float] = []
    for point in parcel_xyz:
        d, idx = tree.query(point, k=stencil_k)
        d = np.atleast_1d(d).astype(float)
        idx = np.atleast_1d(idx).astype(int)
        valid = np.isfinite(d) & (d <= max_distance_mm)
        if not np.any(valid):
            stencils.append([])
            nearest = float(np.nanmin(d)) if np.isfinite(d).any() else float("nan")
            distances.append(float(round(nearest, 3)) if np.isfinite(nearest) else nearest)
            continue
        d = d[valid]
        idx = idx[valid]
        weights = np.exp(-(d ** 2) / (2.0 * sigma_mm ** 2))
        weights = weights / weights.sum()
        stencils.append([
            [int(i + vertex_index_offset), float(round(float(w), 5))]
            for i, w in zip(idx.tolist(), weights.tolist())
        ])
        distances.append(float(round(float(d.min()), 3)))
    return stencils, distances


def _build_mouse_shell_layer(
    mouse_ad,
    *,
    root: str | Path = ".",
    stencil_k: int = 12,
    max_distance_mm: float = 2.0,
    sigma_mm: float = 0.8,
) -> dict:
    """Build a transparent mouse brain shell from the DSURQE atlas when present.

    The atlas surface provides anatomical context for the interactive GUI;
    selected groups are rendered as opaque browser-side alpha hulls/markers on
    top of it. If the external DSURQE files are missing, fall back to a parcel
    convex hull so the mode remains usable on fresh checkouts.

    A sparse per-parcel vertex stencil is attached (analogous to the human
    pial layer) so selections can paint a heat overlay onto the mouse shell.
    Defaults are tuned for the DSURQE CCFv3 frame (mm-scale, whole-brain).
    """
    root = Path(root)
    mask_path = (
        root
        / "data_external/MouseHumanTranscriptomicSimilarity/AMBA/data/imaging/"
          "DSURQE_CCFv3_mask_200um.mnc"
    )
    parcel_xyz = mouse_ad.var[["x", "y", "z"]].to_numpy(dtype=float)

    if mask_path.exists():
        try:
            import nibabel as nib
            from skimage import measure
            from otter.data.anchor_packs._dsurqe import DSURQE_OFFSET_MM

            img = nib.load(str(mask_path))
            mask = np.asarray(img.get_fdata()) > 0
            verts, faces, _, _ = measure.marching_cubes(
                mask.astype(np.float32), level=0.5, step_size=1
            )
            world = (img.affine @ np.c_[verts, np.ones(len(verts))].T).T[:, :3]
            coords = world - DSURQE_OFFSET_MM
            try:
                stencils, distances = _build_parcel_stencils(
                    parcel_xyz,
                    coords,
                    stencil_k=stencil_k,
                    max_distance_mm=max_distance_mm,
                    sigma_mm=sigma_mm,
                )
            except Exception:
                stencils, distances = [], []
            return {
                "available": True,
                "source": "DSURQE CCFv3 200um atlas mask surface",
                "n_vertices": int(len(coords)),
                "n_faces": int(len(faces)),
                "stencil_k": int(stencil_k),
                "max_distance_mm": float(max_distance_mm),
                **_mesh_payload(coords, faces, ndigits=3),
                "parcel_stencil": stencils,
                "surface_distance_mm": distances,
            }
        except Exception:
            # Continue to the parcel hull fallback below.
            pass

    try:
        from scipy.spatial import ConvexHull
    except Exception as exc:  # pragma: no cover - scipy is a core dependency
        return {
            "available": False,
            "source": "scipy.spatial.ConvexHull",
            "message": f"Mouse shell unavailable: {exc}",
        }

    try:
        if len(parcel_xyz) < 4:
            raise ValueError("Need at least four mouse parcels to build a shell")
        hull = ConvexHull(parcel_xyz)
        faces = np.asarray(hull.simplices, dtype=np.int64)
        # Hull fallback: each parcel maps directly onto its own vertex so the
        # stencil is trivial. Still emit it for code-path symmetry with the
        # DSURQE branch.
        stencils = [[[int(i), 1.0]] for i in range(len(parcel_xyz))]
        distances = [0.0] * len(parcel_xyz)
        return {
            "available": True,
            "source": "scipy ConvexHull shell from mouse parcel coordinates",
            "n_vertices": int(len(parcel_xyz)),
            "n_faces": int(len(faces)),
            "stencil_k": 1,
            "max_distance_mm": 0.0,
            **_mesh_payload(parcel_xyz, faces, ndigits=3),
            "parcel_stencil": stencils,
            "surface_distance_mm": distances,
        }
    except Exception as exc:
        return {
            "available": False,
            "source": "scipy.spatial.ConvexHull",
            "message": f"Mouse shell failed: {exc}",
        }


def build_visual_layers(mouse_ad, human_ad, *, root: str | Path = ".") -> dict:
    """Build optional mesh layers used by richer GUI view modes."""
    return {
        "human_surface": _build_human_surface_layer(human_ad),
        "mouse_shell": _build_mouse_shell_layer(mouse_ad, root=root),
    }


def build_gui_payload(
    model_specs: Sequence[dict],
    mouse_ad,
    human_ad,
    *,
    top_k: int = 50,
    trust_path: Optional[str | Path] = None,
    anchor_entries: Optional[Sequence[Any]] = None,
    include_visual_layers: bool = False,
    visual_layers: Optional[dict] = None,
    root: str | Path = ".",
) -> dict:
    """Build the JSON payload consumed by the static GUI.

    ``model_specs`` accepts dictionaries with:
      - ``id`` and ``label``
      - ``pi`` ndarray or ``pi_file`` path
      - optional ``region_eval_file`` path
    """
    root = Path(root)
    mouse = _node_records(mouse_ad, trust_path=trust_path, species="mouse")
    human = _node_records(human_ad, species="human")
    groups = _build_groups(mouse, human, anchor_entries=anchor_entries)

    models = []
    for raw in model_specs:
        spec = _normalise_model_spec(raw, root)
        pi = np.asarray(spec["pi"], dtype=np.float64)
        if pi.shape != (len(mouse["ids"]), len(human["ids"])):
            raise ValueError(
                f"{spec['id']} pi shape {pi.shape} != "
                f"({len(mouse['ids'])}, {len(human['ids'])})"
            )
        region_eval_file = spec.get("region_eval_file")
        if region_eval_file and not Path(region_eval_file).is_absolute():
            region_eval_file = root / region_eval_file
        models.append({
            "id": str(spec["id"]),
            "label": str(spec["label"]),
            "pi_file": str(spec.get("pi_file", spec.get("pi_path", "(in-memory)"))),
            "top_k": int(top_k),
            "top_mouse_to_human": topk_per_row(pi, top_k),
            "top_human_to_mouse": topk_per_col(pi, top_k),
            "mouse_entropy": [float(round(v, 5)) for v in row_entropy(pi).tolist()],
            "human_entropy": [float(round(v, 5)) for v in col_entropy(pi).tolist()],
            "human_col_mass": [float(round(v, 8)) for v in pi.sum(axis=0).tolist()],
            "region_eval": _load_region_eval(region_eval_file),
        })

    return {
        "version": 1,
        "title": "OTTER Mapping Explorer",
        "top_k": int(top_k),
        "mouse": mouse,
        "human": human,
        "groups": groups,
        "models": models,
        "visual_layers": (
            visual_layers
            if visual_layers is not None
            else (build_visual_layers(mouse_ad, human_ad, root=root) if include_visual_layers
                  else _empty_visual_layers())
        ),
        "anchor_packs": [
            {
                "pair_id": int(e.pair_id),
                "label": str(e.label),
                "n_mouse": int(len(e.mouse_indices)),
                "n_human": int(len(e.human_indices)),
            }
            for e in (anchor_entries or [])
        ],
    }


def build_gui_html(payload: dict) -> str:
    embedded = (
        json.dumps(payload, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    return _HTML_TEMPLATE.replace("{{GUI_DATA}}", embedded)


def write_gui(
    payload: dict,
    *,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "gui_data.json"
    html_path = output_dir / "index.html"
    data_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    html_path.write_text(build_gui_html(payload), encoding="utf-8")
    return data_path, html_path


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>OTTER Mapping Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {
    --bg: #f7f7f4;
    --panel: #ffffff;
    --ink: #202124;
    --muted: #62666d;
    --line: #d8d7d0;
    --accent: #c75b12;
    --blue: #2764a5;
    --green: #13795b;
    --amber: #a16207;
    --red: #b42318;
    --gray: #6b7280;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    letter-spacing: 0;
  }
  header {
    height: 54px;
    border-bottom: 1px solid var(--line);
    background: #fff;
    display: grid;
    grid-template-columns: minmax(200px, 300px) 1fr auto;
    align-items: center;
    gap: 16px;
    padding: 0 16px;
  }
  h1 { margin: 0; font-size: 18px; font-weight: 700; }
  label { color: var(--muted); font-size: 12px; display: grid; gap: 4px; }
  select, input, button {
    height: 32px;
    border: 1px solid var(--line);
    border-radius: 6px;
    background: #fff;
    color: var(--ink);
    font: inherit;
    font-size: 13px;
    padding: 0 10px;
  }
  button { cursor: pointer; }
  button.primary { background: var(--ink); color: #fff; border-color: var(--ink); }
  button.icon { width: 32px; padding: 0; font-size: 16px; }
  main {
    height: calc(100vh - 54px);
    display: grid;
    grid-template-columns: 320px minmax(560px, 1fr) 390px;
    gap: 10px;
    padding: 10px;
  }
  aside, section.panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    min-height: 0;
  }
  aside { display: grid; grid-template-rows: auto auto 1fr; gap: 10px; padding: 12px; }
  .searchrow { display: grid; grid-template-columns: 1fr auto; gap: 8px; }
  .filters { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .results { overflow: auto; border-top: 1px solid var(--line); padding-top: 8px; }
  .result {
    width: 100%;
    height: auto;
    min-height: 42px;
    text-align: left;
    margin-bottom: 6px;
    padding: 7px 8px;
    border-radius: 6px;
    background: #fbfbf9;
  }
  .result strong { display: block; font-size: 13px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .result span { display: block; font-size: 12px; color: var(--muted); margin-top: 2px; }
  .brain-grid {
    display: grid;
    grid-template-rows: auto 1fr;
    min-height: 0;
    gap: 10px;
  }
  .toolbar {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 10px 12px;
    min-width: 0;
  }
  .toolbar-row {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    min-width: 0;
  }
  .toolbar-row + .toolbar-row {
    padding-top: 6px;
    border-top: 1px dashed var(--line);
  }
  .seg { display: inline-flex; border: 1px solid var(--line); border-radius: 7px; overflow: hidden; }
  .seg button { border: 0; border-right: 1px solid var(--line); border-radius: 0; background: #fff; }
  .seg button:last-child { border-right: 0; }
  .seg button.active { background: #e9eef4; color: var(--blue); font-weight: 700; }
  .slider-group {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
  }
  .slider-group label {
    color: var(--muted);
    font-size: 11px;
    white-space: nowrap;
  }
  .slider-group input[type="range"] {
    width: 96px;
    height: 18px;
    padding: 0;
    border: 0;
    background: transparent;
    cursor: pointer;
  }
  .slider-group .val {
    color: var(--muted);
    font-size: 11px;
    min-width: 28px;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .color-group {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .color-group label { color: var(--muted); font-size: 11px; white-space: nowrap; }
  .color-group input[type="color"] {
    width: 28px;
    height: 22px;
    padding: 0;
    border: 1px solid var(--line);
    border-radius: 5px;
    background: #fff;
    cursor: pointer;
  }
  .plots {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    min-height: 0;
  }
  .plotbox {
    position: relative;
    background: #fff;
    border: 1px solid var(--line);
    border-radius: 8px;
    min-height: 0;
  }
  .plot-title {
    position: absolute;
    left: 10px;
    top: 8px;
    z-index: 3;
    font-size: 13px;
    font-weight: 700;
    color: var(--muted);
    pointer-events: none;
  }
  #plotMouse, #plotHuman { width: 100%; height: 100%; min-height: 520px; }
  .right {
    display: grid;
    grid-template-rows: auto 1fr 1fr auto;
    gap: 10px;
    padding: 12px;
    overflow: hidden;
  }
  .summary h2, .block h2 { margin: 0 0 8px; font-size: 15px; }
  .summary { border-bottom: 1px solid var(--line); padding-bottom: 10px; }
  .block { min-height: 0; overflow: auto; }
  .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .metric { background: #f7f7f4; border: 1px solid var(--line); border-radius: 7px; padding: 8px; }
  .metric b { display: block; font-size: 16px; margin-bottom: 2px; }
  .metric span { color: var(--muted); font-size: 11px; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th, td { border-bottom: 1px solid #ecebe6; padding: 6px 4px; text-align: left; vertical-align: top; }
  th { color: var(--muted); font-weight: 600; position: sticky; top: 0; background: #fff; }
  .pill { display: inline-block; border-radius: 999px; padding: 2px 7px; font-size: 11px; font-weight: 700; }
  .tier-anchored_and_validated { background: #dff4ea; color: var(--green); }
  .tier-anchored_only { background: #e8eef7; color: var(--blue); }
  .tier-validated_only { background: #fff4cc; color: var(--amber); }
  .tier-structural { background: #eee9ff; color: #6247aa; }
  .tier-low_evidence, .tier-unknown { background: #eeeeec; color: var(--gray); }
  .muted { color: var(--muted); }
  .small { font-size: 12px; }
  .danger { color: var(--red); }
  .mode-legend {
    display: flex;
    align-items: center;
    gap: 7px;
    min-width: 0;
    flex: 1 1 220px;
    color: var(--muted);
    font-size: 11px;
  }
  .swatch { width: 11px; height: 11px; border-radius: 3px; border: 1px solid rgba(0,0,0,0.16); flex: 0 0 auto; }
  .gradient-swatch {
    width: 86px;
    height: 11px;
    border-radius: 3px;
    border: 1px solid rgba(0,0,0,0.16);
    background: linear-gradient(90deg,#440154,#31688e,#35b779,#fde725);
  }
  .footer-actions { display: flex; gap: 8px; align-items: center; justify-content: space-between; }
  @media (max-width: 1120px) {
    main { grid-template-columns: 300px 1fr; grid-template-rows: minmax(520px, 1fr) 430px; }
    .right { grid-column: 1 / -1; grid-template-columns: 1fr 1fr; grid-template-rows: auto 1fr; }
    .summary { border-bottom: 0; }
  }
  @media (max-width: 780px) {
    header { grid-template-columns: 1fr; height: auto; padding: 10px; }
    main { height: auto; grid-template-columns: 1fr; }
    .plots { grid-template-columns: 1fr; }
    .right { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<header>
  <h1>OTTER Mapping Explorer</h1>
  <label>Model
    <select id="modelSelect"></select>
  </label>
  <button id="exportBtn" class="primary">Export Query</button>
</header>
<main>
  <aside>
    <div>
      <label>Mouse query</label>
      <div class="searchrow">
        <input id="searchInput" placeholder="region, parcel id, network, pack" />
        <button id="searchBtn" class="icon" title="Search">⌕</button>
      </div>
    </div>
    <div class="filters">
      <label>Trust
        <select id="trustFilter">
          <option value="">Any</option>
          <option value="anchored_and_validated">Anchored + validated</option>
          <option value="validated_only">Validated only</option>
          <option value="structural">Structural</option>
          <option value="anchored_only">Anchored only</option>
          <option value="low_evidence">Low evidence</option>
        </select>
      </label>
      <label>Hemisphere
        <select id="hemiFilter">
          <option value="">Both</option>
          <option value="L">L</option>
          <option value="R">R</option>
        </select>
      </label>
    </div>
    <div class="results" id="results"></div>
  </aside>
  <div class="brain-grid">
    <div class="toolbar">
      <div class="toolbar-row">
        <span class="small muted">Mouse</span>
        <span class="seg" id="mouseView">
          <button data-mview="points" class="active">Points</button>
          <button data-mview="shell">Shell</button>
          <button data-mview="shell_points">Shell + points</button>
        </span>
        <span class="small muted">Human</span>
        <span class="seg" id="humanView">
          <button data-hview="points" class="active">Points</button>
          <button data-hview="surface">Surface</button>
          <button data-hview="surface_points">Surface + points</button>
        </span>
        <span class="small muted">Color</span>
        <span class="seg" id="colorMode">
          <button data-mode="evidence" class="active">Trust</button>
          <button data-mode="network">Network</button>
          <button data-mode="entropy">Entropy</button>
        </span>
        <span class="small muted" id="selectionLabel">No selection</span>
      </div>
      <div class="toolbar-row">
        <span class="slider-group">
          <label for="surfaceOpacity">Surface opacity</label>
          <input type="range" id="surfaceOpacity" min="0" max="1" step="0.02" value="0.24" />
          <span class="val" id="surfaceOpacityVal">0.24</span>
        </span>
        <span class="slider-group">
          <label for="pointOpacity">Point opacity</label>
          <input type="range" id="pointOpacity" min="0.1" max="1" step="0.02" value="1.00" />
          <span class="val" id="pointOpacityVal">1.00</span>
        </span>
        <span class="slider-group">
          <label for="pointSize">Point size</label>
          <input type="range" id="pointSize" min="0.4" max="2.5" step="0.05" value="1.00" />
          <span class="val" id="pointSizeVal">1.00&times;</span>
        </span>
        <span class="color-group">
          <label for="surfaceTint">Surface tint</label>
          <input type="color" id="surfaceTint" value="#c7ccd0" />
        </span>
        <span class="mode-legend" id="modeLegend"></span>
      </div>
    </div>
    <div class="plots">
      <div class="plotbox"><div class="plot-title" id="mousePlotTitle">Mouse</div><div id="plotMouse"></div></div>
      <div class="plotbox"><div class="plot-title" id="humanPlotTitle">Human</div><div id="plotHuman"></div></div>
    </div>
  </div>
  <section class="panel right">
    <div class="summary">
      <h2 id="queryTitle">Select a mouse parcel or region</h2>
      <div class="metric-grid">
        <div class="metric"><b id="metricN">-</b><span>mouse parcels</span></div>
        <div class="metric"><b id="metricHemi">-</b><span>same-hemi top-1</span></div>
        <div class="metric"><b id="metricMass">-</b><span>top-K mass used</span></div>
      </div>
    </div>
    <div class="block">
      <h2>Ranked Human Regions</h2>
      <table><thead><tr><th>Region</th><th>Mass</th><th>Top parcel</th></tr></thead><tbody id="regionRows"></tbody></table>
    </div>
    <div class="block">
      <h2>Evidence</h2>
      <div id="evidenceBox" class="small muted">Evidence appears after selection.</div>
      <h2 style="margin-top:14px;">Top Human Parcels</h2>
      <table><thead><tr><th>Parcel</th><th>Prob.</th><th>Hemi</th></tr></thead><tbody id="parcelRows"></tbody></table>
    </div>
    <div class="footer-actions">
      <span class="small muted" id="modelSummary"></span>
      <button id="resetBtn">Reset</button>
    </div>
  </section>
</main>
<script>
const DATA = {{GUI_DATA}};
const COLORS = {
  auditory:'#c2410c', brainstem:'#6d5bd0', frontal_dmn:'#1d70a2',
  frontoparietal:'#157f63', limbic:'#a16207', olfactory:'#b7791f',
  salience:'#b83280', sensorimotor:'#0f766e', subcortical:'#64748b',
  temporal_dmn:'#15803d', visual:'#ea580c'
};
const TIER_COLORS = {
  anchored_and_validated:'#16825d', anchored_only:'#2764a5',
  validated_only:'#b7791f', structural:'#7457c6',
  low_evidence:'#8a8f98', unknown:'#b3b5b8'
};
const VIRIDIS = ['#440154','#482878','#3e4a89','#31688e','#26828e','#1f9e89','#35b779','#6ece58','#b5de2b','#fde725'];
// Single viridis colorscale shared by point heat, entropy, and surface meshes
// so both panels read on the same axis. The accent orange (#c75b12) is
// reserved for selection emphasis (alpha hull, hover dot ring), never heat.
const VIRIDIS_SCALE = VIRIDIS.map((c, i) => [i / (VIRIDIS.length - 1), c]);
let state = {
  model: DATA.models[0].id,
  selected: null,
  colorMode: 'evidence',
  mouseView: 'points',         // 'points' | 'shell' | 'shell_points'
  humanView: 'points',         // 'points' | 'surface' | 'surface_points'
  surfaceOpacity: 0.24,        // base mesh opacity (no-heat state)
  surfaceTint: '#c7ccd0',      // base mesh tint
  pointOpacity: 1.0,           // multiplier on baseOpacity('species')
  pointSizeFactor: 1.0,        // multiplier on baseSizes('species')
};
let _cameraLock = false;  // guard against feedback loops for linked-camera sync

function byId(id){ return document.getElementById(id); }
function model(){ return DATA.models.find(m => m.id === state.model) || DATA.models[0]; }
function pct(x){ return Number.isFinite(x) ? `${Math.round(x*100)}%` : '-'; }
function fmt(x, n=3){ return Number.isFinite(x) ? x.toFixed(n) : '-'; }
function esc(s){ return String(s ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
function classToken(s){ return String(s || 'unknown').replace(/[^a-zA-Z0-9_-]/g, '_'); }
function short(s, n=70){ s = String(s || ''); return s.length > n ? s.slice(0,n-1)+'…' : s; }
function uniq(arr){ return [...new Set(arr.filter(v => v !== undefined && v !== null && String(v) !== ''))]; }
function cleanRegion(s){
  s = String(s || '').replace(/^L_/, '').replace(/^R_/, '');
  if (s.includes(';')) s = s.split(';')[0];
  return short(s, 82);
}
function tierPill(t){
  const tier = String(t || 'unknown');
  return `<span class="pill tier-${classToken(tier)}">${esc(tier.replaceAll('_',' '))}</span>`;
}
function colorScale(t){ const i = Math.max(0, Math.min(VIRIDIS.length-1, Math.round(t*(VIRIDIS.length-1)))); return VIRIDIS[i]; }
function quantile(arr, q){
  const vals = arr.filter(Number.isFinite).slice().sort((a,b)=>a-b);
  if (!vals.length) return NaN;
  const pos = (vals.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo);
}
function entropyColors(arr){
  const lo = quantile(arr, 0.05);
  const hi = quantile(arr, 0.95);
  const span = Number.isFinite(hi - lo) && hi > lo ? hi - lo : Math.max(...arr, 1e-9);
  return arr.map(v => colorScale((v - lo) / span));
}

function setupModelSelect(){
  byId('modelSelect').innerHTML = DATA.models.map(m => `<option value="${esc(m.id)}">${esc(m.label)}</option>`).join('');
  byId('modelSelect').value = state.model;
  byId('modelSelect').addEventListener('change', e => {
    state.model = e.target.value;
    drawPlots();
    updateSelection(state.selected);
  });
}

function baseColors(species){
  const d = DATA[species], m = model();
  if (state.colorMode === 'network') return d.network.map(n => COLORS[n] || '#777');
  if (state.colorMode === 'entropy') {
    const arr = species === 'mouse' ? m.mouse_entropy : m.human_entropy;
    return entropyColors(arr);
  }
  if (species === 'mouse') return d.evidence_tier.map(t => TIER_COLORS[t] || TIER_COLORS.unknown);
  return d.network.map(n => COLORS[n] || '#777');
}
function baseSizes(species){
  const d = DATA[species];
  const k = Math.max(0.1, Number(state.pointSizeFactor) || 1.0);
  return d.is_garin_anchor.map(v => (v ? 7 : 4) * k);
}
function baseOpacity(species){
  const k = Math.max(0.05, Math.min(1, Number(state.pointOpacity) || 1.0));
  const base = (state.colorMode === 'entropy')
    ? (species === 'mouse' ? 0.88 : 0.76)
    : (species === 'mouse' ? 0.82 : 0.68);
  return base * k;
}
function applyColorMode(){ drawPlots(); }
function renderModeLegend(){
  const box = byId('modeLegend');
  if (state.colorMode === 'entropy') {
    box.innerHTML = '<span>Low uncertainty</span><span class="gradient-swatch"></span><span>High uncertainty</span>';
    return;
  }
  if (state.colorMode === 'network') {
    const nets = uniq(DATA.mouse.network.concat(DATA.human.network)).slice(0, 5);
    box.innerHTML = nets.map(n => `<span class="swatch" style="background:${COLORS[n] || '#777'}"></span><span>${esc(n.replaceAll('_',' '))}</span>`).join('') + '<span>...</span>';
    return;
  }
  const tiers = uniq(DATA.mouse.evidence_tier).sort();
  box.innerHTML = tiers.map(t => `<span class="swatch" style="background:${TIER_COLORS[t] || TIER_COLORS.unknown}"></span><span>${esc(t.replaceAll('_',' '))}</span>`).join('');
}

function layer(name){ return (DATA.visual_layers || {})[name] || {available:false}; }
function hasLayer(name){ return !!layer(name).available; }
function layout3d(){
  return {margin:{l:0,r:0,t:0,b:0}, paper_bgcolor:'#fff', plot_bgcolor:'#fff',
    scene:{xaxis:{visible:false},yaxis:{visible:false},zaxis:{visible:false},aspectmode:'data'},
    showlegend:false};
}
function mousePointTrace(){
  const mouse = DATA.mouse;
  return {type:'scatter3d', mode:'markers', hoverinfo:'text',
    x:mouse.x, y:mouse.y, z:mouse.z, customdata: mouse.ids.map((_,i)=>i),
    text: mouse.ids.map((id,i)=>`Mouse ${esc(id)}<br>${esc(mouse.region[i])}<br>${esc(mouse.evidence_tier[i] || 'unknown')}`),
    marker:{size:baseSizes('mouse'), color:baseColors('mouse'), opacity:baseOpacity('mouse'), line:{width:0}}};
}
function humanPointTrace(){
  const human = DATA.human;
  return {type:'scatter3d', mode:'markers', hoverinfo:'text',
    x:human.x, y:human.y, z:human.z,
    text: human.ids.map((id,i)=>`Human ${esc(id)}<br>${esc(human.region[i])}`),
    marker:{size:baseSizes('human'), color:baseColors('human'), opacity:baseOpacity('human'), line:{width:0}}};
}
function meshTrace(mesh, opts={}){
  // Surface tint and opacity come from state so the sliders / color picker
  // restyle these in place without rebuilding the plot.
  const color = opts.color != null ? opts.color : (state.surfaceTint || '#c7ccd0');
  const opacity = opts.opacity != null ? opts.opacity : Number(state.surfaceOpacity ?? 0.24);
  return {type:'mesh3d', x:mesh.x, y:mesh.y, z:mesh.z, i:mesh.i, j:mesh.j, k:mesh.k,
    color, opacity,
    flatshading:false, hoverinfo:'skip', lighting:{ambient:0.72, diffuse:0.55, roughness:0.85}};
}
function surfaceHeatTrace(mesh){
  return {type:'mesh3d', x:mesh.x, y:mesh.y, z:mesh.z, i:mesh.i, j:mesh.j, k:mesh.k,
    intensity:new Array(mesh.n_vertices || mesh.x.length).fill(0), colorscale:VIRIDIS_SCALE,
    cmin:0, cmax:1, opacity:0, showscale:false, hoverinfo:'skip',
    lighting:{ambient:0.82, diffuse:0.45, roughness:0.9}};
}
function emptyScatterTrace(){
  return {type:'scatter3d', mode:'markers', x:[], y:[], z:[],
    marker:{size:8, color:[]}, hoverinfo:'text', text:[]};
}
function emptyMeshTrace(){
  return {type:'mesh3d', x:[], y:[], z:[],
    color:'#c75b12', opacity:0, hoverinfo:'skip', alphahull:1.8};
}
function bindMouseClicks(){
  const div = byId('plotMouse');
  if (div.removeAllListeners) div.removeAllListeners('plotly_click');
  div.on('plotly_click', ev => {
    const p = ev.points && ev.points[0];
    if (!p) return;
    // Only the underlying parcel-points trace carries per-point customdata
    // (parcel index). Selection-highlight or shell traces won't.
    if (!p.data || !p.data.customdata) return;
    const idx = Number(p.data.customdata[p.pointNumber]);
    if (Number.isInteger(idx) && idx >= 0) {
      selectMouseIndices([idx], `Mouse parcel ${DATA.mouse.ids[idx]}`);
    }
  });
}

// Trace-index layout for each panel/view combination. The drawPlots()
// builder emits traces in exactly this order so highlight functions can
// reference traces by name instead of positional magic numbers.
const TRACE_LAYOUT = {
  mouse: {
    points:       { points: 0,             selected: 1 },
    shell:        { shell: 0, heat: 1,             hull: 2, selected: 3 },
    shell_points: { shell: 0, heat: 1, points: 2, hull: 3, selected: 4 },
  },
  human: {
    points:         { points: 0,             selected: 1 },
    surface:        { shell: 0, heat: 1,             selected: 2 },
    surface_points: { shell: 0, heat: 1, points: 2, selected: 3 },
  },
};
function mouseLayout(){ return TRACE_LAYOUT.mouse[state.mouseView] || TRACE_LAYOUT.mouse.points; }
function humanLayout(){ return TRACE_LAYOUT.human[state.humanView] || TRACE_LAYOUT.human.points; }
function mouseUsesShell(){
  return (state.mouseView === 'shell' || state.mouseView === 'shell_points') && hasLayer('mouse_shell');
}
function humanUsesSurface(){
  return (state.humanView === 'surface' || state.humanView === 'surface_points') && hasLayer('human_surface');
}

function buildMouseTraces(){
  if (mouseUsesShell()) {
    const shell = layer('mouse_shell');
    const traces = [meshTrace(shell), surfaceHeatTrace(shell)];
    if (state.mouseView === 'shell_points') traces.push(mousePointTrace());
    traces.push(emptyMeshTrace());     // alpha-hull slot for selection groups
    traces.push(emptyScatterTrace());  // selected mouse points (above shell)
    return traces;
  }
  return [mousePointTrace(), emptyScatterTrace()];
}

function buildHumanTraces(){
  if (humanUsesSurface()) {
    const surf = layer('human_surface');
    const traces = [meshTrace(surf), surfaceHeatTrace(surf)];
    if (state.humanView === 'surface_points') traces.push(humanPointTrace());
    traces.push(emptyScatterTrace());  // selection / off-surface points
    return traces;
  }
  return [humanPointTrace(), emptyScatterTrace()];
}

function panelTitle(species){
  if (species === 'mouse') {
    if (state.mouseView === 'shell_points' && hasLayer('mouse_shell')) return 'Mouse (shell + parcels)';
    if (state.mouseView === 'shell'        && hasLayer('mouse_shell')) return 'Mouse (shell)';
    return 'Mouse parcels';
  }
  if (state.humanView === 'surface_points' && hasLayer('human_surface')) return 'Human (surface + parcels)';
  if (state.humanView === 'surface'        && hasLayer('human_surface')) return 'Human (cortical surface)';
  return 'Human parcels';
}

function drawPlots(){
  const layout = layout3d();
  byId('mousePlotTitle').textContent = panelTitle('mouse');
  byId('humanPlotTitle').textContent = panelTitle('human');
  Plotly.react('plotMouse', buildMouseTraces(), layout, {displayModeBar:false, responsive:true});
  Plotly.react('plotHuman', buildHumanTraces(), layout, {displayModeBar:false, responsive:true});
  bindMouseClicks();
  bindLinkedCameras();
  renderModeLegend();
}

// Mirror scene.camera between the two panels so rotating one rotates the
// other. The _cameraLock flag prevents the change/echo from looping back.
function bindLinkedCameras(){
  const mouseDiv = byId('plotMouse'), humanDiv = byId('plotHuman');
  const handler = (src, dst) => ev => {
    if (_cameraLock) return;
    if (!ev || !ev['scene.camera']) return;
    _cameraLock = true;
    Plotly.relayout(dst, {'scene.camera': ev['scene.camera']})
      .finally(() => { _cameraLock = false; });
  };
  if (mouseDiv.removeAllListeners) mouseDiv.removeAllListeners('plotly_relayout');
  if (humanDiv.removeAllListeners) humanDiv.removeAllListeners('plotly_relayout');
  mouseDiv.on('plotly_relayout', handler('plotMouse', 'plotHuman'));
  humanDiv.on('plotly_relayout', handler('plotHuman', 'plotMouse'));
}

function runSearch(){
  const q = byId('searchInput').value.trim().toLowerCase();
  const trust = byId('trustFilter').value;
  const hemi = byId('hemiFilter').value;
  const res = [];
  for (const g of DATA.groups.mouse) {
    if (q && !g.label.toLowerCase().includes(q) && !g.kind.toLowerCase().includes(q)) continue;
    const idxs = filterIndices(g.indices, trust, hemi);
    if (idxs.length) res.push({kind:'group', label:g.label, sub:`${g.kind} · ${idxs.length} parcels`, indices:idxs});
  }
  const m = DATA.mouse;
  for (let i=0; i<m.ids.length && res.length < 80; i++) {
    const blob = `${m.ids[i]} ${m.region[i]} ${m.subregion[i]} ${m.network[i]} ${m.evidence_tier[i]}`.toLowerCase();
    if (q && !blob.includes(q)) continue;
    if (!filterIndices([i], trust, hemi).length) continue;
    res.push({kind:'parcel', label:`${m.ids[i]} · ${cleanRegion(m.region[i])}`, sub:`${m.network[i]} · ${m.hemisphere[i]} · ${m.evidence_tier[i] || 'unknown'}`, indices:[i]});
  }
  renderResults(res.slice(0, 80));
}
function filterIndices(indices, trust, hemi){
  return indices.filter(i => (!trust || DATA.mouse.evidence_tier[i] === trust) && (!hemi || DATA.mouse.hemisphere[i] === hemi));
}
function renderResults(items){
  const box = byId('results');
  if (!items.length) { box.innerHTML = '<div class="small muted">No matches.</div>'; return; }
  box.innerHTML = items.map((r, i) => `<button class="result" data-i="${i}"><strong>${esc(r.label)}</strong><span>${esc(r.sub)}</span></button>`).join('');
  box.querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => {
    const r = items[Number(btn.dataset.i)];
    selectMouseIndices(r.indices, r.label);
  }));
}

function selectMouseIndices(indices, label){
  state.selected = {indices, label};
  updateSelection(state.selected);
}

function updateSelection(sel){
  if (!sel) {
    byId('queryTitle').textContent = 'Select a mouse parcel or region';
    byId('selectionLabel').textContent = 'No selection';
    drawPlots();
    return;
  }
  const indices = sel.indices;
  byId('queryTitle').textContent = short(sel.label, 80);
  byId('selectionLabel').textContent = `${indices.length} mouse parcel${indices.length === 1 ? '' : 's'} selected`;
  const agg = aggregate(indices);
  renderSummary(indices, agg);
  renderActiveHighlights(indices, agg);
}

function aggregate(indices){
  const m = model(), h = DATA.human, mouse = DATA.mouse;
  const byHuman = new Map();
  let usedMass = 0, same = 0, top1Count = 0;
  for (const mi of indices) {
    const row = m.top_mouse_to_human[mi] || [];
    let rowMass = 0;
    if (row.length) {
      const h0 = row[0][0];
      if (h.hemisphere[h0] === mouse.hemisphere[mi]) same++;
      top1Count++;
    }
    for (const [hj, p] of row) {
      byHuman.set(hj, (byHuman.get(hj) || 0) + p / indices.length);
      rowMass += p;
    }
    usedMass += rowMass / indices.length;
  }
  const topHumans = [...byHuman.entries()].sort((a,b)=>b[1]-a[1]).slice(0, 30);
  const byRegion = new Map();
  for (const [hj, p] of byHuman.entries()) {
    const label = cleanRegion(h.subregion[hj] || h.region[hj]);
    if (!byRegion.has(label)) byRegion.set(label, {mass:0, top:null});
    const r = byRegion.get(label);
    r.mass += p;
    if (!r.top || p > r.top.mass) r.top = {idx:hj, mass:p};
  }
  const topRegions = [...byRegion.entries()].map(([label,v]) => ({label, ...v}))
    .sort((a,b)=>b.mass-a.mass).slice(0, 15);
  return {topHumans, topRegions, usedMass, sameHemiRate: top1Count ? same/top1Count : NaN};
}

function renderSummary(indices, agg){
  byId('metricN').textContent = String(indices.length);
  byId('metricHemi').textContent = pct(agg.sameHemiRate);
  byId('metricMass').textContent = pct(agg.usedMass);
  byId('regionRows').innerHTML = agg.topRegions.map(r =>
    `<tr><td>${esc(r.label)}</td><td>${fmt(r.mass, 4)}</td><td>${esc(DATA.human.ids[r.top.idx])}</td></tr>`).join('');
  byId('parcelRows').innerHTML = agg.topHumans.slice(0, 12).map(([j,p]) =>
    `<tr><td>${esc(DATA.human.ids[j])} · ${esc(cleanRegion(DATA.human.region[j]))}</td><td>${fmt(p,4)}</td><td>${esc(DATA.human.hemisphere[j])}</td></tr>`).join('');
  renderEvidence(indices, agg);
  renderModelSummary();
}

function renderEvidence(indices, agg){
  const tiers = {}, n = indices.length;
  let trustSum = 0, trustN = 0, pack = 0, garin = 0, bTop = 0, bN = 0;
  for (const i of indices) {
    const t = DATA.mouse.evidence_tier[i] || 'unknown';
    tiers[t] = (tiers[t] || 0) + 1;
    if (Number.isFinite(DATA.mouse.trust[i])) { trustSum += DATA.mouse.trust[i]; trustN++; }
    if (DATA.mouse.pack_anchored[i]) pack++;
    if (DATA.mouse.garin_anchored[i]) garin++;
    if (Number.isFinite(DATA.mouse.beauchamp_top1[i])) { bTop += DATA.mouse.beauchamp_top1[i]; bN++; }
  }
  const tierHtml = Object.entries(tiers).sort((a,b)=>b[1]-a[1]).map(([t,c]) =>
    `${tierPill(t)} <span class="muted">${c}/${n}</span>`).join('<br>');
  const warning = agg.sameHemiRate < 0.8 ? '<p class="danger">Hemisphere consistency is low for this query. Treat parcel-level partners cautiously.</p>' : '';
  byId('evidenceBox').innerHTML = `
    <p>${tierHtml || 'No tier data'}</p>
    <p><b>Mean internal trust:</b> ${trustN ? fmt(trustSum/trustN,3) : '-'}</p>
    <p><b>Pack anchored:</b> ${pack}/${n} · <b>Garin anchored:</b> ${garin}/${n}</p>
    <p><b>Mean Beauchamp top-1 for covered parcels:</b> ${bN ? pct(bTop/bN) : '-'}</p>
    ${warning}
  `;
}

function renderModelSummary(){
  const r = model().region_eval || {};
  const a = r.aggregate || {};
  const q = a.qualified_top_k || {};
  byId('modelSummary').textContent = q[3] !== undefined
    ? `Model region-level qualified top-3: ${pct(q[3])}`
    : model().label;
}

// Per-species highlight functions live below (renderMouseShellHighlights /
// renderMousePointsHighlights and renderHumanSurfaceHighlights /
// renderHumanPointsHighlights). The old positional helpers
// (renderMousePointSelection, renderHumanPointHighlights, renderParcelHighlights,
// renderMouseAtlasHighlights) were removed when view-mode became per-species.

// --- Mouse highlights ----------------------------------------------------

function renderMouseShellHighlights(indices){
  // Paint a viridis heat overlay onto the mouse shell driven directly by the
  // selected mouse parcels (analogue of renderHumanSurfaceHighlights, which
  // is driven by aggregated top-K human partners). Selected parcels also
  // get an opaque accent-orange alpha-hull/dots layer on top.
  const t = mouseLayout();
  const shell = layer('mouse_shell');
  const mouse = DATA.mouse;
  const sf = Math.max(0.1, Number(state.pointSizeFactor) || 1.0);
  if (shell.available && shell.parcel_stencil) {
    const values = new Array(shell.n_vertices || shell.x.length).fill(0);
    for (const i of indices) {
      const stencil = shell.parcel_stencil[i] || [];
      for (const [v, w] of stencil) values[v] += w;
    }
    const maxV = Math.max(...values, 1e-9);
    Plotly.restyle('plotMouse', {
      intensity:[values], cmax:[maxV], opacity:[0.82]
    }, [t.heat]);
  }
  // Alpha-hull blob over the selection when there are enough points to mesh.
  const blockX = indices.map(i=>mouse.x[i]);
  const blockY = indices.map(i=>mouse.y[i]);
  const blockZ = indices.map(i=>mouse.z[i]);
  if (indices.length >= 4) {
    Plotly.restyle('plotMouse', {
      x:[blockX], y:[blockY], z:[blockZ],
      alphahull:[1.8], color:['#c75b12'], opacity:[0.45]
    }, [t.hull]);
  } else {
    Plotly.restyle('plotMouse', {x:[[]], y:[[]], z:[[]], opacity:[0]}, [t.hull]);
  }
  // Selected-points trace: keep each parcel's base color (trust / network /
  // entropy) so the encoding survives selection; emphasise with a thicker
  // accent ring and larger size (scaled by global pointSizeFactor).
  const baseAll = baseColors('mouse');
  Plotly.restyle('plotMouse', {
    x:[blockX], y:[blockY], z:[blockZ],
    text:[indices.map(i=>`Mouse ${esc(mouse.ids[i])}<br>${esc(mouse.region[i])}`)],
    'marker.color':[indices.map(i => baseAll[i])],
    'marker.size':[indices.map(i => (indices.length === 1 ? 12 : 9) * sf)],
    'marker.line.color':[indices.map(() => '#c75b12')],
    'marker.line.width':[indices.map(() => 2)],
  }, [t.selected]);
}

function renderMousePointsHighlights(indices){
  // Selected mouse parcels in the plain points view: keep base colors,
  // bump size, add an accent ring. Mirror of how the shell view emphasises.
  const t = mouseLayout();
  const mouse = DATA.mouse;
  const baseAll = baseColors('mouse');
  const sf = Math.max(0.1, Number(state.pointSizeFactor) || 1.0);
  Plotly.restyle('plotMouse', {
    x:[indices.map(i=>mouse.x[i])],
    y:[indices.map(i=>mouse.y[i])],
    z:[indices.map(i=>mouse.z[i])],
    text:[indices.map(i=>`Mouse ${esc(mouse.ids[i])}<br>${esc(mouse.region[i])}`)],
    'marker.color':[indices.map(i => baseAll[i])],
    'marker.size':[indices.map(i => (indices.length === 1 ? 11 : 9) * sf)],
    'marker.line.color':[indices.map(() => '#c75b12')],
    'marker.line.width':[indices.map(() => 2)],
  }, [t.selected]);
}

// --- Human highlights ----------------------------------------------------

function renderHumanPointsHighlights(agg){
  const t = humanLayout();
  const human = DATA.human;
  const hs = agg.topHumans.slice(0, 30);
  const maxP = Math.max(...hs.map(x=>x[1]), 1e-9);
  const sf = Math.max(0.1, Number(state.pointSizeFactor) || 1.0);
  Plotly.restyle('plotHuman', {
    x:[hs.map(x=>human.x[x[0]])], y:[hs.map(x=>human.y[x[0]])], z:[hs.map(x=>human.z[x[0]])],
    text:[hs.map(([j,p])=>`Human ${esc(human.ids[j])}<br>${esc(human.region[j])}<br>${fmt(p,4)}`)],
    'marker.color':[hs.map(x=>colorScale(x[1]/maxP))],
    'marker.size':[hs.map(x=>(6 + 14*x[1]/maxP) * sf)]
  }, [t.selected]);
}

function renderHumanSurfaceHighlights(agg){
  const t = humanLayout();
  const surf = layer('human_surface');
  if (!surf.available) { renderHumanPointsHighlights(agg); return; }
  const values = new Array(surf.n_vertices || surf.x.length).fill(0);
  const deep = [];
  for (const [j, p] of agg.topHumans) {
    const stencil = surf.parcel_stencil[j] || [];
    if (!stencil.length) {
      deep.push([j, p]);
      continue;
    }
    for (const [v, w] of stencil) values[v] += p * w;
  }
  const maxV = Math.max(...values, 1e-9);
  Plotly.restyle('plotHuman', {
    intensity:[values], cmax:[maxV], opacity:[0.82]
  }, [t.heat]);
  // Off-surface partners go into the selection-scatter slot so they're
  // still visible above the mesh; on-surface partners are already encoded
  // by the heat overlay.
  const human = DATA.human;
  const sf = Math.max(0.1, Number(state.pointSizeFactor) || 1.0);
  const maxP = Math.max(...deep.map(x=>x[1]), 1e-9);
  Plotly.restyle('plotHuman', {
    x:[deep.map(x=>human.x[x[0]])], y:[deep.map(x=>human.y[x[0]])], z:[deep.map(x=>human.z[x[0]])],
    text:[deep.map(([j,p])=>`Off-surface human ${esc(human.ids[j])}<br>${esc(human.region[j])}<br>${fmt(p,4)}`)],
    'marker.color':[deep.map(x=>colorScale(x[1]/maxP))],
    'marker.size':[deep.map(x=>(7 + 13*x[1]/maxP) * sf)]
  }, [t.selected]);
}

function renderActiveHighlights(indices, agg){
  // Mouse panel: shell views (shell, shell_points) share the same heat +
  // hull + selection-scatter slot layout; differ only by whether the base
  // parcel points are also drawn.
  if (mouseUsesShell()) {
    renderMouseShellHighlights(indices);
  } else {
    renderMousePointsHighlights(indices);
  }
  // Human panel: same idea for surface and surface_points.
  if (humanUsesSurface()) {
    renderHumanSurfaceHighlights(agg);
  } else {
    renderHumanPointsHighlights(agg);
  }
}

function exportQuery(){
  const payload = {model: model().id, selection: state.selected, generated_from: DATA.title};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'otter_query.json'; a.click();
  URL.revokeObjectURL(url);
}

function init(){
  setupModelSelect();
  drawPlots();
  runSearch();
  byId('searchBtn').addEventListener('click', runSearch);
  byId('searchInput').addEventListener('keydown', e => { if(e.key === 'Enter') runSearch(); });
  byId('trustFilter').addEventListener('change', runSearch);
  byId('hemiFilter').addEventListener('change', runSearch);
  byId('resetBtn').addEventListener('click', () => { state.selected = null; byId('searchInput').value = ''; runSearch(); updateSelection(null); });
  byId('exportBtn').addEventListener('click', exportQuery);
  byId('colorMode').querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => {
    byId('colorMode').querySelectorAll('button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.colorMode = btn.dataset.mode;
    applyColorMode();
    updateSelection(state.selected);
  }));
  // Per-species view toggles (3-way: points | surface | surface+points).
  // The two controls are independent, so e.g. mouse=shell + human=surface_points
  // is a valid combination.
  function bindViewToggle(rootId, stateKey, dataAttr){
    const root = byId(rootId);
    if (!root) return;
    root.querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => {
      root.querySelectorAll('button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state[stateKey] = btn.dataset[dataAttr];
      drawPlots();
      updateSelection(state.selected);
    }));
  }
  bindViewToggle('mouseView', 'mouseView', 'mview');
  bindViewToggle('humanView', 'humanView', 'hview');

  // Live controls for transparency, point size, and surface tint. These
  // restyle traces in place so the user gets immediate feedback without
  // rebuilding the plots.
  bindLiveControls();

  renderModelSummary();
}

function bindLiveControls(){
  const opSurf = byId('surfaceOpacity');
  const opSurfVal = byId('surfaceOpacityVal');
  const opPt = byId('pointOpacity');
  const opPtVal = byId('pointOpacityVal');
  const sz = byId('pointSize');
  const szVal = byId('pointSizeVal');
  const tint = byId('surfaceTint');

  if (opSurf) {
    opSurf.addEventListener('input', () => {
      state.surfaceOpacity = Number(opSurf.value);
      if (opSurfVal) opSurfVal.textContent = state.surfaceOpacity.toFixed(2);
      applySurfaceStyle();
    });
  }
  if (opPt) {
    opPt.addEventListener('input', () => {
      state.pointOpacity = Number(opPt.value);
      if (opPtVal) opPtVal.textContent = state.pointOpacity.toFixed(2);
      applyPointStyle();
    });
  }
  if (sz) {
    sz.addEventListener('input', () => {
      state.pointSizeFactor = Number(sz.value);
      if (szVal) szVal.textContent = state.pointSizeFactor.toFixed(2) + '×';
      applyPointStyle();
    });
  }
  if (tint) {
    tint.addEventListener('input', () => {
      state.surfaceTint = tint.value;
      applySurfaceStyle();
    });
  }
}

// Apply current surfaceOpacity / surfaceTint to whichever mesh trace exists
// on each panel. Cheaper than rebuilding the plots on every slider tick.
function applySurfaceStyle(){
  const mt = mouseLayout();
  const ht = humanLayout();
  const op = Number(state.surfaceOpacity);
  const color = state.surfaceTint;
  if (mt.shell != null) {
    Plotly.restyle('plotMouse', {opacity: [op], color: [color]}, [mt.shell]);
  }
  if (ht.shell != null) {
    Plotly.restyle('plotHuman', {opacity: [op], color: [color]}, [ht.shell]);
  }
}

// Apply current pointOpacity / pointSizeFactor to the base-points trace on
// each panel (when visible). Selected-point sizes are governed by their own
// highlight render and rescale on the next updateSelection().
function applyPointStyle(){
  const mt = mouseLayout();
  const ht = humanLayout();
  if (mt.points != null) {
    Plotly.restyle('plotMouse', {
      'marker.size': [baseSizes('mouse')],
      'marker.opacity': [baseOpacity('mouse')],
    }, [mt.points]);
  }
  if (ht.points != null) {
    Plotly.restyle('plotHuman', {
      'marker.size': [baseSizes('human')],
      'marker.opacity': [baseOpacity('human')],
    }, [ht.points]);
  }
  // Re-render selection-scatter so sized-by-prob dots also scale with the
  // global point-size factor.
  if (state.selected) updateSelection(state.selected);
}
window.addEventListener('load', init);
</script>
</body>
</html>
"""
