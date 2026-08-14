"""Pipeline step 08 - build the region-first OTTER GUI.

This creates a static app in ``outputs/gui/``:

    outputs/gui/gui_data.json
    outputs/gui/index.html

The GUI is intentionally static. It embeds compact top-K partner lists,
node metadata, trust tiers, anchor-pack groups, and region-level summaries.
It does not run FGW solves or require a backend.

Pass ``--publish`` to additionally copy the rendered HTML into ``docs/``
(``docs/index.html``) so it can be committed and served via GitHub Pages.
``outputs/`` stays gitignored - only the ``docs/`` copy gets pushed.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otter.data import load_cached                                      # noqa: E402
from otter.viz.gui import build_gui_payload, write_gui                  # noqa: E402

ANN = ROOT / "outputs" / "anndata"
COUP = ROOT / "outputs" / "coupling"
LOG = ROOT / "outputs" / "logs"
GUI = ROOT / "outputs" / "gui"
DOCS = ROOT / "docs"


def _default_anchor_entries(M, H):
    """Anchor entries matching the current recommended all-packs pi.

    Uses the pack registry (``otter.data.anchor_packs.registry``) so the GUI's
    anchor-pack groups stay in lockstep with what ``compose_all.py`` fits.
    """
    try:
        from otter.data.anchor_packs import build_default_pack_entries     # noqa: E402
        return build_default_pack_entries(M.var, H.var, atlas_root=ROOT)
    except Exception as exc:  # pragma: no cover - depends on external atlas files
        print(f"  warning: could not build anchor-pack groups: {exc}")
        return []


def _default_models():
    """The canonical coupling, which is what load_pi() returns."""
    pi = COUP / "pi_canonical.npy"
    if not pi.exists():
        return []
    return [{
        "id": "canonical",
        "label": "OTTER canonical coupling",
        "pi_file": pi,
        "region_eval_file": LOG / "region_level_eval_canonical.json",
    }]


def main(args):
    print(f"Loading AnnData from {ANN}")
    M, _ = load_cached("mouse", cache_dir=ANN)
    H, _ = load_cached("human", cache_dir=ANN)
    print(f"  mouse={len(M.var)} parcels, human={len(H.var)} parcels")

    models = _default_models()
    if args.pi_file:
        for raw in args.pi_file:
            parts = raw.split(":", 2)
            if len(parts) == 1:
                path = COUP / parts[0]
                model_id = path.stem.replace("pi_", "")
                label = model_id.replace("_", " ")
            elif len(parts) == 2:
                model_id, path_s = parts
                path = COUP / path_s
                label = model_id.replace("_", " ")
            else:
                model_id, path_s, label = parts
                path = COUP / path_s
            models.append({"id": model_id, "label": label, "pi_file": path})
    if not models:
        raise SystemExit("No pi files found. Run pipeline/04_solve_production.py first.")

    print("Models:")
    for m in models:
        print(f"  {m['id']:>12s}: {m['pi_file']}")

    trust_path = COUP / "trust_multisource_canonical.npz"
    if not trust_path.exists():
        print(f"  warning: {trust_path} missing; trust tiers will be unknown")
        trust_path = None

    print("Building anchor-pack group metadata...")
    anchor_entries = _default_anchor_entries(M, H)
    print(f"  {len(anchor_entries)} anchor-pack entries")

    print(f"Building GUI payload (top_k={args.top_k})...")
    payload = build_gui_payload(
        models, M, H,
        top_k=args.top_k,
        trust_path=trust_path,
        anchor_entries=anchor_entries,
        include_visual_layers=True,
        root=ROOT,
    )
    visual = payload.get("visual_layers", {})
    for key, layer in visual.items():
        status = "available" if layer.get("available") else "unavailable"
        source = layer.get("source") or layer.get("message", "")
        print(f"  visual layer {key}: {status} ({source})")
    data_path, html_path = write_gui(payload, output_dir=args.output_dir)
    print(f"\nSaved {data_path} ({data_path.stat().st_size / 1024:.1f} KB)")
    print(f"Saved {html_path} ({html_path.stat().st_size / 1024:.1f} KB)")
    print(f"\nOpen {html_path}")

    if args.publish:
        # Publish a committable copy of the rendered HTML into docs/ so it
        # can be served via GitHub Pages. We only copy the HTML (which has
        # the data embedded) - gui_data.json stays in outputs/, which the
        # GUI doesn't actually fetch at runtime.
        publish_path = Path(args.publish_dir) / "index.html"
        publish_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(html_path, publish_path)
        print(f"Published {publish_path} ({publish_path.stat().st_size / 1024:.1f} KB)")
        print(f"  commit this and enable GitHub Pages on the {publish_path.parent.name}/ folder")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=50,
                    help="top partners to embed per node and model")
    ap.add_argument("--output-dir", default=GUI, type=Path)
    ap.add_argument(
        "--pi-file",
        action="append",
        default=[],
        help="optional extra model as filename.npy, id:filename.npy, or id:filename.npy:Label",
    )
    ap.add_argument(
        "--publish",
        action="store_true",
        help="also copy the rendered HTML into docs/index.html for GitHub Pages",
    )
    ap.add_argument(
        "--publish-dir",
        default=DOCS,
        type=Path,
        help="directory to publish into when --publish is set (default: docs/)",
    )
    main(ap.parse_args())
