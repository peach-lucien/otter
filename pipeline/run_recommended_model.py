"""Refit the canonical OTTER coupling and write a provenance sidecar.

The canonical recipe is defined in otter.repro. By default this command writes
pi_canonical_refit.npy and leaves the released pi_canonical.npy untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otter.repro import (  # noqa: E402
    CANONICAL,
    anchor_warped_xyz,
    fit_coupling,
    load_inputs,
    refit_provenance,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/coupling/pi_canonical_refit.npy"),
        help="output .npy path relative to the repository root",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file",
    )
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.suffix != ".npy":
        raise SystemExit("--output must end in .npy")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {output}; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading processed inputs and canonical regional entries")
    mouse, human, costs, entries = load_inputs(ROOT)
    print("Building the Garin-anchor-warped spatial cost")
    spatial_cost = anchor_warped_xyz(mouse, human)

    print(f"Fitting canonical coupling with {len(entries)} regional entries")
    started = time.time()
    pi = fit_coupling(
        mouse,
        human,
        costs,
        entries,
        spatial_cost,
        **CANONICAL,
    )
    elapsed = time.time() - started
    np.save(output, pi)

    comparison = refit_provenance(pi, recipe=CANONICAL)
    sidecar = {
        "output": str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),
        "sha256": _sha256(output),
        "shape": list(pi.shape),
        "elapsed_seconds": elapsed,
        **comparison,
    }
    output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")

    print(f"Saved {output}")
    print(f"Saved {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
