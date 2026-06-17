#!/usr/bin/env python3
"""Fetch HOMER's data + artifacts from the Zenodo archive.

The code lives in Git; the data does not (too large, and mostly third-party).
This script downloads the versioned archive described in ``data_manifest.json``
and unpacks it at the repository root.

    python scripts/fetch_data.py                 # reproduce bundle (default)
    python scripts/fetch_data.py --tier raw      # full raw inputs (for a rebuild)
    python scripts/fetch_data.py --tier all      # both
    python scripts/fetch_data.py --check         # verify what's already present

See DATA.md for what each tier contains. Standard library only — no deps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data_manifest.json"

# A few representative files per tier: if these exist, the tier is already
# unpacked. Keeps the script honest without listing every file.
SENTINELS = {
    "reproduce": [
        "outputs/coupling/pi_fc_plus_SC_with_all_packs.npy",
        "outputs/anndata/mouse.h5ad",
        "outputs/anndata/human.h5ad",
        "data_external/human_genes.npy",
    ],
    "raw": [
        "data_external/MouseHumanTranscriptomicSimilarity/AMBA/data/DSURQE_tree.json",
    ],
}


def _load_manifest() -> dict:
    if not MANIFEST.exists():
        sys.exit(f"manifest not found: {MANIFEST}")
    return json.loads(MANIFEST.read_text())


def _is_present(tier: str) -> bool:
    return all((ROOT / p).exists() for p in SENTINELS[tier])


def _md5(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    print(f"  downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _hook(block: int, block_size: int, total: int) -> None:
        if total > 0:
            pct = min(100, block * block_size * 100 // total)
            print(f"\r  {pct:3d}%  ({total / 1e6:.0f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, _hook)  # noqa: S310 (trusted Zenodo URL)
    print()


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract while refusing any member that escapes ``dest`` (path traversal),
    and skipping macOS junk (AppleDouble ``._*`` sidecars, ``.DS_Store``)."""
    dest = dest.resolve()
    safe = []
    for member in tar.getmembers():
        base = Path(member.name).name
        if base.startswith("._") or base == ".DS_Store":
            continue
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest)):
            sys.exit(f"refusing unsafe path in archive: {member.name}")
        safe.append(member)
    tar.extractall(dest, members=safe)  # noqa: S202 (members validated above)


def fetch_tier(tier: str, manifest: dict, force: bool) -> None:
    if _is_present(tier) and not force:
        print(f"[{tier}] already present — skipping (use --force to re-download)")
        return

    spec = manifest["archives"].get(tier)
    if spec is None:
        sys.exit(f"no '{tier}' archive in manifest")
    url = spec.get("url", "")
    if not url or url.startswith("TODO"):
        sys.exit(
            f"[{tier}] manifest URL not set yet. Upload the archive to Zenodo, "
            f"then put the DOI/URL/md5 into {MANIFEST.name}. See DATA.md."
        )

    archive = ROOT / spec["filename"]
    print(f"[{tier}] {spec['filename']} (~{spec.get('approx_size_mb', '?')} MB)")
    _download(url, archive)

    expected = spec.get("md5", "")
    if expected and not expected.startswith("TODO"):
        print("  verifying md5...")
        got = _md5(archive)
        if got != expected:
            sys.exit(f"  checksum mismatch: expected {expected}, got {got}")
        print("  checksum OK")
    else:
        print("  (no md5 in manifest — skipping verification)")

    print("  unpacking at repo root...")
    with tarfile.open(archive, "r:gz") as tar:
        _safe_extract(tar, ROOT)
    archive.unlink()  # remove the tarball after a successful unpack
    print(f"[{tier}] done")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=["reproduce", "raw", "all"], default="reproduce")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--check", action="store_true", help="report presence and exit")
    args = ap.parse_args()

    manifest = _load_manifest()

    if args.check:
        for t in ("reproduce", "raw"):
            print(f"  {t:10s} {'present' if _is_present(t) else 'missing'}")
        return

    tiers = ["reproduce", "raw"] if args.tier == "all" else [args.tier]
    for t in tiers:
        fetch_tier(t, manifest, args.force)


if __name__ == "__main__":
    main()
