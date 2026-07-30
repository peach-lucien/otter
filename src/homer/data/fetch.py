"""Fetch HOMER's data + artifacts from the versioned Zenodo archive.

The code lives in Git; the data does not (too large, and mostly third-party).
This module downloads the archive described in ``data_manifest.json`` (at the
repository root) and unpacks it in place. It is used two ways:

* as a CLI, via ``scripts/fetch_data.py`` (a thin wrapper around :func:`main`);
* as a guard, via :func:`ensure_data`, which library functions call before they
  read a data file, so a user who forgot to fetch gets a clear prompt (or a
  clear error in non-interactive use) instead of a bare ``FileNotFoundError``.

Set ``HOMER_AUTO_FETCH=1`` to download missing data without prompting.
See ``DATA.md`` for what each tier contains. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

# Representative files per tier: if these exist, the tier is already unpacked.
# Files whose presence means a tier is already unpacked. These decide whether fetch_tier()
# skips the download, so they must include anything a NEWER archive added: a user who fetched
# v1.2.0 has every pre-warp file on disk, and if the canonical coupling is not listed here they
# are told "already present, skipping" and never receive v1.3.0. Add an entry whenever the
# archive gains a file the notebooks require.
SENTINELS = {
    "reproduce": [
        # added in v1.3.0 - what load_pi() returns and what every notebook needs
        "outputs/coupling/pi_canonical.npy",
        "outputs/coupling/trust_multisource_canonical.npz",
        "outputs/anndata/_schaefer_order.txt",
        "outputs/coupling/mouse_tpl_100um.nii.gz",
        # present since v1.0.0
        "outputs/coupling/pi_fc_plus_SC_with_all_packs.npy",
        "outputs/coupling/pi_fc_plus_SC.npy",
        "outputs/anndata/mouse.h5ad",
        "outputs/anndata/human.h5ad",
        "outputs/anndata/full_costs.npz",
        "data_external/human_genes.npy",
    ],
    "raw": [
        "data_external/MouseHumanTranscriptomicSimilarity/AMBA/data/DSURQE_tree.json",
    ],
}


class DataNotFound(FileNotFoundError):
    """Raised when required data is absent and not (or cannot be) fetched."""


def find_root(start: Path | None = None) -> Path:
    """Locate the repo root: the nearest ancestor containing data_manifest.json.

    Searches upward from ``start`` (default: this file), then from the current
    working directory. Falls back to the package's parents[3] (src/homer/data ->
    repo root) so an editable install still works if the manifest is missing.
    """
    candidates = []
    here = (start or Path(__file__)).resolve()
    candidates.extend([here, *here.parents])
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    for d in candidates:
        if d.is_dir() and (d / "data_manifest.json").exists():
            return d
    return Path(__file__).resolve().parents[3]


def _load_manifest(root: Path) -> dict:
    mf = root / "data_manifest.json"
    if not mf.exists():
        raise DataNotFound(f"data_manifest.json not found at {root}")
    return json.loads(mf.read_text())


def _is_present(tier: str, root: Path) -> bool:
    return all((root / p).exists() for p in SENTINELS[tier])


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
            raise DataNotFound(f"refusing unsafe path in archive: {member.name}")
        safe.append(member)
    tar.extractall(dest, members=safe)  # noqa: S202 (members validated above)


def fetch_tier(tier: str, *, root: Path | None = None, force: bool = False) -> None:
    """Download, verify, and unpack one tier ('reproduce' or 'raw')."""
    root = root or find_root()
    manifest = _load_manifest(root)

    if _is_present(tier, root) and not force:
        print(f"[{tier}] already present, skipping (force=True to re-download)")
        return

    spec = manifest["archives"].get(tier)
    if spec is None:
        raise DataNotFound(f"no '{tier}' archive in manifest")
    url = spec.get("url", "")
    if not url or url.startswith("TODO"):
        raise DataNotFound(
            f"[{tier}] manifest URL not set. Add the Zenodo DOI/URL/md5 to "
            f"{root / 'data_manifest.json'} (see DATA.md)."
        )

    archive = root / spec["filename"]
    print(f"[{tier}] {spec['filename']} (~{spec.get('approx_size_mb', '?')} MB)")
    _download(url, archive)

    expected = spec.get("md5", "")
    if expected and not expected.startswith("TODO"):
        print("  verifying md5...")
        got = _md5(archive)
        if got != expected:
            archive.unlink(missing_ok=True)
            raise DataNotFound(f"  checksum mismatch: expected {expected}, got {got}")
        print("  checksum OK")
    else:
        print("  (no md5 in manifest, skipping verification)")

    print("  unpacking at repo root...")
    with tarfile.open(archive, "r:gz") as tar:
        _safe_extract(tar, root)
    archive.unlink(missing_ok=True)
    print(f"[{tier}] done")


def _bundle_mb(tier: str, root: Path) -> str:
    try:
        return str(_load_manifest(root)["archives"][tier].get("approx_size_mb", "?"))
    except Exception:
        return "?"


def ensure_data(required, *, tier: str = "reproduce", what: str = "this step",
                root: Path | None = None) -> None:
    """Make sure ``required`` files exist; otherwise prompt to fetch them.

    ``required`` is an iterable of repo-relative paths. If any are missing:

    * with ``HOMER_AUTO_FETCH=1`` set, download the tier automatically;
    * else, in an interactive terminal, ask the user whether to download now;
    * else (or if they decline), raise :class:`DataNotFound` with the exact
      command to run.
    """
    root = root or find_root()
    missing = [str(p) for p in required if not (root / p).exists()]
    if not missing:
        return

    size = _bundle_mb(tier, root)
    listing = "\n".join(f"  - {p}" for p in missing[:8])
    auto = os.environ.get("HOMER_AUTO_FETCH", "").strip().lower() in ("1", "true", "yes")

    if not auto and sys.stdin.isatty() and sys.stdout.isatty():
        print(f"\nHOMER needs data for {what} that isn't present yet:\n{listing}")
        resp = input(
            f"Download the '{tier}' data bundle (~{size} MB) from Zenodo now? [y/N] "
        )
        auto = resp.strip().lower() in ("y", "yes")

    if not auto:
        raise DataNotFound(
            f"Data required for {what} is not present:\n{listing}\n\n"
            f"Fetch it with:\n"
            f"    python scripts/fetch_data.py" + (f" --tier {tier}" if tier != "reproduce" else "") + "\n\n"
            f"Or set HOMER_AUTO_FETCH=1 to download automatically when needed. "
            f"See DATA.md."
        )

    fetch_tier(tier, root=root)
    still = [str(p) for p in required if not (root / p).exists()]
    if still:
        raise DataNotFound(
            "Fetch completed but these files are still missing:\n"
            + "\n".join(f"  - {p}" for p in still)
        )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="fetch_data", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=["reproduce", "raw", "all"], default="reproduce")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument("--check", action="store_true", help="report presence and exit")
    args = ap.parse_args(argv)

    root = find_root()
    if args.check:
        for t in ("reproduce", "raw"):
            print(f"  {t:10s} {'present' if _is_present(t, root) else 'missing'}")
        return

    tiers = ["reproduce", "raw"] if args.tier == "all" else [args.tier]
    for t in tiers:
        fetch_tier(t, root=root, force=args.force)


if __name__ == "__main__":
    main()
