# Regional correspondence packs

OTTER's released coupling uses 26 curated regional correspondences grouped into 15 packs, in addition to the 21 Garin point anchors. The pack registry is the source of truth for the released composition:

- `src/otter/data/anchor_packs/registry.py` defines the included packs.
- `src/otter/data/anchor_packs/` contains their builders.
- [`docs/04_anchor_packs.md`](../../docs/04_anchor_packs.md) lists the anatomical sources and parcel definitions.

The canonical regional-entry membership and fitting order are defined by
`src/otter/data/anchor_packs/registry.py`. Use
`build_default_pack_entries()` when refitting the canonical model through
`otter.repro`.

Run from the repository root:

```bash
PYTHONPATH=src python experiments/anchor_packs/compose_all.py
```

The remaining scripts in this directory fit individual-pack variants. Four packs (`perirhinal`, `auditory`, `somatosensory` and `ppc`) are available through the registry but do not have standalone runners.
