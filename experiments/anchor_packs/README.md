# Anchor pack experiments

Per-pack runner scripts that fit a variant of π with one or more anchor packs added on top of the 21 Garin point anchors. Each script writes its fitted π to `outputs/coupling/pi_fc_plus_SC_with_<pack_name>.npy` and a Beauchamp + region-level evaluation log to `outputs/logs/`.

See `docs/04_anchor_packs.md` for the citations and biological motivation behind each pack.

## Which packs are "default"

The canonical coupling (`outputs/coupling/pi_canonical.npy`, what `load_pi()` returns) and the retired pre-warp `pi_fc_plus_SC_with_all_packs.npy` are both composed from the packs flagged `default=True` in the pack registry, `src/otter/data/anchor_packs/registry.py`, the single source of truth. `compose_all.py`, the GUI builder (`pipeline/08_build_gui.py`), and the multi-source trust step (`pipeline/08a_multisource_trust.py`) all read that registry, so it cannot drift from what is actually fitted.

**All 15 packs are in the recommended composition** (26 region-anchor entries); a multi-benchmark comparison favoured the full set. The canonical coupling adds the anchor-warped spatial cost on top of that composition. To change which packs are composed, flip the `default` flag in the registry and re-run `compose_all.py` (or `pipeline/run_recommended_model.py`).

## Per-pack runners

Each runner below fits a single-pack ablation variant. All of these packs are also in the recommended composition; `compose_all.py` fits all 15 at once.

| Script | Pack contents | Note |
|---|---|---|
| `biccn_motor.py` | Mouse M1 + M2/PMd → human area 4 / area 6 (Bakken 2021) | |
| `tectum.py` | Mouse SC + IC → human tectal regions (Isa 2021) | |
| `olfactory.py` | Mouse piriform + AON → human olfactory cortex (Mori 2014) | |
| `amygdala.py` | Mouse cortical subplate → human amygdala (Janak & Tye 2015) | |
| `hippocampal.py` | Mouse Sub/CA1/CA3/DG → human hippocampal subfields (Strange 2014) | |
| `cingulate.py` | Mouse subgenual ACC + RSC → human ACC sub-areas (Vogt 2012) | Beauchamp ACG trade-off |
| `lateral_pfc.py` | Mouse OFC + PrL → human OFC + dlPFC (Wallis 2011, Carlén 2017) | dlPFC entry contested |
| `entorhinal.py` | Mouse entorhinal area → human entorhinal cortex (Franjic 2021) | |
| `striatum.py` | Mouse CPu dorsolateral/ventromedial → human putamen + caudate (Voorn 2004) | |
| `visual.py` | Mouse LM → human V2 (Wang & Burkhalter 2007) | Beauchamp cuneus trade-off |
| `pag.py` | Mouse PAG → human PAG (Ezra 2015) | |
| `compose_all.py` | All 15 registry packs in one fit | builds the pre-warp coupling (retired) |

The trade-offs (cingulate, somatosensory, visual lower a coarse Beauchamp metric for one region; the dlPFC entry is anatomically contested) are documented per pack in `docs/04_anchor_packs.md`. Four further packs, `perirhinal`, `auditory`, `somatosensory`, `ppc`, are in the recommended composition and exist as builders in `src/otter/data/anchor_packs/` but have no standalone runner here; compose them programmatically via the registry.

## Reproduce production-with-all-packs π

```bash
PYTHONPATH=src python experiments/anchor_packs/compose_all.py
```

This produces `outputs/coupling/pi_fc_plus_SC_with_all_packs.npy`, the **pre-warp** coupling. It is superseded by `pi_canonical.npy`, which adds the anchor-warped spatial cost; `load_pi()` returns the canonical one. Use this recipe only to reproduce the pre-warp comparison. To run the whole recommended-model pipeline (solve → compose → bootstrap → trust → GUI) end to end:

```bash
PYTHONPATH=src python pipeline/run_recommended_model.py
```

## Reproduce individual pack ablations

```bash
PYTHONPATH=src python experiments/anchor_packs/biccn_motor.py
# … etc for each runner script above
```

Per-pack outputs support ablations, such as the effect of removing the cingulate pack on DMN-DMN row-mass, and the multi-source trust map.
