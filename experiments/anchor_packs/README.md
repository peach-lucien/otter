# Anchor pack experiments

Per-pack runner scripts that fit a variant of π with one or more anchor packs added on top of the 21 Garin point anchors. Each script writes its fitted π to `outputs/coupling/pi_fc_plus_SC_with_<pack_name>.npy` and a Beauchamp + region-level evaluation log to `outputs/logs/`.

See `docs/04_anchor_packs.md` for the citations and biological motivation behind each pack.

## Per-pack runners

| Script | Pack contents | Default? |
|---|---|:---:|
| `biccn_motor.py` | BICCN M1 → human area 4 (Bakken 2021) | ✅ |
| `tectum.py` | Mouse SC + IC → human tectal regions (May 2006) | ✅ |
| `olfactory.py` | Mouse piriform + AON → human olfactory cortex (Mori 2014) | ✅ |
| `cingulate.py` | Mouse Cg1/Cg2/RSC → human ACC sub-areas (Vogt 2019) | opt-in |
| `amygdala.py` | Mouse amygdalar nuclei → human amygdala (Janak & Tye 2015) | ✅ |
| `hippocampal.py` | Mouse CA1/CA3/DG/Sub → human hippocampal subfields (Strange 2014) | ✅ |
| `lateral_pfc.py` | Mouse PrL → human dlPFC/vlPFC/OFC (Wallis 2012, Carlén 2017) | opt-in |
| `entorhinal.py` | Mouse ERh → human EC (Burwell 1995) | ✅ |
| `striatum.py` | Mouse CPu → human putamen + caudate (Voorn 2004) | opt-in |
| `visual.py` | Mouse V1+V2 extrastriate → human V1 + LO (Wang 2011) | ✅ |
| `pag.py` | Mouse PAG → human PAG (Linnman 2012) | ✅ |
| `compose_all.py` | All default packs in one fit | — |

Opt-in packs are biologically defensible but hurt one Beauchamp validation metric — see `docs/04_anchor_packs.md` for the trade-off discussion.

## Reproduce production-with-all-packs π

```bash
PYTHONPATH=src python experiments/anchor_packs/compose_all.py
```

This produces `outputs/coupling/pi_fc_plus_SC_with_all_packs.npy` — the recommended π for downstream queries.

## Reproduce individual pack ablations

```bash
PYTHONPATH=src python experiments/anchor_packs/biccn_motor.py
# … etc for each pack
```

Per-pack outputs are useful for ablations ("does removing the cingulate pack change DMN-DMN row-mass?") and for the multi-source trust map.
