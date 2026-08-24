# Curated regional correspondence entries

The canonical coupling includes 26 regional correspondence entries grouped into
15 modules. Each entry links a set of mouse parcels to a set of human parcels
and contributes a soft cross-species anatomical cost. These entries complement,
but are distinct from, the 21 Garin homology classes used as point anchors and
to fit the spatial warp.

The registry in src/otter/data/anchor_packs/registry.py is the source of truth
for the canonical composition. build_default_pack_entries() returns the entries
in their fitting order.

## Registry

| Pair IDs | Module | Correspondence | Primary source |
|---|---|---|---|
| 30–31 | biccn_motor | M1 and M2/PMd | Bakken et al., Nature (2021), doi:10.1038/s41586-021-03465-8 |
| 32–33 | tectum | superior and inferior colliculus | Isa et al., Current Biology (2021); Winer & Schreiner (2005) |
| 34–35 | olfactory | piriform cortex and anterior olfactory nucleus | Mori (2014) |
| 36–37 | cingulate | subgenual ACC and retrosplenial cortex | Vogt et al., Brain Structure and Function (2012), doi:10.1007/s00429-012-0411-8 |
| 38 | amygdala | cortical subplate/amygdala | Janak & Tye, Nature (2015), doi:10.1038/nature14188 |
| 39–42 | hippocampal | subiculum, CA1, CA3 and dentate gyrus | Strange et al., Nature Reviews Neuroscience (2014); Iglesias et al., NeuroImage (2015) |
| 45 | lateral_pfc | orbitofrontal cortex | Wallis, Nature Neuroscience (2011), doi:10.1038/nn.2956 |
| 47–48 | striatum | dorsolateral and ventromedial caudoputamen | Voorn et al., Trends in Neurosciences (2004), doi:10.1016/j.tins.2004.06.006 |
| 49 | entorhinal | entorhinal cortex | Franjic et al., Neuron (2021), doi:10.1016/j.neuron.2021.10.036 |
| 52 | visual | mouse LM to human V2 | Wang & Burkhalter, Journal of Comparative Neurology (2007), doi:10.1002/cne.21286 |
| 54 | pag | periaqueductal gray | Ezra et al., Human Brain Mapping (2015), doi:10.1002/hbm.22855 |
| 55 | perirhinal | perirhinal cortex | Burwell et al., Hippocampus (1995), doi:10.1002/hipo.450050503 |
| 56–57 | auditory | auditory core and belt | Hackett et al., Journal of Comparative Neurology (2001); Kaas & Hackett, PNAS (2000) |
| 58–60 | somatosensory | face, hand and leg S1 representations | Penfield & Boldrey (1937); Seelke et al., PLOS ONE (2012) |
| 61 | ppc | posterior parietal cortex/BA7 | Whitlock, Current Biology (2017) |

The optional prelimbic-to-dlPFC entry (pair ID 46) is not part of the canonical
registry because that cross-species correspondence is contested.

## Programmatic use

    from otter.data.anchor_packs import build_default_pack_entries

    entries = build_default_pack_entries(
        mouse.var,
        human.var,
        atlas_root=".",
    )

Pass entries as region_anchors when fitting MultimodalFGW, or use the canonical
wrapper in pipeline/run_recommended_model.py. The default outside-region
penalty is 0.15.

## Adding an entry

A new module should expose a builder returning RegionAnchorEntry objects,
reserve non-conflicting pair IDs, cite the comparative-anatomy basis and include
tests for its mouse and human parcel sets. Add it to the registry only when it is
intended to change the canonical recipe; doing so requires refitting the
coupling and all dependent analyses.
