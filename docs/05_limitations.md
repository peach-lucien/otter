# Limitations

## Anatomical supervision

OTTER extends a curated anatomical frame through connectivity and spatial
structure. It does not discover homology without supervision. In
leave-one-supervision-unit-out refits, broad regional correspondence is more
stable than parcel-exact recovery; fine-scale assignments should therefore be
interpreted cautiously outside directly supported territories.

The Beauchamp correspondences are not supplied as OTTER constraints, but some
benchmark territories overlap the Garin-derived spatial scaffold or curated
regional entries. Target-wise supervision-withheld refits are the appropriate
generalisation analysis.

## Spatial and parcellation limits

The mouse and human atlases have different native coordinate systems. The
spatial cost is defined after a thin-plate-spline warp fitted to the Garin
landmarks and is therefore itself anatomically supervised.

The analysis is limited to the released 1,864-parcel mouse and 2,094-parcel
human representations. Cerebellum and medulla are excluded, and several nuclei
or cortical subdivisions are aggregated. OTTER cannot support claims finer than
the input parcellations.

## Group-average correspondence

The coupling is fitted to group-average connectomes. It represents a species-
level correspondence and does not model individual variation, development,
disease state or experimental condition.

## Coupling weights are not confidence

The concentration of a coupling row depends on the solver's proximal weight and
stopping point. A high top weight should not be interpreted as a calibrated
probability that the target is correct. Explorer labels are descriptive interface
metadata, not validation or confidence tiers.

## Cross-modal transfer

Successful transfer of a spatial map supports consistency at the scale and
modality tested. It does not prove cell-type equivalence, causal conservation or
one-to-one homology. The principal-gradient analysis is an internal consistency
test because it is derived from the same connectomes that enter OTTER. Marker-
expression maps are proxies and should not be described as cell abundance.

## Association cortex

Mouse-based reconstruction of human functional connectivity is lower in parts
of association cortex. This concerns connectional organisation: it does not show
that a human area has no molecular or anatomical counterpart. Claims in these
territories should be matched to the modality being translated.

## Contested correspondences

Some cross-species correspondences remain debated. In particular, the optional
prelimbic-to-dlPFC entry is excluded from the canonical regional-entry registry.
Users who introduce additional regional supervision should document both its
comparative-anatomy source and any competing interpretation.
