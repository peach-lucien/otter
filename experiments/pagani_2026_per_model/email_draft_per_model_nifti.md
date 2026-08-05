# Draft email to Silvia, request for per-model connectivity maps

Subject: OTTER follow-up, per-model degree-centrality maps?

Dear Silvia,

Thank you very much, the package was exactly what we needed. For the record, on
our side everything checked out: the clean `sorted_etiology_by_feature_matrix.csv`
matched your Fig 1c and let us recover the hyper/hypo subtype split directly from
the data (n=9 hyper / n=11 hypo), the Fig 1d occurrence maps and the 13
conserved-region masks are all in Allen space and align with our mouse atlas, and
the chd8 functional templates registered cleanly.

We've now reproduced the subtype-level cross-species mapping with our
optimal-transport coupling, and the mouse hyper/hypo signatures translate to the
matching human subtypes, so we'd love to push to the **per-model** level next.

For that, the one thing that would unblock us is the **per-model voxelwise
weighted-degree-centrality maps**, i.e. the mutant-vs-WT global connectivity
maps behind Fig 1a/b, one NIfTI per model (all 20), in the chd8 functional space
(or Allen space, whichever is native to your pipeline).

To explain why the `sorted_etiology_by_feature_matrix.csv` can't substitute here:
its 1,491 columns are a downsampled, dendrogram-sorted reduction, and there's no
feature-index → voxel key in the supplement, so we can't place each value back in
the brain to route it through the coupling. The full-resolution per-model maps
your pipeline produces would let us map each model individually to human-parcel
space.

No rush at all, and thank you again for sharing so much already.

Best wishes,
Robert
