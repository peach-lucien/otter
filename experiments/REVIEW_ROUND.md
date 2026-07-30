# Review-round analyses

The analyses added during the review rounds, ported from the scratch harness they were first written
in. Each writes a log under `outputs/logs/` that the manuscript cites, and each stamps that log with
the coupling or the recipe that produced it.

All of them import the fitting recipe from `homer.repro` rather than restating it, so an arm fitted
here is the same arm the figure scripts use.

## Order

Three of these share couplings, so they run in sequence. The rest are independent.

```
section2_supervision/02_ablation_ladder.py        writes outputs/coupling/pi_ladder_*.npy
  section2_supervision/03_downstream_by_arm.py    reads them
  section3_transfer/04_gradient_components.py     reads them
```

The ladder saves each arm's coupling instead of letting the downstream scripts refit. Refitting
would give couplings that differ from the ones scored in Figure 2a, and the arms would no longer be
comparable across figures.

| script | log | manuscript |
|---|---|---|
| `pipeline/02b_build_splithalf_fc.py` | `outputs/splithalf/*.npz` | input to the split-half refit |
| `section1_stability/01_split_half_refit.py` | `out_a2_splithalf.json` | section 1, ED 2 |
| `section2_supervision/02_ablation_ladder.py` | `out_a1_ladder.json` | section 2, Fig. 2a, ED 3 |
| `section2_supervision/03_downstream_by_arm.py` | `out_a1c_downstream.json` | section 5, ED 5 |
| `section3_transfer/04_gradient_components.py` | `out_c1_gradient.json` | section 3, ED 6 |

Five scripts are still to be ported. They cover the held-out three-config comparison
(`out_a1b_loro`), the section 5 controls (`out_a3_section5`), the two null models and effective
resolution (`out_c2_nulls`), the epsilon and modality robustness checks (`out_a1d_robust`), and the
ABIDE case-control analysis (`abide_magel2_casecontrol`). `out_g2_regret` has no surviving producer
and needs one written.

## Provenance

A script that loads a coupling stamps the file it opened:

```python
from homer.repro import provenance, stamp
OUT.write_text(json.dumps(stamp(out, **provenance()), indent=2))
```

A script that fits its own coupling records the recipe and its measured distance from the release,
because borrowing the released sha would claim an input the run never used:

```python
from homer.repro import refit_provenance, stamp
OUT.write_text(json.dumps(stamp(out, **refit_provenance(pi, recipe=config)), indent=2))
```

`tools/check_repro_harness.py` reports any cited log that carries no stamp, and resolves every stamp
to a file in `outputs/coupling/`.
