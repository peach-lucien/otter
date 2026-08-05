# Review-round analyses

The analyses added during the review rounds, ported from the scratch harness they were first written
in. Each writes a log under `outputs/logs/` that the manuscript cites, and each stamps that log with
the coupling or the recipe that produced it.

All of them import the fitting recipe from `otter.repro` rather than restating it, so an arm fitted
here is the same arm the figure scripts use.

## Order

Three of these share couplings, so they run in sequence. The rest are independent.

```
section2_supervision/02_ablation_ladder.py        writes outputs/coupling/pi_ladder_*.npy
  section2_supervision/03_downstream_by_arm.py    reads them        NOT YET WRITTEN
  section3_transfer/04_gradient_components.py     reads them        NOT YET WRITTEN

section2_supervision/05_heldout_three_config.py   writes out_a1b_loro.json
  section2_supervision/06_regret.py               reads it
```

Because neither consumer of the ladder exists yet, whether `02_ablation_ladder.py` actually saves
its per-arm couplings has not been exercised. Check that `outputs/coupling/pi_ladder_*.npy` appear
before relying on the ordering above.

The ladder saves each arm's coupling instead of letting the downstream scripts refit. Refitting
would give couplings that differ from the ones scored in Figure 2a, and the arms would no longer be
comparable across figures.

| script | log | manuscript | state |
|---|---|---|---|
| `pipeline/02b_build_splithalf_fc.py` | `outputs/splithalf/*.npz` | input to the split-half refit | written |
| `section1_stability/01_split_half_refit.py` | `out_a2_splithalf.json` | section 1, ED 2 | written |
| `section2_supervision/02_ablation_ladder.py` | `out_a1_ladder.json` | section 2, Fig. 2a, ED 3 | written |
| `section2_supervision/05_heldout_three_config.py` | `out_a1b_loro.json`, `heldout_three_config_canonical.json` | section 2, Fig. 2c, ED 4 | written, not yet run |
| `section2_supervision/06_regret.py` | `out_g2_regret.json` | section 2 | written, reproduces the committed log |
| `section2_supervision/03_downstream_by_arm.py` | `out_a1c_downstream.json` | section 5, ED 5 | **not written** |
| `section3_transfer/04_gradient_components.py` | `out_c1_gradient.json` | section 3, ED 6 | **not written** |

An earlier version of this file listed `03_downstream_by_arm.py` and `04_gradient_components.py`
as ported. Neither exists, and `experiments/section3_transfer/` is not a directory. Their logs are
committed and cited, so the manuscript rests on numbers the repo cannot regenerate.

Five analyses still have no producer at all. They are the section 5 controls (`out_a3_section5`), the two
null models and effective resolution (`out_c2_nulls`), the epsilon and modality robustness checks
(`out_a1d_robust`), the ABIDE case-control analysis (`abide_magel2_casecontrol`), and the section 5
downstream and gradient scripts named above.

Every producer takes `--check`, which scores without writing and compares against the committed
log. A port that does not reproduce its log is a fault in the port. Report it rather than editing
the manuscript to agree with the re-run.

## Provenance

A script that loads a coupling stamps the file it opened:

```python
from otter.repro import provenance, stamp
OUT.write_text(json.dumps(stamp(out, **provenance()), indent=2))
```

A script that fits its own coupling records the recipe and its measured distance from the release,
because borrowing the released sha would claim an input the run never used:

```python
from otter.repro import refit_provenance, stamp
OUT.write_text(json.dumps(stamp(out, **refit_provenance(pi, recipe=config)), indent=2))
```

`tools/check_repro_harness.py` reports any cited log that carries no stamp, and resolves every stamp
to a file in `outputs/coupling/`.
