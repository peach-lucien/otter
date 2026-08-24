# Sensorimotor-association coverage diagnostic

This optional analysis compares human-side coupling mass with the cortical sensorimotor-association axis motivated by [Buckner and Krienen (2013)](https://doi.org/10.1016/j.tics.2013.09.017). It is a coverage diagnostic rather than a test of regional homology.

The model-based reconstruction measure is implemented in `experiments/section5_coverage_rigor/`; direct coupling mass from this directory should not be interpreted as reconstruction accuracy or as evidence that a region lacks a mouse counterpart.

Run from the repository root:

```bash
PYTHONPATH=src python experiments/buckner_krienen_2013_tethering/01_tethering_test.py
PYTHONPATH=src python experiments/buckner_krienen_2013_tethering/02_plot.py
```

Outputs:

- `outputs/logs/buckner_krienen_2013_tethering.json`
- `outputs/figures/buckner_krienen_2013_tethering.png`
