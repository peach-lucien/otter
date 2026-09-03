# OTTER notebooks

Download the released data before running the notebooks:

```bash
python scripts/fetch_data.py
```

Run notebooks from the repository root so that data paths resolve consistently.

| Notebook | Purpose |
|---|---|
| [`01_quickstart.ipynb`](01_quickstart.ipynb) | Load the canonical coupling and query mouse–human correspondences |
| [`02_methodology.ipynb`](02_methodology.ipynb) | Inspect the coupling, cost terms and fitting parameters |
| [`03_coupling.ipynb`](03_coupling.ipynb) | Summarise coupling concentration and broad anatomical organisation |
| [`04_cost_terms_and_supervision.ipynb`](04_cost_terms_and_supervision.ipynb) | Reproduce cost-term and supervision-withheld analyses |
| [`05_map_transfer.ipynb`](05_map_transfer.ipynb) | Route functional, microstructural and marker-expression maps |
| [`06_vs_transbrain.ipynb`](06_vs_transbrain.ipynb) | Reproduce the comparative-method analysis |
| [`07_coverage.ipynb`](07_coverage.ipynb) | Compute mouse-based reconstruction of human connectivity |
| [`08_disease.ipynb`](08_disease.ipynb) | Demonstrate translation of a phenotype map |
| [`09_disease_dimensions.ipynb`](09_disease_dimensions.ipynb) | Summarise Alzheimer phenotype confirmation, TMS symptom-circuit dissociation and Parkinson stage validation |

The Beauchamp scoring frame used in notebook 04 contains 19 scorable region pairs. The pairs
inform hyperparameter evaluation but are not supplied as anatomical correspondence constraints.
Some benchmark territories overlap OTTER's anatomical scaffold, so the notebook also includes
target-wise supervision-withheld refits.

Use `load_pi()` rather than hard-coding a coupling filename. It loads `pi_canonical.npy` by
default. Logs read by the notebooks record the coupling SHA-256 and should be checked against
`pi_provenance()` before values are compared.

Some analyses require model refits or third-party software and therefore read the corresponding
provenance-stamped result from `outputs/logs/` by default. The producing script is identified in
the notebook when a refit is optional. Explorer labels shown by the notebooks are interface
metadata, not confidence or validation tiers.
