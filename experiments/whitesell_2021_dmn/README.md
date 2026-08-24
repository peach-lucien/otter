# Whitesell 2021 default-mode network routing

This auxiliary analysis routes a mouse default-mode parcel set derived from
[Whitesell et al. (2021)](https://doi.org/10.1016/j.neuron.2021.01.011) through
the canonical OTTER coupling. Translated mass is summarized over the human
Yeo-7 networks and subcortex and compared descriptively with the existing OTTER
mouse-network definitions.

The analysis evaluates routing only. The Whitesell parcel set is not used as an
anchor pack, and differences between alternative mouse-network definitions are
not treated as a formal model comparison.

## Files

| File | Purpose |
|---|---|
| `01_whitesell_dmn_refinement.py` | Aggregates translated mass from the Whitesell parcel set over human networks. |

## Inputs and output

The script uses the canonical coupling, cached mouse and human parcellations,
and the existing Pagani and Coletta network-summary logs. These inputs are
available from the Zenodo reproduce bundle.

It writes `outputs/logs/whitesell_2021_dmn_refinement.json`.

## Run

From the repository root:

```bash
PYTHONPATH=src python experiments/whitesell_2021_dmn/01_whitesell_dmn_refinement.py
```
