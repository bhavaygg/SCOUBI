# SCOUBI

SCOUBI (Signaling-informed Characterization of Unresolved Biological Interfaces) is a Python toolkit for identifying and profiling axon-dendrite interface regions in spatial transcriptomics data. It starts from transcript coordinates and cell annotations, builds a spatial bin representation, learns axon-versus-dendrite structure with marker genes and ligand-receptor communication signals, then summarizes expression and interaction patterns at the resulting interface map.

## What SCOUBI Provides

- Spatial binning of transcript-level data into extrasomatic and cell-associated regions
- Lightweight neural-network annotation of bins as axonic or dendritic
- Interface detection from neighboring axon and dendrite predictions
- Gene-level enrichment, empirical background testing, and region-aware proportion tests
- Interface-level expression and communication profiling by cell type or region
- Plotting helpers for overlays, enrichment scatter plots, radial summaries, and hierarchical tree views

## Installation

SCOUBI requires Python 3.10 or newer.

```bash
pip install scoubi
```

For local development from a checkout:

```bash
pip install -e .
```

## Quickstart

```python
import torch
import scoubi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adata = scoubi.io.load_data(
    "data/data.parquet",
    cell_type="data/cell_types.csv",
    region="data/regions.csv",
)

adata = scoubi.pp.bin_data(adata, binsize=2)
adata = scoubi.md.train(adata, axon_markers, dendrite_markers, device=device)
adata = scoubi.tl.overview(adata, device=device)
adata = scoubi.tl.axon_dendrite_enrichment(adata)
adata = scoubi.tl.distance(adata)
adata = scoubi.tl.expression_profile(adata, key="region")
adata = scoubi.tl.communication_profile(adata, key="region")

adata.summarize()
```

## Module Map

| Module | Alias | Purpose |
|---|---|---|
| `scoubi.io` | - | Data loading and AnnData helpers |
| `scoubi.preprocess` | `scoubi.pp` | Spatial binning |
| `scoubi.model` | `scoubi.md` | Axon/dendrite bin annotation |
| `scoubi.tools` | `scoubi.tl` | Downstream analysis utilities |
| `scoubi.plotting` | `scoubi.pl` | Visualization helpers |

## Data and Runtime Notes

- `scoubi.md.train()` can use a bundled ligand-receptor reference table when `pairs=None`.
- Large example datasets such as `data.h5ad` are not shipped as install-time package data.
- GPU acceleration is optional; most workflows can run on CPU, although training and convolution-heavy steps are faster on CUDA when available.

## Tutorial

The guided walkthrough lives in [tutorial.ipynb](tutorial.ipynb). It covers data loading, model training, spatial overview generation, enrichment analysis, interface profiling, communication analysis, and save/load workflows.

## License

Released under the MIT License. See [LICENSE](LICENSE).
