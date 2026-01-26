# Gen_Unfold: Diffusion Model for Protein Mechanical Unfolding Curves

## Overview

Gen_Unfold provides a conditional diffusion-based workflow for generating protein mechanical force–extension (F–E) curves and extracting mechanical properties (e.g., unfolding force) from the generated distributions. The repository includes core model code, data processing utilities, evaluation/analysis helpers, and example scripts for training, inference, and plotting.

## Motivation

Single-molecule force spectroscopy (SMFS) provides valuable insights into protein mechanics at the single-molecule level. However, experimental data acquisition can be time-consuming and challenging, and the resulting F-E curves are inherently stochastic and noisy. Molecular dynamics (MD) simulations can generate large datasets but are computationally expensive, especially for slow unfolding events or large proteins.

This project explores using generative models, specifically Diffusion Models, to learn the complex distribution of F-E curves from available data (simulated). A well-trained conditional model can potentially:
- Generate realistic synthetic F-E curves for various proteins and conditions.
- Supplement limited experimental data for downstream analysis.
- Aid in understanding the relationship between protein sequence, unfolding conditions, and mechanical response.
- Facilitate the inference of mechanical properties directly from the generated distributions.

## Repository Layout (Current)

```
.
├── README.md
├── requirements.txt
├── data                      # (Optional) Dataset directory for training
├── scripts
│   ├── run.py                # Training/inference entrypoint (edit paths before use)
│   ├── inference.ipynb       # Notebook example for inference
│   ├── results.py            # Plotting and evaluation results in the paper
│   ├── data                  # Sample data (curves, labels, metadata)
│   ├── 1emb                  # Example PDB-derived features and outputs
│   └── sample_results        # Example output figures
├── src
│   ├── data_processing       # Dataset loading, preprocessing, feature extraction
│   ├── models                # Diffusion model architectures and schedulers
│   ├── training              # Trainers, losses, and training helpers
│   ├── evaluation            # Metrics and visualization utilities
│   ├── analysis              # Curve/property analysis utilities
│   ├── utils.py              # Pipeline, training, and inference glue
│   └── __init__.py
└── trained_models
    └── best_model.pt         # Example checkpoint (config.yaml expected alongside)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# venv\Scripts\activate.bat  # On Windows
pip install -r requirements.txt
```

> If you plan to use protein language models (e.g., ESM-based encoders), additional dependencies may be required depending on your encoder choice.

## Usage

### 1) Training

Edit paths in `scripts/run.py` to point to your local configuration and data, then run:

```bash
python scripts/run.py
```

The training entrypoint calls `train_pipeline(...)` from `src/utils.py`, which expects a YAML config path. This repository does not include a default config file, so you need to provide your own configuration and data paths.

### 2) Inference / Curve Generation

The inference entrypoint uses `curve_prediction(...)` from `src/utils.py` and expects:

- A pretrained model directory containing both `best_model.pt` and `config.yaml`.
- A PDB/mmCIF file (or PDB ID path) plus chain information.

Update `scripts/run.py` or your own script accordingly:

```python
from Gen_Unfold.src import curve_prediction

curve_prediction(
    pretrained_model_path="/path/to/checkpoint_dir",
    pdb_id_or_path="/path/to/1emb.cif",
    chain="A",
    feature_path="/path/to/feature_dir",
    save_path="/path/to/predictions.npy",
    num_samples=1024,
    device="cuda",
)
```

### 3) Plotting & Evaluation

Use `scripts/results.py` to generate comparison plots and metrics. The default helpers assume the sample arrays under `scripts/data` and will output figures such as violin plots and curve comparisons.

```bash
python scripts/results.py
```

Example plots are available in `scripts/sample_results/`.

## Notes

- `trained_models/best_model.pt` is an example checkpoint. The inference pipeline expects a `config.yaml` alongside the checkpoint; make sure your trained model directory contains both.
- Sample data used for plotting is under `scripts/data/`.
- The repository is research-oriented; some paths in scripts are placeholders and should be updated to match your local environment.

## License

Specify your project's license here.
