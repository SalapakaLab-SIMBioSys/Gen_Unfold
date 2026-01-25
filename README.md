# Project Title: Diffusion Model for Protein Mechanical Unfolding Force-Extension Curve Generation and Characteristic Inference

## Overview

This project implements a deep learning framework based on Conditional Diffusion Models to generate synthetic Force-Extension (F-E) curves for protein mechanical unfolding experiments (such as those performed with AFM-SMFS, optical tweezers, or magnetic tweezers). The model is designed to condition the generation process on the protein's amino acid sequence and experimental conditions (e.g., pulling speed).

The generated F-E curves can then be used to infer and analyze key mechanical properties and unfolding mechanisms of the proteins, complementing experimental and simulation approaches.

## Motivation

Single-molecule force spectroscopy (SMFS) provides valuable insights into protein mechanics at the single-molecule level. However, experimental data acquisition can be time-consuming and challenging, and the resulting F-E curves are inherently stochastic and noisy. Molecular dynamics (MD) simulations can generate large datasets but are computationally expensive, especially for slow unfolding events or large proteins.

This project explores using generative models, specifically Diffusion Models, to learn the complex distribution of F-E curves from available data (experimental and/or simulated). A well-trained conditional model can potentially:
- Generate realistic synthetic F-E curves for various proteins and conditions.
- Supplement limited experimental data for downstream analysis.
- Aid in understanding the relationship between protein sequence, unfolding conditions, and mechanical response.
- Facilitate the inference of mechanical properties directly from the generated distributions.

## Project Structure

The codebase is organized into the following directories:
.

├── data/                 # Raw and processed data files

├── notebooks/            # Jupyter notebooks for exploration and development

├── src/                  # Source code

│   ├── data_processing/  # Modules for loading, preprocessing, and preparing data

│   ├── models/           # Modules for the Diffusion Model architecture and components

│   ├── training/         # Modules for the training loop, losses, and optimizers

│   ├── inference/        # Modules for generating new data using the trained model

│   └── analysis/         # Modules for analyzing F-E curves and extracting properties

│   └── evaluation/       # Modules for evaluating model performance and visualizing results

├── config/               # YAML configuration files for data, model, and training parameters

├── scripts/              # Executable scripts for running workflow steps

├── trained_models/       # Directory to save trained model checkpoints

├── results/              # Directory to save analysis results and generated curves

├── README.md             # Project overview and instructions (this file)

└── requirements.txt      # Python dependencies

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate.bat # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you plan to use Protein Language Models (PLMs) like ESM-2 and have `protein_encoding_type: raw` in your model config, you might need to additionally install the `transformers` library: `pip install transformers`.*

## Usage

The project workflow typically involves the following steps:

1.  **Prepare your raw data:** Place your raw experimental or simulation data files in the `data/raw/` directory. Ensure your raw data format is consistent and can be loaded by the preprocessing script (you may need to adapt `scripts/preprocess_data.py` and `src/data_processing/utils.py` to your specific format).

2.  **Configure the pipeline:** Update the YAML files in the `config/` directory (`data_config.yaml`, `model_config.yaml`, `training_config.yaml`) with your specific data paths, preprocessing settings, model hyperparameters, and training parameters.

3.  **Preprocess the data:** Run the preprocessing script to convert your raw data into a standardized format suitable for training and inference.
    ```bash
    python scripts/preprocess_data.py --config config/data_config.yaml
    ```
    This will save processed data files (e.g., `fe_curves.npy`, `sequences.csv`, `conditions.npy`) into the directory specified in `data_config.yaml`.

4.  **Train the model:** Run the training script to train the Conditional Diffusion Model.
    ```bash
    python scripts/train.py --data_config config/data_config.yaml --model_config config/model_config.yaml --training_config config/training_config.yaml --device cuda # or cpu
    ```
    Training progress will be logged to TensorBoard in the `runs/` directory, and model checkpoints will be saved in the `checkpoints/` directory. You can resume training from a checkpoint using the `--checkpoint` argument.

5.  **Generate synthetic curves:** Use the trained model to generate new F-E curves for specific protein sequences and conditions. Create an input file (e.g., CSV) containing the sequences and conditions you want to generate for.
    ```bash
    python scripts/generate.py --data_config config/data_config.yaml --model_config config/model_config.yaml --checkpoint checkpoints/best_model.pt --input_data data/generation_inputs.csv --output results/generated_curves.npy --num_samples_per_input 10 --device cuda # or cpu
    ```
    Adjust `best_model.pt` to the path of your desired checkpoint. `generation_inputs.csv` should contain columns for sequences and conditions as defined in your `data_config.yaml`.

6.  **Analyze and evaluate results:** Compare the generated curves against true curves (from your processed data) and evaluate their quality using various metrics and visualizations.
    ```bash
    python scripts/analyze_results.py --data_config config/data_config.yaml --true_data data/processed/fe_curves.npy --generated_data results/generated_curves.npy --output_dir results/evaluation_analysis
    ```
    This will save evaluation metrics (e.g., to a JSON file) and plots to the specified output directory.

## Configuration

The `config/` directory contains YAML files that control the behavior of the pipeline:

- `data_config.yaml`: Specifies paths to raw and processed data, as well as parameters for data loading and preprocessing (e.g., curve length, normalization, sequence/condition encoding types). Also includes parameters for mechanical property analysis used during evaluation.
- `model_config.yaml`: Defines the architecture and hyperparameters of the Conditional Diffusion Model, including parameters for the protein encoder, condition encoder, time embedding, and the denoising network itself.
- `training_config.yaml`: Sets parameters for the training process, such as epochs, batch size, optimizer settings, learning rate scheduler, loss function weights, device, and logging/checkpointing intervals.

**Important:** Ensure consistency between parameters across configuration files, especially regarding data dimensions and encoding types (e.g., `fe_curve_length`, `fe_curve_channels`, `protein_encoding_type`, `condition_columns`, `protein_input_dim`).

## Adapting to Your Data and Model

This codebase provides a flexible structure, but you will need to adapt specific parts to your research and data:

-   **Raw Data Loading:** Modify the data loading logic in `scripts/preprocess_data.py` and potentially `src/data_processing/utils.py` to correctly read your raw data files and structure them by individual F-E curves.
-   **Preprocessing Details:** Refine the `standardize_fe_curve`, `encode_protein_sequences`, and `encode_conditions` functions in `src/data_processing/preprocessing.py` to match your specific preprocessing requirements (e.g., handling different force units, specific normalization methods, detailed sequence encoding).
-   **Denoising Model Architecture:** Implement the core `ConditionalDenoisingModel` within `src/models/diffusion_model.py`. Replace the placeholder layers with your chosen architecture (e.g., U-Net variant, Transformer blocks, custom residual network) and ensure it effectively incorporates the conditional inputs (time, protein embedding, condition embedding).
-   **Protein Language Model Integration:** If using `protein_encoding_type: raw`, implement the actual loading and usage of the PLM (e.g., using the `transformers` library) within `src/models/protein_encoder.py` and create a custom `collate_fn` in `src/data_processing/` if processing raw strings in batches is needed.
-   **Differentiable Property Extraction:** If enabling `MechanicalPropertyLoss` or `GeneratedCurveMatchingLoss` during training, implement differentiable methods for calculating relevant properties (e.g., peak force, unfolding energy) in `src/training/losses.py` or a utility module. This is often the most challenging part if these losses are applied during the iterative diffusion training steps.
-   **Analysis Parameters:** Tune the parameters for peak finding and WLC fitting in `data_config.yaml` (`analysis_params`) to work well with your specific F-E curve characteristics.
-   **Visualization:** Customize plotting functions in `src/evaluation/visualizer.py` as needed for your analysis and presentation.

## Contributing

If you find issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

[Specify your project's license here, e.g., MIT, Apache 2.0]