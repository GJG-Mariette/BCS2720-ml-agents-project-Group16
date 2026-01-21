# Project Overview

This repository contains our BSc Computer Science (BCS2720 M2.1) AI & Machine Learning project. We built a pipeline to preprocess reinforcement-learning training run data, train machine-learning models, and evaluate performance using cross-validation. The primary goal is to make it possible for anyone to rerun our experiments and reproduce the reported metrics and figures.

**Research questions**
- RQ1: Can we predict the wall-clock training duration (seconds) of an RL run from its configuration and metadata?
- RQ2: Can we predict final agent performance (final mean reward) based on training configuration?
- RQ3: Can we predict peak RAM usage during training based on environment configuration and hyperparameters?

## Repository Structure

```text
.
├── config/                      # ML-Agents training configs (YAML)
├── data/                        # Datasets (raw/processed). See Data section below.
├── data-collect/                # Scripts/utilities used to collect run data
├── results/                     # Generated results (tables/figures/metrics)
├── colab/                       # Google Colab notebooks/scripts (if used)
├── docs/                        # Additional documentation (optional)
├── localized_docs/              # Localized documentation resources (not required for reproduction)
├── utils/                       # Helper scripts/utilities

# Unity ML-Agents / Unity project components (upstream + project)
├── Project/                     # Unity project (main)
├── DevProject/                  # Unity dev project (if applicable)
├── PerformanceProject/          # Unity performance project (if applicable)
├── ml-agents/                   # ML-Agents source (submodule / upstream)
├── ml-agents-envs/              # ML-Agents Python environments (upstream)
├── ml-agents-plugin-examples/   # Plugin examples (upstream)
├── ml-agents-trainer-plugin/    # Trainer plugin (upstream)
├── com.unity.ml-agents/         # Unity package
├── com.unity.ml-agents.extensions/
├── protobuf-definitions/
├── unity-volume/

# Tooling / CI / metadata
├── .github/                     # GitHub workflows/config
├── .yamato/                     # Unity CI (Yamato)
├── Dockerfile                   # Docker environment (if used for reproducibility)
├── colab_requirements.txt       # Python deps for Colab
├── test_requirements.txt        # Test dependencies
└── README.md                    # Main entry point (this file)
```


## Setup (Reproduce via Google Colab)

We reproduce all experiments using Google Colab notebooks stored in Google Drive under:

`RL_Predictor_Project/`

The Drive folder contains:
- `01_Data_Raw/` (raw logged data)
- `02_Data_Processed/` (processed dataset used for ML)
- `03_Notebooks - The Colab files (one for each Research Question)/` (Colab notebooks)
- `04_Models/` (saved trained models)
- `05_Visuals/` (generated plots)
- `06_Results/` (generated metric tables / outputs)

### Prerequisites
- A Google account (for Colab)
- Access to the shared Google Drive folder `RL_Predictor_Project`

### 1) Open the notebooks (in this order)
Notebooks are located in:
`RL_Predictor_Project/03_Notebooks - The Colab files (one for each Research Question)/`

Run in this order:

1. `Data_Preparation.ipynb` — loads raw data and produces the processed dataset in `02_Data_Processed/`
2. `Predictor_Duration.ipynb` — **RQ1** duration prediction (training duration)
3. `Predictor_Performance.ipynb` — **RQ2** final performance prediction (final mean reward)
4. `Predictor_RAM.ipynb` — **RQ3** peak RAM usage prediction

*(Optional / experimental notebooks)*
- `Time_Predictor_V2.ipynb`
- `Time_Predictor_V2_sac.ipynb`

### 2) Mount Google Drive (in every notebook)
Each notebook starts by mounting Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```
### 3) Install dependencies

```bash
pip install -r colab_requirements.txt
```

### 4) Data used for reproduction

The notebooks expect the following folders inside `RL_Predictor_Project/`:

- `01_Data_Raw/` — raw logged reinforcement learning data  
- `02_Data_Processed/` — processed dataset used for machine learning

Run `Data_Preparation.ipynb` first to generate or update the processed dataset from the raw data.

### 5) Reproduce the experiments (RQ1–RQ3)

After the data is prepared, run the following notebooks top-to-bottom:

- **RQ1 (Training duration):** `Predictor_Duration.ipynb`
- **RQ2 (Final performance):** `Predictor_Performance.ipynb`
- **RQ3 (Peak RAM usage):** `Predictor_RAM.ipynb`

The optional notebooks (`Time_Predictor_V2.ipynb`, `Time_Predictor_V2_sac.ipynb`) contain alternative or experimental versions and are not required to reproduce the main report results.

### 6) Outputs

Running the notebooks generates the following outputs:

- Trained models saved in: `04_Models/`
- Figures and plots saved in: `05_Visuals/`
- Metrics and result tables saved in: `06_Results/`

These outputs correspond to the results reported in the final project report.


