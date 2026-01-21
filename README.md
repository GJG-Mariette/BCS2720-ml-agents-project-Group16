# Group 16 MAI Project Documentation

[![fork badge](https://img.shields.io/badge/fork_of-ML_Agents-yellow)](https://github.com/DennisSoemers/ml-agents)

[![original readme badge](https://img.shields.io/badge/ML_agents_readme-reference-green)](./docs/Readme.md)

**This repository is based on the [fork of ML-agents](https://github.com/DennisSoemers/ml-agents).** The goal of this repository is to collect data, and do a research related to predicting performance of ML-Agents, be it actual performance of an agent, or performance of your machine with how long it will take to finish. It will be achieved performing multiple data collection strategies, using the intuitive project layout and performing different ML strategies. The goal of this project holds a very high relevance to the RL in the modern world being a powerful paradigm. owever, RL training
presents significant challenges: runs can take minutes to days, and the impact of hyperparameter choices on duration, resource usage, and performance remains unclear without expensive empirical testing.

# Preliminaries

To ensure stable and consisten results, make sure 3.10.10<=python<=3.10.12 version range because it proved to be consistent in Windows, macOS and Linux.

## Setting Up a virtual enviroment

Setting up a virtual enviroment is highly recommended due to the module like project structure

Windows:
```
python -m venv path\to\venv\
```

macOS and Linus:
```
python -m venv path/to/venv/
```

## Running Virutal Enviroment

Windows:
```
path\to\venv\Scripts\activate
```

macOS and Linus:
```
.path/to/venv/bin/activate
```
## Installing ML-Agents and Data-Colletion

Windows:
```
python -m pip install -e .\data-collection
python -m pip install -e .\ml-agents-envs
python -m pip install -e .\ml-agents
```

macOS and Linus:
```
pip install -e ./data-collection
pip install -e ./ml-agents-envs
pip install -e ./ml-agents
```
Just make sure you install ml-agents-envs before ml-agents

# How To Run

## Data Collection
To run the data collection, you can follow the [README for the ML-Agents](./docs/Readme.md) to start the training. The data is collected automatically.

### Predictions

To run predictions you have to run the models from the `/data-collection/data_collection/models/model-files/` by addingthe path into a config at `/data-collection/data_collection/models/prediction_models/`. You can edit which features to use and which model to use from the `\model-files` folder.
```
predict_data path/to/config.yaml
```

**Note** current prediction config is made to work for Ram and Performance prediction, it would be needed to change the config in order to predict duration by adding more features.

# Repository Structure
The project forks the stable ML-Agents branch (fix-numpy-release-21-branch) to ensure compatibility and repro-
ducibility. Main directories:
- `/configs` – YAML configuration files defining experimental hyperparameters
- `/data-collection` – Python module automation for training, logging, and parsing data collection and config generation for trainig.
- `/data` – Collected CSV data
- `/data-collection/data_collection/models/model-files/` – Trained ML models
- `/data-collection/data_collection/models/prediction_models/` – Prediction configs

# Objectives

This project will have two main objectives:
- Engineering objective
- Research objective

The research objective will be related to the scientific research part. While engineering is related to the coding part.

## Engineering

Build a clean, well-documented public GitHub repository containing our data collection infrastructure, experimental results, trained ML models, and comprehensive documentation.

## Research

- RQ1: Can we predict the wall-clock training duration (seconds) of an RL run from its configuration and metadata?
- RQ2: Can we predict final agent performance (final mean reward) based on training configuration?
- RQ3: Can we predict peak RAM usage during training based on environment configuration and hyperparameters?


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




