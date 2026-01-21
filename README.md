# Group 16 MAI Project Documentation

[![fork badge](https://img.shields.io/badge/fork_of-ML_Agents-yellow)](https://github.com/DennisSoemers/ml-agents)

[![original readme badge](https://img.shields.io/badge/ML_agents_readme-reference-green)](./docs/Readme.md)

**Note**. This README might be ahead of the current version if something isn't implemented just yet for example ML model training.

**This repository is based on the [fork of ML-agents](https://github.com/DennisSoemers/ml-agents).** The goal of this repository is to collect data, and do a research related to predicting performance of ML-Agents, be it actual performance of an agent, or performance of your machine with how long it will take to finish. It will be achieved performing multiple data collection strategies, using the intuitive project layout and performing different ML strategies. The goal of this project holds a very high relevance to the RL in the modern world being a powerful paradigm. owever, RL training
presents significant challenges: runs can take minutes to days, and the impact of hyperparameter choices on duration, resource usage, and performance remains unclear without expensive empirical testing.

# How To Run

## Data Collection
To run the data collection, you can follow the [README for the ML-Agents](./docs/Readme.md) to start the training. The data is collected automatically.
<!---
Assumption that our way of doing it wil be the same just with data collection and is automated at some point.
-->

## ML Agent ML Prediction Model Training

### Training
To run the training, run:

```
"insert training script path" data.csv
```

You can chose to use your own or data already present in the `/data` folder.

CSV needs to follow a specfic format mentioned in the **Research Questions paragraph** with correct input features. The model will be saved in the `/models` folder.

### Predictions

To run predictions you have to run the models from the `/models` folder and input the data in a CSV format using command.
```
"insert prediction script path" predict_data.csv
```
<!---
Just a throwaway text probably will be done differently
--->

# Repository Structure
The project forks the stable ML-Agents branch (fix-numpy-release-21-branch) to ensure compatibility and repro-
ducibility. Main directories:
- `/config` – YAML configuration files defining experimental hyperparameters
- `/scripts` – Python automation for training, logging, and parsing
- `/data` – Collected CSV/JSON results
- `/docs` – Setup notes and experiment logs
- `/models` – Trained ML models

# Objectives

This project will have two main objectives:
- Research objective
- Engineering objective

The research objective will be related to the scientific research part. While engineering is related to the coding part.

## Research

**The main research objective is to apply supervised ML to predict deep RL training run properties in Unity ML-Agents.** We will collect data from 15 environments across 4 algorithms `(PPO, SAC, POCA, Imitation Learning)` with systematic hyperparameter variations, then train regression models to predict training duration, agent performance, and RAM usage.

## Engineering

Build a clean, well-documented public GitHub repository containing our data collection infrastructure, experimental results, trained ML models, and comprehensive documentation.


