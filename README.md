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
"insert command" data.csv
```

You can chose to use your own or data already present in the `/data` folder.

CSV needs to follow a specfic format mentioned in the **Research Questions paragraph** with correct input features. The model will be saved in the `/models` folder.

### Predictions

To run predictions you have to run the models from the `/models` folder and input the data in a CSV format.
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

# Research Questions and Motivation
**This project applies supervised ML to predict properties of deep RL training runs in Unity ML-Agents.** Rather than focusing on RL algorithms themselves, we build meta-learning models that predict training outcomes from configuration parameters and hardware specifications. We address three core research questions guiding our data collection and model development.
## Research Question 1: Training Duration Prediction
Can we predict wall-clock training duration based on configuration parameters and
hardware specifications?

### Motivation
RL training is computationally expensive, with runs ranging from minutes to days. Accurate
duration estimates enable effective resource scheduling and budget allocation. Currently, estimating training time requires expensive pilot experiments or rough heuristics—a predictive model would save significant time and energy costs in large-scale RL research.

### Input features
- environment: (Categorical)\
Available options:
    - 3DBall
    - GridWorld
    - Basic
    - PushBlock
    - Hallway
    - VisualHallway
    - Reacher
    - Bouncer
    - FoodCollector
    - Pyramids
    - Walker
    - Crawler
    - Worm
    - SoccerTwos
    - CooperativePushBlock

- algorithm: (Categorical)\
Available options:
    - PPO
    - SAC
    - POCA
    - Imitation Learning

- learning_rate: (Float)\
Range: 0.0001 – 0.001

- batch_size: (Integer)\
Possible values: (32, 64, 128, 256)

- hidden_units: (Integer)\
Possible values: (64, 128, 256, 512)

- num_layers: (Integer)\
Possible values: (1, 2, 3)

- max_steps: (Integer)\
Range: 250,000 – 1,000,000

- ram_gb: (Float)\
Total system RAM (in GB)

- cpu_cores: (Integer)\
Number of CPU cores

### Target Variable:
- training_duration_seconds: Float\
Wall-clock time from start to completion

## Research Question 2: Final Performance Prediction

Can we predict final agent performance (mean cumulative reward) based on training
configuration?

### Motivation

Hyperparameter optimization is critical in RL but requires running multiple expensive training
runs. A model that estimates final performance without full training would significantly accelerate hyperparameter search, enabling quick identification of promising configurations and reducing computational overhead by orders of magnitude.

### Input Features

- environment: (Categorical)\
Available options:
    - 3DBall
    - GridWorld
    - Basic
    - PushBlock
    - Hallway
    - VisualHallway
    - Reacher
    - Bouncer
    - FoodCollector
    - Pyramids
    - Walker
    - Crawler
    - Worm
    - SoccerTwos
    - CooperativePushBlock

- algorithm: (Categorical)\
Available options:
    - PPO
    - SAC
    - POCA
    - Imitation Learning

- learning_rate: (Float)\
Range: 0.0001 – 0.001

- batch_size: (Integer)\
Possible values: (32, 64, 128, 256)

- hidden_units: (Integer)\
Possible values: (64, 128, 256, 512)

- num_layers: (Integer)\
Possible values: (1, 2, 3)

- max_steps: (Integer)\
Range: 250,000 – 1,000,000

- time_horizon: Integer\
Steps before policy update

- buffer_size: Integer\
 Experience replay buffer size (particularly relevant for SAC)

 ### Target Variable:
- final_mean_reward: Float\
Average cumulative reward over last 100 episodes

## Research Question 3: Resource Usage Prediction

Can we predict peak RAM usage during training based on environment configuration
and hyperparameters?

### Motivation

Memory constraints are a common bottleneck in RL research. Out-of-memory errors can cause training failures after hours of computation, wasting significant resources. Predictive RAM models enable researchers to determine hardware sufficiency in advance and guide decisions about environment design, parallelization strategies, and hardware purchasing.

### Input Features

- environment: (Categorical)\
Available options:
    - 3DBall
    - GridWorld
    - Basic
    - PushBlock
    - Hallway
    - VisualHallway
    - Reacher
    - Bouncer
    - FoodCollector
    - Pyramids
    - Walker
    - Crawler
    - Worm
    - SoccerTwos
    - CooperativePushBlock

- algorithm: (Categorical)\
Available options:
    - PPO
    - SAC
    - POCA
    - Imitation Learning

- learning_rate: (Float)\
Range: 0.0001 – 0.001

- batch_size: (Integer)\
Possible values: (32, 64, 128, 256)

- hidden_units: (Integer)\
Possible values: (64, 128, 256, 512)

- num_layers: (Integer)\
Possible values: (1, 2, 3)

- num_parallel_agents: Integer\
Number of parallel environment instances, if applicable

 ### Target Variable:
- peak_ram_mb: Float\
Maximum RAM usage in megabytes during entire training run

