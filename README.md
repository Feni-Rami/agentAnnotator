# Annotator Agent System with AI-Economist Integration

This repository contains the code for a multi-agent system that integrates Reinforcement Learning (RL) with an AI-Economist framework to manage and optimize the behavior of annotators in a data annotation task. The project aims to balance technical performance and economic stability within the system.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Files](#key-files)
  - [`agent_annotator.py`](#agent_annotatorpy)
  - [`central_server.py`](#central_serverpy)
  - [`fraud_detection.py`](#fraud_detectionpy)
  - [`main.py`](#mainpy)
- [Reinforcement Learning and AI-Economist Integration](#reinforcement-learning-and-ai-economist-integration)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)

## Project Overview

This project involves a simulated environment where multiple annotators (agents) are tasked with labeling data. The system uses Reinforcement Learning (RL) to optimize the performance of these agents, while the AI-Economist framework introduces economic dynamics such as taxation, income redistribution, and resource trading. The combined approach ensures that annotators not only improve their technical accuracy but also operate within an economically stable and fair environment.

## Key Files

### `agent_annotator.py`

This file contains the implementation of the `AnnotatorAgent` class, which defines the behavior of each annotator in the system. The key features include:

- **Annotation Process**: The agent makes decisions based on the RL policy, balancing between exploration and exploitation.
- **Economic Interactions**: The agent's income is subject to taxation, and it can participate in resource trading.
- **Performance Metrics**: The agent tracks its performance in terms of accuracy, consistency, and improvement, which are used to adjust its learning and economic strategies.

### `central_server.py`

The `CentralServer` class in this file manages the overall environment in which the annotators operate. It is responsible for:

- **Task Assignment**: Assigns data annotation tasks to the agents and evaluates their performance.
- **Economic Management**: Applies taxation, redistributes income, and manages economic stability within the system.
- **Malicious Detection**: Identifies and mitigates the impact of malicious agents that might provide incorrect annotations.

### `fraud_detection.py`

This file contains the `FraudDetectionEnvWithEconomics` class, which extends the environment to include specific mechanisms for detecting fraudulent or malicious behavior among annotators. It integrates both RL and economic principles to:

- **Identify Malicious Behavior**: Uses clustering and anomaly detection techniques to identify agents that behave suspiciously.
- **Adjust Economic Policies**: Modifies economic policies, such as increasing taxes or redistributing resources, to reduce the impact of detected fraud.

### `main.py`

The `main.py` file is the entry point for running the project. It sets up the environment, initializes the annotator agents, and starts the simulation. Key functionalities include:

- **Training the Agents**: Runs episodes where agents learn to improve their annotation quality and economic standing through RL.
- **Evaluation**: Tracks and logs the performance of agents over time, including their response to economic incentives and penalties.
- **Visualization**: Generates graphs and visual reports to analyze the performance and economic metrics.

## Reinforcement Learning and AI-Economist Integration

### Reinforcement Learning (RL)

RL is used in this project to optimize the annotation process of each agent. The agents learn from their interactions with the environment, adjusting their strategies to maximize rewards based on the accuracy and consistency of their annotations. The Q-learning algorithm is employed to balance short-term actions with long-term strategy, allowing agents to continuously improve their performance over multiple episodes.

### AI-Economist (AI-Eco)

The AI-Economist framework introduces economic principles into the system. Agents are subject to taxation, can engage in resource trading, and receive income redistribution based on their performance. The economic environment ensures that resources are distributed fairly among agents, preventing any single agent from gaining undue advantage or facing significant disadvantage. The integration of AI-Economist with RL ensures that agents are not only technically proficient but also operate within a sustainable and fair economic framework.

## Requirements

To run this project, you'll need to install the following dependencies:

- Python 3.7 or higher
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `gym` (for setting up the RL environment)
- `tensorflow` or `pytorch` (depending on your preference for the RL backend)
- `rllib` (if using RLlib for the reinforcement learning algorithm)

You can install these dependencies using `pip`:

```bash
pip install numpy scikit-learn matplotlib seaborn gym tensorflow rllib

