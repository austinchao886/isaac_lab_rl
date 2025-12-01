# Isaac Lab RL Training

## Brief Introduction

This repository contains a reinforcement learning (RL) training framework for the ANYmal quadruped robot using Isaac Lab. The project demonstrates a structured approach to RL training with the following key characteristics:

- **Robot Platform**: ANYmal quadruped dog robot
- **Training Framework**: Isaac Lab with RSL-RL PPO algorithm
- **Environment**: Fully containerized Docker environment for reproducibility
- **Isaac Sim Version**: 4.5

The repository is organized to provide a clean separation between environment configuration, agent setup, and training scripts, making it easy to modify and extend for different RL tasks. All training and testing are performed within a Docker container to ensure consistent execution across different systems.

## Repository Structure

This project uses the **ManagerBasedRL** training environment configuration from Isaac Lab. All core training code is located in the `scripts/` directory, which contains two main components:

### `agent/`
This directory defines how the RL policy learns from the environment. The current implementation uses **PPO (Proximal Policy Optimization)** as the policy learning algorithm, configured through the RSL-RL framework.

### `configs/`
This directory contains all configurations for the ManagerBasedRL environment, including:
- Environment parameters and settings
- Robot configuration
- Scene setup and observation/action spaces

## Training and Inference Scripts

### `train.py` - Policy Training
This script handles the training of the RL policy. Key features include:

- **Configurable Environments**: Use the `--config` parameter to specify which environment configuration to use:
  ```bash
  --config scripts/configs/env_cfg.py
  ```
  This design allows you to maintain multiple environment configurations and easily switch between them for testing different parameter sets.

- **Auto-Registration**: The script includes a registration phase that automatically registers the task using the environment config name, eliminating the need to manually specify task names. See the `register_env_task()` function for implementation details.

- **Configuration Backup**: The `save_configuration()` function automatically copies the agent and config files to the `logs/` directory for each training run, ensuring you can always trace back which configurations were used for any given training session.

### `play.py` - Policy Inference
This script performs inference using trained policies. The key parameter is:

- **`--log_dir`**: Specify the training run directory you want to visualize:
  ```bash
  --log_dir logs/DogWalk-ManagerBasedRLEnv-v1.0-test251001/rsl_rl/ppo_runner/2025-10-01_14.44.23
  ```
  The script will load the trained model from the specified directory and run inference in the simulation environment.

## Visualization

### TensorBoard
To visualize training metrics and progress, use the provided `tensorboard.sh` script:

```bash
source tensorboard.sh
```

This will launch TensorBoard and display all training runs logged in the `logs/` directory, allowing you to monitor reward curves, loss values, and other training metrics in real-time.
