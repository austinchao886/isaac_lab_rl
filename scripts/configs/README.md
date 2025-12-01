# Configuration Directory

This directory defines all configurations for the ManagerBasedRL environment. The configuration is organized into three main components:

## `robot_cfg.py`
Defines the robot USD file settings and robot-specific parameters. This configuration specifies:
- Path to the robot USD asset
- Robot articulation settings
- Joint configurations and properties

## `scene_cfg.py`
Defines the complete simulation scene setup, including:
- Robot instantiation in the scene
- Terrain generation and configuration
- All sensors (cameras, IMU, contact sensors, etc.)
- Environmental objects and obstacles

## `env_cfg.py`
Defines the complete ManagerBasedRL environment configuration. This is the core configuration that brings everything together:

- **Actions**: Action space definition and control settings
- **Observations**: Observation space and sensor data processing
- **Rewards**: Reward terms and their weights for policy learning
- **Events**: Random events and perturbations during training
- **Curriculum**: Curriculum learning schedules and difficulty progression
- **Termination conditions**: Episode ending criteria
- **Command generators**: Goal/command generation for the robot

These three files work together to create a complete, modular environment configuration that can be easily modified and extended for different training scenarios.
