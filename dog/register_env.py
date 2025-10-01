# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Register the custom dog walking RL task with gymnasium.
"""

import gymnasium as gym

# Import your environment config
# from tutorials.leraning_RL.test_20250930.dog.configs.env_cfg import DogWalkEnvCfg

gym.register(
	id="DogWalk-ManagerBasedRLEnv-v1.0-test251001",
	entry_point="isaaclab.envs:ManagerBasedRLEnv",
	disable_env_checker=True,
	kwargs={
		"env_cfg_entry_point": "dog.configs.env_cfg:DogWalkEnvCfg",
		# Optionally specify agent config if needed, e.g. PPO
		"rsl_rl_cfg_entry_point": "dog.agent.rsl_rl_ppo_cfg:VanilaPPORunnerCfg",
	},
)

print("kike")