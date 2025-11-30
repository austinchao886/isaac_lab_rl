# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from zoneinfo import ZoneInfo

from isaaclab.app import AppLauncher

# add argparse arguments
import numpy as np
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=10, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="")
parser.add_argument("--log_dir", type=str, default="", help="")
parser.add_argument("--agent", type=str, default='rsl_rl_cfg_entry_point', help="Name of the agnet register for the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=1000000, help="RL Policy training iterations.")
parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
# parser.add_argument("--headless",type=bool, default=True)

# append RSL-RL cli arguments

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime
import importlib
import inspect
from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from pathlib import Path
import json 
import time


def register_env_task(env_task, log_dir):
    config_dir = [s for s in Path(log_dir).resolve().__str__().split("/") if len(s) > 0]
    entry_cfg_point = '.'.join(config_dir)
    gym.register(
        id=f"{env_task}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{entry_cfg_point}.configs.{env_task}:PandaEvnCfg",
            "rsl_rl_cfg_entry_point": f"{entry_cfg_point}.configs.rsl_rl_ppo_cfg:VanilaPPORunnerCfg",
        },
    )

def get_latest_pt(path_input):
    from pathlib import Path
    print("Getting latest pt from path:", path_input)
    idx_pth = [int(f.stem.split("_")[-1]) for f in Path(path_input).iterdir() if "model_" in f.stem]
    return Path(path_input).__str__() + f"/model_{np.max(idx_pth)}.pt"
    return Path(path_input).__str__() + f"/model_400.pt"


def main():
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    # Taiwan's IANA zone ID, UTC+8 with no DST :contentReference[oaicite:1]{index=1}
    log_dir = args_cli.log_dir
    print("Log dir:", log_dir)
    resume_path = get_latest_pt(log_dir)
    print("Resume path:", resume_path)
    REG_TASK = resume_path.split("/")[-5]
    print(f"Registered task from log dir: {REG_TASK}")
        
    tw_tz = ZoneInfo("Asia/Taipei")  
    register_env_task(REG_TASK, log_dir)
    
    env_cfg = parse_env_cfg(
        REG_TASK, device=args_cli.device, num_envs=args_cli.num_envs
    )
    env_cfg.scene.env_spacing = 2.0
    
    agent_cfg = load_cfg_from_registry(
        REG_TASK, args_cli.agent)
    
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = f"{Path(__file__).parent}/logs/{REG_TASK}/rsl_rl/{agent_cfg.experiment_name}"
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    
    # create isaac environment
    env = gym.make(REG_TASK, cfg=env_cfg)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    obs = obs[0]
    timestep = 0
    pos_range = (-0.5, 0.5)
    quat_range = (-1.0, 1.0)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # random actions
            # actions = torch.zeros(10, 7)
            # actions[:, 0:3] = (pos_range[1] - pos_range[0]) * torch.rand(10, 3) + pos_range[0]
            # actions[:, 3:7] = (quat_range[1] - quat_range[0]) * torch.rand(10, 4) + quat_range[0]

            # actions = torch.zeros(10,14)
            # actions[:] =( pos_range[1] - pos_range[0]) * torch.rand(10, 14) + pos_range[0]

            # print(actions.shape)
            print(f"actions:{actions}")
            print(f"action_shape:{actions.shape}")
            obs, _, _, _ = env.step(actions)  
            # print(f"observation:{obs[0]}")

        # sleep_time = dt - (time.time() - start_time)
        # if args_cli.real_time and sleep_time > 0:
        #     time.sleep(sleep_time)    

    # close the simulator
    env.close()



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
