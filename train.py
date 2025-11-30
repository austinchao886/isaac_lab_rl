import argparse
import sys
from zoneinfo import ZoneInfo

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--num_envs", type=int, default=500, help="Number of environments to simulate.")
parser.add_argument("--env_spacing", type=int, default=2, help="The distance between each env.")
parser.add_argument("--task", type=str, default=f'None', help="")
parser.add_argument("--config", type=str, default=f'scripts/configs/env_cfg.py', help="")
parser.add_argument("--agent", type=str, default='rsl_rl_cfg_entry_point', help="Name of the agnet register for the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=1000000, help="RL Policy training iterations.")
parser.add_argument("--noshow", type=int, default=True, help="Whether to launch the simulator in headless mode.")
# append RSL-RL cli arguments

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import  RslRlVecEnvWrapper
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from pathlib import Path
import dog

def save_configurations(log_dir, REG_TASK):
    root = Path(__file__).parent          
    robot_cfg_file = f"{root}/scripts/configs/robot_cfg.py"
    scene_cfg_file = f"{root}/scripts/configs/scene_cfg.py"
    env_cfg_file = f"{root}/scripts/configs/{REG_TASK}_env_cfg.py"
    ppo_cfg_file = f"{root}/scripts/agent/rsl_rl_ppo_cfg.py"

    # copy files
    new_path = lambda x, log_dir: f"{log_dir}/{Path(x).name}"
    configs_dir = f"{log_dir}/configs"
    os.makedirs(configs_dir, exist_ok=True)
    shutil.copyfile(robot_cfg_file, new_path(robot_cfg_file, configs_dir))
    shutil.copyfile(scene_cfg_file, new_path(scene_cfg_file, configs_dir))
    shutil.copyfile(env_cfg_file, new_path(env_cfg_file, configs_dir))      
    shutil.copyfile(ppo_cfg_file, new_path(ppo_cfg_file, configs_dir))

def register_env_task(env_task):
    print(f"You will register and use this env-task: {env_task}")
    gym.register(
        id=f"{env_task}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"scripts.configs.{env_task}:DogWalkEnvCfg",
            "rsl_rl_cfg_entry_point": "scripts.agents.rsl_rl_ppo_cfg:VanilaPPORunnerCfg",
        },
    )

def main():
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    tw_tz = ZoneInfo("Asia/Taipei")                      # Taiwan's IANA zone ID, UTC+8 with no DST :contentReference[oaicite:1]{index=1}
    config_fn = args_cli.config
    print(f"Config file name: {config_fn}")
    REG_TASK = Path(config_fn).resolve().stem
    print(f"Registered task name: {REG_TASK}")
    
    register_env_task(REG_TASK)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    
    agent_cfg = load_cfg_from_registry(
        args_cli.task, args_cli.agent)
    
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = f"{Path(__file__).parent}/logs/{args_cli.task}/rsl_rl/{agent_cfg.experiment_name}"
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now(tw_tz).strftime("%Y-%m-%d_%H.%M.%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    save_configurations(log_dir,REG_TASK)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
