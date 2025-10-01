import gymnasium as gym
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)  
simulation_app = app_launcher.app

import dog  # trigger __init__.py → register_env.py
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from dog.configs.env_cfg import DogWalkEnvCfg

# List all registered environments
all_envs = list(gym.envs.registry.keys())
print(f"Total registered environments: {len(all_envs)}")
print("First 10 environments:", all_envs[:10])  # show only part if too many

task_name = "DogWalk-ManagerBasedRLEnv-v1.0-test251001"

if task_name in all_envs:
    print(f"[OK] Registered: {task_name}")
else:
    print(f"[ERROR] {task_name} not found, please check if register_env.py correctly calls gym.register()")

try:
    env_cfg = DogWalkEnvCfg()
    env = gym.make(task_name, cfg=env_cfg)
    print(f"[OK] Successfully created environment: {env}")
    obs, info = env.reset()
    print("[OK] Reset successful. Observation keys and shapes:")
    for k, v in obs.items():
        print(f"  {k}: {v.shape}")
    env.close()
except Exception as e:
    print(f"[ERROR] Failed to create environment: {e}")

simulation_app.close()
