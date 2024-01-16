import gymnasium as gym
import mujoco as mujoco
from sb3_contrib import TRPO
import numpy as np
import datetime
import os
import shutil
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback
import snake_v14
from parallel_env import ParallelEnvs

#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#█████████████████████████████████████████████████████████████████████████████████████████████████████████████████
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
snake_ver = 14

TOTAL_TIMESTEPS = 7e6

num_process = 1             #in triton, numbers of cores used

num_env_per_process = 1     #env per core

#ex: 5 process * 10 env per process = 50 snakes in total

#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------
#█████████████████████████████████████████████████████████████████████████████████████████████████████████████████
#-----------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    env_id = f"Snake-v{snake_ver}"
    policy_type = "MlpPolicy"


    now = datetime.datetime.now()
    now_save = now
    now_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    filename = f"trained_model"        #----------------------------------------filename

    tensorboard_log_dir = "tr_results/tendorboard_logs"
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)

    config = {
        "policy_type": policy_type,
        "total_timesteps": TOTAL_TIMESTEPS,
        "env_name": env_id,
    }

    run = wandb.init(
        project="04_goal_rndm",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )


    vec_env = ParallelEnvs(env_id, num_process, num_env_per_process)

    model = TRPO(policy_type, vec_env, verbose=1, tensorboard_log=f"{tensorboard_log_dir}/log_{filename}")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar= True, callback=WandbCallback())


    model.save(f"{filename}")

    print(f"saved in {filename}")



