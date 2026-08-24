import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from envs.ur5e_grasp_env import UR5eGraspEnv

LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# initaitae enviroment for training . 

env = UR5eGraspEnv(render_mode=None)
check_env(env, warn=True)

eval_env = UR5eGraspEnv(render_mode=None)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=5_000,
    n_eval_episodes=10,
    deterministic=True,
)

# select policy : soft actor critic with Mlp policy 

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    learning_rate=3e-4,
    buffer_size=300_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",
    device="cpu",
)

# start training and save the final trained model .

print("Starting SAC training on UR5e reach-grasp-lift task...")
model.learn(total_timesteps=300_000, callback=eval_callback, progress_bar=True)
model.save(f"{MODEL_DIR}/ur5e_grasp_final")
print("Training done. Model saved.")