

import os
from envs.ur_reach_env import URReachEnv
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

EXISTING_MODEL = f"{MODEL_DIR}/ur5e_reach_final"  # or models/best_model, no .zip suffix

env = URReachEnv(render_mode=None)

eval_env = URReachEnv(render_mode=None)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
)

print(f"Loading existing model from {EXISTING_MODEL}...")
model = SAC.load(EXISTING_MODEL, env=env, device="cpu")

ADDITIONAL_TIMESTEPS = 300_000

print(f"Continuing training for {ADDITIONAL_TIMESTEPS} more timesteps...")
model.learn(
    total_timesteps=ADDITIONAL_TIMESTEPS,
    callback=eval_callback,
    progress_bar=True,
    reset_num_timesteps=False,  # keeps the step counter continuous in tensorboard
)

model.save(f"{MODEL_DIR}/ur5e_reach_continued")
print("Continued training done. Model saved.")