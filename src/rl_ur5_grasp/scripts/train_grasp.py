from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env

from rl_ur5_grasp.envs.ur5e_grasp_env import UR5eGraspEnv

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent

MODEL_DIR = PACKAGE_DIR / "models"
LOG_DIR = PACKAGE_DIR / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

#  instantiate and check environment

env = UR5eGraspEnv(render_mode=None)
check_env(env, warn=True)

eval_env = UR5eGraspEnv(render_mode=None)


eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(MODEL_DIR),
    log_path=str(LOG_DIR),
    eval_freq=5_000,
    n_eval_episodes=10,
    deterministic=True,
)


#  select policy and configure SAC
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=str(LOG_DIR),
    learning_rate=3e-4,
    buffer_size=300_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    ent_coef="auto",
    device="cpu",
)


print("Starting SAC training on UR5e reach-grasp-lift task...")
model.learn(total_timesteps=300_000, callback=eval_callback, progress_bar=True)
model.save(MODEL_DIR / "ur5e_grasp_final")
print("Training done. Model saved successfully.")