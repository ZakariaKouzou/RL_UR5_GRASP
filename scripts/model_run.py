import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loop_rate_limiters import RateLimiter
from stable_baselines3 import SAC

from envs.ur5e_grasp_env import IK_DT, UR5eGraspEnv

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def find_latest_model():
    checkpoints = list(MODEL_DIR.glob("*.zip"))
    if not checkpoints:
        raise FileNotFoundError(f"No .zip checkpoints found in {MODEL_DIR}")
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


MODEL_PATH = find_latest_model()
print(f"Loading most recently updated model: {MODEL_PATH.name}")

env = UR5eGraspEnv(render_mode='human')
model = SAC.load(MODEL_PATH)

# The env's internal IK/physics step already advances the sim by IK_DT
# seconds each call, so pace the *policy* loop at the same rate -- otherwise
# the viewer just blasts through episodes as fast as Python can loop.
rate = RateLimiter(frequency=1.0 / IK_DT, warn=False)

obs, _ = env.reset()
episode = 0
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    rate.sleep()
    if terminated or truncated:
        episode += 1
        status = "SUCCESS" if info["success"] else ("timed out" if truncated else "ended")
        print(
            f"Episode {episode} {status}: "
            f"dist={info['distance']:.3f}  grasped={info['grasped']}  lift={info['lift']:.3f}"
        )
        obs, _ = env.reset()