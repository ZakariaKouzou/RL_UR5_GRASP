from pathlib import Path
from loop_rate_limiters import RateLimiter
from stable_baselines3 import SAC

# Updated package imports
from rl_ur5_grasp.envs.ur5e_grasp_env import IK_DT, UR5eGraspEnv

# Define project model directory relative to package location
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parent / "models"


def find_latest_model(model_dir: Path) -> Path:
    checkpoints = list(model_dir.glob("*.zip"))
    if not checkpoints:
        raise FileNotFoundError(f"No .zip checkpoints found in {model_dir}")
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def main():
    model_path = find_latest_model(MODEL_DIR)
    print(f"Loading most recently updated model: {model_path.name}")

    env = UR5eGraspEnv(render_mode="human")
    model = SAC.load(model_path)

    # Pace policy loop at physics rate
    rate = RateLimiter(frequency=1.0 / IK_DT, warn=False)

    obs, _ = env.reset()
    episode = 0

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            rate.sleep()

            if terminated or truncated:
                episode += 1
                status = (
                    "SUCCESS"
                    if info.get("success")
                    else ("timed out" if truncated else "ended")
                )
                print(
                    f"Episode {episode} {status}: "
                    f"dist={info.get('distance', 0.0):.3f} "
                    f"grasped={info.get('grasped', False)} "
                    f"lift={info.get('lift', 0.0):.3f}"
                )
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()