# RL_UR5_GRASP

A Deep Reinforcement Learning pipeline using **SAC** (Soft Actor-Critic), **Gymnasium**, and **MuJoCo** to train a UR5e robotic arm — fitted with a Robotiq 2F-85 gripper — to autonomously reach, grasp, and lift a cube. End-effector control is handled through **mink** (differential IK), so the policy operates in Cartesian action space instead of raw joint torques.

<!--
🎥 DEMO VIDEO
Drop your video file directly into this README on GitHub.com while editing —
GitHub will upload it and auto-insert the embed link here for you.
Alternatively, keep it in the repo (e.g. assets/demo.mp4) and link it:
[Watch the demo](assets/demo.mp4)
-->
## Demo

_Add your demo video here._

<!--
🧩 BLOCK DIAGRAM
Same idea — drag an image into the GitHub editor and it'll insert something like:
![architecture](https://github.com/user-attachments/assets/your-image-id)
Or save it to the repo and reference it:
![architecture](assets/architecture.png)
-->
## Architecture

_Add your block diagram here._

---

## Overview

The agent controls the arm's end-effector position (dx, dy, dz) plus a gripper open/close command. Each action is converted into joint targets via a mink IK solver, stepped through MuJoCo physics, and rewarded based on distance to the cube, successful grasp contact, and lift height.

**Task:** reach → grasp → lift a randomly spawned cube above a success threshold, within an 8-second episode.

## Features

- Custom `gymnasium.Env` (`UR5eGraspEnv`) built directly from MuJoCo Menagerie's UR5e + Robotiq 2F-85 models
- Cartesian end-effector control via `mink` IK (no manual joint-space tuning)
- Randomized cube spawn zone for generalization
- Contact-force-based grasp detection (both gripper pads must register normal force)
- Dense reward shaping (distance + grasp bonus + lift bonus + success bonus)
- SAC training via Stable-Baselines3, with periodic evaluation and best-model checkpointing
- Standalone IK sanity-check scripts (waypoint following, circular tracing) for debugging the arm setup before training

## Project Structure

```
RL_UR5_GRASP/
├── envs/
│   └── ur5e_grasp_env.py     # Custom Gymnasium environment (UR5eGraspEnv)
├── scripts/
│   ├── spawn_view.py         # Sanity check: spawn the arm, open the viewer
│   ├── mink_test.py          # IK demo: cycle through fixed waypoints
│   ├── mink_circulare.py     # IK demo: trace a continuous circle
│   ├── train_grasp.py        # Train a SAC policy from scratch
│   ├── train_continue.py     # Resume training from a saved checkpoint
│   └── model_run.py          # Load the latest checkpoint and watch it run
├── models/                   # Saved checkpoints (best_model.zip, ur5e_grasp_final.zip)
└── logs/                     # Evaluation logs / TensorBoard logs
```

## Requirements

- Python 3.10+
- [MuJoCo](https://github.com/google-deepmind/mujoco)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (provides the UR5e and Robotiq 2F-85 models)
- [mink](https://github.com/kevinzakka/mink) (IK solver)
- [Gymnasium](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- `loop_rate_limiters`
- `numpy`

## Installation

1. **Clone this repo**

   ```bash
   git clone https://github.com/ZakariaKouzou/RL_UR5_GRASP.git
   cd RL_UR5_GRASP
   ```

2. **Set up a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate        # on Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install mujoco mink gymnasium stable-baselines3 numpy loop_rate_limiters
   ```

4. **Get the robot models (MuJoCo Menagerie)**

   The environment expects the UR5e and Robotiq 2F-85 model files. Clone Menagerie into your home directory (the default the code looks for), or point `MUJOCO_MENAGERIE_PATH` at wherever you keep it:

   ```bash
   git clone https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie
   # or, if you keep it elsewhere:
   export MUJOCO_MENAGERIE_PATH=/path/to/mujoco_menagerie
   ```

## Usage

**1. Sanity-check the setup** — spawns the bare UR5e arm in the MuJoCo viewer to confirm paths/models are correct:

```bash
python scripts/spawn_view.py
```

**2. Try the IK solver on its own** — before touching RL, these confirm mink is driving the arm correctly:

```bash
python scripts/mink_test.py        # cycles through fixed waypoints
python scripts/mink_circulare.py   # traces a continuous circle
```

**3. Train the SAC agent from scratch:**

```bash
python scripts/train_grasp.py
```

This runs 300k timesteps, evaluates every 5k steps, and saves the best-performing checkpoint to `models/best_model.zip` plus the final model to `models/ur5e_grasp_final.zip`. Training logs are written to `logs/` (viewable with `tensorboard --logdir logs`).

**4. Watch a trained policy run:**

```bash
python scripts/model_run.py
```

This automatically loads the most recently saved checkpoint in `models/` and runs it in the MuJoCo viewer, printing per-episode results (success, final distance, grasp status, lift height).

> **Note:** `train_continue.py` is left over from an earlier reach-only environment (`URReachEnv`) and isn't wired up to the current `UR5eGraspEnv` — treat it as a reference for how to resume training from a checkpoint rather than a ready-to-run script.

## Environment Details

| | |
|---|---|
| **Observation space** | 22-dim vector: joint positions (6) + joint velocities (6) + gripper state (1) + end-effector position (3) + cube position (3) + relative cube offset (3) + grasp flag (1) |
| **Action space** | 4-dim continuous: Δx, Δy, Δz end-effector displacement + gripper command, each in `[-1, 1]` |
| **Reward** | Dense: negative distance to cube, small action penalty, +1 for a valid two-pad grasp, +5 × lift height while grasping, +20 on success |
| **Success condition** | Cube grasped (contact force on both gripper pads) **and** lifted above 0.15 m |
| **Episode length** | Up to 8 simulated seconds, or early termination on success/dropped cube |

