
# RL_UR5_GRASP

A Deep Reinforcement Learning pipeline using **SAC** (Soft Actor-Critic), **Gymnasium**, and **MuJoCo** to train a UR5e robotic arm — fitted with a Robotiq 2F-85 gripper — to autonomously reach, grasp, and lift a cube. End-effector control is handled through **mink** (differential IK), so the policy operates in Cartesian action space instead of raw joint position reducing the actions space .


## Demo



https://github.com/user-attachments/assets/4800bec8-f3b6-4d7a-bf31-a58c7b3f5888


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
├── pyproject.toml               # Package configuration & dependencies
├── uv.lock                      # Dependency lock file
├── .gitignore                   # Git ignore rules for logs, models, and environments
├── README.md                    # Project documentation
└── src/
└── rl_ur5_grasp/
├── init.py          # Package initialization
├── envs/
│   ├── init.py
│   └── ur5e_grasp_env.py# Custom Gymnasium environment (UR5eGraspEnv)
├── scripts/
│   ├── spawn_view.py    # Sanity check: spawn the arm, open the viewer
│   ├── mink_test.py     # IK demo: cycle through fixed waypoints
│   ├── mink_circulare.py# IK demo: trace a continuous circle
│   ├── train_grasp.py   # Train a SAC policy from scratch
│   └── model_run.py     # Load the latest checkpoint and watch it run
├── models/              # directory that holds the model    } 
└── logs/                # TensorBoard & evaluation logs     } generated once trained
```

## Requirements

- Python 3.10+
- [MuJoCo](https://github.com/google-deepmind/mujoco)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (provides the UR5e and Robotiq 2F-85 models)
- [mink](https://github.com/kevinzakka/mink) (IK solver)
- [Gymnasium](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [`loop_rate_limiters`](https://github.com/kevinzakka/loop-rate-limiters)
- [`numpy`](https://numpy.org/)

## Installation

1. **Clone this repo**

   ```bash
   git clone https://github.com/ZakariaKouzou/RL_UR5_GRASP.git
   cd RL_UR5_GRASP
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the package in editable mode**
   
   ```bash
   pip install -e .
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
python3 src/rl_ur5_grasp/scripts/spawn_view.py
```

**2. Try the IK solver on its own** — before touching RL, these confirm mink is driving the arm correctly:

```bash
python3 src/rl_ur5_grasp/scripts/mink_test.py        # cycles through fixed waypoints
python3 src/rl_ur5_grasp/scripts/mink_circulare.py   # traces a continuous circle
```

**3. Train the SAC agent from scratch:**

```bash
python3 src/rl_ur5_grasp/scripts/train_grasp.py
```

This runs 300k timesteps, evaluates every 5k steps, and saves the best-performing checkpoint to `models/best_model.zip` plus the final model to `models/ur5e_grasp_final.zip`. Training logs are written to `logs/` (viewable with `tensorboard --logdir logs`).

**4. Watch a trained policy run:**

```bash
python scripts/model_run.py
```

This automatically loads the most recently saved checkpoint in `models/` and runs it in the MuJoCo viewer, printing per-episode results (success, final distance, grasp status, lift height).

## Environment Details

| | |
|---|---|
| **Observation space** | 22-dim vector: joint positions (6) + joint velocities (6) + gripper state (1) + end-effector position (3) + cube position (3) + relative cube offset (3) + grasp flag (1) |
| **Action space** | 4-dim continuous: Δx, Δy, Δz end-effector displacement + gripper command, each in `[-1, 1]` |
| **Reward** | Dense: negative distance to cube, small action penalty, +1 for a valid two-pad grasp, +5 × lift height while grasping, +20 on success |
| **Success condition** | Cube grasped (contact force on both gripper pads) **and** lifted above 0.15 m |
| **Episode length** | Up to 8 simulated seconds, or early termination on success/dropped cube |

## Results

Training progress and evaluation metrics are logged under `logs/` and can be inspected with TensorBoard:

```bash
tensorboard --logdir logs
```

the training results : 

<img width="2131" height="834" alt="47573161-0968-490c-8de6-3d82ff73153f" src="https://github.com/user-attachments/assets/cb934a1d-c16a-4a53-aada-d53c5e5691c0" />

Training converged cleanly: reward climbed from -47 to 191 over 300k steps, with a clear plateau around 100k–130k steps followed by a sharp jump once the policy learned to actually grasp and lift the cube rather than just hover nearby. Eval reward tracked closely (-29 → 195), showing no overfitting to the training env .
