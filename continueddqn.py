import gymnasium as gym
import highway_env  # registers highway-fast-v0
import json
import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

# Load base config
with open("config.json", "r") as f:
    config = json.load(f)


def make_env(rank: int, env_config: dict):
    def _init():
        env = gym.make("highway-fast-v0", render_mode=None, config=env_config)
        env = Monitor(env, f"logs/dqn_{rank}")
        return env

    return _init


if __name__ == "__main__":
    # Training settings
    SAVE_DIR = "highway_meta_continued"
    N_ENV = 16
    TOTAL_TIMESTEPS = 10_000_000  # original total horizon

    # Resume from existing checkpoint at 5M steps
    RESUME_CHECKPOINT = "highway_dqn_meta/checkpoints/dqn_model_5000000_steps.zip"
    RESUME_TIMESTEP = 5_000_000

    # Exploration schedule after resume: 0.8 -> 0.03 over 3M steps (5M -> 8M)
    EXPL_START = 0.8
    EXPL_END = 0.03
    EXPL_DECAY_STEPS = 3_000_000
    EXPL_FRACTION = EXPL_DECAY_STEPS / TOTAL_TIMESTEPS  # 0.3 of full 10M

    # Config overrides (same as dqn_implementation.py)
    CONFIG_OVERRIDES = {
        "collision_reward": -20.0,
    }

    # Apply overrides
    env_config = config.copy()
    for key, value in CONFIG_OVERRIDES.items():
        env_config[key] = value
        print(f"Config override: {key} = {value}")

    os.makedirs("logs", exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/checkpoints", exist_ok=True)

    vec_env = SubprocVecEnv([make_env(i, env_config) for i in range(N_ENV)])

    # Load checkpoint
    print(f"Resuming from {RESUME_CHECKPOINT}")
    model = DQN.load(
        RESUME_CHECKPOINT,
        env=vec_env,
        device="cuda",
        tensorboard_log=f"{SAVE_DIR}/",
    )
    # Set timestep to resume point
    model.num_timesteps = RESUME_TIMESTEP

    # Install a resume-aware exploration schedule:
    # eps = 0.8 at step 5M, decays linearly to 0.03 by step 8M, then stays there.
    def resumed_eps(_progress_remaining: float) -> float:
        current_step = model.num_timesteps
        if current_step <= RESUME_TIMESTEP:
            return EXPL_START
        if current_step >= RESUME_TIMESTEP + EXPL_DECAY_STEPS:
            return EXPL_END
        pct = (current_step - RESUME_TIMESTEP) / EXPL_DECAY_STEPS
        return EXPL_START + pct * (EXPL_END - EXPL_START)
    model.exploration_rate = EXPL_START
    model.exploration_initial_eps = EXPL_START
    model.exploration_final_eps = EXPL_END
    model.exploration_schedule = resumed_eps
    model.exploration_fraction = EXPL_DECAY_STEPS / TOTAL_TIMESTEPS  # logging only
    print(f"Exploration: {EXPL_START} -> {EXPL_END} from step {RESUME_TIMESTEP:,} to {RESUME_TIMESTEP + EXPL_DECAY_STEPS:,}")
    print("Using device:", model.device)

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // N_ENV,
        save_path=f"{SAVE_DIR}/checkpoints/",
        name_prefix="dqn_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=50,
        callback=checkpoint_callback,
        reset_num_timesteps=False,  # continue from resume step
    )

    model.save(f"{SAVE_DIR}/model")
    with open(f"{SAVE_DIR}/config.json", "w") as f:
        json.dump(env_config, f, indent=2)
    print(f"Model and config saved to {SAVE_DIR}/")

    vec_env.close()
