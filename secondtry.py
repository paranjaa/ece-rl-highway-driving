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
    SAVE_DIR = "highway_second_try"
    N_ENV = 16

    # Resume settings (set to None to start from scratch)
    RESUME_CHECKPOINT = "highway_second_try/checkpoints/dqn_model_10000000_steps.zip"
    RESUME_TIMESTEP = 10_000_000  # Steps already trained in the checkpoint
    ADDITIONAL_TIMESTEPS = 10_000_000  # Train for 10M more steps
    FIXED_RESUME_EPSILON = 0.12  # Keep epsilon fixed after resuming

    if RESUME_CHECKPOINT:
        TOTAL_TIMESTEPS = RESUME_TIMESTEP + ADDITIONAL_TIMESTEPS
    else:
        TOTAL_TIMESTEPS = 10_000_000  # Default horizon when training from scratch

    # Exploration (used only when training from scratch)
    EXPL_START = 0.8
    EXPL_END = 0.08
    EXPL_DECAY_STEPS = 5_000_000
    EXPL_FRACTION = EXPL_DECAY_STEPS / TOTAL_TIMESTEPS  # 0.5 when TOTAL_TIMESTEPS=10M

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

    if RESUME_CHECKPOINT:
        print(f"Resuming training from checkpoint: {RESUME_CHECKPOINT}")
        model = DQN.load(
            RESUME_CHECKPOINT,
            env=vec_env,
            device="cuda",
            tensorboard_log=f"{SAVE_DIR}/",
        )
        # Ensure timestep counter is set and log intent
        model.num_timesteps = RESUME_TIMESTEP
        print(
            f"Checkpoint stored timesteps: {model.num_timesteps:,}. "
            f"Training to {TOTAL_TIMESTEPS:,} total."
        )

        # Force a fixed exploration rate for resumed training
        model.exploration_rate = FIXED_RESUME_EPSILON
        model.exploration_initial_eps = FIXED_RESUME_EPSILON
        model.exploration_final_eps = FIXED_RESUME_EPSILON
        model.exploration_fraction = 1.0
        model.exploration_schedule = (
            lambda _progress_remaining: FIXED_RESUME_EPSILON
        )
        print(f"Exploration fixed at epsilon = {FIXED_RESUME_EPSILON}")
    else:
        # Fresh model from scratch
        model = DQN(
            "MlpPolicy",
            vec_env,
            device="cuda",
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            buffer_size=150_000,
            learning_starts=20_000,
            batch_size=256,
            gamma=0.80,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1_000,
            exploration_fraction=EXPL_FRACTION,
            exploration_initial_eps=EXPL_START,
            exploration_final_eps=EXPL_END,
            verbose=1,
            tensorboard_log=f"{SAVE_DIR}/",
        )
        print(
            f"Exploration: {EXPL_START} -> {EXPL_END} over first {EXPL_DECAY_STEPS:,} steps "
            f"({EXPL_FRACTION:.0%} of training)"
        )

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
        reset_num_timesteps=not bool(RESUME_CHECKPOINT),
    )

    model.save(f"{SAVE_DIR}/model")
    with open(f"{SAVE_DIR}/config.json", "w") as f:
        json.dump(env_config, f, indent=2)
    print(f"Model and config saved to {SAVE_DIR}/")

    vec_env.close()
