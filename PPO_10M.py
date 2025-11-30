import gymnasium as gym
import highway_env
import json
import os

#using PPO instead of DQN
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


from stable_baselines3.common.callbacks import ProgressBarCallback


with open("config.json", "r") as f:
    config = json.load(f)


def make_env(rank: int, env_config: dict):
    """
    Returns a function that creates the environment.
    Needed for SubprocVecEnv.
    """
    def _init():
        env = gym.make("highway-fast-v0", render_mode=None, config=env_config)
        env = Monitor(env, f"{SAVE_DIR}/logs_{rank}")
        return env
    return _init


if __name__ == "__main__":

    #doing this again with the proper config, from Jibran

    SAVE_DIR = "ppo_10M_new_run"
    # Number of parallel environments (have 20 CPUs, don't want to overdo it')
    N_ENV = 18  
    # Total training steps
    TOTAL_TIMESTEPS = 10_000_000  

  
    #no need to override the config, it's the right one now
    CONFIG_OVERRIDES = {}

    env_config = config.copy()
    env_config.update(CONFIG_OVERRIDES)


    #make sure the directories exist first (was a real pain before)
    os.makedirs("logs", exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/checkpoints", exist_ok=True)


    #make vectorized environment
    vec_env = SubprocVecEnv([make_env(i, env_config) for i in range(N_ENV)])


    model = PPO(
        "MlpPolicy",
        vec_env,
        device="cuda",
        #could probably tune these some more?
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.80,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=SAVE_DIR,
        verbose=1,
    )

    print("Using device:", model.device)


    #same checkpoint callback as other ones
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // N_ENV,  # Save every 100k steps (adjusted for parallel envs)
        save_path=f"{SAVE_DIR}/checkpoints/",
        name_prefix="ppo_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    #also adding this, took ~12 hours last time, 
    progress_callback = ProgressBarCallback()


    #start training, same as others
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=50,
        callback=[checkpoint_callback, progress_callback],
        reset_num_timesteps=True,
    )


    #save final model and print 
    model.save(f"{SAVE_DIR}/final_model")
    with open(f"{SAVE_DIR}/config.json", "w") as f:
        json.dump(env_config, f, indent=2)

    vec_env.close()
    print(f"Training complete. Model and config saved to {SAVE_DIR}/")

