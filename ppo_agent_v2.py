import gymnasium as gym
import numpy as np
import json

from gymnasium import spaces
import highway_env
from enum import Enum
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.callbacks import CheckpointCallback


MODEL_SAVE_PATH = "./models/PPO_v2/{filename}"

def set_up_model(env):
    batch_size = 256
    cpu_cores = 8

    model = PPO(
        policy="MlpPolicy",
        # default is 2 layers of 64 width
        policy_kwargs=dict(net_arch=[256, 256]),
        batch_size=batch_size,
        n_steps=batch_size * cpu_cores,
        n_epochs=5,  # Default 10
        learning_rate=5e-4,
        gamma=0.995,  # Originally 0.8
        gae_lambda=0.93,  # Default 0.95
        # Potentially different architectures for policy and value networks
        #policy_kwargs=dict(dict(pi=[256, 256], vf=[256, 256])),
        env=env,
        seed=1508,
        verbose=1,
        tensorboard_log="./tensorboard/PPO_v2/",
        device="cpu"
    )

    return model


def run(config, filename, train=True, train_duration=50000):
    filepath = MODEL_SAVE_PATH.format(filename=filename)

    if train:
        env = make_vec_env("highway-fast-v0", n_envs=8, vec_env_cls=SubprocVecEnv, env_kwargs={"config": config})
        model = set_up_model(env)
        callbacks = [
            CheckpointCallback(
                save_freq=5000,
                save_path="./models/PPO_v2/checkpoints/",
                name_prefix=filename,
                save_replay_buffer=True,
                save_vecnormalize=True,
            ),
            ProgressBarCallback(),
        ]

        print("Training new model")
        trained_model = model.learn(total_timesteps=train_duration, callback=callbacks)
        print(f"Saving trained model to {filepath}")
        trained_model.save(filepath)

    else:
        env = gym.make(
            "highway-fast-v0",
            config=config,
            render_mode="human",
        )
        model = set_up_model(env)

        print(f"Loading trained model from: {filepath}")
        trained_model = model.load(filepath)

        print("Testing model")
        test_seeds = [i for i in range(20)] # We want repeatable results so we set the seeds to known values
        rewards = []
        steps = []

        for test_seed in test_seeds:
            state, _ = env.reset(seed=test_seed)

            ended = False
            truncated = False
            num_steps = 0
            episode_reward = 0

            while not ended and not truncated:
                action, _ = trained_model.predict(state, deterministic=True)
                #action = env.action_space.sample()
                next_state, reward, ended, truncated, _ = env.step(action)

                state = next_state
                episode_reward += reward
                num_steps += 1

                env.render()

            print(f"Episode reward: {episode_reward} \t Steps: {num_steps}")
            rewards.append(episode_reward)
            steps.append(num_steps)

        env.close()

        rewards = np.array(rewards)
        steps = np.array(steps)
        print(f"Average reward across {len(test_seeds)} trajectories: {np.mean(rewards)}")
        print(f"Average number of steps across {len(test_seeds)} trajectories: {np.mean(steps)}")

if __name__ == "__main__":
    ############ PARAMETERS ############
    vehicle_density = 1.5
    train = True
    train_duration = 1e5
    ############ ========== ############

    with open("config.json", "r") as fh:
        config = json.load(fh)

    config["normalize_reward"] = True
    config["vehicles_density"] = vehicle_density

    filename = f"vd_{vehicle_density}".replace(".", "_") + "_trial_1"

    print(f"Running with config {config}")

    run(config, filename=filename, train=train, train_duration=train_duration)
