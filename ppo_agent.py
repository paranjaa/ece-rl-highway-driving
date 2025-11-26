import gymnasium as gym
import numpy as np

from gymnasium import spaces
import highway_env
from enum import Enum
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import ProgressBarCallback
from stable_baselines3.common.callbacks import CheckpointCallback


MODEL_SAVE_PATH = "./models/PPO/{filename}.pth"

def set_up_model(env):
    batch_size = 64
    cpu_cores = 8

    model = PPO(
        policy="MlpPolicy",
        # default is 2 layers of 64 width
        policy_kwargs=dict(net_arch=[256, 256]),
        batch_size=batch_size,
        n_steps=batch_size * cpu_cores,
        learning_rate=5e-4,
        gamma=0.8,
        # Potentially different architectures for policy and value networks
        #policy_kwargs=dict(dict(pi=[256, 256], vf=[256, 256])),
        env=env,
        #device="cpu",
        seed=1508,
        verbose=1,
        tensorboard_log="./tensorboard/PPO/",
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
                save_path="./models/PPO/checkpoints/",
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
                action, _ = trained_model.predict(state)
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
    class ObservationTypes(Enum):
        KINEMATICS = "Kinematics"
        LIDAR = "LidarObservation"

    class ActionTypes(Enum):
        DISCRETE = "DiscreteAction"
        DISCRETE_META = "DiscreteMetaAction"

    filenames = {
        ObservationTypes.KINEMATICS: {
            ActionTypes.DISCRETE: "act_D__obs_K",
            ActionTypes.DISCRETE_META: "act_DMA__obs_K",
        },
        ObservationTypes.LIDAR: {
            ActionTypes.DISCRETE: "act_D__obs_LDR",
            ActionTypes.DISCRETE_META: "act_DMA__obs_LDR",
        },
    }

    observation_config = {
        ObservationTypes.KINEMATICS: {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 10,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "feature_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": False,
                "order": "sorted"
            }
        },
        ObservationTypes.LIDAR: {
            "observation": {
                "type": "LidarObservation",
                "cells": 128
            }
        }

    }

    action_config = {
        ActionTypes.DISCRETE: {
            "action": {
                "type": "DiscreteAction",
                "longitudinal": True,
                "lateral": True,
                "actions_per_axis": 10,          # 10 accel bins x 10 steering bins = 100 actions
                "acceleration_range": [-3.0, 3.0],   # m/s^2
                "steering_range":     [-0.3, 0.3],   # rad
            }
        },
        ActionTypes.DISCRETE_META: {
            "action": {
                "type": "DiscreteMetaAction",
            }
        }
    }

    config = {
        "policy_frequency": 5,
        "lanes_count": 5,
        "vehicles_count": 50,
        "vehicles_density": 1.0,
        "initial_spacing": 10,
        "offroad_terminal": True,
        "collision_reward": -1.0,
        "high_speed_reward": 0.7,
        #"right_lane_reward": 0.1,
        "lane_change_reward": -0.1,
    }

    ############ PARAMETERS ############
    observation_type = ObservationTypes.KINEMATICS
    action_type = ActionTypes.DISCRETE_META
    train = False
    train_duration = 1e6
    ############ ========== ############

    config.update(observation_config[observation_type])
    config.update(action_config[action_type])
    filename = filenames[observation_type][action_type]

    print(f"Running with config {config}")

    run(config, filename=filename, train=train, train_duration=train_duration)
