import gymnasium as gym
import numpy as np
import json
import highway_env

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from stable_baselines3 import DQN
from benchmark import MODEL_TYPE

def set_up_model(env, model_type:MODEL_TYPE):
    batch_size = 256
    cpu_cores = 8

    if model_type == MODEL_TYPE.DQN:
        model = DQN(
            "MlpPolicy",
            env,
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
            exploration_fraction=0.5,
            exploration_initial_eps=0.8,
            exploration_final_eps=0.08,
            verbose=1,
            tensorboard_log="./tensorboard/DQN/",
        )
    elif model_type == MODEL_TYPE.PPO:
        model = PPO(
            policy="MlpPolicy",
            # default is 2 layers of 64 width
            policy_kwargs=dict(net_arch=[256, 256]),
            batch_size=batch_size,
            n_steps=batch_size * cpu_cores,
            n_epochs=5,  # Default 10
            learning_rate=3e-4,
            gamma=0.8,  # Originally 0.8
            gae_lambda=0.95,  # Default 0.95
            # Potentially different architectures for policy and value networks
            #policy_kwargs=dict(dict(pi=[256, 256], vf=[256, 256])),
            env=env,
            seed=1508,
            verbose=1,
            tensorboard_log="./tensorboard/PPO_v2/",
            device="cpu"
        )
    else:
        print(f"Unknown model type {model_type}")
        exit(-1)

    return model


def run(config, model_type:MODEL_TYPE, filepath):
    env = make_vec_env(
        "highway-fast-v0",
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"config": config})
    model = set_up_model(env, model_type)

    print(f"Loading trained model from: {filepath}")
    trained_model = model.load(filepath)

    print("Testing model")
    test_seeds = [i for i in range(20)] # We want repeatable results so we set the seeds to known values
    rewards = []
    steps = []
    pmfs = []

    for test_seed in test_seeds:
        model.env.seed(test_seed)
        state = model.env.reset()

        dones = False
        num_steps = 0
        episode_reward = 0
        pmf = []

        while not dones:
            action, _ = trained_model.predict(state, deterministic=True)
            _state, _ = model.policy.obs_to_tensor(state)

            if model_type == MODEL_TYPE.DQN:
                q_values = model.policy.q_net(_state)
                _action_pmf = q_values.softmax(dim=1).squeeze().detach().numpy()
                pmf.append(_action_pmf)
            elif model_type == MODEL_TYPE.PPO:
                pmf.append(model.policy.get_distribution(_state).distribution.probs.detach().numpy())

            next_state, reward, dones, _ = model.env.step(action)

            state = next_state
            episode_reward += reward
            num_steps += 1

            env.render()

        print(f"Episode reward: {episode_reward} \t Steps: {num_steps}")
        rewards.append(episode_reward)
        steps.append(num_steps)
        pmfs.append(pmf)

    env.close()

    rewards = np.array(rewards)
    steps = np.array(steps)
    print(f"Average reward across {len(test_seeds)} trajectories: {np.mean(rewards)}")
    print(f"Average number of steps across {len(test_seeds)} trajectories: {np.mean(steps)}")

    # Only plot the first trajectory
    pmf = pmfs[0]
    labels=["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]

    # Stacked line graph
    x = np.arange(len(pmf))
    y = np.asarray(pmf).squeeze().transpose()
    print(x.shape, y.shape)
    plt.stackplot(x, *y, baseline="zero", labels=labels)
    plt.legend()
    plt.show()

    # Separated line graph
    for index, y_i in enumerate(y):
        plt.plot(x, y_i, label=labels[index])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    ############ PARAMETERS ############
    model_type = MODEL_TYPE.DQN

    # filepath = "models/PPO_v2/vd_2_0_trial_1.zip"
    filepath = "models/DQN/checkpoints/dqn_model_10000000_steps.zip"

    vehicle_density = 2.0
    ############ ========== ############

    with open("config.json", "r") as fh:
        config = json.load(fh)

    config["vehicles_density"] = vehicle_density
    print(f"Running with config {config}")

    run(config, model_type, filepath)
