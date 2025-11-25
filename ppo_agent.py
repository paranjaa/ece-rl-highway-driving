import gymnasium as gym
from gymnasium import spaces
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback


MODEL_SAVE_PATH = "./models/PPO/agent.pth"


def set_up_env(config):
    env = gym.make(
        "highway-v0",
        config=config,
        render_mode="human"
    )

    return env


def set_up_model(env):
    model = PPO(
        policy="MlpPolicy",
        env=env,
        #device="cpu",
        seed=1508,
        verbose=1,
    )

    return model


def run(env, model, train=True):
    if train:
        print("Training new model")
        trained_model = model.learn(total_timesteps=5000, progress_bar=True)
        print(f"Saving trained model to {MODEL_SAVE_PATH}")
        trained_model.save(MODEL_SAVE_PATH)
    else:
        print(f"Loading trained model from: {MODEL_SAVE_PATH}")
        trained_model = model.load(MODEL_SAVE_PATH)

    print("Testing model")
    state = env.reset(seed=1508)

    ended = False
    truncated = False
    num_steps = 0
    episode_reward = 0


    while not ended and not truncated:
        action = trained_model.predict(state)
        next_state, reward, ended, truncated, _ = env.step(action)

        state = next_state
        episode_reward += reward
        num_steps += 1

        env.render()

    print(f"Episode reward: {episode_reward} \t Steps: {num_steps}")

    env.close()

if __name__ == "__main__":
    config = {
        "lanes_count": 5,
        "vehicles_count": 50,
        "vehicles_density": 2.0,
        "initial_spacing": 10,
        "offroad_terminal": True,
        "action": {
            "type": "ContinuousAction",
        },
        "observation:": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "absolute": False,
            "order": "sorted"
        }
    }

    env = set_up_env(config)
    model = set_up_model(env)

    run(env, model, train=True)