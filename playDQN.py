# playDQN.py
# Play trained DQN models (supports both SB3 DQN and custom Double DQN)
# Usage:
#   python playDQN.py                          # Play SB3 DQN model
#   python playDQN.py --ddqn                   # Play final Double DQN model
#   python playDQN.py --ddqn --checkpoint 600000  # Play Double DQN checkpoint

import gymnasium as gym
import highway_env
import time
import json
import argparse
import os

import torch
import torch.nn as nn
import numpy as np

from stable_baselines3 import DQN


# ─────────────────────────────────────────────────────────────
# Q-Network (must match doubledqntrain.py)
# ─────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# Load config
# ─────────────────────────────────────────────────────────────
def load_config(model_dir):
    """Load config from model directory or fall back to config.json"""
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    elif os.path.exists("config.json"):
        with open("config.json", 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError("No config.json found!")


# ─────────────────────────────────────────────────────────────
# Play SB3 DQN
# ─────────────────────────────────────────────────────────────
def play_sb3_dqn(model_path, config, num_episodes=5):
    print(f"Loading SB3 DQN model from {model_path}")
    
    # Override for playback (MUST override training settings)
    config["simulation_frequency"] = 15
    config["policy_frequency"] = 5
    config["real_time_rendering"] = True
    config["offscreen_rendering"] = False
    
    print(f"Rendering config: real_time={config['real_time_rendering']}, offscreen={config['offscreen_rendering']}")
    
    env = gym.make("highway-fast-v0", render_mode="human", config=config)
    model = DQN.load(model_path)
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = truncated = False
        total_reward = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(0.03)  # ~30 FPS playback
        
        print(f"Episode {episode + 1}: reward = {total_reward:.2f}")
    
    env.close()


# ─────────────────────────────────────────────────────────────
# Play Double DQN
# ─────────────────────────────────────────────────────────────
def play_ddqn(model_path, config, num_episodes=5):
    print(f"Loading Double DQN model from {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Override for playback (MUST override training settings)
    config["simulation_frequency"] = 15
    config["policy_frequency"] = 5
    config["real_time_rendering"] = True  # Enable real-time rendering
    config["offscreen_rendering"] = False  # Disable offscreen
    
    print(f"Rendering: real_time={config['real_time_rendering']}, offscreen={config['offscreen_rendering']}")
    print(f"Traffic: vehicles={config.get('vehicles_count', 'default')}, density={config.get('vehicles_density', 'default')}")
    
    env = gym.make("highway-fast-v0", render_mode="human", config=config)
    
    # Get dimensions
    obs, _ = env.reset()
    obs_flat = obs.flatten()
    obs_dim = obs_flat.shape[0]
    n_actions = env.action_space.n
    
    # Load model
    policy_net = QNetwork(obs_dim, n_actions).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()
    
    print(f"Observation dim: {obs_dim}, Actions: {n_actions}")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs_flat = obs.flatten()
        done = truncated = False
        total_reward = 0
        
        while not (done or truncated):
            with torch.no_grad():
                obs_t = torch.tensor(obs_flat, device=device, dtype=torch.float32).unsqueeze(0)
                q_vals = policy_net(obs_t)
                action = q_vals.argmax(dim=1).item()
            
            obs, reward, done, truncated, info = env.step(action)
            obs_flat = obs.flatten()
            total_reward += reward
            env.render()
        
        print(f"Episode {episode + 1}: reward = {total_reward:.2f}")
    
    env.close()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ═══════════════════════════════════════════════════════════
    # SETTINGS - Change these to configure playback
    # ═══════════════════════════════════════════════════════════
    USE_DDQN = False              # True = Double DQN, False = SB3 DQN
    CHECKPOINT = 10000000          # Checkpoint timestep (None = use final model)
    NUM_EPISODES = 5             # Number of episodes to play
    
    # Model directory (where checkpoints and config.json are stored)
    MODEL_DIR = "highway_second_try"
    
    # Environment overrides (set to None to use config defaults)
    VEHICLES_COUNT = 30          # Number of vehicles (None = use config)
    VEHICLES_DENSITY = 1.0     # Traffic density (None = use config)
    DURATION = 40              # Episode duration in seconds (None = use config)
    # ═══════════════════════════════════════════════════════════
    
    if USE_DDQN:
        # Double DQN (custom PyTorch model)
        if CHECKPOINT:
            checkpoint_dir = f"{MODEL_DIR}/checkpoints/checkpoint_{CHECKPOINT}_steps"
            model_path = os.path.join(checkpoint_dir, "policy_net.pth")
        else:
            model_path = os.path.join(MODEL_DIR, "policy_net.pth")
        
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Available checkpoints:")
            checkpoints_dir = f"{MODEL_DIR}/checkpoints"
            if os.path.exists(checkpoints_dir):
                for d in sorted(os.listdir(checkpoints_dir)):
                    print(f"  - {d}")
            exit(1)
        
        config = load_config(MODEL_DIR)
    else:
        # SB3 DQN
        if CHECKPOINT:
            model_path = f"{MODEL_DIR}/checkpoints/dqn_model_{CHECKPOINT}_steps"
        else:
            model_path = f"{MODEL_DIR}/model"
        
        if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Available checkpoints:")
            checkpoints_dir = f"{MODEL_DIR}/checkpoints"
            if os.path.exists(checkpoints_dir):
                for f in sorted(os.listdir(checkpoints_dir)):
                    if f.endswith(".zip"):
                        print(f"  - {f}")
            exit(1)
        
        config = load_config(MODEL_DIR)
        
    # Apply environment overrides AFTER loading config
    if VEHICLES_COUNT is not None:
        config["vehicles_count"] = VEHICLES_COUNT
    if VEHICLES_DENSITY is not None:
        config["vehicles_density"] = VEHICLES_DENSITY
    if DURATION is not None:
        config["duration"] = DURATION
    
    # Play the model
    if USE_DDQN:
        play_ddqn(model_path, config, NUM_EPISODES)
    else:
        play_sb3_dqn(model_path, config, NUM_EPISODES)
