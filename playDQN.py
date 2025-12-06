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
import pygame

from stable_baselines3 import DQN
ACTION_NAMES = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}

PLAYBACK_POLICY_FREQ = 15  # Hz for smooth yet realistic playback
PROB_TEMPERATURE = 0.25    # Softmax temperature for display smoothing


class ProbabilityOverlay:
    """Draw a stylized probability bar chart inside a reserved pygame panel."""

    def __init__(self, n_actions: int, panel_width: int):
        self.n_actions = n_actions
        self.width = panel_width
        pygame.font.init()
        self.title_font = pygame.font.SysFont("Segoe UI Semibold", 20)
        self.label_font = pygame.font.SysFont("Segoe UI", 14)
        
        # Define the new color scheme
        self.color_default = (220, 220, 220)  # Off-white
        self.color_selected = (255, 60, 60)   # Bright Red
        
        self.panel_height = 170
        self.margin = 24
        self.spacing = 14

    def draw(self, target_surface: pygame.Surface, probs: np.ndarray, top_offset: int):
        panel = pygame.Surface((self.width, self.panel_height), pygame.SRCALPHA)
        panel.fill((8, 10, 20, 235))

        max_bar_height = self.panel_height - 80
        base_y = self.panel_height - 45
        total_spacing = self.spacing * (self.n_actions - 1)
        bar_width = max(20, (self.width - (2 * self.margin) - total_spacing) / self.n_actions)
        best_idx = int(np.argmax(probs))

        title_text = self.title_font.render(
            f"DDQN policy — best: {ACTION_NAMES.get(best_idx, best_idx)} ({probs[best_idx]:.2f})",
            True,
            (235, 235, 235),
        )
        panel.blit(title_text, (self.margin, 12))

        for i in range(self.n_actions):
            prob = float(probs[i])
            bar_height = max_bar_height * prob
            x = self.margin + i * (bar_width + self.spacing)
            y = base_y - bar_height
            rect = pygame.Rect(x, y, bar_width, bar_height)

            # Use red for the selected bar, white for others
            bar_color = self.color_selected if i == best_idx else self.color_default
            pygame.draw.rect(panel, bar_color, rect, border_radius=8)

            label = ACTION_NAMES.get(i, f"A{i}")
            label_surface = self.label_font.render(label, True, (210, 210, 210))
            panel.blit(label_surface, (x + (bar_width - label_surface.get_width()) / 2, base_y + 8))

            prob_surface = self.label_font.render(f"{prob * 100:4.1f}%", True, (180, 180, 180))
            panel.blit(prob_surface, (x + (bar_width - prob_surface.get_width()) / 2, y - 24))

        target_surface.blit(panel, (0, top_offset))


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
    playback_policy_freq = PLAYBACK_POLICY_FREQ
    config["simulation_frequency"] = playback_policy_freq * 3
    config["policy_frequency"] = playback_policy_freq
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
    playback_policy_freq = PLAYBACK_POLICY_FREQ
    config["simulation_frequency"] = playback_policy_freq * 3
    config["policy_frequency"] = playback_policy_freq
    config["real_time_rendering"] = False
    config["offscreen_rendering"] = True
    
    print(f"Rendering: real_time={config['real_time_rendering']}, offscreen={config['offscreen_rendering']}")
    print(f"Traffic: vehicles={config.get('vehicles_count', 'default')}, density={config.get('vehicles_density', 'default')}")
    
    env = gym.make("highway-fast-v0", render_mode="rgb_array", config=config)
    
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

    # Determine frame size and prep custom pygame window (env frame + overlay panel)
    frame = env.render()
    frame_height, frame_width = frame.shape[:2]
    overlay = ProbabilityOverlay(n_actions, frame_width)
    total_height = frame_height + overlay.panel_height
    screen = pygame.display.set_mode((frame_width, total_height))
    pygame.display.set_caption("Highway-env — DDQN playback")
    policy_freq = config.get("policy_frequency", PLAYBACK_POLICY_FREQ)
    policy_interval = 1.0 / policy_freq if policy_freq > 0 else 0.0
    render_fps = 60
    render_interval = 1.0 / render_fps
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs_flat = obs.flatten()
        done = truncated = False
        total_reward = 0

        current_probs = np.ones(n_actions, dtype=np.float32) / n_actions
        current_frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        next_action_time = time.perf_counter()
        next_render_time = next_action_time

        while not (done or truncated):
            now = time.perf_counter()

            if policy_interval == 0.0 or now >= next_action_time:
                with torch.no_grad():
                    obs_t = torch.tensor(obs_flat, device=device, dtype=torch.float32).unsqueeze(0)
                    q_vals = policy_net(obs_t)
                    probs = torch.softmax(q_vals / PROB_TEMPERATURE, dim=1).squeeze(0).cpu().numpy()
                    action = int(torch.argmax(q_vals, dim=1).item())

                obs, reward, done, truncated, info = env.step(action)
                obs_flat = obs.flatten()
                total_reward += reward

                frame = env.render()
                current_frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                current_probs = probs

                next_action_time = now + policy_interval if policy_interval > 0 else now

            if now >= next_render_time:
                screen.blit(current_frame_surface, (0, 0))
                overlay.draw(screen, current_probs, frame_height)
                pygame.display.flip()
                pygame.event.pump()
                next_render_time = now + render_interval

            target_time = min(
                next_action_time if policy_interval > 0 else float("inf"),
                next_render_time
            )
            sleep_duration = target_time - now
            if sleep_duration > 0:
                time.sleep(min(sleep_duration, 0.005))

        print(f"Episode {episode + 1}: reward = {total_reward:.2f}")
    
    env.close()
    pygame.display.quit()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ═══════════════════════════════════════════════════════════
    # SETTINGS - Change these to configure playback
    # ═══════════════════════════════════════════════════════════
    USE_DDQN = True               # True = Double DQN, False = SB3 DQN
    CHECKPOINT = None             # Checkpoint timestep (None = use final model)
    NUM_EPISODES = 5              # Number of episodes to play
    
    # Model directory (where checkpoints and config.json are stored)
    MODEL_DIR = "models/DDQN_density2"     # <--- Changed
    
    # Environment overrides (set to None to use config defaults)
    VEHICLES_COUNT = 30           # Number of vehicles (None = use config)
    VEHICLES_DENSITY = 1.5      # <--- Changed to match density2 training
    DURATION = 80                 # Episode duration in seconds (None = use config)
    # ═══════════════════════════════════════════════════════════
    
    if USE_DDQN:
        # Double DQN (custom PyTorch model)
        if CHECKPOINT:
            # Try specific checkpoint path first
            checkpoint_dir = f"{MODEL_DIR}/checkpoints/checkpoint_{CHECKPOINT}_steps"
            model_path = os.path.join(checkpoint_dir, "policy_net.pth")
            
            # If not found, try without "checkpoints" subdir (for manually moved folders)
            if not os.path.exists(model_path):
                 checkpoint_dir = f"{MODEL_DIR}/checkpoint_{CHECKPOINT}_steps"
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
            model_path = f"{MODEL_DIR}"
        
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
