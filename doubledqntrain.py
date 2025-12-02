# doubledqntrain.py
# Double DQN implementation from scratch for highway-env
# Uses config.json for environment configuration
# Vectorized with SubprocVecEnv for parallel environment stepping
#
# Logging:
#   - Training metrics (rewards, lengths, losses) are saved to training_logs.npz
#     every LOG_DUMP_INTERVAL steps (default: 10k steps) and at the end of training.
#     This allows monitoring progress during long training runs without waiting
#     until completion. All history is preserved in each dump.
#
# Performance:
#   - offscreen_rendering is forced to True to disable GUI rendering overhead
#     and improve training speed. Set env_config["offscreen_rendering"] = False
#     in the code if you need visual rendering for debugging.

import gymnasium as gym
import numpy as np
import highway_env
import os
import json
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from stable_baselines3.common.vec_env import SubprocVecEnv

# ─────────────────────────────────────────────────────────────
# Load config from config.json
# ─────────────────────────────────────────────────────────────
with open('config.json', 'r') as f:
    config = json.load(f)

# ─────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────
N_ENV = 9
BUFFER_SIZE = 150_000
BATCH_SIZE = 256
GAMMA = 0.99
LR = 5e-4
TARGET_UPDATE_INTERVAL = 1000
LEARNING_STARTS = 20000
TRAIN_FREQ = 4
EXPLORATION_FRACTION = 0.5   # 50% of 10M steps -> 5M exploration steps
EXPLORATION_INITIAL_EPS = 0.8
EXPLORATION_FINAL_EPS = 0.08
TOTAL_TIMESTEPS = 10_000_000
LOG_INTERVAL = 50  # log every N episodes
CHECKPOINT_INTERVAL = 100_000  # save every 100k steps
LOG_DUMP_INTERVAL = 10_000  # dump metrics to disk every N steps
SAVE_DIR = "doubledqn_density2"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────
# Q-Network
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
# Replay Buffer
# ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Push a single transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def push_batch(self, states, actions, rewards, next_states, dones):
        """Push a batch of transitions (from vectorized env)."""
        for i in range(len(states)):
            self.buffer.append((states[i], actions[i], rewards[i], next_states[i], dones[i]))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────
# Epsilon schedule
# ─────────────────────────────────────────────────────────────
def get_epsilon(timestep: int, start_timestep: int = 0, fixed_epsilon: float = None) -> float:
    """Get epsilon value, accounting for resumed training."""
    if fixed_epsilon is not None:
        return fixed_epsilon
    effective_timestep = timestep - start_timestep
    fraction = min(1.0, effective_timestep / (EXPLORATION_FRACTION * TOTAL_TIMESTEPS))
    return EXPLORATION_INITIAL_EPS + fraction * (EXPLORATION_FINAL_EPS - EXPLORATION_INITIAL_EPS)


# ─────────────────────────────────────────────────────────────
# Double DQN update
# ─────────────────────────────────────────────────────────────
def update(
    policy_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    replay_buffer: ReplayBuffer,
):
    states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

    states_t = torch.tensor(states, device=DEVICE)
    actions_t = torch.tensor(actions, device=DEVICE).unsqueeze(1)
    rewards_t = torch.tensor(rewards, device=DEVICE)
    next_states_t = torch.tensor(next_states, device=DEVICE)
    dones_t = torch.tensor(dones, device=DEVICE)

    # Current Q values
    q_values = policy_net(states_t).gather(1, actions_t).squeeze(1)

    # Double DQN: use policy_net to select actions, target_net to evaluate
    with torch.no_grad():
        # Action selection with policy network
        next_actions = policy_net(next_states_t).argmax(dim=1, keepdim=True)
        # Action evaluation with target network
        next_q_values = target_net(next_states_t).gather(1, next_actions).squeeze(1)
        # Bellman target
        target_q_values = rewards_t + GAMMA * next_q_values * (1 - dones_t)

    # Loss
    loss = F.mse_loss(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()

    return loss.item()


# ─────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────
def make_env(rank: int, env_config: dict):
    def _init():
        env = gym.make("highway-fast-v0", render_mode=None, config=env_config)
        return env
    return _init


# ─────────────────────────────────────────────────────────────
# Checkpoint saving
# ─────────────────────────────────────────────────────────────
def save_checkpoint(policy_net, target_net, timestep, save_dir, env_config):
    """Save model checkpoint with config."""
    checkpoint_path = os.path.join(save_dir, f"checkpoint_{timestep}_steps")
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save(policy_net.state_dict(), os.path.join(checkpoint_path, "policy_net.pth"))
    torch.save(target_net.state_dict(), os.path.join(checkpoint_path, "target_net.pth"))
    # Save config with checkpoint
    with open(os.path.join(checkpoint_path, "config.json"), "w") as f:
        json.dump(env_config, f, indent=2)
    print(f"Checkpoint saved at {timestep} steps")


# ─────────────────────────────────────────────────────────────
# Load checkpoint
# ─────────────────────────────────────────────────────────────
def load_checkpoint(checkpoint_path, policy_net, target_net):
    """Load model from checkpoint."""
    policy_path = os.path.join(checkpoint_path, "policy_net.pth")
    target_path = os.path.join(checkpoint_path, "target_net.pth")
    
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Checkpoint not found: {policy_path}")
    
    policy_net.load_state_dict(torch.load(policy_path, map_location=DEVICE))
    target_net.load_state_dict(torch.load(target_path, map_location=DEVICE))
    print(f"Loaded checkpoint from {checkpoint_path}")


# ─────────────────────────────────────────────────────────────
# Save training logs
# ─────────────────────────────────────────────────────────────
def save_training_logs(save_dir, episode_rewards, episode_lengths, losses):
    """Save training metrics to disk."""
    log_path = os.path.join(save_dir, "training_logs.npz")
    np.savez(
        log_path,
        episode_rewards=np.array(episode_rewards),
        episode_lengths=np.array(episode_lengths),
        losses=np.array(losses),
    )


# ─────────────────────────────────────────────────────────────
# Main training loop (vectorized)
# ─────────────────────────────────────────────────────────────
def main(resume_checkpoint: int = None, start_timestep: int = 0, config_overrides: dict = None, fixed_epsilon: float = None):
    """
    Main training function.
    
    Args:
        resume_checkpoint: Checkpoint timestep to resume from (e.g., 800000)
        start_timestep: Starting timestep for logging (usually same as resume_checkpoint)
        config_overrides: Dict of config values to override
        fixed_epsilon: If set, use this fixed epsilon value instead of the schedule
    """
    # Apply config overrides
    env_config = config.copy()
    if config_overrides:
        for key, value in config_overrides.items():
            env_config[key] = value
            print(f"Config override: {key} = {value}")
    
    # Force offscreen rendering for performance (disables GUI rendering overhead)
    env_config["offscreen_rendering"] = True
    
    os.makedirs(f"{SAVE_DIR}/logs", exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(f"{SAVE_DIR}/checkpoints", exist_ok=True)

    # Create vectorized environment
    print(f"Creating {N_ENV} parallel environments...")
    vec_env = SubprocVecEnv([make_env(i, env_config) for i in range(N_ENV)])

    # Get observation and action dimensions from a single env
    single_env = gym.make("highway-fast-v0", render_mode=None, config=env_config)
    obs_sample, _ = single_env.reset()
    obs_flat = obs_sample.flatten()
    obs_dim = obs_flat.shape[0]
    n_actions = single_env.action_space.n
    single_env.close()

    print(f"Observation dim: {obs_dim}, Number of actions: {n_actions}")
    print(f"Using device: {DEVICE}")

    # Networks
    policy_net = QNetwork(obs_dim, n_actions).to(DEVICE)
    target_net = QNetwork(obs_dim, n_actions).to(DEVICE)
    
    # Load checkpoint if resuming
    if resume_checkpoint is not None:
        checkpoint_path = f"{SAVE_DIR}/checkpoints/checkpoint_{resume_checkpoint}_steps"
        load_checkpoint(checkpoint_path, policy_net, target_net)
        print(f"Resuming training from timestep {start_timestep}")
    else:
        target_net.load_state_dict(policy_net.state_dict())
    
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    # Logging
    episode_rewards = []
    episode_lengths = []
    losses = []

    # Per-env episode tracking
    env_episode_rewards = np.zeros(N_ENV)
    env_episode_lengths = np.zeros(N_ENV, dtype=np.int32)

    # Training state
    timestep = start_timestep
    episode = 0
    last_checkpoint = start_timestep
    last_log_dump = start_timestep

    # Initial reset
    obs = vec_env.reset()  # shape: (N_ENV, vehicles_count, features)
    obs_flat = obs.reshape(N_ENV, -1)  # shape: (N_ENV, obs_dim)

    end_timestep = start_timestep + TOTAL_TIMESTEPS
    print(f"Starting training from {start_timestep} to {end_timestep} timesteps...")
    print(f"Environment config: vehicles={env_config.get('vehicles_count')}, density={env_config.get('vehicles_density')}, high_speed_reward={env_config.get('high_speed_reward')}")

    while timestep < end_timestep:
        # Epsilon-greedy action selection for all envs
        epsilon = get_epsilon(timestep, start_timestep, fixed_epsilon)
        
        # Get actions for all envs
        if random.random() < epsilon:
            # Random actions for all envs
            actions = np.array([vec_env.action_space.sample() for _ in range(N_ENV)])
        else:
            # Greedy actions from policy network
            with torch.no_grad():
                obs_t = torch.tensor(obs_flat, device=DEVICE, dtype=torch.float32)
                q_vals = policy_net(obs_t)
                actions = q_vals.argmax(dim=1).cpu().numpy()

        # Step all environments
        next_obs, rewards, dones, infos = vec_env.step(actions)
        next_obs_flat = next_obs.reshape(N_ENV, -1)

        # Store transitions for all envs
        replay_buffer.push_batch(
            obs_flat, 
            actions, 
            rewards, 
            next_obs_flat, 
            dones.astype(np.float32)
        )

        # Update per-env episode stats
        env_episode_rewards += rewards
        env_episode_lengths += 1
        timestep += N_ENV  # We took N_ENV steps

        # Check for done envs
        for i in range(N_ENV):
            if dones[i]:
                episode_rewards.append(env_episode_rewards[i])
                episode_lengths.append(env_episode_lengths[i])
                episode += 1
                
                # Reset tracking for this env
                env_episode_rewards[i] = 0
                env_episode_lengths[i] = 0

                # LOGGING: Moved inside the 'done' check
                if episode > 0 and episode % LOG_INTERVAL == 0 and len(episode_rewards) >= LOG_INTERVAL:
                    mean_reward = np.mean(episode_rewards[-LOG_INTERVAL:])
                    mean_length = np.mean(episode_lengths[-LOG_INTERVAL:])
                    mean_loss = np.mean(losses[-1000:]) if losses else 0.0
                    print(
                        f"Episode {episode} | "
                        f"Timestep {timestep:,} | "
                        f"Mean Reward: {mean_reward:.2f} | "
                        f"Mean Length: {mean_length:.1f} | "
                        f"Epsilon: {epsilon:.3f} | "
                        f"Loss: {mean_loss:.4f} | "
                        f"Buffer: {len(replay_buffer):,}"
                    )

        # Move to next state
        obs_flat = next_obs_flat

        # Train (multiple gradient steps to compensate for vectorized collection)
        if timestep >= start_timestep + LEARNING_STARTS and len(replay_buffer) >= BATCH_SIZE:
            # Do more gradient steps since we collect N_ENV transitions per iteration
            n_updates = max(1, N_ENV // TRAIN_FREQ)
            for _ in range(n_updates):
                loss = update(policy_net, target_net, optimizer, replay_buffer)
                losses.append(loss)

        # Update target network
        if timestep // TARGET_UPDATE_INTERVAL > (timestep - N_ENV) // TARGET_UPDATE_INTERVAL:
            target_net.load_state_dict(policy_net.state_dict())

        # Save checkpoint every 100k steps
        if timestep // CHECKPOINT_INTERVAL > last_checkpoint // CHECKPOINT_INTERVAL:
            checkpoint_step = (timestep // CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL
            save_checkpoint(policy_net, target_net, checkpoint_step, f"{SAVE_DIR}/checkpoints", env_config)
            last_checkpoint = timestep
        
        # Dump training logs periodically (every LOG_DUMP_INTERVAL steps)
        if timestep - last_log_dump >= LOG_DUMP_INTERVAL:
            save_training_logs(SAVE_DIR, episode_rewards, episode_lengths, losses)
            last_log_dump = timestep

    vec_env.close()

    # Save final model
    torch.save(policy_net.state_dict(), f"{SAVE_DIR}/policy_net.pth")
    torch.save(target_net.state_dict(), f"{SAVE_DIR}/target_net.pth")
    print(f"Model saved to {SAVE_DIR}/")

    # Save config alongside the model
    with open(f"{SAVE_DIR}/config.json", "w") as f:
        json.dump(env_config, f, indent=2)
    print(f"Config saved to {SAVE_DIR}/config.json")

    # Save final training logs
    save_training_logs(SAVE_DIR, episode_rewards, episode_lengths, losses)
    print(f"Final training logs saved to {SAVE_DIR}/training_logs.npz")


if __name__ == "__main__":
    # ═══════════════════════════════════════════════════════════
    # TRAINING SETTINGS - Change these to configure training
    # ═══════════════════════════════════════════════════════════
    
    # Resume from checkpoint (set to None to train from scratch)
    RESUME_CHECKPOINT = None           # <--- Changed
    START_TIMESTEP = 0                 # <--- Changed
    TOTAL_TIMESTEPS = 10_000_000       # Train for 10M steps
    FIXED_EPSILON = None               # <--- Changed to use schedule
    
    # Config overrides (set to None to use config.json defaults)
    CONFIG_OVERRIDES = {
        "collision_reward": -20.0,
        "vehicles_density": 2.0,       # <--- Added
    }
    # ═══════════════════════════════════════════════════════════
    
    main(
        resume_checkpoint=RESUME_CHECKPOINT,
        start_timestep=START_TIMESTEP,
        config_overrides=CONFIG_OVERRIDES,
        fixed_epsilon=FIXED_EPSILON,
    )
