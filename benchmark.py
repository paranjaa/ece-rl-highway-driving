"""
benchmark.py

Evaluate the trained SB3 DQN policy across multiple highway-env configs.
The model is loaded from the 10M-step checkpoint used in playDQN.py.
For each config we run 50 evaluation episodes using a 4-env SubprocVecEnv
without rendering, and report:
    • Average episode reward
    • Collision rate (% of episodes that ended in a crash)
    • Average ego speed (m/s) averaged over each episode
"""

import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import highway_env  # noqa: F401 (register env)
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv

# ─────────────────────────────────────────────────────────────
# Paths / constants
# ─────────────────────────────────────────────────────────────
BASE_CONFIG_PATH = "config.json"
MODEL_DIR = "highway_second_try"
CHECKPOINT_STEPS = 10_000_000
MODEL_PATH = os.path.join(MODEL_DIR, f"checkpoints/dqn_model_{CHECKPOINT_STEPS}_steps")

N_ENVS = 4
EPISODES_PER_CONFIG = 50

ACTION_NAMES = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def load_base_config() -> Dict:
    with open(BASE_CONFIG_PATH, "r") as f:
        return json.load(f)


def make_vec_env(run_config: Dict) -> SubprocVecEnv:
    def make_env(rank: int):
        def _init():
            cfg = copy.deepcopy(run_config)
            cfg["offscreen_rendering"] = True
            cfg["real_time_rendering"] = False
            env = gym.make("highway-fast-v0", render_mode=None, config=cfg)
            return env

        return _init

    return SubprocVecEnv([make_env(i) for i in range(N_ENVS)])


def maybe_get_speed(info: Dict, obs_chunk) -> float:
    if isinstance(info, dict) and "speed" in info:
        return float(info["speed"])
    # Fallback: estimate from observation (ego vehicle is first row)
    if obs_chunk is not None and len(obs_chunk) > 0:
        ego = obs_chunk[0]
        if len(ego) >= 4:
            vx, vy = ego[2], ego[3]
            return float(np.hypot(vx, vy))
    return 0.0


def evaluate_config(model: DQN, run_config: Dict, label: str) -> Tuple[float, float, float, np.ndarray, int]:
    env = make_vec_env(run_config)

    reset_out = env.reset()
    if isinstance(reset_out, tuple):
        obs, _info = reset_out
    else:
        obs = reset_out

    obs = np.array(obs)
    n_actions = int(env.action_space.n)
    action_counts = np.zeros(n_actions, dtype=np.int64)

    ep_returns: List[float] = []
    ep_avg_speeds: List[float] = []
    ep_collisions: List[bool] = []

    curr_return = np.zeros(N_ENVS, dtype=np.float32)
    curr_steps = np.zeros(N_ENVS, dtype=np.int32)
    curr_speed_sum = np.zeros(N_ENVS, dtype=np.float32)
    curr_collision = np.zeros(N_ENVS, dtype=bool)

    completed_eps = 0

    while completed_eps < EPISODES_PER_CONFIG:
        actions, _ = model.predict(obs, deterministic=True)
        for action in np.asarray(actions).reshape(-1):
            action_counts[int(action)] += 1
        step_out = env.step(actions)
        if len(step_out) == 4:
            next_obs, rewards, dones, infos = step_out
        else:
            next_obs, rewards, dones, _, infos = step_out  # gymnasium compatibility

        rewards = np.array(rewards)
        dones = np.array(dones)
        next_obs = np.array(next_obs)

        curr_return += rewards
        curr_steps += 1

        # infos is a list of dicts per env
        for idx, info in enumerate(infos):
            ego_obs = next_obs[idx]
            speed = maybe_get_speed(info, ego_obs)
            curr_speed_sum[idx] += speed
            if isinstance(info, dict) and info.get("crashed", False):
                curr_collision[idx] = True

        for idx, done in enumerate(dones):
            if done:
                avg_speed = curr_speed_sum[idx] / max(curr_steps[idx], 1)
                ep_returns.append(float(curr_return[idx]))
                ep_avg_speeds.append(float(avg_speed))
                ep_collisions.append(bool(curr_collision[idx]))

                curr_return[idx] = 0.0
                curr_steps[idx] = 0
                curr_speed_sum[idx] = 0.0
                curr_collision[idx] = False

                completed_eps += 1
                if completed_eps >= EPISODES_PER_CONFIG:
                    break

        obs = next_obs

    env.close()

    avg_return = float(np.mean(ep_returns))
    collision_rate = float(np.mean(ep_collisions) * 100.0)
    avg_speed = float(np.mean(ep_avg_speeds))
    total_actions = action_counts.sum()
    return avg_return, collision_rate, avg_speed, action_counts, total_actions


def build_configs(base_cfg: Dict) -> List[Tuple[str, Dict]]:
    configs = []
    densities = [1.0, 1.5, 2.0]

    # Duration 40 first (as base), then duration 20 variants
    for density in densities:
        cfg = copy.deepcopy(base_cfg)
        cfg["vehicles_density"] = density
        cfg["duration"] = base_cfg.get("duration", 40)
        cfg["real_time_rendering"] = False
        label = f"density={density:.1f}, duration={cfg['duration']}s"
        configs.append((label, cfg))

    for density in densities:
        cfg = copy.deepcopy(base_cfg)
        cfg["vehicles_density"] = density
        cfg["duration"] = 20
        cfg["real_time_rendering"] = False
        label = f"density={density:.1f}, duration=20s"
        configs.append((label, cfg))

    return configs


if __name__ == "__main__":
    base_config = load_base_config()
    configs = build_configs(base_config)

    if not os.path.exists(MODEL_PATH + ".zip") and not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}(.zip)")

    model = DQN.load(MODEL_PATH, device="cuda")

    summary = []
    for label, cfg in configs:
        metrics = evaluate_config(model, cfg, label)
        summary.append((label, *metrics))

    print("=" * 70)
    print("Highway DQN Benchmark Summary (50 eps/config, n_env=4)")
    print(f"Model checkpoint: {MODEL_PATH}")
    print("=" * 70)
    for label, avg_ret, coll_rate, avg_speed, action_counts, total_actions in summary:
        print(f"{label}")
        print(f"  Avg Reward     : {avg_ret:8.3f}")
        print(f"  Collision Rate : {coll_rate:8.2f}%")
        print(f"  Avg Speed      : {avg_speed:8.3f} m/s")
        print(f"  Action distribution (total actions = {total_actions}):")
        for idx, count in enumerate(action_counts):
            name = ACTION_NAMES.get(idx, f"A{idx}")
            pct = (count / total_actions * 100.0) if total_actions > 0 else 0.0
            print(f"      [{idx}] {name:10s}: {pct:6.2f}% ({count} actions)")
        print("-" * 60)
